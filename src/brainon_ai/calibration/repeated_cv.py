from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from .pipeline import _ece,_threshold_report,_write_plot
from .scoring import calculate_prediction_score

def _transform(x,name):
    x=np.asarray(x,float)
    if name=='identity': return x
    if name=='log1p': return np.log1p(np.maximum(0,x))
    raise ValueError(name)

def _fit(X,y,constraints,seed):
    if not constraints:
        m=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed).fit(X,y)
        return m.coef_[0],float(m.intercept_[0])
    n=X.shape[1]; counts=np.bincount(y,minlength=2); weights=np.where(y==1,len(y)/(2*counts[1]),len(y)/(2*counts[0]))
    def objective(v):
        z=v[0]+X@v[1:]; loss=np.logaddexp(0,z)-y*z
        return float(np.mean(weights*loss)+0.5*np.sum(v[1:]**2)/len(y))
    bounds=[(None,None)]+[(0,None) if constraints.get(i)==1 else (None,0) if constraints.get(i)==-1 else (None,None) for i in range(n)]
    result=minimize(objective,np.zeros(n+1),method='L-BFGS-B',bounds=bounds)
    if not result.success: raise RuntimeError(result.message)
    return result.x[1:],float(result.x[0])

def run_repeated_cv(frame:pd.DataFrame, specs:list[dict], output:Path, model_id:str, checkpoint_sha256:str|None, target:dict, constrained=False, repeats=10, seed=42):
    names=[x['name'] for x in specs]; X=np.column_stack([_transform(frame[x['name']],x.get('transform','identity')) for x in specs]); y=frame.acceptable_segmentation.to_numpy(int)
    counts=np.bincount(y,minlength=2); folds=min(5,int(counts.min()))
    if folds<2: raise RuntimeError(f'insufficient classes: {counts.tolist()}')
    cv=RepeatedStratifiedKFold(n_splits=folds,n_repeats=repeats,random_state=seed); sums=np.zeros(len(y)); seen=np.zeros(len(y)); coefficients=[]; fold_rows=[]
    constraints={i:x.get('sign') for i,x in enumerate(specs) if x.get('sign') in (-1,1)} if constrained else {}
    for fold,(train,val) in enumerate(cv.split(X,y)):
        scaler=StandardScaler().fit(X[train]); Xt=scaler.transform(X[train]); Xv=scaler.transform(X[val]); coef,intercept=_fit(Xt,y[train],constraints,seed+fold); prob=1/(1+np.exp(-(intercept+Xv@coef))); sums[val]+=prob;seen[val]+=1;coefficients.append(coef)
        if len(np.unique(y[val]))==2: fold_rows.append({'fold':fold,'auroc':roc_auc_score(y[val],prob),'auprc':average_precision_score(y[val],prob),'brier':brier_score_loss(y[val],prob),'ece':_ece(y[val],prob)})
    oof=sums/seen; threshold,criterion,confusions=_threshold_report(y,oof); pred=(oof>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    metrics={'reliability_auroc':float(roc_auc_score(y,oof)),'reliability_auprc':float(average_precision_score(y,oof)),'reliability_sensitivity':float(recall_score(y,pred,zero_division=0)),'reliability_specificity':float(recall_score(1-y,1-pred,zero_division=0)),'reliability_f1':float(f1_score(y,pred,zero_division=0)),'reliability_ppv':float(precision_score(y,pred,zero_division=0)),'reliability_npv':float(precision_score(1-y,1-pred,zero_division=0)),'reliability_brier':float(brier_score_loss(y,oof)),'reliability_ece':_ece(y,oof)}
    fold_df=pd.DataFrame(fold_rows); metrics['fold_metrics']={c:{'mean':float(fold_df[c].mean()),'std':float(fold_df[c].std(ddof=1))} for c in ['auroc','auprc','brier','ece']}
    co=np.asarray(coefficients); stability=pd.DataFrame([{'feature':n,'coefficient_mean':float(co[:,i].mean()),'coefficient_std':float(co[:,i].std(ddof=1)),'coefficient_min':float(co[:,i].min()),'coefficient_max':float(co[:,i].max()),'positive_sign_ratio':float((co[:,i]>1e-10).mean()),'negative_sign_ratio':float((co[:,i]<-1e-10).mean())} for i,n in enumerate(names)])
    scaler=StandardScaler().fit(X); coef,intercept=_fit(scaler.transform(X),y,constraints,seed); artifact={'schema_version':'1.0.0','score_version':'1.0.0','score_type':'segmentation_reliability','method':'sign_constrained_logistic_regression' if constrained else 'logistic_regression','model_id':model_id,'features':[{'name':x['name'],'transform':x.get('transform','identity'),'mean':float(scaler.mean_[i]),'std':float(scaler.scale_[i]),'coefficient':float(coef[i])} for i,x in enumerate(specs)],'intercept':intercept,'target':{'name':'acceptable_segmentation','definition_version':str(target.get('version','1.0.0')),'status':target.get('status','MISSING')},'training':{'case_count':len(y),'acceptable_cases':int(counts[1]),'unacceptable_cases':int(counts[0]),'acceptable_ratio':float(counts[1]/len(y)),'unacceptable_ratio':float(counts[0]/len(y)),'cv_method':'repeated_stratified_kfold','cv_folds':folds,'cv_repeats':repeats},'segmentation_model':{'model_id':model_id,'checkpoint_sha256':checkpoint_sha256},'threshold_selection':{'selected_threshold':threshold,'criterion':criterion,'selected_on':'mean_repeated_oof'},'metrics':metrics,'deployment_eligible':False}
    runtime=np.array([calculate_prediction_score({n:r[n] for n in names},artifact)/100 for _,r in frame.iterrows()]); direct=1/(1+np.exp(-(intercept+scaler.transform(X)@coef)))
    if not np.allclose(runtime,direct,atol=1e-10): raise AssertionError('runtime parity failed')
    output.mkdir(parents=True,exist_ok=True); result=frame.copy();result['repeated_oof_probability']=oof;result['prediction_score']=100*oof;result.to_csv(output/'score_features.csv',index=False);stability.to_csv(output/'coefficient_stability.csv',index=False);fold_df.to_csv(output/'cv_fold_metrics.csv',index=False);confusions.to_csv(output/'confusion_matrix_by_threshold.csv',index=False)
    (output/'score_model.json').write_text(json.dumps(artifact,indent=2)+'\n'); (output/'score_evaluation.json').write_text(json.dumps({'metrics':metrics,'class_distribution':artifact['training'],'threshold_selection':artifact['threshold_selection']},indent=2)+'\n')
    definitions={'foreground_uncertainty':'Mean binary entropy over predicted foreground voxels.','tta_consistency':'Mean Dice between original and inverse-transformed TTA masks.','spatial_coherence':'Largest component fraction of predicted foreground.','predicted_component_count':'26-connected predicted component count.','component_probability_std':'SD of component mean probabilities.','ensemble_disagreement':'Mean voxelwise SD across component probabilities inside final ensemble foreground.','largest_component_confidence':'Mean or top-10% probability within largest predicted component.','fp_like_component_count':'Count of components satisfying the configured small/low-confidence rule.'}
    schema={'schema_version':'1.0.0','model_id':model_id,'features':[{'name':x['name'],'definition':definitions[x['name']],'dtype':'float','required':True,'transform':x.get('transform','identity'),'empty_prediction_behavior':0.0} for x in specs]};(output/'score_feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    edges=np.linspace(0,1,min(10,len(y))+1); rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        z=(oof>=lo)&((oof<hi) if hi<1 else (oof<=hi))
        if z.any(): rows.append({'bin_lower':lo,'bin_upper':hi,'predicted_probability':oof[z].mean(),'observed_success_rate':y[z].mean(),'case_count':z.sum()})
    cal=pd.DataFrame(rows);cal.to_csv(output/'calibration_curve.csv',index=False);_write_plot(cal,'predicted_probability','observed_success_rate',output/'calibration_curve.png','Repeated-CV calibration')
    thresholds=np.r_[np.inf,np.sort(np.unique(oof))[::-1],-np.inf];roc=pd.DataFrame({'threshold':thresholds,'sensitivity':[recall_score(y,oof>=t,zero_division=0) for t in thresholds],'specificity':[recall_score(1-y,oof<t,zero_division=0) for t in thresholds]});roc.to_csv(output/'roc_curve.csv',index=False);_write_plot(roc,'specificity','sensitivity',output/'roc_curve.png','Repeated-CV ROC')
    pts=np.sort(np.unique(oof))[::-1];pr=pd.DataFrame({'threshold':pts,'precision':[precision_score(y,oof>=t,zero_division=0) for t in pts],'recall':[recall_score(y,oof>=t,zero_division=0) for t in pts]});pr.to_csv(output/'pr_curve.csv',index=False);_write_plot(pr,'recall','precision',output/'pr_curve.png','Repeated-CV PR')
    return artifact,stability
