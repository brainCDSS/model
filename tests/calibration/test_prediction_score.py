import json,math
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from brainon_ai.calibration.features import dice,extract_features
from brainon_ai.calibration.scoring import calculate_prediction_score

FEATURES=("foreground_top10_mean","foreground_uncertainty","spatial_coherence","tta_consistency")
def artifact(mean=None,std=None,coef=None,intercept=0,version='v1'):
 mean=mean or [0]*4; std=std or [1]*4; coef=coef or [1]*4
 return {'model':{'version':version},'intercept':intercept,'features':[{'name':n,'mean':mean[i],'std':std[i],'coefficient':coef[i]} for i,n in enumerate(FEATURES)]}
def test_feature_calculation_and_connected_components():
 p=np.zeros((4,4,4),np.float32); p[0,0,0]=.9; p[0,0,1]=.8; p[3,3,3]=.6
 f=extract_features(p,.5,[p]); assert f['foreground_top10_mean']==pytest.approx(.9); assert f['spatial_coherence']==pytest.approx(2/3); assert f['tta_consistency']==1
def test_empty_foreground_policy():
 p=np.zeros((3,3,3)); f=extract_features(p,.5,[p]); assert f=={'foreground_top10_mean':0.,'foreground_uncertainty':0.,'spatial_coherence':0.,'tta_consistency':1.}
def test_entropy_clipping_is_finite():
 p=np.ones((2,2,2)); assert math.isfinite(extract_features(p,.5,[p])['foreground_uncertainty'])
def test_tta_inverse_flip():
 x=np.zeros((4,3,2),bool); x[0]=1; assert dice(x,np.flip(np.flip(x,0),0))==1
def test_missing_tta_rejected():
 with pytest.raises(ValueError): extract_features(np.ones((2,2,2)),.5,[])
def test_json_roundtrip_and_determinism():
 a=json.loads(json.dumps(artifact())); f=dict.fromkeys(FEATURES,.2); assert calculate_prediction_score(f,a)==calculate_prediction_score(f,a)
def test_missing_feature_error():
 with pytest.raises(KeyError): calculate_prediction_score({},artifact())
def test_zero_std_safe():
 assert math.isfinite(calculate_prediction_score(dict.fromkeys(FEATURES,1.),artifact(std=[0,0,0,0])))
def test_sigmoid_overflow_safe():
 f=dict.fromkeys(FEATURES,1e300); assert calculate_prediction_score(f,artifact())==100.; assert calculate_prediction_score(f,artifact(coef=[-1]*4))==0.
def test_model_version_mismatch():
 with pytest.raises(ValueError): calculate_prediction_score(dict.fromkeys(FEATURES,0.),artifact(),expected_model_version='v2')
def test_sklearn_json_reference_parity():
 rng=np.random.default_rng(2); X=rng.normal(size=(50,4)); y=(X[:,0]-X[:,1]+.2*X[:,2]>0).astype(int); sc=StandardScaler().fit(X); lr=LogisticRegression(max_iter=1000).fit(sc.transform(X),y); a=artifact(sc.mean_.tolist(),sc.scale_.tolist(),lr.coef_[0].tolist(),float(lr.intercept_[0])); ref=np.array([calculate_prediction_score(dict(zip(FEATURES,r)),a)/100 for r in X]); assert np.allclose(ref,lr.predict_proba(sc.transform(X))[:,1],atol=1e-12)
