import json,math
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from brainon_ai.calibration.features import dice,extract_features,ensemble_disagreement,component_reliability_features
from brainon_ai.calibration.scoring import calculate_prediction_score

FEATURES=("foreground_top10_mean","foreground_uncertainty","tta_consistency","spatial_coherence","component_probability_std","predicted_component_count")
def artifact(mean=None,std=None,coef=None,intercept=0,version='v1'):
 mean=mean or [0]*len(FEATURES); std=std or [1]*len(FEATURES); coef=coef or [1]*len(FEATURES)
 return {'model':{'version':version},'intercept':intercept,'features':[{'name':n,'mean':mean[i],'std':std[i],'coefficient':coef[i]} for i,n in enumerate(FEATURES)]}
def test_feature_calculation_and_connected_components():
 p=np.zeros((4,4,4),np.float32); p[0,0,0]=.9; p[0,0,1]=.8; p[3,3,3]=.6
 f=extract_features(p,.5,[p]); assert f['foreground_top10_mean']==pytest.approx(.9); assert f['spatial_coherence']==pytest.approx(2/3); assert f['tta_consistency']==1; assert f['predicted_component_count']==2; assert f['component_probability_std']>0
def test_empty_foreground_policy():
 p=np.zeros((3,3,3)); f=extract_features(p,.5,[p]); assert f == {'foreground_top10_mean':0., 'foreground_uncertainty':0., 'tta_consistency':1., 'spatial_coherence':0., 'component_probability_std':0., 'predicted_component_count':0}
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
 assert math.isfinite(calculate_prediction_score(dict.fromkeys(FEATURES,1.),artifact(std=[0]*len(FEATURES))))
def test_sigmoid_overflow_safe():
 f=dict.fromkeys(FEATURES,1e300); assert calculate_prediction_score(f,artifact())==100.; assert calculate_prediction_score(f,artifact(coef=[-1]*len(FEATURES)))==0.
def test_model_version_mismatch():
 with pytest.raises(ValueError): calculate_prediction_score(dict.fromkeys(FEATURES,0.),artifact(),expected_model_version='v2')
def test_sklearn_json_reference_parity():
 rng=np.random.default_rng(2); X=rng.normal(size=(50,len(FEATURES))); y=(X[:,0]-X[:,1]+.2*X[:,2]>0).astype(int); sc=StandardScaler().fit(X); lr=LogisticRegression(max_iter=1000).fit(sc.transform(X),y); a=artifact(sc.mean_.tolist(),sc.scale_.tolist(),lr.coef_[0].tolist(),float(lr.intercept_[0])); ref=np.array([calculate_prediction_score(dict(zip(FEATURES,r)),a)/100 for r in X]); assert np.allclose(ref,lr.predict_proba(sc.transform(X))[:,1],atol=1e-12)


def test_ensemble_disagreement():
 p=np.ones((2,2,2),dtype=float)*.6; a=np.ones_like(p)*.4; b=np.ones_like(p)*.8
 assert ensemble_disagreement([a,b],p,.5)==pytest.approx(.2)

def test_mra_fp_like_component_count():
 p=np.zeros((5,5,5),dtype=float);p[0,0,0]=.6;p[3:5,3:5,3:5]=.9
 f=component_reliability_features(p,.5,.001,small_volume_ml=.005,low_probability=.7,rule='or',largest_mode='mean')
 assert f['fp_like_component_count']==1 and f['largest_component_confidence']==pytest.approx(.9)

def test_checkpoint_and_model_binding_rejected():
 a=artifact();a['model_id']='X';a['segmentation_model']={'model_id':'X','checkpoint_sha256':'abc'};f=dict.fromkeys(FEATURES,0.)
 with pytest.raises(ValueError):calculate_prediction_score(f,a,expected_model_id='Y')
 with pytest.raises(ValueError):calculate_prediction_score(f,a,expected_checkpoint_sha256='def')
