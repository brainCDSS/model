from __future__ import annotations
import math

def _sigmoid(z:float)->float:
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))

def calculate_prediction_score(features:dict, calibration:dict, expected_model_version:str|None=None)->float:
    if expected_model_version is not None and calibration.get("model",{}).get("version")!=expected_model_version:
        raise ValueError("model version mismatch")
    z=float(calibration["intercept"])
    for spec in calibration["features"]:
        name=spec["name"]
        if name not in features: raise KeyError(f"missing feature: {name}")
        std=float(spec["std"])
        if not math.isfinite(std) or abs(std)<1e-12: std=1.0
        value=float(features[name])
        if not math.isfinite(value): raise ValueError(f"non-finite feature: {name}")
        z+=float(spec["coefficient"])*(value-float(spec["mean"]))/std
    return float(min(100.0,max(0.0,100.0*_sigmoid(z))))
