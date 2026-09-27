from __future__ import annotations
import math

def _sigmoid(z: float) -> float:
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))

def _transform(value: float, name: str) -> float:
    if name == "identity": return value
    if name == "log1p": return math.log1p(max(0.0, value))
    raise ValueError(f"unsupported feature transform: {name}")

def calculate_prediction_score(features: dict, calibration: dict, expected_model_version: str | None = None, expected_model_id: str | None = None, expected_checkpoint_sha256: str | None = None) -> float:
    """Returns a per-analysis segmentation reliability score in [0, 100]."""
    model = calibration.get("segmentation_model", calibration.get("model", {}))
    if expected_model_version is not None and model.get("version") not in (None, expected_model_version):
        raise ValueError("model version mismatch")
    if expected_model_id is not None and model.get("model_id", calibration.get("model_id")) != expected_model_id:
        raise ValueError("model id mismatch")
    if expected_checkpoint_sha256 is not None and model.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("checkpoint binding mismatch")
    if calibration.get("score_type") not in (None, "segmentation_reliability"):
        raise ValueError("unsupported score_type")
    z = float(calibration["intercept"])
    for spec in calibration["features"]:
        name = spec["name"]
        if name not in features: raise KeyError(f"missing feature: {name}")
        value = float(features[name])
        if not math.isfinite(value): raise ValueError(f"non-finite feature: {name}")
        value = _transform(value, spec.get("transform", "identity"))
        std = float(spec["std"])
        if not math.isfinite(std) or abs(std) < 1e-12: std = 1.0
        z += float(spec["coefficient"]) * (value - float(spec["mean"])) / std
    return float(min(100.0, max(0.0, 100.0 * _sigmoid(z))))
