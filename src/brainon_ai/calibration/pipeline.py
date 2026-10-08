from __future__ import annotations
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BASELINE_FEATURE_NAMES, extract_features, foreground_probability, load_label, segmentation_metrics
from .scoring import calculate_prediction_score


def _case_id(path: Path) -> str:
    return path.name.removesuffix('.npz')


def _label_path(label_dir: Path, cid: str) -> Path:
    for name in (f"{cid}.nii.gz", f"{cid}_mask.nii.gz", f"{cid}.nii"):
        p = label_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"label not found for {cid} in {label_dir}")


def _target(metric: dict, cfg: dict) -> int:
    """Config-driven acceptable-segmentation target; GT is never a feature."""
    rules = cfg.get("positive_case_rules", cfg.get("rules", {})) if metric["gt_positive"] else cfg.get("negative_case_rules", {})
    if not rules:
        raise ValueError("quality target config needs positive_case_rules/rules")
    comparisons = {
        "dice_min": lambda actual, value: actual >= float(value),
        "lesion_recall_min": lambda actual, value: actual >= float(value),
        "fp_components_max": lambda actual, value: actual <= int(value),
    }
    for key, ok in comparisons.items():
        if key in rules and not ok(metric[{"dice_min":"dice", "lesion_recall_min":"lesion_recall", "fp_components_max":"fp_count"}[key]], rules[key]):
            return 0
    return 1


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if selected.any():
            total += selected.mean() * abs(float(p[selected].mean()) - float(y[selected].mean()))
    return float(total)


def _transform(value: float, name: str) -> float:
    if name == "identity":
        return float(value)
    if name == "log1p":
        return float(np.log1p(max(0.0, value)))
    raise ValueError(f"unsupported feature transform: {name}")


def _threshold_report(y: np.ndarray, p: np.ndarray) -> tuple[float, str, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append({"threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                     "sensitivity": sensitivity, "specificity": specificity,
                     "f1": f1_score(y, pred, zero_division=0), "youden_j": sensitivity + specificity - 1})
    report = pd.DataFrame(rows)
    best = report.sort_values(["youden_j", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best.threshold), "max_oof_youden_j_then_f1", report


def _write_plot(frame: pd.DataFrame, x: str, y: str, output: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(frame[x], frame[y], marker="o")
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)
    except Exception as exc:  # plot is useful but not a training dependency
        warnings.warn(f"Could not create {output.name}: {exc}")


def fit_calibration(model: dict, target_cfg: dict, output_dir: Path, allow_provisional: bool = False, random_state: int = 42) -> dict:
    if target_cfg.get("status") != "approved" and not allow_provisional:
        raise RuntimeError("target thresholds are provisional; review config or pass --allow-provisional-target")
    pred_dir = Path(model["prediction_dir"])
    label_dir = Path(model["label_dir"])
    tta_dirs = [Path(x) for x in model.get("tta_prediction_dirs", [])]
    if not tta_dirs:
        raise RuntimeError("No actual TTA probability directory configured; tta_consistency will not be fabricated")
    feature_specs = model.get("score_features") or [{"name": n, "transform": "log1p" if n == "predicted_component_count" else "identity"} for n in BASELINE_FEATURE_NAMES]
    feature_names = [spec["name"] for spec in feature_specs]
    rows = []
    for pp in sorted(pred_dir.glob("*.npz")):
        cid = _case_id(pp)
        prob = foreground_probability(pp)
        lp = _label_path(label_dir, cid)
        gt, spatial = load_label(lp)
        if prob.shape != gt.shape:
            raise ValueError(f"{cid}: probability {prob.shape} != label {gt.shape}; export must be original GT space")
        tta = []
        for directory in tta_dirs:
            q = directory / f"{cid}.npz"
            if not q.is_file():
                raise FileNotFoundError(q)
            tta.append(foreground_probability(q))
        feature = extract_features(prob, float(model["threshold"]), tta, int(model.get("connectivity", 26)))
        metric = segmentation_metrics(prob, gt, float(model["threshold"]))
        rows.append({"case_id": cid, **feature, **metric, "acceptable_segmentation": _target(metric, target_cfg),
                     "label_path": str(lp), "voxel_volume_ml": float(np.prod(spatial["spacing_mm"])) / 1000})
    frame = pd.DataFrame(rows)
    if len(frame) < 10:
        raise RuntimeError(f"too few calibration cases: {len(frame)}")
    transformed = np.column_stack([[_transform(v, spec.get("transform", "identity")) for v in frame[spec["name"]]] for spec in feature_specs])
    y = frame.acceptable_segmentation.to_numpy(int)
    counts = np.bincount(y, minlength=2)
    if min(counts) < 2:
        raise RuntimeError(f"quality target needs both classes with >=2 cases; counts={counts.tolist()}")
    if min(counts) < 5:
        warnings.warn(f"quality class is sparse: acceptable={counts[1]}, unacceptable={counts[0]}; reliability model is unstable")
    folds = min(5, int(min(counts)))
    cv = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", penalty="l2", random_state=random_state))
    oof = cross_val_predict(estimator, transformed, y, cv=cv, method="predict_proba")[:, 1]
    estimator.fit(transformed, y)
    scaler = estimator.named_steps["standardscaler"]
    lr = estimator.named_steps["logisticregression"]
    if (np.asarray(scaler.var_) < 1e-12).any():
        warnings.warn("near-zero variance score feature detected")
    selected_threshold, threshold_criterion, confusion = _threshold_report(y, oof)
    selected_pred = (oof >= selected_threshold).astype(int)
    metrics = {
        "reliability_auroc": float(roc_auc_score(y, oof)),
        "reliability_auprc": float(average_precision_score(y, oof)),
        "reliability_sensitivity": float(recall_score(y, selected_pred, zero_division=0)),
        "reliability_specificity": float(recall_score(1-y, 1-selected_pred, zero_division=0)),
        "reliability_f1": float(f1_score(y, selected_pred, zero_division=0)),
        "reliability_ppv": float(precision_score(y, selected_pred, zero_division=0)),
        "reliability_npv": float(precision_score(1-y, 1-selected_pred, zero_division=0)),
        "reliability_brier": float(brier_score_loss(y, oof)),
        "reliability_ece": _ece(y, oof),
    }
    artifact = {
        "schema_version": "1.0.0", "score_version": "1.0.0", "score_type": "segmentation_reliability",
        "method": "logistic_regression", "model_id": model["model_id"],
        "features": [{"name": spec["name"], "transform": spec.get("transform", "identity"),
                      "mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i]), "coefficient": float(lr.coef_[0, i])}
                     for i, spec in enumerate(feature_specs)],
        "intercept": float(lr.intercept_[0]),
        "target": {"name": "acceptable_segmentation", "definition_version": str(target_cfg.get("version", "1.0.0")),
                   "status": target_cfg.get("status", "MISSING"), "definition": target_cfg},
        "threshold_selection": {"criterion": threshold_criterion, "selected_threshold": selected_threshold, "selected_on": "stratified_oof"},
        "training": {"dataset_id": model.get("dataset_id", "MISSING"), "split": model.get("calibration_split", "validation"),
                     "case_count": len(frame), "acceptable_cases": int(counts[1]), "unacceptable_cases": int(counts[0]),
                     "cv_method": "StratifiedKFold", "cv_folds": folds},
        "segmentation_model": {"model_id": model["model_id"], "checkpoint_sha256": model.get("checkpoint_sha256")},
        "prediction_threshold": float(model["threshold"]), "connectivity": int(model.get("connectivity", 26)),
        "empty_prediction_policy": {"foreground_top10_mean": 0.0, "foreground_uncertainty": 0.0,
          "spatial_coherence": 0.0, "component_probability_std": 0.0, "predicted_component_count": 0,
          "tta_consistency": "Dice(mask, inverse-transformed TTA mask); empty-empty=1"},
        "metrics": metrics, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
    }
    frame["oof_reliability_probability"] = oof
    frame["prediction_score"] = 100 * oof
    reference = np.array([calculate_prediction_score({name: row[name] for name in feature_names}, artifact) for _, row in frame.iterrows()]) / 100
    if not np.allclose(reference, estimator.predict_proba(transformed)[:, 1], atol=1e-10):
        raise AssertionError("score_model.json full-fit calculation != sklearn")
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {"schema_version": "1.0.0", "score_type": "segmentation_reliability", "features": [
        {"name": spec["name"], "dtype": "float", "required": True, "transform": spec.get("transform", "identity"),
         "definition": {"foreground_top10_mean": "Mean of highest 10% foreground probabilities at p >= segmentation threshold.",
           "foreground_uncertainty": "Mean binary entropy -p ln p -(1-p) ln(1-p) over foreground voxels.",
           "tta_consistency": "Mean Dice between original foreground mask and each inverse-transformed TTA foreground mask.",
           "spatial_coherence": "Largest 26-connected foreground component voxel count divided by foreground voxel count.",
           "component_probability_std": "Population standard deviation of mean probability across 26-connected foreground components.",
           "predicted_component_count": "Number of 26-connected foreground components before log1p transform."}[spec["name"]]}
        for spec in feature_specs]}
    calibration_curve = []
    for lo, hi in zip(np.linspace(0, 1, min(10, len(y)) + 1)[:-1], np.linspace(0, 1, min(10, len(y)) + 1)[1:]):
        selected = (oof >= lo) & ((oof < hi) if hi < 1 else (oof <= hi))
        if selected.any(): calibration_curve.append({"bin_lower": float(lo), "bin_upper": float(hi), "predicted_probability": float(oof[selected].mean()), "observed_success_rate": float(y[selected].mean()), "case_count": int(selected.sum())})
    calib_df = pd.DataFrame(calibration_curve)
    roc = pd.DataFrame({"threshold": np.r_[np.inf, np.sort(np.unique(oof))[::-1], -np.inf]})
    roc["sensitivity"] = [recall_score(y, oof >= x, zero_division=0) for x in roc.threshold]
    roc["specificity"] = [recall_score(1-y, oof < x, zero_division=0) for x in roc.threshold]
    pr = pd.DataFrame({"threshold": np.sort(np.unique(oof))[::-1]})
    pr["precision"] = [precision_score(y, oof >= x, zero_division=0) for x in pr.threshold]
    pr["recall"] = [recall_score(y, oof >= x, zero_division=0) for x in pr.threshold]
    (output_dir / "score_model.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "score_feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "score_evaluation.json").write_text(json.dumps({"metrics": metrics, "class_distribution": artifact["training"], "threshold_selection": artifact["threshold_selection"]}, ensure_ascii=False, indent=2) + "\n")
    frame.to_csv(output_dir / "score_features.csv", index=False); calib_df.to_csv(output_dir / "calibration_curve.csv", index=False)
    roc.to_csv(output_dir / "roc_curve.csv", index=False); pr.to_csv(output_dir / "pr_curve.csv", index=False); confusion.to_csv(output_dir / "confusion_matrix_by_threshold.csv", index=False)
    _write_plot(calib_df, "predicted_probability", "observed_success_rate", output_dir / "calibration_curve.png", "Reliability calibration (OOF)")
    _write_plot(roc, "specificity", "sensitivity", output_dir / "roc_curve.png", "Reliability ROC (OOF)")
    _write_plot(pr, "recall", "precision", output_dir / "pr_curve.png", "Reliability PR (OOF)")
    manifest = {"model_id": model["model_id"], "model_version": str(model["version"]), "files": {}}
    for path in output_dir.iterdir():
        if path.is_file() and path.name != "manifest.json": manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return artifact
