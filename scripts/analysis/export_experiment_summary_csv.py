#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.ndimage import label as cc_label

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "experiment_summary"
FIELDS = [
    "task", "experiment_id", "category", "status", "dataset", "input",
    "network", "method_change", "loss", "sampling", "optimizer", "learning_rate",
    "epochs_planned", "epoch_completed", "selected_epoch", "selection_rule",
    "evaluation_scope", "positive_cases", "negative_cases",
    "macro_dice_positive", "voxel_recall_positive", "voxel_precision_positive",
    "case_detection_rate", "negative_case_fp_rate",
    "lesion_recall", "lesion_precision", "lesion_f1", "fp_per_scan",
    "result_path", "config_path",
]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return round(value, 6)
    return value


def summary_metrics(path):
    data = read_json(path)
    items = data.get("metric_per_case", [])
    positive, negative = [], []
    for item in items:
        metric = item["metrics"]["1"]
        (positive if float(metric.get("n_ref", 0)) > 0 else negative).append(metric)
    def mean_ratio(rows, numerator, denominator):
        values = []
        for row in rows:
            den = float(row[numerator]) + float(row[denominator])
            if den:
                values.append(float(row[numerator]) / den)
        return float(np.mean(values)) if values else None
    dice = [
        float(row["Dice"]) for row in positive
        if math.isfinite(float(row["Dice"]))
    ]
    return {
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "macro_dice_positive": float(np.mean(dice)) if dice else None,
        "voxel_recall_positive": mean_ratio(positive, "TP", "FN"),
        "voxel_precision_positive": mean_ratio(positive, "TP", "FP"),
        "case_detection_rate": (
            sum(float(row["TP"]) > 0 for row in positive) / len(positive)
            if positive else None
        ),
        "negative_case_fp_rate": (
            sum(float(row["n_pred"]) > 0 for row in negative) / len(negative)
            if negative else None
        ),
    }


def lesion_metrics(summary_path):
    data = read_json(summary_path)
    total_gt = total_pred = detected_gt = false_positive = scans = 0
    for item in data.get("metric_per_case", []):
        reference = Path(item["reference_file"])
        prediction = summary_path.parent / reference.name
        if not prediction.is_file() or not reference.is_file():
            continue
        gt = np.asanyarray(nib.load(str(reference)).dataobj) > 0
        pred = np.asanyarray(nib.load(str(prediction)).dataobj) > 0
        gt_cc, gt_count = cc_label(gt, structure=np.ones((3, 3, 3), dtype=np.uint8))
        pred_cc, pred_count = cc_label(pred, structure=np.ones((3, 3, 3), dtype=np.uint8))
        total_gt += int(gt_count)
        total_pred += int(pred_count)
        detected_gt += sum(
            bool(np.any(pred[gt_cc == component]))
            for component in range(1, gt_count + 1)
        )
        false_positive += sum(
            not bool(np.any(gt[pred_cc == component]))
            for component in range(1, pred_count + 1)
        )
        scans += 1
    recall = detected_gt / total_gt if total_gt else None
    precision = (total_pred - false_positive) / total_pred if total_pred else None
    f1 = (
        2 * recall * precision / (recall + precision)
        if recall is not None and precision is not None and recall + precision else None
    )
    return {
        "lesion_recall": recall,
        "lesion_precision": precision,
        "lesion_f1": f1,
        "fp_per_scan": false_positive / scans if scans else None,
    }


def loss_text(config):
    loss = config.get("loss", {})
    name = loss.get("name", "")
    details = []
    for key in ("alpha", "beta", "tversky_alpha", "tversky_beta", "gamma", "smooth"):
        if key in loss:
            details.append(f"{key}={loss[key]}")
    return name + (f" ({'; '.join(details)})" if details else "")


def sampling_text(config):
    sampling = config.get("sampling", {})
    trainer = config.get("trainer", {})
    foreground = trainer.get("oversample_foreground_percent")
    if not sampling.get("enabled", False):
        return f"standard foreground oversampling={foreground}"
    strategy = sampling.get("strategy", "case_weighted")
    return f"{strategy}; foreground={foreground}; bins={sampling.get('volume_thresholds_ml')}"


def method_text(config, experiment_id):
    network = config.get("network", {}).get("name") or config.get("model", {}).get("name", "")
    if experiment_id == "H-N00": return "nnU-Net baseline"
    if experiment_id == "H-N10": return "Residual encoder M"
    if experiment_id == "H-N20": return "Residual encoder M + attention"
    if experiment_id == "H-N30": return "Default encoder + attention"
    if experiment_id == "H-N40": return "BHSD-only dataset"
    if experiment_id == "H-N50": return "Two-window CT input"
    if experiment_id == "H-N60": return "Small-lesion case sampling"
    if experiment_id == "H-N70": return "Two-window CT + small-lesion sampling"
    if experiment_id == "H-N80": return "Tversky-CE loss"
    if experiment_id == "H-N90": return "Component-centered sampling"
    if experiment_id == "H-N100": return "Tversky-CE + component-centered sampling"
    if experiment_id == "H-N110": return "Dice-Focal loss"
    if experiment_id == "H-N120": return "Tversky-Focal loss"
    if experiment_id == "H-N130": return "Case-weighted + component-centered sampling"
    if experiment_id == "H-N140": return "DiceCE smooth ablation (smooth=0)"
    if experiment_id.startswith("H-T"):
        loss = config.get("loss", {})
        tr = config.get("trainer", {})
        return f"Tversky/optimizer tuning: alpha={loss.get('alpha')}, beta={loss.get('beta')}, {tr.get('optimizer',{}).get('name')}, lr={tr.get('learning_rate')}"
    if experiment_id == "H-S00": return "SegResNet baseline"
    if experiment_id == "H-S10": return "SegResNet + attention gate"
    if experiment_id == "H-S20": return "SegResNet + multiscale context"
    if experiment_id == "H-S30": return "Config placeholder; not configured"
    aneurysm = {
        "A-B00": "nnU-Net baseline",
        "A-B01": "DiceCE smooth ablation (smooth=0)",
        "A-B02": "Foreground oversampling 0.33 to 0.50",
        "A-B03": "Oversampling 0.50 + Tversky-Focal",
        "A-B04": "Tversky-Focal with baseline oversampling 0.33",
        "A-B05": "A-B04 + component-centered foreground patches",
    }
    return aneurysm.get(experiment_id, network)


def locate_fold(config, task):
    plan = config.get("plans", {}).get("output_identifier")
    if not plan:
        return None
    dataset_id = int(config["experiment"]["dataset_id"])
    fold = int(config["experiment"].get("fold", 0))
    base = ROOT / ("runs/aneurysm_binary/nnunet" if task == "brain_aneurysm" else "runs/nnunet")
    matches = list(base.glob(f"Dataset{dataset_id:03d}_*/**/*__{plan}__*/fold_{fold}"))
    exact = [path for path in matches if path.name == f"fold_{fold}"]
    return exact[0] if exact else None


def choose_summary(fold):
    selection = read_json(fold / "final_selection.json")
    if selection.get("selected", {}).get("summary"):
        selected = selection["selected"]
        return Path(selected["summary"]), int(selected["epoch"]), "dual_criteria"
    history = read_json(fold / "full_validation_history.json")
    if isinstance(history, list) and history:
        best = max(history, key=lambda item: float(item.get("macro_dice_positive", -1)))
        return Path(best["summary"]), int(best["epoch"]), "best_full_macro_dice"
    summary = fold / "validation" / "summary.json"
    if summary.is_file():
        # Older experiments save only one full-volume validation summary. Their
        # best Dice checkpoint still preserves the validation epoch, so expose
        # it rather than leaving the selected epoch blank.
        best_dice = fold / "checkpoint_best_dice.pth"
        if best_dice.is_file():
            try:
                state = torch.load(best_dice, map_location="cpu", weights_only=False)
                epoch = state.get("current_epoch")
                if epoch is not None:
                    return summary, int(epoch), "best_patch_validation_dice"
            except (OSError, RuntimeError, KeyError, TypeError):
                pass
        return summary, None, "saved_validation_checkpoint"
    return None, None, "no_full_volume_result"


def nnunet_row(path, task, category):
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiment_id = path.stem
    exp, trainer = config.get("experiment", {}), config.get("trainer", {})
    dataset_id = exp.get("dataset_id")
    dataset_name = exp.get("dataset_name") or ({1:"ICHBinary",2:"BHSDOnly",3:"ICHBinary2Window"}.get(dataset_id,""))
    input_name = "MRA" if task == "brain_aneurysm" else ("CT two-window" if dataset_id == 3 else "CT")
    row = {
        "task": task, "experiment_id": experiment_id, "category": category,
        "status": "not_run", "dataset": f"Dataset{int(dataset_id):03d}_{dataset_name}",
        "input": input_name, "network": config.get("network", {}).get("name", "default"),
        "method_change": method_text(config, experiment_id), "loss": loss_text(config),
        "sampling": sampling_text(config), "optimizer": trainer.get("optimizer",{}).get("name",""),
        "learning_rate": trainer.get("learning_rate"), "epochs_planned": trainer.get("epochs"),
        "epoch_completed": "", "selected_epoch": "", "selection_rule": "",
        "evaluation_scope": "", "result_path": "", "config_path": str(path),
    }
    fold = locate_fold(config, task)
    if fold is None:
        return row
    marker = read_json(fold / "ich_config.json")
    actual = marker.get("config") if isinstance(marker, dict) else None
    if isinstance(actual, dict):
        config = actual
        trainer = config.get("trainer", {})
        exp = config.get("experiment", exp)
        row.update({
            "network": config.get("network", {}).get("name", row["network"]),
            "method_change": method_text(config, experiment_id),
            "loss": loss_text(config),
            "sampling": sampling_text(config),
            "optimizer": trainer.get("optimizer", {}).get("name", ""),
            "learning_rate": trainer.get("learning_rate", ""),
            "epochs_planned": trainer.get("epochs", ""),
        })
    progress = read_json(fold / "progress.json")
    row["status"] = progress.get("status", "result_found")
    row["epoch_completed"] = progress.get("current_epoch", "")
    summary, selected_epoch, rule = choose_summary(fold)
    row["selection_rule"] = rule
    row["selected_epoch"] = selected_epoch if selected_epoch is not None else ""
    row["result_path"] = str(fold)
    if summary and summary.is_file():
        row.update(summary_metrics(summary))
        row["evaluation_scope"] = "full-volume validation"
        if task == "brain_aneurysm":
            row.update(lesion_metrics(summary))
    return row


def segresnet_rows():
    rows = []
    for experiment_id in ("H-S00", "H-S10", "H-S20", "H-S30"):
        path = ROOT / "configs/experiments/hemorrhage" / f"{experiment_id}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        run = ROOT / "runs/segresnet" / experiment_id
        progress = read_json(run / "progress.json")
        summaries = []
        for summary_path in (run / "validation").glob("epoch_*/summary.json"):
            data = read_json(summary_path)
            if data:
                summaries.append((float(data.get("composite", -1)), summary_path, data))
        best = max(summaries, default=None, key=lambda item: item[0])
        training = config.get("training", {})
        model = config.get("model", {})
        row = {
            "task":"brain_hemorrhage", "experiment_id":experiment_id,
            "category":"architecture_experiment", "status":progress.get("status","not_configured" if not config else "not_run"),
            "dataset":"mixed hemorrhage validation", "input":"CT",
            "network":model.get("name",""), "method_change":method_text(config,experiment_id),
            "loss":loss_text(config), "sampling":"patch sampler",
            "optimizer":training.get("optimizer",""), "learning_rate":training.get("learning_rate",""),
            "epochs_planned":progress.get("total_epochs",training.get("max_epochs","")),
            "epoch_completed":progress.get("current_epoch",""), "selected_epoch":"",
            "selection_rule":"best composite (0.5 Dice + 0.5 Recall)",
            "evaluation_scope":"full-volume validation" if best else "",
            "result_path":str(run) if run.exists() else "", "config_path":str(path),
        }
        if best:
            data = best[2]
            row["selected_epoch"] = int(best[1].parent.name.split("_")[-1])
            row.update({
                "positive_cases":data.get("positive_cases"),
                "negative_cases":data.get("empty_gt_cases"),
                "macro_dice_positive":data.get("dice_positive_macro"),
                "voxel_recall_positive":data.get("recall_positive_macro"),
                "voxel_precision_positive":data.get("precision_positive_macro"),
                "case_detection_rate":None,
                "negative_case_fp_rate":data.get("empty_gt_false_positive_rate"),
            })
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key)) for key in FIELDS})


def main():
    hemorrhage = []
    for path in sorted((ROOT / "configs/experiments/hemorrhage").glob("H-N*.yaml")):
        hemorrhage.append(nnunet_row(path, "brain_hemorrhage", "architecture_or_method_experiment"))
    hemorrhage.extend(segresnet_rows())
    for path in sorted((ROOT / "configs/experiments/hemorrhage/tuning").glob("H-*.yaml")):
        hemorrhage.append(nnunet_row(path, "brain_hemorrhage", "hyperparameter_tuning"))
    for path in sorted((ROOT / "configs/experiments/hemorrhage/kfold").glob("H-*.yaml")):
        hemorrhage.append(nnunet_row(path, "brain_hemorrhage", "five_fold_validation"))

    aneurysm = [
        nnunet_row(path, "brain_aneurysm", "method_experiment")
        for path in sorted((ROOT / "configs/experiments/aneurysm_binary").glob("A-B*.yaml"))
    ]
    import re
    def order(row):
        parts = re.split(r"(\d+)", row["experiment_id"])
        return [int(part) if part.isdigit() else part for part in parts]
    hemorrhage.sort(key=order)
    aneurysm.sort(key=order)
    write_csv(OUTPUT / "hemorrhage_all_experiments.csv", hemorrhage)
    write_csv(OUTPUT / "aneurysm_all_experiments.csv", aneurysm)
    write_csv(OUTPUT / "hemorrhage_aneurysm_all_experiments.csv", hemorrhage + aneurysm)
    definitions = [
        {"metric":"macro_dice_positive","scope":"positive cases","reason":"primary segmentation overlap; avoids empty-case inflation"},
        {"metric":"voxel_recall_positive","scope":"positive cases","reason":"voxel-level false-negative sensitivity"},
        {"metric":"voxel_precision_positive","scope":"positive cases","reason":"voxel-level over-segmentation control"},
        {"metric":"case_detection_rate","scope":"hemorrhage positive cases","reason":"whether any hemorrhage voxel was detected"},
        {"metric":"negative_case_fp_rate","scope":"hemorrhage negative cases","reason":"case-level false-alarm control"},
        {"metric":"lesion_recall","scope":"aneurysm lesions","reason":"fraction of GT connected components with any overlap"},
        {"metric":"lesion_precision","scope":"aneurysm predicted lesions","reason":"fraction of predicted components overlapping GT"},
        {"metric":"lesion_f1","scope":"aneurysm lesions","reason":"balance of lesion recall and precision"},
        {"metric":"fp_per_scan","scope":"aneurysm all scans","reason":"mean non-overlapping predicted components per scan; no size filter"},
    ]
    with (OUTPUT / "metric_definitions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric","scope","reason"])
        writer.writeheader()
        writer.writerows(definitions)
    print(f"hemorrhage={len(hemorrhage)} aneurysm={len(aneurysm)}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
