#!/usr/bin/env python3
"""Validate original-space calibration exports and write a non-patient manifest."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import nibabel as nib
import numpy as np
from brainon_ai.calibration.features import foreground_probability
from brainon_ai.calibration.registry import resolve_model

def label_path(directory: Path, cid: str) -> Path:
    for name in (f"{cid}.nii.gz", f"{cid}_mask.nii.gz", f"{cid}.nii"):
        p=directory/name
        if p.is_file(): return p
    raise FileNotFoundError(cid)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-id",required=True); ap.add_argument("--registry",type=Path,default=Path("configs/calibration/models.yaml")); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]; m=resolve_model(a.model_id,a.registry,repo); root=Path(m["prediction_dir"]).parent
    rows=[]; probs=sorted(p for p in Path(m["prediction_dir"]).glob("*.npz") if not p.name.endswith(".normalizing.npz"))
    for prob_file in probs:
        cid=prob_file.name.removesuffix(".npz"); gt_file=label_path(Path(m["label_dir"]),cid); pred_file=root/"predictions"/f"{cid}.nii.gz"
        gt=nib.load(str(gt_file)); p=foreground_probability(prob_file)
        row={"case_id":cid,"probability_shape":list(p.shape),"gt_shape":list(gt.shape),"prediction_path":str(pred_file),"status":"READY"}
        if tuple(p.shape)!=tuple(gt.shape): row["status"]="BLOCKED: probability_shape"
        if not pred_file.is_file(): row["status"]="BLOCKED: prediction_missing"
        else:
            pred=nib.load(str(pred_file))
            if tuple(pred.shape)!=tuple(gt.shape) or not np.allclose(pred.affine,gt.affine,atol=1e-4): row["status"]="BLOCKED: prediction_space"
        metadata = {"case_id": cid, "probability_path": str(prob_file), "prediction_path": str(pred_file),
                    "gt_label_path": str(gt_file), "shape": list(gt.shape),
                    "spacing_mm": [float(x) for x in gt.header.get_zooms()[:3]],
                    "affine": gt.affine.tolist(), "validation_status": row["status"]}
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "metadata" / f"{cid}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(row)
    tta_counts=[len(list(Path(d).glob("*.npz"))) for d in m.get("tta_prediction_dirs",[])]
    summary={"model_id":a.model_id,"probability_cases":len(rows),"tta_probability_cases":tta_counts,"ready_cases":sum(r["status"]=="READY" for r in rows),"all_tta_available":bool(rows) and all(x==len(rows) for x in tta_counts)}
    root.mkdir(parents=True,exist_ok=True)
    with (root/"case_manifest.csv").open("w",newline="") as f: csv.DictWriter(f,fieldnames=["case_id","probability_shape","gt_shape","prediction_path","status"]).writeheader(); csv.DictWriter(f,fieldnames=["case_id","probability_shape","gt_shape","prediction_path","status"]).writerows(rows)
    (root/"export_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2)); return 0 if summary["ready_cases"]==len(rows) and summary["all_tta_available"] else 2
if __name__=="__main__": raise SystemExit(main())
