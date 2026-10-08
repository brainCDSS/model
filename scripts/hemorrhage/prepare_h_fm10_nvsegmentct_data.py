#!/usr/bin/env python3
"""Create a MONAI Decathlon datalist for H-FM10 from the existing nnU-Net fold-0 split."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True,help="Dataset001_ICHBinary nnUNet_raw folder")
    ap.add_argument("--splits",type=Path,required=True,help="Dataset001_ICHBinary/splits_final.json")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    split=json.loads(args.splits.read_text(encoding="utf-8"))[0]
    train, val=set(split["train"]), set(split["val"])
    if train & val: raise ValueError("train/val overlap")
    records=[]
    for case_id in sorted(train|val):
        image=args.dataset_root/"imagesTr"/f"{case_id}_0000.nii.gz"
        label=args.dataset_root/"labelsTr"/f"{case_id}.nii.gz"
        if not image.is_file() or not label.is_file(): raise FileNotFoundError(f"missing pair: {case_id}")
        # MONAI datafold_read: entries whose fold equals 0 are validation.
        records.append({"fold": 1 if case_id in train else 0,"image":str(image),"label":str(label),"case_id":case_id})
    payload={"description":"H-FM10 uses existing Dataset001_ICHBinary fold-0 split.","labels":{"background":0,"intracranial_hemorrhage":1},"training":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"train":len(train),"val":len(val),"output":str(args.output)},ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
