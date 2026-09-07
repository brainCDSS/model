#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import yaml

def main():
 p=argparse.ArgumentParser();p.add_argument("--preprocessed-root",required=True,type=Path);p.add_argument("--overwrite",action="store_true");a=p.parse_args();repo=Path(__file__).resolve().parents[2];c=yaml.safe_load((repo/"configs/experiments/hemorrhage/H-N140.yaml").read_text());e,q=c["experiment"],c["plans"]
 dataset=a.preprocessed_root.resolve()/f"Dataset{int(e['dataset_id']):03d}_ICHBinary";src=dataset/f"{q['base_identifier']}.json";dst=dataset/f"{q['output_identifier']}.json"
 if not src.is_file():raise FileNotFoundError(src)
 if dst.is_file() and not a.overwrite:
  payload=json.loads(dst.read_text())
  if payload.get("plans_name")!=q["output_identifier"]:raise RuntimeError(f"plans_name mismatch: {dst}")
  print(f"[ok] {dst}");return 0
 shutil.copy2(src,dst);payload=json.loads(dst.read_text());payload["plans_name"]=q["output_identifier"];dst.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n");print(f"[created] {dst}");return 0
if __name__=="__main__":raise SystemExit(main())
