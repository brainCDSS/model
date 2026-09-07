#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True,type=Path); p.add_argument("--data-root",required=True,type=Path); a=p.parse_args()
    c=yaml.safe_load(a.config.read_text(encoding="utf-8")); e,q=c["experiment"],c["plans"]
    d=a.data_root.resolve()/"nnUNet_preprocessed"/f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"
    src=d/f"{q['base_identifier']}.json"; dst=d/f"{q['output_identifier']}.json"
    if not src.is_file(): raise FileNotFoundError(f"base plans 없음: {src}")
    payload=json.loads(src.read_text(encoding="utf-8")); payload["plans_name"]=q["output_identifier"]
    dst.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(dst); return 0
if __name__=="__main__": raise SystemExit(main())
