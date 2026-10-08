#!/usr/bin/env python3
"""Stage existing original-space I-E03 validation probabilities; does not fabricate TTA."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/ischemia_ensemble/I-E03_N120_S50_S101/N120_060_S101_020_S50_020/probabilities"
DEST = REPO / "runs/calibration_exports/I-E03"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    probabilities = sorted(SOURCE.glob("*.npz"))
    if a.limit: probabilities = probabilities[:a.limit]
    if not probabilities: raise RuntimeError(f"No I-E03 probabilities in {SOURCE}")
    for prob in probabilities:
        cid = prob.name.removesuffix(".npz")
        mask = SOURCE / f"{cid}.nii.gz"
        if not mask.is_file(): raise FileNotFoundError(mask)
        for source, target in ((prob, DEST/"probabilities"/prob.name), (mask, DEST/"predictions"/mask.name)):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not a.overwrite: continue
            shutil.copy2(source, target)
        print(cid, flush=True)
    print("Base I-E03 probabilities staged. TTA ensemble probabilities are intentionally NOT created by this script.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
