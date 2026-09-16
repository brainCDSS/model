#!/usr/bin/env python3
"""Create a notebook for manual review of A-B04's missed sub-0.01 mL lesions."""
import json
from pathlib import Path


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


cells = [
    cell("markdown", """# A-B04 epoch 290: missed lesions below 0.01 mL

This notebook reviews the six validation lesions below 0.01 mL that have no overlap with the A-B04 epoch 290 prediction. It shows the original MRA, GT, and prediction around each GT component. A one-voxel label should be treated as a possible annotation fragment until reviewed.

Run with the `model/.venv` kernel. Green is GT and red is prediction.
"""),
    cell("code", """from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'datasets').is_dir() and (p / 'model').is_dir())
FOLD = ROOT / 'model/runs/aneurysm_binary/nnunet/Dataset503_TopAneuMRABinary/nnUNetTrainerAneurysmBinaryBaseline__AneurysmBinaryPlansTverskyFocalV1__3d_fullres/fold_0'
PRED_DIR = FOLD / 'full_validation/epoch_0290'
PRE = ROOT / 'datasets/aneurysm/v2/model_ready/nnunet/nnUNet_preprocessed/Dataset503_TopAneuMRABinary'
RAW = ROOT / 'datasets/aneurysm/v2/model_ready/nnunet/nnUNet_raw/Dataset503_TopAneuMRABinary/imagesTr'
OUT = ROOT / 'model/runs/aneurysm_binary/A-B04/small_missed_review_epoch_0290'
OUT.mkdir(parents=True, exist_ok=True)
REVIEW_PATH = OUT / 'manual_review.csv'
STRUCTURE = ndimage.generate_binary_structure(3, 3)
summary = json.loads((PRED_DIR / 'summary.json').read_text())
cases = {Path(x['prediction_file']).name.removesuffix('.nii.gz'): x for x in summary['metric_per_case']}
"""),
    cell("code", """rows = []
for case_id, item in cases.items():
    gt_img = nib.load(item['reference_file'])
    pred_img = nib.load(PRED_DIR / f'{case_id}.nii.gz')
    gt = np.asanyarray(gt_img.dataobj) > 0
    pred = np.asanyarray(pred_img.dataobj) > 0
    gt_labels, n = ndimage.label(gt, STRUCTURE)
    voxel_ml = abs(np.linalg.det(gt_img.affine[:3, :3])) / 1000
    for lesion_id in range(1, n + 1):
        mask = gt_labels == lesion_id
        volume_ml = int(mask.sum()) * voxel_ml
        if volume_ml < .01 and not np.any(pred & mask):
            center = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
            rows.append(dict(case_id=case_id, lesion_id=lesion_id,
                             gt_voxels=int(mask.sum()), gt_volume_ml=volume_ml,
                             x=int(center[0]), y=int(center[1]), z=int(center[2])))
missed = pd.DataFrame(rows).sort_values('gt_volume_ml').reset_index(drop=True)
missed.index.name = 'candidate'
assert len(missed) == 6
missed.to_csv(OUT / 'missed_small_lesions.csv')
display(missed)
"""),
    cell("markdown", """## Visual review

Use `show_candidate(0)` through `show_candidate(5)`. The rows are the three orthogonal planes through the lesion center. The columns show MRA, MRA+GT, and MRA+GT+prediction. Use offsets to inspect adjacent slices, for example `show_candidate(0, offsets=(0, 0, 2))`.
"""),
    cell("code", """def _align(img, target, order):
    img = nib.as_closest_canonical(img)
    if img.shape == target.shape and np.allclose(img.affine, target.affine, atol=1e-5):
        return np.asanyarray(img.dataobj)
    return np.asanyarray(resample_from_to(img, target, order=order).dataobj)

def load_candidate(index):
    row = missed.loc[index]
    gt_native = nib.load(PRE / 'gt_segmentations' / f'{row.case_id}.nii.gz')
    pred_native = nib.load(PRED_DIR / f'{row.case_id}.nii.gz')
    labels, _ = ndimage.label(np.asanyarray(gt_native.dataobj) > 0, STRUCTURE)
    lesion_native = nib.Nifti1Image((labels == row.lesion_id).astype(np.uint8), gt_native.affine)
    target = nib.as_closest_canonical(pred_native)
    raw = _align(nib.load(RAW / f'{row.case_id}_0000.nii.gz'), target, 1)
    gt = _align(gt_native, target, 0) > 0
    lesion = _align(lesion_native, target, 0) > 0
    pred = np.asanyarray(target.dataobj) > 0
    center = np.rint(np.argwhere(lesion).mean(axis=0)).astype(int)
    values = raw[np.isfinite(raw) & (raw != 0)]
    limits = np.percentile(values, [1, 99.5]) if values.size else (0, 1)
    return row, raw, gt, lesion, pred, center, limits

def show_candidate(index=0, offsets=(0, 0, 0), radius=45, save=False):
    row, raw, gt, lesion, pred, center, limits = load_candidate(index)
    pos = np.clip(center + np.asarray(offsets), 0, np.array(raw.shape)-1).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for axis in range(3):
        dims = [d for d in range(3) if d != axis]
        planes = [np.take(a, pos[axis], axis=axis).T for a in (raw, gt, lesion, pred)]
        for col in range(3):
            ax = axes[axis, col]
            ax.imshow(planes[0], cmap='gray', origin='lower', vmin=limits[0], vmax=limits[1])
            if col >= 1 and planes[1].any() and not planes[1].all():
                ax.contour(planes[1], levels=[.5], colors=['lime'], linewidths=1)
            if col == 2 and planes[3].any() and not planes[3].all():
                ax.contour(planes[3], levels=[.5], colors=['red'], linewidths=1)
            ax.set_xlim(max(0, center[dims[0]]-radius), min(raw.shape[dims[0]]-1, center[dims[0]]+radius))
            ax.set_ylim(max(0, center[dims[1]]-radius), min(raw.shape[dims[1]]-1, center[dims[1]]+radius))
            ax.set_title(f"{'XYZ'[axis]}={pos[axis]} | {['MRA', 'GT: green', 'Prediction: red'][col]}")
    fig.suptitle(f"#{index} {row.case_id}, lesion {row.lesion_id}: {row.gt_voxels} voxels, {row.gt_volume_ml:.6f} mL")
    plt.tight_layout()
    if save:
        fig.savefig(OUT / f'candidate_{index:02d}.png', dpi=140)
    plt.show()

show_candidate(0)
"""),
    cell("markdown", """## Manual decision

Suggested categories: `visually identifiable`, `subtle but plausible`, `annotation fragment/artifact suspected`, `registration issue`, `uncertain`. Record the image feature and slice direction in notes. This CSV does not change the training labels.
"""),
    cell("code", """CATEGORIES = ['visually identifiable', 'subtle but plausible', 'annotation fragment/artifact suspected', 'registration issue', 'uncertain']
def review_table():
    if REVIEW_PATH.exists():
        return pd.read_csv(REVIEW_PATH, keep_default_na=False)
    table = missed.reset_index().copy()
    table['category'] = 'uncertain'
    table['notes'] = ''
    return table

def record_review(index, category, notes=''):
    if category not in CATEGORIES:
        raise ValueError(CATEGORIES)
    row = missed.loc[index]
    table = review_table()
    match = (table.case_id == row.case_id) & (table.lesion_id == row.lesion_id)
    if match.sum() != 1:
        raise ValueError('Candidate list and review CSV do not match')
    table.loc[match, ['category', 'notes']] = [category, notes]
    table.to_csv(REVIEW_PATH, index=False)
    display(table.loc[match])

# record_review(0, 'visually identifiable', 'Bright focal bulge visible on ... plane')
display(review_table())
"""),
]

for i, item in enumerate(cells):
    item["id"] = f"small-missed-{i:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).resolve().parents[2] / "notebooks/hy/aneurysm_binary/06_A-B04_small_missed_lesion_review.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(destination)
