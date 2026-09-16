# CDSS AI Prediction Score calibration

This pipeline calibrates a case-level probability that the final segmentation is acceptable. It does not replace or modify voxel probabilities.

## Repository integration

- nnU-Net validation probabilities are `.npz` files with `probabilities` or `softmax` arrays in restored label space. nnU-Net prediction/export and trained mirroring-axis conventions are reused.
- Ischemia SegResNet probabilities are `probabilities[class, X, Y, Z]`; labels remain in original NIfTI array space and inference uses sliding windows.
- Ground truth is resolved by case ID as `<case>.nii.gz` or `<case>_mask.nii.gz`.
- Spacing and affine come from the matched NIfTI label. Components use 3D 26-connectivity.
- Model IDs and artifact locations are in `configs/calibration/models.yaml`.

## Features

1. `foreground_top10_mean`: mean of the highest `ceil(10%)` probabilities inside the thresholded foreground; empty prediction = 0.
2. `foreground_uncertainty`: mean clipped binary entropy inside foreground; empty prediction = 0.
3. `spatial_coherence`: largest 26-connected foreground component / total foreground voxels; empty prediction = 0.
4. `tta_consistency`: mean Dice between the original mask and actual inverse-flipped TTA masks; empty-empty Dice = 1. Missing TTA is an error and is never replaced by a fabricated constant.

SegResNet identifies the left-right voxel axis from the NIfTI affine, flips the input, performs inference, and inverses the flip. nnU-Net uses the first trained allowed mirroring axis in preprocessed space and inverses logits before nnU-Net export to original space. No rotation, resize, or interpolating TTA is used.

## Target and leakage

Task thresholds are separate YAML files under `configs/calibration/tasks/`. They combine Dice, lesion recall and false-positive components. Current values are explicitly provisional and require `--allow-provisional-target`; they are not clinically approved.

Only validation/calibration predictions are fitted. Test data is never loaded. Brier, log loss, ECE and AUROC use stratified out-of-fold predictions. The deployable coefficient is then fitted on all calibration cases. A dedicated calibration split or external validation remains preferred.

## Commands

```bash
PYTHONPATH=src .venv/bin/python scripts/calibration/list_models.py
PYTHONPATH=src .venv/bin/python scripts/calibration/run_model_calibration.py   --model-id I-S40 --allow-provisional-target
```

The one-command runner creates missing original/TTA probabilities and then writes:

```text
artifacts/calibration/<task>/<model-id>/<checkpoint-version>/
  calibration.json
  calibration_metrics.json
  calibration_predictions.csv
  calibration_curve.csv
  manifest.json
```

The version contains the checkpoint SHA-256 prefix. A different weight file therefore creates a different calibration version.

## CDSS reference score

```python
from brainon_ai.calibration.scoring import calculate_prediction_score
score = calculate_prediction_score(features, calibration_json, expected_model_version=model_version)
```

The function matches features by name, standardizes with JSON mean/std, applies coefficients and a stable sigmoid, and returns a float from 0 to 100. CDSS does not require sklearn.
