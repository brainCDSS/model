# Metrics

Canonical names are `dice_fg_case_mean`, `dice_fg_global`, `lesion_dice_mean`, `lesion_precision`, `lesion_recall`, `lesion_f1`, `fp_per_scan`, `froc_score`, `hd95_mm`, `nsd`, `avd_ml`, `rvd_percent`, and `absolute_lesion_count_difference`.

Binary Dice excludes background; the default aggregate is mean foreground Dice over positive patients. Negative cases are excluded from that average and reported with patient specificity, FP/scan, and false-positive volume. Lesions use 3D 26-connectivity. A prediction matches one ground-truth lesion when at least 10% of that target overlaps; matching is one-to-one. HD95 uses physical NIfTI spacing in millimetres. Per-case and per-lesion tables are retained.

ISLES-compatible reporting includes Dice, AVD, ALCD and lesion-wise F1. Aneurysm FROC is mean sensitivity at 0.5, 1.0 and 2.0 FP/scan (the production evaluator adapter will interpolate its operating curve). Test results never influence threshold or checkpoint selection.

