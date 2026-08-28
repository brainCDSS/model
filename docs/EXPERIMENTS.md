# Experiments

YAML IDs define runs—Git branches do not. Hemorrhage: H-N00/H-S00/H-S10/H-S20/H-S30. Ischemia: I-N00/I-S00/I-S10/I-S20/I-S30. Aneurysm: A-N00/A-S00/A-S10/A-S20/A-S30/A-S40.

Each run records resolved YAML, Git SHA, data release, split, seed, Python/package/CUDA/GPU details, metrics and immutable checkpoints. MLflow is optional; local run artifacts remain authoritative when it is unavailable. Threshold and minimum component size are selected on validation only and frozen before test evaluation.

Balanced checkpoint selection first keeps checkpoints within 0.01 of the task primary optimum, then optimizes the secondary metric and finally the tie-breaker. The YAML files contain each task's exact metric ordering. `last.ckpt`, best variants and `top_k/` are local aliases; cloud synchronization uses immutable `periodic/epoch-NNNN.ckpt`, `selected/best-*.ckpt`, then updates `pointers/latest.json` last.

