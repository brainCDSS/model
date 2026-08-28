# BrainOn AI model repository

This repository is the source of truth for BrainOn medical-image segmentation training, experiments, evaluation, visualization, and model releases. It is intentionally independent from BrainOn application repositories. Python 3.11 is recommended (supported: `>=3.10,<3.13`).

## Quick start

Clone into the documented `model/` directory and work on `dev`; validated changes reach `main` through a `dev -> main` pull request.

```bash
git clone https://github.com/brainCDSS/model.git model
cd model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch separately for your CPU/CUDA platform, then:
python -m pip install -e ".[dev,notebook,storage]"
brainon doctor
brainon storage doctor --backend local
brainon data make-synthetic --task all
brainon smoke-test --task all
pytest -q
```

Do not use a common dependency to replace a working CUDA-specific PyTorch wheel. Follow the [PyTorch selector](https://pytorch.org/get-started/locally/) for local/server CUDA. Colab already provides PyTorch. MLflow, MONAI, and nnU-Net are independent extras: `.[mlflow]`, `.[monai]`, `.[nnunet]`.

## Windows VSCode PowerShell

```powershell
git clone https://github.com/brainCDSS/model.git model
Set-Location model
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the correct PyTorch wheel first.
python -m pip install -e ".[dev,notebook,storage]"
$env:BRAINON_CACHE_ROOT = "$PWD\.cache\brainon"
$env:BRAINON_RUN_ROOT = "$PWD\runs"
brainon data make-synthetic --task all
brainon smoke-test --task all
pytest -q
```

## Jupyter and Colab

Local or remote Jupyter uses the installed package, never copied training code:

```bash
python -m pip install -e ".[notebook]"
jupyter lab
```

In Colab set `REPO_URL=https://github.com/brainCDSS/model.git` and `BRANCH=main` for shared/released use or `BRANCH=dev` for development validation. Run `notebooks/00_colab_setup.ipynb`. Private-repository tokens belong only in Colab Secrets or a protected environment variable and must never be printed or saved in notebook output. See [docs/COLAB.md](docs/COLAB.md).

## Commands

```bash
brainon data list
brainon data pull --release hemorrhage_binary_v1.0.0
brainon data verify --release hemorrhage_binary_v1.0.0
brainon data prepare --dataset Dataset101_HemorrhageBinary
brainon data make-synthetic --task all
brainon train --config configs/experiments/hemorrhage/H-S00-smoke.yaml
brainon evaluate --config configs/experiments/hemorrhage/H-S00-smoke.yaml --checkpoint runs/.../best_balanced.ckpt
brainon predict --config configs/experiments/hemorrhage/H-S00-smoke.yaml --checkpoint runs/.../best_balanced.ckpt
brainon visualize --run-id RUN_ID
brainon run sync --run-id RUN_ID --dry-run
brainon run resume --run-id RUN_ID
brainon model export --run-id RUN_ID --version 1.0.0
brainon model pull --task hemorrhage --version 1.0.0
```

`BRAINON_CACHE_ROOT` automatically contains `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. Training always uses a local cache. Downloads use `.partial`, verify SHA-256, and only then produce readiness markers. Dataset acquisition is never used to bypass licenses or DUAs.

## Empty repository initialization

Only use this sequence if the remote truly has no commits. The two branches must share the same initial commit; do not create orphan branches.

```bash
git switch -c main
git add .
git commit -m "Initialize BrainOn AI project"
git push -u origin main
git switch -c dev
git push -u origin dev
```

Normal work happens on `dev`. Use temporary `feature/<topic>` branches only for concurrent or large structural work, merge them into `dev`, then delete them. Never create branches for individual hyperparameter runs; use experiment YAML IDs, MLflow, and versioned GCS run paths.

## GCS and application integration

Copy `.env.example` to an untracked `.env` or inject environment variables. Authentication uses Application Default Credentials. Bucket/IAM scripts are examples and are never run automatically. Final releases live at `gs://PROJECT_ID-brainon-models/releases/<task>/<version>/`. BrainOn inference consumes only the GCS model URI and versioned manifest conforming to `contracts/model_release.schema.json`; no training code or checkpoints are copied into the application repository.

More detail: [datasets](docs/DATASETS.md), [experiments](docs/EXPERIMENTS.md), [metrics](docs/METRICS.md), and [storage](docs/STORAGE.md).

