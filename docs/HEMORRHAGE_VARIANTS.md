# Hemorrhage non-baseline variants

The batch runner executes the four non-baseline experiments sequentially on one GPU:

1. `H-N10`: nnU-Net ResEnc M
2. `H-N20`: nnU-Net ResEnc M with attention-gated skip connections
3. `H-S10`: SegResNet with attention-gated skip connections
4. `H-S20`: SegResNet with multi-scale dilated bottleneck context

Preview paths and commands without changing data or starting training:

```bash
python scripts/run_hemorrhage_variants.py --dry-run
```

Run all four on GPU 0:

```bash
mkdir -p runs/hemorrhage-variants
nohup python scripts/run_hemorrhage_variants.py --gpu 0 \
  > runs/hemorrhage-variants/launcher.log 2>&1 &
```

Monitor the batch:

```bash
tail -f runs/hemorrhage-variants/launcher.log
cat runs/hemorrhage-variants/batch_status.json
```

Resume after an interruption:

```bash
python scripts/run_hemorrhage_variants.py --gpu 0 --resume
```

Run selected experiments only:

```bash
python scripts/run_hemorrhage_variants.py --only H-N20 H-S20 --gpu 0
```

The runner creates the official nnU-Net ResEnc M plans on first use. It stops immediately if an experiment fails. Per-experiment logs are stored under `runs/hemorrhage-variants/logs/`.
