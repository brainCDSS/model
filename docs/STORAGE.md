# Storage

The `file://` and `gs://` backends share list, exists, download and upload operations. Downloads are staged as `.partial`, hashes are checked before rename, and retries protect GCS transfers. Sync is non-deleting and supports dry-run. No delete operation is exposed.

Training writes checkpoints locally first. Cloud layouts use immutable object names and pointer-last publication:

```text
checkpoints/periodic/epoch-0050.ckpt
checkpoints/selected/best-primary.ckpt
checkpoints/selected/best-balanced.ckpt
pointers/latest.json
```

The model release root is `gs://PROJECT_ID-brainon-models/releases/<task>/<version>/`. The inference service receives only the release manifest and GCS URI.

