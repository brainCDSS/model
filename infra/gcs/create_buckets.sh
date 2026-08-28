#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-${BRAINON_GCP_PROJECT:-}}"
REGION="${2:-${BRAINON_GCP_REGION:-}}"
if [[ -z "$PROJECT_ID" || -z "$REGION" ]]; then
  echo "Usage: $0 PROJECT_ID REGION" >&2
  exit 2
fi

for suffix in raw datasets experiments models; do
  bucket="gs://${PROJECT_ID}-brainon-${suffix}"
  gcloud storage buckets create "$bucket" \
    --project="$PROJECT_ID" --location="$REGION" --default-storage-class=STANDARD \
    --uniform-bucket-level-access --public-access-prevention
done

