# GCS infrastructure examples

These scripts are reviewed examples only; this project never invokes them. They require project ID and region as arguments or environment variables, create four Standard-class buckets with uniform bucket-level access and public access prevention, and do not create service-account keys. Authenticate locally with Application Default Credentials:

```bash
gcloud auth application-default login
```

Review IAM and lifecycle examples before applying them. No destructive rsync or delete operation is included.

