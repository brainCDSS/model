# IAM example

Grant identities only the least privilege required. Human data curators may need `roles/storage.objectUser` on raw/datasets; training identities need object viewer on datasets and object creator/user on experiments; release automation needs object user on models; inference needs object viewer on models. Prefer workload identity or user ADC. Do not create or commit service-account JSON keys.

Review bindings with your security owner before running commands. Bucket-level IAM is required because uniform bucket-level access is enabled.

