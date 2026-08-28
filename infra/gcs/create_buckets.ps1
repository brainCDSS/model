param(
    [string]$ProjectId = $env:BRAINON_GCP_PROJECT,
    [string]$Region = $env:BRAINON_GCP_REGION
)
$ErrorActionPreference = "Stop"
if (-not $ProjectId -or -not $Region) { throw "Provide -ProjectId and -Region (or environment variables)." }

foreach ($Suffix in @("raw", "datasets", "experiments", "models")) {
    $Bucket = "gs://$ProjectId-brainon-$Suffix"
    gcloud storage buckets create $Bucket `
        --project=$ProjectId --location=$Region --default-storage-class=STANDARD `
        --uniform-bucket-level-access --public-access-prevention
}

