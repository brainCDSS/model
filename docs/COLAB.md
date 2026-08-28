# Colab

Set `BRANCH=main` for shared/released execution or `dev` for development validation.

```python
REPO_URL = "https://github.com/brainCDSS/model.git"
BRANCH = "main"
!git clone --branch "$BRANCH" "$REPO_URL" model
%cd model
!python -m pip install -e ".[notebook,storage]"
```

For a private repository, read a GitHub token from Colab Secrets into an environment variable for the clone operation and immediately remove it; never embed it in the URL cell source, print it, or retain cell output. Set `BRAINON_CACHE_ROOT` to local Colab storage and the four GCS URIs through secrets/environment variables. Use ADC/DWI or other data only after authorized acquisition.
