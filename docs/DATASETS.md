# Datasets

The NIfTI release is shared by MONAI and nnU-Net. Restricted originals remain outside Git and are acquired manually from official sources after accepting the applicable license or DUA.

| Task | Inputs | Release | nnU-Net dataset | Source |
|---|---|---|---|---|
| Hemorrhage | NCCT | `hemorrhage_binary_v1.0.0` | `Dataset101_HemorrhageBinary` | BHSD, INSTANCE 2022, BCIHM |
| Ischemia | DWI + ADC | `isles2022_dwi_adc_v1.0.0` | `Dataset102_IschemiaDwiAdc` | ISLES 2022 train 250; FLAIR retained as extra |
| Aneurysm | MRA | `topaneu2026_mra_v1.0.0` | `Dataset103_AneurysmMra` | TopAneu 2026; vessel labels/masks retained as extras |

All task labels are binary foreground. Manifest columns are fixed in `registry/datasets`. IDs use source prefixes such as `bhsd_00001`, `instance_00001`, and `bcihm_00001`; manifests contain no PHI. Pull only required task/split objects into the local cache, write `*.partial`, verify SHA-256, and create `READY.json` only after complete verification.

