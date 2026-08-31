# BrainCDSS Model Repository 개발 가이드

> 대상 저장소: `brainCDSS/model`
> 기준 브랜치: `dev`
> 목적: CT 출혈, MRI 허혈성 뇌졸중, MRA 동맥류 segmentation 모델을 재현 가능하게 개발·비교·평가·릴리스하기 위한 저장소 구조와 작업 위치를 정리한다.

---

## 1. 이 저장소의 역할

이 저장소는 **AI 모델 개발의 Source of Truth**이다.

여기에서 관리하는 것은 다음과 같다.

- 데이터셋 메타데이터와 split 정보
- 데이터 품질검사 및 전처리 코드
- nnU-Net 실행/확장 코드
- SegResNet 등 직접 관리하는 모델 코드
- 실험 설정 YAML
- 학습·평가·추론 공통 로직
- 평가 지표
- 시각화
- 테스트
- 모델 release metadata와 inference output contract

반대로 다음은 Git에 직접 저장하지 않는다.

- 실제 CT/MRI/MRA 원본 데이터
- 전체 전처리 결과 데이터
- 대용량 `.ckpt`, `.pth`, nnU-Net checkpoint
- 실제 운영 환자 데이터
- 머신별 절대경로

대용량 데이터와 모델 artifact는 로컬 cache 또는 GCS를 사용한다.

---

# 2. 전체 구조

```text
model/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
│
├─ configs/
│  ├─ data/
│  ├─ experiments/
│  │  ├─ hemorrhage/
│  │  ├─ ischemia/
│  │  └─ aneurysm/
│  ├─ preprocessing/
│  ├─ runtime/
│  └─ storage/
│
├─ contracts/
│  ├─ inference_output.schema.json
│  └─ model_release.schema.json
│
├─ docs/
│  ├─ DATASETS.md
│  ├─ EXPERIMENTS.md
│  ├─ METRICS.md
│  ├─ STORAGE.md
│  ├─ COLAB.md
│  ├─ PREPROCESSING.md
│  ├─ HEMORRHAGE_VARIANTS.md
│  └─ REPOSITORY_MODEL_DEVELOPMENT_GUIDE.md
│
├─ infra/
├─ notebooks/
│
├─ registry/
│  ├─ datasets/
│  ├─ manifests/
│  ├─ models/
│  └─ splits/
│
├─ scripts/
│  ├─ preprocessing/
│  └─ run_hemorrhage_variants.py
│
├─ src/
│  └─ brainon_ai/
│     ├─ cli.py
│     ├─ common/
│     ├─ data/
│     ├─ engine/
│     ├─ experiments/
│     ├─ export/
│     ├─ metrics/
│     ├─ models/
│     ├─ nnunet/
│     ├─ storage/
│     └─ visualization/
│
├─ tests/
├─ pyproject.toml
├─ requirements-nnunet-cu130.txt
├─ .env.example
└─ README.md
```

---

# 3. 가장 중요한 원칙

저장 위치를 결정할 때 아래 기준으로 생각하면 된다.

```text
실제로 재사용되는 Python 로직
→ src/brainon_ai/

실험마다 바뀌는 값
→ configs/

사람이 직접 실행하는 보조 명령/일괄 작업
→ scripts/

데이터/실험/모델의 버전 정보
→ registry/

설명과 작업 규칙
→ docs/

모델 결과 형식
→ contracts/

검증 코드
→ tests/
```

즉 **노트북이나 scripts 안에 핵심 학습 로직을 복사해서 만들지 않는다.**

핵심 로직은 `src/brainon_ai/`에 한 번만 구현하고, notebook·CLI·script에서는 그것을 호출한다.

---

# 4. `configs/` — 모델과 실험의 설정값

## `configs/data/`

질환별 데이터 입력 정의를 둔다.

현재:

```text
configs/data/
├─ hemorrhage.yaml
├─ ischemia.yaml
└─ aneurysm.yaml
```

여기에 들어갈 내용 예:

- modality
- channel 수
- dataset 이름
- label 정의
- manifest 위치
- dataset release ID

예시 개념:

```yaml
task: ischemia
modalities:
  - DWI
  - ADC
channels: 2
```

**학습률, batch size 같은 모델 학습 설정은 여기 넣지 않는다.**

---

## `configs/preprocessing/`

직접 관리하는 전처리 pipeline의 parameter를 둔다.

예:

```text
configs/preprocessing/
├─ hemorrhage_single_window.yaml
└─ hemorrhage_multi_window.yaml
```

들어갈 수 있는 내용:

- orientation
- target spacing
- interpolation
- HU clipping/window
- normalization
- binary mask 변환

중요:

- **SegResNet용 직접 전처리 설정**은 여기서 관리한다.
- **nnU-Net의 자동 preprocessing plan 자체를 여기서 직접 다시 구현하지 않는다.**
- nnU-Net은 `nnUNetPlans.json`과 framework preprocessing 결과를 기준으로 재현한다.

현재 두 hemorrhage preprocessing YAML은 비어 있으므로 실제 전처리 실험을 사용한다면 내용 확정이 필요하다.

---

## `configs/experiments/`

**가장 많이 수정하게 될 곳**이다.

모델 구조 코드를 복사해서 실험하지 않고, 가능한 변경은 YAML로 표현한다.

```text
configs/experiments/
├─ hemorrhage/
│  ├─ H-N00.yaml
│  ├─ H-N10.yaml
│  ├─ H-N20.yaml
│  ├─ H-S00.yaml
│  ├─ H-S10.yaml
│  └─ ...
├─ ischemia/
│  ├─ I-N00.yaml
│  ├─ I-S00.yaml
│  └─ ...
└─ aneurysm/
   ├─ A-N00.yaml
   ├─ A-S00.yaml
   └─ ...
```

권장 ID 규칙:

```text
H = Hemorrhage
I = Ischemia
A = Aneurysm

N = nnU-Net
S = SegResNet
```

예:

```text
H-N00 = 출혈 nnU-Net baseline
H-S00 = 출혈 SegResNet baseline
I-N00 = 허혈 nnU-Net baseline
I-S00 = 허혈 SegResNet baseline
A-N00 = 동맥류 nnU-Net baseline
A-S00 = 동맥류 SegResNet baseline
```

실험 설정 예:

- model architecture option
- loss
- optimizer
- learning rate
- patch size
- batch size
- augmentation
- epochs
- threshold
- seed
- checkpoint selection metric

즉 **실험 차이는 우선 YAML로 표현하고, 실제 network logic 변경이 필요한 경우에만 `src/`를 수정한다.**

---

# 5. `src/brainon_ai/` — 실제 모델 개발 코드

이 폴더가 저장소의 핵심이다.

---

## `common/`

모든 모델에서 공유하는 범용 기능.

현재:

```text
common/
├─ checksums.py
├─ config.py
└─ runtime.py
```

역할:

- SHA-256
- config loading
- 실행 환경
- seed/runtime/device 공통 처리

### 여기에 넣을 것

CT/MRI/MRA, nnU-Net/SegResNet 모두에서 공통으로 사용하는 코드.

### 넣지 말 것

```text
hemorrhage 전용 preprocessing
SegResNet 전용 loss
nnU-Net custom trainer
```

공통이 아닌 것은 각 domain에 둔다.

---

# 6. `data/` — 데이터 처리 로직

현재:

```text
src/brainon_ai/data/
├─ hemorrhage/
├─ ischemia/
├─ aneurysm/
├─ preprocessing.py
├─ quality.py
├─ registry.py
└─ synthetic.py
```

## `quality.py`

데이터 품질검사 코드를 둔다.

예:

- image-mask pairing
- shape mismatch
- spacing mismatch
- affine mismatch
- orientation
- NaN/Inf
- intensity 범위
- mask label
- empty mask
- foreground voxel

**모델 학습 전에 수행하는 QA 로직**은 여기다.

---

## `preprocessing.py`

재사용 가능한 실제 preprocessing function을 둔다.

예:

```text
RAS orientation
signed CT correction
binary mask conversion
resampling
clipping/window
normalization
```

중요:

`scripts/preprocessing/*.py`와 역할이 다르다.

```text
src/brainon_ai/data/preprocessing.py
= 실제 재사용 가능한 전처리 함수

scripts/preprocessing/*.py
= 사용자가 실행하는 workflow/entry script
```

---

## `data/hemorrhage`, `data/ischemia`, `data/aneurysm`

질환별로 데이터 loader나 modality-specific 처리 차이가 커질 경우 사용한다.

예:

```text
hemorrhage
- CT HU handling

ischemia
- DWI + ADC channel pairing

aneurysm
- MRA-specific manifest/filtering
```

현재 질환별 폴더는 대부분 skeleton 수준이므로 향후 질환별 logic이 필요할 때 확장한다.

---

# 7. `models/` — 직접 관리하는 모델 코드

현재 구조:

```text
src/brainon_ai/models/
├─ common/
├─ hemorrhage/
│  └─ segresnet/
├─ ischemia/
└─ aneurysm/
```

현재 실제 SegResNet 구현은 hemorrhage 쪽이 가장 많이 작성되어 있다.

```text
models/hemorrhage/segresnet/
├─ network.py
├─ losses.py
├─ data.py
├─ metrics.py
├─ inference.py
├─ evaluate.py
├─ checkpointing.py
├─ pruning.py
├─ reproducibility.py
├─ sanity.py
├─ config.py
└─ ...
```

각 파일 역할:

| 파일 | 역할 |
|---|---|
| `network.py` | SegResNet architecture 생성 |
| `losses.py` | Dice/Tversky/CE 등 loss |
| `data.py` | Dataset/DataLoader/transform |
| `inference.py` | sliding-window inference 등 |
| `evaluate.py` | validation/test evaluation |
| `metrics.py` | 모델 전용 metric aggregation |
| `checkpointing.py` | checkpoint save/load/best model |
| `pruning.py` | early stopping/pruning |
| `reproducibility.py` | seed/determinism |
| `sanity.py` | 학습 전 sanity check |
| `config.py` | SegResNet config validation/loading |

---

## 허혈성 SegResNet을 만들 때

권장:

```text
src/brainon_ai/models/ischemia/segresnet/
```

단, hemorrhage SegResNet 코드를 통째로 복사하기 전에 공통 부분을 확인한다.

```text
models/common/
```

으로 뺄 수 있는 기능은 공통화하고, 다음처럼 modality/task 차이만 질환 폴더에 둔다.

```text
ischemia
- in_channels = 2 (DWI + ADC)
- ischemia-specific augmentation
- task-specific config
```

---

## 동맥류 SegResNet을 만들 때

권장:

```text
src/brainon_ai/models/aneurysm/segresnet/
```

MRA-specific 입력, augmentation, patch 전략 등이 필요하면 여기서 관리한다.

---

# 8. `nnunet/` — nnU-Net을 사용하는 코드

현재:

```text
src/brainon_ai/nnunet/
├─ config.py
├─ runner.py
├─ losses/
├─ networks/
├─ pruning/
└─ trainers/
```

nnU-Net은 SegResNet과 관리 방식이 다르다.

## 핵심 원칙

**nnU-Net 공식 source code 전체를 이 저장소에 복사하지 않는다.**

`nnunetv2` package는 dependency로 설치하고 우리 프로젝트에서 필요한 부분만 wrapper/custom extension으로 둔다.

```text
runner.py
→ nnU-Net 명령 실행 orchestration

config.py
→ 프로젝트용 nnU-Net 설정 처리

trainers/
→ custom nnUNetTrainer가 필요할 때

losses/
→ nnU-Net loss를 실제로 변경할 때

networks/
→ nnU-Net architecture extension이 필요할 때

pruning/
→ pruning/early stopping extension이 필요할 때
```

### Baseline nnU-Net

기본 nnU-Net을 그대로 사용한다면 `network.py`를 새로 만들 필요가 없다.

```text
Dataset 생성
→ fingerprint
→ experiment planning
→ preprocessing
→ nnUNetTrainer 학습
```

을 framework가 수행한다.

### nnU-Net 구조를 변경할 때

예:

```text
Attention 추가
custom loss
custom trainer
pruning
```

이 경우에만 `src/brainon_ai/nnunet/` 내부에 구현한다.

---

# 9. `scripts/` — 실행용 스크립트

현재:

```text
scripts/
├─ preprocessing/
│  ├─ build_nnunet_raw.py
│  ├─ preprocess_hemorrhage.py
│  ├─ validate_preprocessing.py
│  ├─ generate_release_metadata.py
│  └─ legacy/
└─ run_hemorrhage_variants.py
```

여기는 **사람이 직접 실행하는 workflow entry point**다.

### `build_nnunet_raw.py`

원본/통합 데이터셋을 nnU-Net 표준 구조로 변환한다.

```text
nnUNet_raw/
└─ DatasetXXX_Name/
   ├─ imagesTr/
   ├─ labelsTr/
   └─ dataset.json
```

### `preprocess_hemorrhage.py`

출혈 데이터 전처리 pipeline 실행용 script.

### `validate_preprocessing.py`

전처리 후 결과가 정상인지 검증.

### `generate_release_metadata.py`

전처리 data release의 metadata 생성.

### 원칙

`scripts/`에 중요한 알고리즘을 길게 구현하기보다는 `src/brainon_ai/` 함수를 호출하도록 유지한다.

---

# 10. `engine/` — 공통 학습 engine

현재 `trainer.py`가 있다.

역할:

- epoch loop
- optimizer step
- AMP
- validation 호출
- checkpoint 연결
- logging

모델마다 trainer loop를 복사해서 만들지 않고 가능한 범위에서 여기의 공통 engine을 사용한다.

단, nnU-Net은 자체 training engine이 있기 때문에 별도로 `nnunet/` wrapper/custom trainer를 사용한다.

---

# 11. `metrics/` — 공통 평가 지표

현재:

```text
src/brainon_ai/metrics/segmentation.py
```

여기에는 모델에 관계없이 공통으로 비교할 지표를 둔다.

권장 공통 지표:

- Dice
- lesion-only Dice
- Recall/Sensitivity
- Precision
- lesion F1
- HD95
- volume error
- empty-GT false positive rate

nnU-Net과 SegResNet을 공정하게 비교하려면 **최종 evaluation metric 계산은 가능한 한 동일한 공통 구현을 사용**하는 것이 좋다.

---

# 12. `visualization/`

현재:

```text
visualization/plots.py
```

역할:

- training curve
- metric plot
- prediction overlay
- 비교 그래프
- 발표용 결과 figure

학습이나 metric 계산 로직은 넣지 않는다.

---

# 13. `registry/` — 재현성의 핵심

```text
registry/
├─ datasets/
├─ manifests/
├─ splits/
└─ models/
```

## `datasets/`

데이터 release 정보.

예:

```text
hemorrhage_binary_v1.0.0
isles2022_dwi_adc_v1.0.0
aneurysm_mra_v1.0.0
```

## `manifests/`

case 단위 파일 목록과 source 정보를 관리.

예:

```text
case_id,image_path,label_path,source
```

실제 의료영상 파일 자체가 아니라 **어떤 파일이 어떤 실험에 사용됐는지 재현하기 위한 목록**이다.

## `splits/`

train/val/test split을 버전 관리한다.

```text
train
validation
test
```

모델마다 임의로 split을 다시 만들지 않는다.

nnU-Net과 SegResNet 비교에서는 특히 **동일 split 사용**이 중요하다.

## `models/`

최종 모델의 version/metadata 기록.

checkpoint binary 자체는 Git에 저장하지 않는다.

---

# 14. `contracts/` — 다른 시스템과의 연결 규격

현재:

```text
contracts/
├─ inference_output.schema.json
└─ model_release.schema.json
```

## `inference_output.schema.json`

모델 추론 결과가 Backend/Inference 서비스에 어떤 형태로 전달되는지 정의한다.

예:

```text
case_id
model_version
mask URI
volume_ml
confidence
shape
spacing
processing time
```

## `model_release.schema.json`

최종 모델 release package metadata 형식.

모델을 바꿔도 Backend가 학습 코드를 알 필요가 없도록 **model artifact + version + manifest 계약**만 맞춘다.

---

# 15. `export/`

현재:

```text
src/brainon_ai/export/release.py
```

역할:

```text
학습 완료
→ best checkpoint 선택
→ release metadata 작성
→ schema 검증
→ GCS release 경로 생성
```

최종 배포 모델은 예를 들어 다음처럼 관리한다.

```text
gs://PROJECT_ID-brainon-models/releases/<task>/<version>/
```

GitHub에는 checkpoint를 올리지 않는다.

---

# 16. `storage/`

현재:

```text
src/brainon_ai/storage/backends/
```

역할:

- local filesystem
- GCS

같은 storage 차이를 모델 코드에서 분리한다.

모델 코드는 가능하면

```python
open("gs://...")
```

처럼 GCS를 직접 다루지 않고 storage abstraction을 통해 사용한다.

---

# 17. `notebooks/`

용도:

- 데이터 탐색
- visualization
- 빠른 실험
- Colab entry

하지만 notebook에만 존재하는 학습/전처리 코드를 만들면 안 된다.

잘못된 구조:

```text
notebook 안에 500줄짜리 SegResNet 학습 코드
```

권장:

```python
from brainon_ai... import train
```

처럼 package를 호출한다.

Notebook은 **실험 UI**이고 실제 source는 `src/`다.

---

# 18. `tests/`

현재 다음 검증이 존재한다.

```text
tests/
├─ test_checksums.py
├─ test_config.py
├─ test_core.py
├─ test_metrics.py
├─ test_model_forward.py
├─ test_smoke_train.py
└─ test_storage_local.py
```

새 기능을 추가하면 해당 테스트도 같이 추가한다.

예:

```text
새 loss
→ loss unit test

새 model
→ forward shape test

새 preprocessing
→ synthetic NIfTI preprocessing test

새 config
→ config parsing/validation test
```

실제 의료 데이터를 CI에 넣지 않고 synthetic fixture를 사용한다.

---

# 19. 모델별 실제 개발 위치

## A. CT 출혈

입력:

```text
CT 1 channel
```

### nnU-Net

```text
configs/experiments/hemorrhage/H-Nxx.yaml
src/brainon_ai/nnunet/
scripts/preprocessing/build_nnunet_raw.py
registry/splits/
```

### SegResNet

```text
configs/experiments/hemorrhage/H-Sxx.yaml
src/brainon_ai/models/hemorrhage/segresnet/
src/brainon_ai/data/preprocessing.py
src/brainon_ai/metrics/segmentation.py
```

현재 세 질환 중 SegResNet 구현이 가장 진행된 영역이다.

---

## B. MRI 허혈성 뇌졸중

입력:

```text
DWI + ADC = 2 channels
```

### nnU-Net

```text
configs/data/ischemia.yaml
configs/experiments/ischemia/I-Nxx.yaml
src/brainon_ai/nnunet/
scripts/preprocessing/build_nnunet_raw.py
registry/splits/
```

nnU-Net dataset에서는 예를 들어:

```text
case001_0000.nii.gz = DWI
case001_0001.nii.gz = ADC
```

처럼 channel을 명확하게 유지한다.

### SegResNet

향후 실제 구현 위치:

```text
src/brainon_ai/models/ischemia/segresnet/
```

필요한 차이:

- `in_channels=2`
- DWI/ADC pairing validation
- MRI-specific normalization
- task-specific augmentation

현재 `models/ischemia/`는 skeleton 수준이므로 구현이 필요하다.

---

## C. MRA 동맥류

입력:

```text
MRA 1 channel
```

### nnU-Net

```text
configs/data/aneurysm.yaml
configs/experiments/aneurysm/A-Nxx.yaml
src/brainon_ai/nnunet/
scripts/preprocessing/build_nnunet_raw.py
registry/splits/
```

### SegResNet

향후 실제 구현 위치:

```text
src/brainon_ai/models/aneurysm/segresnet/
```

MRA는 CT보다 3D volume이 크고 spacing 특성이 다르므로 MRA-specific patch/augmentation/preprocessing은 aneurysm data/model 영역에 둔다.

현재 `models/aneurysm/` 역시 skeleton 수준이다.

---

# 20. 새로운 모델 실험을 추가하는 순서

모델 하나를 만들 때 다음 순서를 기준으로 한다.

```text
1. Dataset 확정
        ↓
2. QA
        ↓
3. Dataset release / manifest 확정
        ↓
4. train/val/test split 확정
        ↓
5. preprocessing 확정
        ↓
6. model baseline 구현
        ↓
7. experiment YAML 작성
        ↓
8. synthetic/smoke test
        ↓
9. full training
        ↓
10. validation/test evaluation
        ↓
11. visualization
        ↓
12. experiment 결과 기록
        ↓
13. best model 선정
        ↓
14. model release export
        ↓
15. GCS upload
        ↓
16. inference integration
```

---

# 21. 작업별 파일 위치 빠른 표

| 하고 싶은 작업 | 위치 |
|---|---|
| 데이터 품질검사 추가 | `src/brainon_ai/data/quality.py` |
| 공통 preprocessing 함수 추가 | `src/brainon_ai/data/preprocessing.py` |
| 질환 전용 데이터 처리 | `src/brainon_ai/data/<task>/` |
| nnU-Net raw 구조 생성 | `scripts/preprocessing/build_nnunet_raw.py` |
| SegResNet 구조 변경 | `src/brainon_ai/models/<task>/segresnet/network.py` |
| SegResNet loss 변경 | `src/brainon_ai/models/<task>/segresnet/losses.py` |
| SegResNet Dataset/transform 변경 | `src/brainon_ai/models/<task>/segresnet/data.py` 또는 `data/<task>/` |
| nnU-Net custom trainer | `src/brainon_ai/nnunet/trainers/` |
| nnU-Net custom loss | `src/brainon_ai/nnunet/losses/` |
| nnU-Net network extension | `src/brainon_ai/nnunet/networks/` |
| 실험 hyperparameter 변경 | `configs/experiments/<task>/` |
| 전처리 parameter 변경 | `configs/preprocessing/` |
| 공통 metric 추가 | `src/brainon_ai/metrics/` |
| training engine 수정 | `src/brainon_ai/engine/` |
| plot/overlay 추가 | `src/brainon_ai/visualization/` |
| dataset version 기록 | `registry/datasets/` |
| case manifest 기록 | `registry/manifests/` |
| split 기록 | `registry/splits/` |
| model metadata 기록 | `registry/models/` |
| 모델 release 생성 | `src/brainon_ai/export/` |
| GCS/local 처리 | `src/brainon_ai/storage/` |
| inference 결과 규격 수정 | `contracts/inference_output.schema.json` |
| release 규격 수정 | `contracts/model_release.schema.json` |
| 테스트 추가 | `tests/` |
| 설명 문서 | `docs/` |

---

# 22. 코드와 설정을 어디에 둘지 판단하는 예

## 예 1. learning rate를 `1e-4 → 5e-5`로 변경

```text
코드 수정 X
configs/experiments/...yaml 수정 O
```

## 예 2. SegResNet에 Attention module 추가

```text
src/brainon_ai/models/<task>/segresnet/network.py
```

그리고 새 실험 YAML 생성.

## 예 3. DiceCE → TverskyCE 변경

```text
loss 구현
→ models/<task>/segresnet/losses.py

사용 여부/weight
→ configs/experiments/<task>/<experiment>.yaml
```

## 예 4. CT HU window 변경

```text
전처리 구현
→ data/preprocessing.py

window 값
→ configs/preprocessing/*.yaml

실험 연결
→ configs/experiments/hemorrhage/*.yaml
```

## 예 5. nnU-Net baseline 실행

```text
nnU-Net source code 복사 X

build_nnunet_raw
→ nnUNetv2_plan_and_preprocess
→ nnUNetv2_train
```

프로젝트 wrapper가 필요하면 `src/brainon_ai/nnunet/runner.py`를 사용한다.

---

# 23. 현재 저장소에서 정리해야 할 부분

## 1. 루트의 `model` 항목

현재 저장소 루트에 다시 `model`이라는 gitlink가 존재한다.

```text
model/
└─ model  ← nested Git repository 흔적
```

Git mode `160000`으로 기록된 항목이며 정상적인 source directory가 아니다.

의도적인 submodule이 아니라면 제거해야 한다.

저장소 자체가 이미 `brainCDSS/model`이므로 내부에 다시 `model` repository를 둘 필요가 없다.

---

## 2. `docs/PREPROCESSING.md`

현재 빈 파일이다.

최종적으로 다음을 기록하는 문서로 사용하는 것을 권장한다.

```text
공통 QA
CT preprocessing
MRI preprocessing
MRA preprocessing
SegResNet preprocessing
nnU-Net preprocessing
interpolation
normalization
orientation
release version
```

---

## 3. `configs/preprocessing/*.yaml`

현재 hemorrhage preprocessing YAML들이 비어 있다.

실제 pipeline이 확정되면 config 값을 넣어 재현성을 확보해야 한다.

---

## 4. `models/ischemia`, `models/aneurysm`

현재 사실상 skeleton이다.

SegResNet baseline을 돌릴 계획이라면 다음이 필요하다.

```text
models/ischemia/segresnet/
models/aneurysm/segresnet/
```

단 hemorrhage 코드를 그대로 2번 복사하기 전에 공통화 가능한 부분은 `models/common/` 또는 공통 engine으로 이동한다.

---

# 24. Branch 운영

현재 branch:

```text
main
└─ 안정/공유 기준

dev
└─ 실제 개발 및 검증
```

일반 작업:

```text
dev
↓
테스트/실험 검증
↓
Pull Request
↓
main
```

큰 구조 변경이 동시에 진행될 때만:

```text
feature/<topic>
→ dev
→ main
```

실험 하나마다 Git branch를 만들지 않는다.

```text
H-S10
H-S20
H-N10
```

같은 실험 ID와 YAML, run metadata를 사용한다.

---

# 25. 최종적으로 기억할 구조

모델을 개발하면서 가장 자주 볼 위치는 사실 아래 다섯 곳이다.

```text
1. src/brainon_ai/data/
   → 데이터 품질검사/전처리

2. src/brainon_ai/models/
   → SegResNet 등 직접 구현하는 모델

3. src/brainon_ai/nnunet/
   → nnU-Net 실행/확장

4. configs/experiments/
   → 실험별 설정

5. registry/
   → 어떤 데이터/split/model을 썼는지 기록
```

그리고 전체 흐름은 다음 한 줄로 기억하면 된다.

```text
Data
→ QA
→ Preprocessing
→ Dataset/Split Registry
→ Model
→ Experiment Config
→ Train
→ Evaluate
→ Release
→ GCS
→ Inference
```

이 구조를 유지하면 **CT/MRI/MRA × nnU-Net/SegResNet의 6개 baseline 및 후속 실험을 같은 규칙 안에서 관리할 수 있다.**
