# TopAneu nnU-Net A-N00-TYPE

## 1. 실험 목적

TopAneu 2026 MRA 데이터의 `type_masks`를 이용하여
뇌동맥류 Type segmentation의 Standard nnU-Net Baseline을 구축한다.

Location mask 실험과 동일한 학습·검증·Resume 기준을 유지하여
Location target과 Type target의 성능을 독립적으로 비교할 수 있도록 한다.

---

## 2. 실험 ID

`A-N00-TYPE`

---

## 3. Target

`type`

Type mask는 다음 4개 output을 사용한다.

- Background
- Type label 1
- Type label 2
- Type label 3

Type 1/2/3의 실제 임상 명칭은 Dataset502 생성 단계에서
원본 Type metadata를 기준으로 확정하며 임의로 작성하지 않는다.

---

## 4. Dataset

nnU-Net Dataset ID:

`Dataset502`

예정 Dataset name:

`Dataset502_TopAneuMRA_Type`

원본:

- MRA Images: 307건
- Type masks: 307건

TYPE STEP 0 기준 Image-Type mask 누락은 0건이다.

---

## 5. Split 정책

Location mask 실험과 동일한 frozen split V2를 사용한다.

전체:

307명

구성:

- Train: 215명
- Validation: 46명
- Test: 46명

Seed:

`20260828`

Train/Validation split은 기존 고정 split을 그대로 사용하며
Type 실험을 위해 새로 random split하지 않는다.

Test 46명은 Train 및 Validation에 포함하지 않는다.

---

## 6. Baseline Model

Standard nnU-Net baseline을 사용한다.

- Architecture: Standard nnU-Net PlainConvUNet
- Configuration: `3d_fullres`
- Fold: `0`
- Trainer: `nnUNetTrainerAneurysmTypeBaseline`
- Output classes: 4

Type baseline의 목적은 추가 Attention, Residual Encoder,
특수 Loss 등을 적용하기 전에 Standard nnU-Net 자체의 기준 성능을
확보하는 것이다.

---

## 7. Training 기준

Location A-N00 Baseline과 동일하게 유지한다.

- Max epochs: 300
- Train iterations / epoch: 250
- Online validation iterations / epoch: 50
- Learning rate: 0.01
- Weight decay: 0.00003
- Foreground oversampling: 0.33

Type baseline 단계에서는 A-N00R1에서 사용했던
Sparse53 보정 Loss를 적용하지 않는다.

먼저 Standard nnU-Net baseline 성능을 확인한다.

---

## 8. Checkpoint 기준

Best checkpoint monitor:

`ema_fg_dice`

Best checkpoint:

`checkpoint_best.pth`

Epoch complete checkpoint:

매 epoch 저장

Resume:

활성화

`min_delta`는 EarlyStopping patience reset 판단에 사용하고,
Best checkpoint는 실제 최고 `ema_fg_dice`를 기준으로 선정한다.

---

## 9. EarlyStopping

- Enabled: true
- Monitor: `ema_fg_dice`
- Start epoch: 0
- Patience: 30
- min_delta: 0.0001

---

## 10. Strict Resume

Colab Runtime 중단에 대응하기 위해
Location A-N00에서 확정한 Strict Resume 정책을 유지한다.

### Train

마지막으로 완료된 optimizer iteration 다음부터 재개한다.

### Online Validation

마지막으로 완료된 validation iteration 다음부터 재개한다.

### Final Validation

마지막으로 완료된 환자 다음부터 재개한다.

Resume 대상에는 가능한 범위에서 다음 상태를 포함한다.

- Model
- Optimizer
- GradScaler
- Logger
- Epoch
- Phase
- Partial loss
- RNG state
- Best metric

Data augmentation worker는 Resume 일관성을 위해 0을 사용한다.

---

## 11. Final Validation

Full-volume Patient Validation은 매 epoch 수행하지 않는다.

학습 종료 후:

`checkpoint_best.pth`

를 이용하여 Validation 46명에 대해 딱 1회 수행한다.

Foreground binary patient metric에서는:

Type label 1~3을 foreground로 병합한다.

즉:

`labels_1_to_3_merged`

를 사용한다.

Type class별 Macro Dice는 각 환자의 GT에 실제 존재하는
Type class만 동일 가중치로 평균한다.

주요 metric:

- Patient Mean Dice
- Patient Median Dice
- Patient Macro Dice
- Patient Mean Macro Dice
- Patient Mean Precision
- Patient Mean Recall
- Macro Dice
- Macro Precision
- Macro Recall
- Negative-case FP Rate

완료 marker:

`final_validation/COMPLETE.json`

---

## 12. Python / CUDA 환경

Location nnU-Net 실험과 동일한 환경 기준을 사용한다.

Python:

`3.12.13`

Colab venv:

`/content/venv312`

Requirements:

`requirements-nnunet-cu130.txt`

Base Colab Python이나 별도 임의 pip 환경을 사용하지 않는다.

---

## 13. 관련 파일

Config:

`configs/experiments/aneurysm/A-N00-TYPE.yaml`

Trainer:

`src/brainon_ai/nnunet/trainers/nnUNetTrainerAneurysmTypeBaseline.py`

Launcher:

`scripts/training/run_a_n00_type.py`

Final Validation 확인:

`scripts/validation/check_a_n00_type_final_validation.py`

Colab Notebook:

`notebooks/21_A-N00_TypeMask_nnunet_baseline.ipynb`

---

## 14. GitHub 저장 정책

GitHub에는 재현과 검토에 필요한 파일을 저장한다.

본학습 전 포함:

- YAML
- Trainer
- Launcher
- Validation 확인 코드
- Colab Notebook
- 학습기준 Docs

학습 이후 실제 생성된 경우 포함 가능:

- 결과 README
- JSON
- CSV
- 소형 Training Log
- Final Validation Summary

GitHub 제외:

- `.pth` checkpoint
- 원본 MRA
- Type mask 원본
- nnU-Net preprocessed 대용량 데이터
- Prediction volume

실제 결과가 생성되기 전에는 결과값을 임의로 작성하지 않는다.

---

## 15. 본학습 전 필수 단계

작업사항1 이후 다음 순서로 진행한다.

1. Python 3.12.13 venv 준비
2. Dataset502 생성
3. 기존 frozen V2 split 적용
4. Dataset502 integrity 확인
5. nnU-Net plan/preprocess
6. 필수 smoke test
7. Standard nnU-Net 본학습 시작

본학습 전 불필요한 추가 실험은 수행하지 않는다.
