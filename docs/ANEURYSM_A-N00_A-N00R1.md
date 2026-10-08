# TopAneu nnU-Net A-N00 / A-N00R1

## 1. 실험 목적

TopAneu 2026 MRA 데이터를 이용한 뇌동맥류 Location segmentation에서
nnU-Net baseline인 `A-N00`과 corrected retrain인 `A-N00R1`의
학습 기준, 변경사항, Resume 정책 및 최종 검증 기준을 기록한다.

---

## 2. 실험 구분

### A-N00

Standard nnU-Net baseline.

- 질환: 뇌동맥류
- Target: Location
- Dataset: `Dataset501_TopAneuMRA_Location`
- Configuration: `3d_fullres`
- Fold: `0`
- Trainer: `nnUNetTrainerAneurysmEarlyStopping`
- Output: 53 classes

### A-N00R1

A-N00의 구조 자체는 유지하면서 sparse 53-class 학습 조건을 보완한
corrected retrain 실험이다.

- Architecture는 Standard nnU-Net PlainConvUNet 유지
- Dataset / Split 유지
- Output 53 classes 유지
- Trainer를 `nnUNetTrainerAneurysmSparse53Retrain`으로 변경
- Loss 및 foreground sampling 정책 보완
- EarlyStopping 시작 시점을 조정

---

## 3. 데이터 및 Split 기준

전체 환자 수는 307명이다.

- Train: 215명
- Validation: 46명
- Test: 46명

Dataset501은 Location multiclass segmentation 용도이다.

- Background: 1 class
- Location: 52 classes
- 총 Output: 53 classes

Train / Validation split은 고정된 `splits_final.json`을 사용한다.

Test 46명은 학습 및 Validation split에 포함하지 않는다.

---

## 4. A-N00 학습 기준

### 학습

- Max epochs: 300
- Train iterations / epoch: 250
- Online validation iterations / epoch: 50
- Learning rate: 0.01
- Weight decay: 0.00003
- Foreground oversampling: 0.33

### Checkpoint

- Best monitor: `ema_fg_dice`
- `checkpoint_best.pth`: 최고 `ema_fg_dice` 기준
- Epoch complete checkpoint: 매 epoch 저장
- Resume: 사용

### EarlyStopping

- Monitor: `ema_fg_dice`
- Patience: 30
- min_delta: 0.0001
- Start epoch: 0

`min_delta`는 EarlyStopping patience reset 판단에 사용하며,
Best checkpoint 자체는 실제 최고 `ema_fg_dice`를 기준으로 선택한다.

---

## 5. A-N00R1 Corrected Retrain 변경사항

A-N00R1에서는 Standard nnU-Net 구조를 변경하지 않는다.

주요 변경사항은 학습 정책이다.

### Loss

`GT-present-only Dice + Cross Entropy`

기존 multiclass Dice 계산에서 해당 patch의 GT에 실제로 존재하지 않는
foreground class가 Dice 평균에 영향을 주는 문제를 줄이기 위해,
GT에 실제 존재하는 foreground class만 Dice 계산에 포함한다.

GT foreground가 존재하지 않는 경우 해당 foreground Dice는
인위적인 보상으로 사용하지 않는다.

### Foreground Sampling

- Probabilistic foreground sampling 사용
- `oversample_foreground_percent: 0.75`

### EarlyStopping

- Monitor: `ema_fg_dice`
- Start epoch: 80
- Patience: 40
- min_delta: 0.0001

### 그대로 유지되는 항목

- Standard nnU-Net PlainConvUNet
- Dataset501
- `3d_fullres`
- Fold 0
- Train 215명
- Validation 46명
- 53 outputs
- Max epochs 300
- Train iterations 250
- Online validation iterations 50
- Strict Resume
- Final 46-patient full-volume validation

---

## 6. Strict Resume 정책

Colab runtime interruption에 대응하기 위해
완료된 최소 작업 단위 다음부터 재개한다.

### Train

마지막으로 정상 완료된 optimizer iteration 다음 iteration부터 재개한다.

### Online Validation

마지막으로 정상 완료된 validation iteration 다음부터 재개한다.

### Final Patient Validation

마지막으로 완료된 환자 다음 환자부터 재개한다.

Resume에는 다음 상태를 함께 보존한다.

- Model
- Optimizer
- GradScaler
- Logger
- Epoch
- 현재 Phase
- 부분 Loss
- RNG State
- Best metric

Data augmentation worker는 strict resume 일관성을 위해 0으로 사용한다.

---

## 7. Final Validation 정책

Full-volume patient validation은 매 epoch 수행하지 않는다.

모든 학습이 끝난 후 `checkpoint_best.pth`를 이용하여
Validation 환자 46명에 대해 딱 1회 수행한다.

주요 평가 항목:

- Patient Mean Dice
- Patient Median Dice
- Patient Mean Macro Dice
- Patient Mean Precision
- Patient Mean Recall
- Macro Dice
- Macro Precision
- Macro Recall
- Negative-case FP Rate

Patient Macro Dice는 각 환자의 GT에 실제 존재하는 Location class만
동일 가중치로 평균한다.

Final Validation 진행 상태 역시 환자 단위로 저장하여
runtime interruption 후 다음 미완료 환자부터 이어서 수행한다.

완료 시:

`final_validation/COMPLETE.json`

을 기록한다.

---

## 8. 설정 파일

Baseline:

`configs/experiments/aneurysm/A-N00.yaml`

Corrected Retrain:

`configs/experiments/aneurysm/A-N00R1.yaml`

A-N00R1 YAML은 팀 공통으로 사용 중인 A-N00 YAML의
항목 구조, 순서 및 주석 형식을 최대한 그대로 유지한다.

A-N00R1에서 실제 달라진 값만 변경한다.

---

## 9. Trainer

A-N00:

`src/brainon_ai/nnunet/trainers/nnUNetTrainerAneurysmEarlyStopping.py`

A-N00R1:

`src/brainon_ai/nnunet/trainers/nnUNetTrainerAneurysmSparse53Retrain.py`

---

## 10. GitHub 결과 저장 정책

GitHub에는 재현과 검토에 필요한 소형 산출물만 저장한다.

포함 가능:

- 학습 설정
- Trainer
- Launcher
- Validation 확인 코드
- Colab Notebook
- 학습기준 문서
- JSON
- CSV
- 소형 학습 Log
- 결과 요약

제외:

- `checkpoint_latest.pth`
- `checkpoint_best.pth`
- `checkpoint_final.pth`
- Dataset501 실제 영상 및 Mask
- nnU-Net preprocessed 대용량 파일
- Prediction volume

Final Validation이 완료되지 않은 상태의 중간 metric은
최종 모델 성능으로 기록하지 않는다.
