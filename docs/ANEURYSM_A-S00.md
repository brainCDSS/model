# A-S00 Standard SegResNet Location Baseline

`A-S00`은 TopAneu 2026 MRA Location Segmentation을 위한
Standard MONAI SegResNet baseline이다.

## 데이터 계약

- 입력: MRA 1 channel
- 출력: Background 1 + Location 52 = 53 classes
- Train / Validation / Test: 215 / 46 / 46
- Split seed: `20260828`
- Split 재생성 금지
- A-S00 개발 manifest에는 Train 215명과 Validation 46명만 포함
- Test 46명은 별도 최종 평가 전까지 사용하지 않음
- `type_masks`는 A-S00에서 사용하지 않음
- Empty Location mask는 유효한 음성 환자로 유지

SegResNet 입력은 다음 폴더만 사용한다.

- `SegResNet_TopAneu_MRA/images`
- `SegResNet_TopAneu_MRA/location_masks`
- `SegResNet_TopAneu_MRA/location_jsons`

`Dataset501_TopAneuMRA_Location`은 nnU-Net용이므로
SegResNet 학습 입력으로 사용하지 않는다.

## 모델·학습 계약

- Standard MONAI `SegResNet`
- Attention 또는 custom decoder 없음
- Patch `[X,Y,Z] = [128,192,80]`
- Train 250 iterations/epoch
- Online Patch Validation 50 iterations/epoch
- Maximum 300 epochs
- Best와 EarlyStopping monitor: `ema_fg_dice`
- Patience 30, min delta `0.0001`
- 매 epoch full-volume validation 금지
- 완료된 마지막 unit 다음부터 strict resume
- Data augmentation workers: 0

Physical batch size는 RTX 5090 full-patch capacity test가
통과한 뒤 확정한다. 통과 전에는 production YAML을 commit하지 않는다.

## 최종 Validation 계약

학습이 종료된 뒤 `checkpoint_best.pth`로 Validation 46명을
full-volume inference한다. 환자마다 prediction은 한 번만 생성하고
같은 prediction에서 모든 지표를 계산한다.

필수 결과:

- `final_validation/per_patient_metrics.csv`
- `final_validation/per_class_metrics.csv`
- `final_validation/summary.json`
- `final_validation/COMPLETE.json`

최종 완료 기준은 `checkpoint_final.pth`가 아니라
`final_validation/COMPLETE.json`이다.

## Notebook 계약

GitHub에는 데이터 준비·QA 노트북과 모델 구조·학습 노트북을 함께
저장한다. Notebook은 `src/brainon_ai/`의 전체 source를 열람하고
실행하는 작업 인터페이스이며, 핵심 학습 로직을 중복 구현하지 않는다.

인증 URL, 인증 코드, SSH key, token, 실제 NIfTI, checkpoint 및
대용량 학습 로그는 Git에 저장하지 않는다.

## A-S00 attempt-02 — foreground-collapse correction

첫 번째 실행은 train/validation DiceCE가 약 1.0으로 수렴하고
online foreground Dice가 31 epoch 동안 0.0을 유지하여
background collapse로 판정하였다.

모델 구조와 확정 계약은 유지한다.

- Standard MONAI SegResNet
- Location 53 classes
- Train/Validation/Test: 215/46/46
- Split seed: 20260828
- Type mask: 사용하지 않음
- Patch: 112 x 160 x 64
- Physical batch: 1
- Train/online validation: 250/50 iterations
- Max epoch: 300
- Monitor: ema_fg_dice
- Patience/min_delta: 30/0.0001
- Strict unit resume 유지
- Full-volume validation은 best checkpoint로 마지막에 46명 1회

교정 범위는 양성 환자에서 foreground crop을 항상 선택하도록
conditional foreground fraction을 1.0으로 변경하고, DiceCE의
CE weight를 0.25로 낮추는 것이다. 첫 번째 결과는 보존하며
attempt-02에서 별도로 실행한다.

## A-S00 attempt-04 — collapse-safe training

attempt-01과 attempt-02에서 foreground Dice가 31 epoch 동안
0.0으로 유지되어 background collapse로 판정하였다.

유지 사항:

- Standard MONAI SegResNet
- Location 53 classes
- 고정 split 215/46/46, seed 20260828
- Type mask 사용 금지
- Patch 112 x 160 x 64, physical batch 1
- Train/online validation 250/50 unit
- Max epoch 300
- monitor ema_fg_dice, min_delta 0.0001
- strict unit resume
- 마지막 checkpoint_best 기준 validation 46명 1회

교정 사항:

- train 250개 중 125개 forced-foreground unit
- validation 50개 중 25개 forced-foreground unit
- GT-present foreground class만 Dice 계산
- CE background target weight 0.001
- Dice/CE 비중 1.0/0.5
- Early Stopping patience 80
- 실제 양성 patch one-patch overfit 통과 후 본학습
- attempt-04에 결과 별도 보존
