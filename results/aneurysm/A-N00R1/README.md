# TopAneu A-N00R1 현재 결과 요약

## 실험

- 실험 ID: `A-N00R1`
- Dataset: `Dataset501_TopAneuMRA_Location`
- Trainer: `nnUNetTrainerAneurysmSparse53Retrain`
- Configuration: `3d_fullres`
- Fold: `0`

## 학습 기준

- Architecture: Standard PlainConvUNet
- Outputs: 53
- Train / Validation: 215 / 46
- Max epochs: 300
- Train iterations / epoch: 250
- Online validation iterations / epoch: 50
- Loss: GT-present-only Dice + CE
- Foreground sampling: probabilistic 0.75
- EarlyStopping start: 80
- Patience: 40
- min_delta: 0.0001
- Best monitor: `ema_fg_dice`
- Strict Resume: 적용

## 현재 실행 상태

- checkpoint_latest.pth: `True`
- checkpoint_best.pth: `True`
- checkpoint_final.pth: `False`
- Final Validation COMPLETE: `False`

> Final Validation이 완료되기 전의 값은 최종 모델 성능으로 확정하지 않는다.

## GitHub에 포함한 소형 결과 산출물

- `results/aneurysm/A-N00R1/A_N00R1_console.log` (35731 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__dataset.json` (2002 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__dataset_fingerprint.json` (50987 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__plans.json` (16112 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__fold_0__debug.json` (21121 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__fold_0__patient_validation__training_history.csv` (1503 bytes)
- `results/aneurysm/A-N00R1/Dataset501_TopAneuMRA_Location__nnUNetTrainerAneurysmSparse53Retrain__nnUNetPlans__3d_fullres__fold_0__training_log_2026_9_2_19_02_40.txt` (10932 bytes)

## GitHub 제외 대상

- checkpoint `.pth`
- Dataset501 실제 영상/Mask
- nnU-Net preprocessed 대용량 파일
- prediction volume

