# CT 데이터 품질 검증 실습 노트북

이 프로젝트는 BHSD와 PhysioNet CT-ICH의 CT/mask를 원본 변경 없이 검사하고, 이진 출혈 분할 데이터로 통합 가능한지 단계별로 판단합니다.

## 준비

1. VS Code에서 이 `CT_Data_Quality_Notebooks` 폴더를 엽니다.
2. VS Code의 **Python** 및 **Jupyter** 확장을 설치합니다.
3. VS Code 터미널에서 아래 명령을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name ct-data-quality --display-name "Python (CT Data Quality)"
```

4. 첫 노트북을 열고 오른쪽 위 **커널 선택**에서 `Python (CT Data Quality)`를 선택합니다.
5. 데이터 위치가 다르면 `project_config.json`의 `data_root`만 수정합니다.

현재 기본 데이터 위치:

```text
C:\path\to\CTdata
```

## 실행 순서

노트북은 다음 순서로 위에서부터 **모두 실행**합니다.

1. `00_환경설정_및_파일인벤토리.ipynb`
   - CT/mask 수량, 1:1 pairing, PhysioNet SHA256 검증
2. `01_NIfTI_공간정합_검증.ipynb`
   - NIfTI header, shape, spacing, qform/sform, effective affine, overlay
3. `02_강도_라벨_정합성_검증.ipynb`
   - slope/inter scaling, CT 분위수, 허용 label, 빈 mask, 작은 병변, PhysioNet CSV 비교
4. `03_통합게이트_및_최종판정.ipynb`
   - 12개 통합 게이트, canonical manifest, 이진/다중 클래스 최종 판정
5. `04_예외케이스_판정_및_통합정책.ipynb`
   - signed-12, affine, z-spacing 예외의 수동 판정 기록과 통합 정책 확정
6. `04_1_예외케이스_시각검토.ipynb`
   - 04에서 탐지한 예외의 메타데이터, overlay, 강도 변환 전후, z-spacing 시각 검토
7. `05_그룹기반_split_및_통합_manifest.ipynb`
   - 환자/검사 그룹 누수를 막는 split과 전처리 대상 manifest 생성
8. `06_공통전처리_및_NIfTI_생성.ipynb`
   - SegResNet용 승인 예외 처리, RAS 표준화, 공통 spacing, HU clipping, binary mask 생성
9. `06_1_nnUNet용_최소교정_및_raw_생성.ipynb`
   - nnU-Net용 승인 예외 처리, binary mask, 원본 spacing 보존, nnUNet_raw 및 고정 split 생성
10. `07_전처리_회귀검증.ipynb`
   - 변환 후 shape·affine·label·병변·부피 보존 검사와 실패 overlay 생성
11. `08_통합준비_최종판정.ipynb`
   - G1~G10 재판정과 기준선 모델 실험 시작 가능 여부 결정

앞 번호의 결과 CSV를 다음 번호가 읽으므로 순서를 지켜야 합니다.

## 결과 읽기

모든 산출물은 `results/`에 생성됩니다.

- `00_*`: 파일 목록, pairing, 체크섬 결과
- `01_geometry_checks.csv`: 사례별 공간 정보와 affine 비교
- `01_overlay_*.png`: 대표 CT-mask 겹침 그림
- `02_value_label_checks.csv`: 사례별 강도·라벨·병변 통계
- `02_physionet_slice_consistency.csv`: CSV와 mask slice 비교
- `03_integration_gates.csv`: 게이트별 PASS/FAIL/REVIEW/PENDING
- `03_canonical_manifest_draft.csv`: 원본을 복사하지 않는 통합 manifest 초안
- `03_final_decision.json`: 최종 판정과 GO 조건
- `04_case_review_registry.csv`: 예외 케이스별 승인 조치와 근거
- `04_integration_policy.json`: 목표 spacing, HU 범위, split, 파생 데이터 위치
- `04_1_review_figures/`: 예외 사례별 affine, 강도, z-spacing 검토 그림
- `05_integration_manifest.csv`: group split과 전처리 자격을 포함한 통합 manifest
- `06_preprocessing_manifest.csv`: 생성된 파생 NIfTI와 실행 상태
- `06_1_nnunet_raw_manifest.csv`: nnU-Net raw 형식으로 생성된 최소 교정 NIfTI와 split
- `06_1_nnunet_raw_checks.csv`: nnU-Net CT-label shape·affine·binary label 검사
- `06_1_nnunet_paths.json`: nnUNet_raw 및 Dataset 경로
- `07_preprocessing_regression.csv`: 전처리 후 케이스별 회귀검사 결과
- `08_integration_gates.csv`: 예외 처리와 전처리 결과가 반영된 통합 게이트
- `08_final_readiness.json`: 기준선 모델 실험 준비 여부

## 중요한 해석 원칙

- `FAIL`: 교정·격리·재검증 전 통합 금지
- `REVIEW`: 자동 판단이 불가능하여 원본 메타데이터·라이선스·시각 검토 필요
- `PENDING`: 전처리 또는 모델 실험을 구현한 뒤 판정 가능
- `PASS_BINARY_ONLY`: 이진 출혈 mask로는 통합 가능하지만 5종 voxel label 직접 통합은 불가능

원본 NIfTI는 수정하지 않습니다. 06 노트북만 승인된 정책에 따라 `processed_root`에 파생 NIfTI를 생성하며, 기본값은 `RUN_PREPROCESSING=False`입니다.
