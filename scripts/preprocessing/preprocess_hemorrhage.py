#!/usr/bin/env python
# coding: utf-8

# # 06. SegResNet용 공통 전처리와 파생 NIfTI 생성
# 
# 승인된 케이스만 대상으로 signed-12 예외 처리, CT-mask grid 정합, RAS 방향 표준화, 공통 spacing `(0.5, 0.5, 2.5) mm` resampling, HU clipping, binary mask 변환을 수행합니다. 전체 volume shape은 환자별로 가변 상태를 유지하고 학습 단계에서만 고정 patch를 추출합니다. 원본은 수정하지 않으며 `processed_root`에 새 파일을 만듭니다.
# 
# CT는 linear interpolation, mask는 nearest-neighbor interpolation을 사용합니다. 이 고정 spacing 출력은 SegResNet용이며, nnU-Net용 최소 교정 데이터는 `06_1`에서 별도로 생성합니다.

# In[17]:


from pathlib import Path
import json
import sys

import pandas as pd
from pprint import pprint as display

def find_project_root(start=Path(__file__).resolve().parent / 'legacy'):
    for candidate in [start, *start.parents]:
        if (candidate / 'project_config.json').is_file() and (candidate / 'ctprep_utils.py').is_file():
            return candidate
    raise FileNotFoundError('project_config.json이 있는 프로젝트 폴더를 여세요.')

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
from ctprep_utils import as_bool, preprocess_pair

CONFIG = json.loads((PROJECT_ROOT / 'project_config.json').read_text(encoding='utf-8'))
RESULTS = Path(__file__).resolve().parents[3] / 'datasets/hemorrhage/v1/metadata/common'
policy = json.loads((RESULTS / '04_integration_policy.json').read_text(encoding='utf-8'))
manifest = pd.read_csv(RESULTS / '05_integration_manifest.csv')
registry = pd.read_csv(RESULTS / '04_case_review_registry.csv').fillna('')
if policy.get('target_orientation') != 'RAS':
    raise ValueError('현재 구현은 target_orientation=RAS만 지원합니다.')
print('파생 데이터 위치:', policy['processed_root'])
print('전처리 가능:', int(manifest.eligible_for_preprocessing.map(as_bool).sum()), '/', len(manifest))


# ## 1. 실행 설정
# 
# 출력 용량과 정책을 확인한 뒤 `RUN_PREPROCESSING=True`로 변경하세요. spacing 정책이 5.0 mm에서 2.5 mm로 바뀌었으므로 기존 파생 NIfTI를 갱신하려면 `OVERWRITE=True`가 필요합니다. 출력 shape은 고정하지 않습니다.

# In[18]:


RUN_PREPROCESSING = False
OVERWRITE = False  # 5.0 mm로 생성된 기존 SegResNet 파생 파일을 2.5 mm로 갱신
MAX_CASES = None  # 시험 실행은 3처럼 지정

target_spacing = tuple(float(x) for x in policy['target_spacing_mm'])
hu_clip = tuple(float(x) for x in policy['hu_clip'])
processed_root = Path(policy['processed_root'])
print('target spacing:', target_spacing)
print('HU clip:', hu_clip)
print('overwrite:', OVERWRITE)


# ## 2. 승인된 예외 조치 확인

# In[19]:


registry['approved_bool'] = registry.approved.map(as_bool)
approved = registry.loc[registry.approved_bool].copy()
action_map = approved.set_index(['source', 'case_id', 'issue']).action.to_dict()

def approved_action(source, case_id, issue):
    value = action_map.get((source, case_id, issue))
    return None if value in {None, '', 'KEEP_CONFIRMED'} else value

display(approved[['source', 'case_id', 'issue', 'action', 'reviewer', 'rationale']])


# ## 3. 파생 NIfTI 생성
# 
# 각 source 아래 `images/`, `masks/`에 같은 case ID로 저장하고 실행 결과 manifest를 남깁니다.

# In[20]:


if not RUN_PREPROCESSING:
    print('안전 정지: RUN_PREPROCESSING=False입니다. 설정을 검토한 뒤 True로 바꾸세요.')
else:
    selected = manifest.loc[manifest.eligible_for_preprocessing.map(as_bool)].copy()
    if MAX_CASES is not None:
        selected = selected.head(int(MAX_CASES))
    records, errors = [], []
    for number, row in enumerate(selected.itertuples(), start=1):
        output_image = processed_root / row.source / 'images' / f'{row.case_id}.nii.gz'
        output_mask = processed_root / row.source / 'masks' / f'{row.case_id}.nii.gz'
        try:
            result = preprocess_pair(
                Path(row.image_path), Path(row.mask_path), output_image, output_mask,
                target_spacing=target_spacing, hu_clip=hu_clip,
                affine_tolerance=float(CONFIG['affine_tolerance']),
                intensity_action=approved_action(row.source, row.case_id, 'INTENSITY_SIGNED12'),
                affine_action=approved_action(row.source, row.case_id, 'AFFINE_MISMATCH'),
                overwrite=OVERWRITE,
            )
            records.append({
                'source': row.source, 'case_id': row.case_id, 'group_id': row.group_id, 'split': row.split,
                'original_image_path': row.image_path, 'original_mask_path': row.mask_path,
                **result,
            })
        except Exception as exc:
            errors.append({'source': row.source, 'case_id': row.case_id, 'error': repr(exc)})
        if number % 25 == 0 or number == len(selected):
            print(f'진행: {number}/{len(selected)}')

    processed = pd.DataFrame(records)
    error_frame = pd.DataFrame(errors, columns=['source', 'case_id', 'error'])
    processed.to_csv(RESULTS / '06_preprocessing_manifest.csv', index=False, encoding='utf-8-sig')
    error_frame.to_csv(RESULTS / '06_preprocessing_errors.csv', index=False, encoding='utf-8-sig')
    print('성공/기존:', len(processed), '오류:', len(error_frame))
    if len(processed):
        display(processed.groupby(['source', 'split', 'status']).size().rename('cases').reset_index())
    if len(error_frame):
        display(error_frame)

