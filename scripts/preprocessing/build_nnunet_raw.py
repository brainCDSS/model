#!/usr/bin/env python
# coding: utf-8

# # 06_1. nnU-Net용 최소 교정과 raw dataset 생성
# 
# `04`의 승인 조치와 `05`의 동일한 환자 split을 사용해 nnU-Net v2 형식의 raw dataset을 만듭니다.
# 
# 적용: signed-12 복원, 승인된 affine 조치, binary mask, 손실 없는 RAS 축 재배열.
# 
# 생략: 공통 spacing resampling, HU clipping, normalization. 이 작업은 nnU-Net fingerprint/planning/preprocessing에 맡깁니다. 원본은 수정하지 않습니다.

# In[1]:


from pathlib import Path
import json
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from pprint import pprint as display

def find_project_root(start=Path(__file__).resolve().parent / 'legacy'):
    for candidate in [start, *start.parents]:
        if (candidate / 'project_config.json').is_file() and (candidate / 'ctprep_utils.py').is_file():
            return candidate
    raise FileNotFoundError('project_config.json이 있는 프로젝트 폴더를 여세요.')

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
from ctprep_utils import as_bool, prepare_nnunet_pair
from ctqa_utils import write_json

CONFIG = json.loads((PROJECT_ROOT / 'project_config.json').read_text(encoding='utf-8'))
RESULTS = Path(__file__).resolve().parents[3] / 'datasets/hemorrhage/v1/metadata/common'
policy = json.loads((RESULTS / '04_integration_policy.json').read_text(encoding='utf-8'))
manifest = pd.read_csv(RESULTS / '05_integration_manifest.csv').fillna('')
registry = pd.read_csv(RESULTS / '04_case_review_registry.csv').fillna('')
print('전체 manifest:', len(manifest))
print('nnU-Net 대상:', int(manifest.eligible_for_preprocessing.map(as_bool).sum()))


# ## 1. 출력 설정
# 
# 먼저 `MAX_CASES=3`으로 시험하고 생성 결과를 확인하세요. 전체 생성 시 `MAX_CASES=None`으로 바꿉니다. Dataset ID는 로컬 nnU-Net 환경에서 중복되지 않아야 합니다.

# In[ ]:


RUN_NNUNET_EXPORT = False
OVERWRITE = False
MAX_CASES = None

DATASET_ID = 1
DATASET_NAME = 'ICHBinary'
dataset_folder = f'Dataset{DATASET_ID:03d}_{DATASET_NAME}'
base_root = Path(policy['processed_root']).parent
nnunet_raw_root = base_root / 'nnUNet_raw'
dataset_root = nnunet_raw_root / dataset_folder
print('nnUNet raw root:', nnunet_raw_root)
print('dataset:', dataset_root)


# ## 2. 승인 조치와 nnU-Net case ID
# 
# BHSD와 PhysioNet의 ID 충돌을 막기 위해 source 접두어를 붙입니다. train과 validation은 `imagesTr/labelsTr`에 함께 두고, `splits_final.json`으로 고정 split을 전달합니다. test는 `imagesTs/labelsTs`에 둡니다.

# In[11]:


registry['approved_bool'] = registry.approved.map(as_bool)
approved = registry.loc[registry.approved_bool].copy()
action_map = approved.set_index(['source', 'case_id', 'issue']).action.to_dict()

def approved_action(source, case_id, issue):
    value = action_map.get((source, case_id, issue))
    return None if value in {None, '', 'KEEP_CONFIRMED'} else value

def nnunet_case_id(source, case_id):
    prefix = 'BHSD' if source == 'BHSD' else 'PHYS'
    safe = re.sub(r'[^A-Za-z0-9_]', '_', str(case_id))
    return f'{prefix}_{safe}'

eligible = manifest.loc[manifest.eligible_for_preprocessing.map(as_bool)].copy()
eligible['nnunet_case_id'] = [nnunet_case_id(s, c) for s, c in zip(eligible.source, eligible.case_id)]
if eligible.nnunet_case_id.duplicated().any():
    raise RuntimeError('nnU-Net case ID가 중복됩니다.')
display(eligible[['source', 'case_id', 'split', 'nnunet_case_id']].head())


# ## 3. nnUNet_raw 생성
# 
# 영상은 `{case_id}_0000.nii.gz`, label은 `{case_id}.nii.gz`로 저장합니다. test label은 nnU-Net 학습 입력은 아니지만 외부 평가를 위해 `labelsTs`에 보관합니다.

# In[12]:


if not RUN_NNUNET_EXPORT:
    print('안전 정지: RUN_NNUNET_EXPORT=False입니다. 설정을 확인한 뒤 True로 바꾸세요.')
else:
    selected = eligible.copy()
    if MAX_CASES is not None:
        limit = int(MAX_CASES)
        # 시험 실행에서도 train/validation/test를 가능하면 각각 한 건 포함합니다.
        anchors = pd.concat([selected.loc[selected.split == s].head(1) for s in ['train', 'validation', 'test']])
        remainder = selected.loc[~selected.index.isin(anchors.index)]
        selected = pd.concat([anchors, remainder]).head(limit)
    records, errors = [], []
    for number, row in enumerate(selected.itertuples(), start=1):
        is_test = row.split == 'test'
        image_dir = dataset_root / ('imagesTs' if is_test else 'imagesTr')
        label_dir = dataset_root / ('labelsTs' if is_test else 'labelsTr')
        output_image = image_dir / f'{row.nnunet_case_id}_0000.nii.gz'
        output_mask = label_dir / f'{row.nnunet_case_id}.nii.gz'
        try:
            result = prepare_nnunet_pair(
                Path(row.image_path), Path(row.mask_path), output_image, output_mask,
                affine_tolerance=float(CONFIG['affine_tolerance']),
                intensity_action=approved_action(row.source, row.case_id, 'INTENSITY_SIGNED12'),
                affine_action=approved_action(row.source, row.case_id, 'AFFINE_MISMATCH'),
                overwrite=OVERWRITE,
            )
            records.append({
                'source': row.source, 'case_id': row.case_id, 'group_id': row.group_id,
                'split': row.split, 'nnunet_case_id': row.nnunet_case_id, **result,
            })
        except Exception as exc:
            errors.append({'source': row.source, 'case_id': row.case_id, 'error': repr(exc)})
        if number % 25 == 0 or number == len(selected):
            print(f'진행: {number}/{len(selected)}')

    export_manifest = pd.DataFrame(records)
    error_frame = pd.DataFrame(errors, columns=['source', 'case_id', 'error'])
    export_manifest.to_csv(RESULTS / '06_1_nnunet_raw_manifest.csv', index=False, encoding='utf-8-sig')
    error_frame.to_csv(RESULTS / '06_1_nnunet_raw_errors.csv', index=False, encoding='utf-8-sig')

    written_ids = set(export_manifest.nnunet_case_id) if len(export_manifest) else set()
    train_ids = eligible.loc[eligible.split == 'train', 'nnunet_case_id']
    val_ids = eligible.loc[eligible.split == 'validation', 'nnunet_case_id']
    fixed_split = [{
        'train': [x for x in train_ids if x in written_ids],
        'val': [x for x in val_ids if x in written_ids],
    }]
    dataset_json = {
        'channel_names': {'0': 'CT'},
        'labels': {'background': 0, 'hemorrhage': 1},
        'numTraining': int((export_manifest.split != 'test').sum()) if len(export_manifest) else 0,
        'file_ending': '.nii.gz',
    }
    write_json(dataset_root / 'dataset.json', dataset_json)
    write_json(dataset_root / 'splits_final.json', fixed_split)
    write_json(RESULTS / '06_1_nnunet_paths.json', {
        'nnUNet_raw': str(nnunet_raw_root),
        'dataset_root': str(dataset_root),
        'dataset_id': DATASET_ID,
        'dataset_name': DATASET_NAME,
        'splits_note': 'nnU-Net preprocessing 후 dataset_root/splits_final.json을 nnUNet_preprocessed의 동일 Dataset 폴더로 복사',
    })
    print('성공/기존:', len(export_manifest), '오류:', len(error_frame))
    if len(export_manifest):
        display(export_manifest.groupby(['split', 'status']).size().rename('cases').reset_index())
    if len(error_frame):
        display(error_frame)


# ## 4. 출력 검증
# 
# 생성된 각 CT-label 쌍의 shape, affine, label `{0,1}`을 다시 검사합니다.

# In[13]:


export_manifest_path = RESULTS / '06_1_nnunet_raw_manifest.csv'
if not export_manifest_path.is_file():
    print('아직 export manifest가 없습니다.')
else:
    exported = pd.read_csv(export_manifest_path)
    checks = []
    for row in exported.itertuples():
        image = nib.load(row.image_path)
        label = nib.load(row.mask_path)
        labels = np.unique(label.get_fdata(dtype=np.float32))
        checks.append({
            'nnunet_case_id': row.nnunet_case_id,
            'shape_match': tuple(image.shape) == tuple(label.shape),
            'affine_match': bool(np.allclose(image.affine, label.affine, atol=float(CONFIG['affine_tolerance']), rtol=0)),
            'valid_binary_label': bool(set(labels.tolist()).issubset({0.0, 1.0})),
        })
    checks = pd.DataFrame(checks)
    checks.to_csv(RESULTS / '06_1_nnunet_raw_checks.csv', index=False, encoding='utf-8-sig')
    display(checks)
    print('전체 통과:', bool(len(checks) and checks[['shape_match', 'affine_match', 'valid_binary_label']].all().all()))


# ## 5. nnU-Net 실행 연결
# 
# 전체 export 후 환경변수 `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`를 설정하고 다음을 실행합니다.
# 
# ```powershell
# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
# ```
# 
# 고정 train/validation split을 사용하려면 생성된 `splits_final.json`을 preprocessing 후 만들어진 `nnUNet_preprocessed/Dataset001_ICHBinary/` 폴더에 복사합니다. 외부 test 평가는 `imagesTs`와 `labelsTs`를 사용합니다.
