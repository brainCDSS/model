#!/usr/bin/env python
# coding: utf-8

# # 07. SegResNet용 전처리 회귀검증
# 
# 06에서 만든 파생 CT-mask의 shape, affine, spacing, 유한값, binary label, 병변 보존과 부피 변화를 검사합니다. 전처리 성공 로그만으로 통과시키지 않고 실제 출력 파일을 다시 읽어 검증합니다.

# In[15]:


from pathlib import Path
import json
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
from ctprep_utils import binary_mask_volume_mm3
from ctqa_utils import save_overlay, write_json

RESULTS = Path(__file__).resolve().parents[3] / 'datasets/hemorrhage/v1/metadata/common'
policy = json.loads((RESULTS / '04_integration_policy.json').read_text(encoding='utf-8'))
processed_path = RESULTS / '06_preprocessing_manifest.csv'
if not processed_path.is_file():
    raise RuntimeError('06 노트북에서 전처리를 먼저 실행하세요.')
processed = pd.read_csv(processed_path)
if processed.empty:
    raise RuntimeError('검증할 전처리 결과가 없습니다.')
print('검증 대상:', len(processed))


# ## 1. 케이스별 회귀검사

# In[16]:


target_spacing = np.asarray(policy['target_spacing_mm'], dtype=float)
volume_limit = float(policy['max_mask_volume_relative_change'])
rows, errors = [], []
for number, item in enumerate(processed.itertuples(), start=1):
    try:
        ct_img = nib.load(str(item.image_path))
        mask_img = nib.load(str(item.mask_path))
        original_mask_img = nib.load(str(item.original_mask_path))
        ct = ct_img.get_fdata(dtype=np.float32)
        mask = mask_img.get_fdata(dtype=np.float32)
        original_mask = original_mask_img.get_fdata(dtype=np.float32)
        labels = sorted(float(x) for x in np.unique(mask))
        before_positive = int((original_mask > 0).sum())
        after_positive = int((mask > 0).sum())
        before_volume = binary_mask_volume_mm3(original_mask_img)
        after_volume = binary_mask_volume_mm3(mask_img)
        if before_volume == 0:
            relative_change = 0.0 if after_volume == 0 else float('inf')
        else:
            relative_change = abs(after_volume - before_volume) / before_volume
        shape_match = tuple(ct_img.shape) == tuple(mask_img.shape)
        affine_match = bool(np.allclose(ct_img.affine, mask_img.affine, atol=1e-5, rtol=0))
        spacing = np.asarray(nib.affines.voxel_sizes(ct_img.affine), dtype=float)
        spacing_match = bool(np.allclose(spacing, target_spacing, atol=1e-4, rtol=0))
        orientation = ''.join(nib.aff2axcodes(ct_img.affine))
        orientation_pass = orientation == policy['target_orientation']
        finite_pass = bool(np.isfinite(ct).all() and np.isfinite(mask).all())
        label_pass = set(labels).issubset({0.0, 1.0})
        lesion_preserved = not (before_positive > 0 and after_positive == 0)
        volume_pass = bool(np.isfinite(relative_change) and relative_change <= volume_limit)
        case_pass = all([shape_match, affine_match, spacing_match, orientation_pass, finite_pass, label_pass, lesion_preserved, volume_pass])
        rows.append({
            'source': item.source, 'case_id': item.case_id, 'split': item.split,
            'shape_match': shape_match, 'affine_match': affine_match, 'spacing_match': spacing_match,
            'orientation': orientation, 'orientation_pass': orientation_pass,
            'finite_pass': finite_pass, 'mask_labels': json.dumps(labels), 'label_pass': label_pass,
            'positive_voxels_before': before_positive, 'positive_voxels_after': after_positive,
            'lesion_preserved': lesion_preserved,
            'mask_volume_before_mm3': before_volume, 'mask_volume_after_mm3': after_volume,
            'mask_volume_relative_change': relative_change, 'volume_pass': volume_pass,
            'case_pass': case_pass,
        })
    except Exception as exc:
        errors.append({'source': item.source, 'case_id': item.case_id, 'error': repr(exc)})
    if number % 25 == 0 or number == len(processed):
        print(f'진행: {number}/{len(processed)}')

checks = pd.DataFrame(rows)
error_frame = pd.DataFrame(errors, columns=['source', 'case_id', 'error'])
checks.to_csv(RESULTS / '07_preprocessing_regression.csv', index=False, encoding='utf-8-sig')
error_frame.to_csv(RESULTS / '07_regression_errors.csv', index=False, encoding='utf-8-sig')
display(checks.groupby('source').agg(cases=('case_id', 'count'), failures=('case_pass', lambda s: int((~s).sum()))).reset_index())


# ## 2. 실패 케이스 overlay
# 
# 실패 케이스는 최대 20건까지 파생 CT-mask overlay를 저장합니다. 수치만 보고 affine 또는 병변 보존 문제를 승인하지 마세요.

# In[17]:


overlay_dir = RESULTS / '07_regression_overlays'
failed = checks.loc[~checks.case_pass].head(20)
for row in failed.itertuples():
    item = processed.loc[(processed.source == row.source) & (processed.case_id == row.case_id)].iloc[0]
    ct = nib.load(str(item.image_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(item.mask_path)).get_fdata(dtype=np.float32)
    save_overlay(ct, mask, overlay_dir / f'{row.source}_{row.case_id}.png')
print('실패 overlay:', len(failed), '건 →', overlay_dir)


# ## 3. G10 전처리 회귀 게이트

# In[18]:


summary = {
    'evaluated_cases': int(len(checks)),
    'read_errors': int(len(error_frame)),
    'shape_failures': int((~checks.shape_match).sum()),
    'affine_failures': int((~checks.affine_match).sum()),
    'spacing_failures': int((~checks.spacing_match).sum()),
    'orientation_failures': int((~checks.orientation_pass).sum()),
    'nonfinite_failures': int((~checks.finite_pass).sum()),
    'invalid_label_cases': int((~checks.label_pass).sum()),
    'lesion_loss_cases': int((~checks.lesion_preserved).sum()),
    'volume_review_cases': int((~checks.volume_pass).sum()),
    'regression_pass': bool(len(error_frame) == 0 and checks.case_pass.all()),
}
write_json(RESULTS / '07_preprocessing_gate.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))

