"""통합용 CT/mask 전처리 공통 함수.

원본 NIfTI는 읽기만 하며, 승인된 예외 처리와 공통 resampling 결과는
별도의 파생 데이터 폴더에 저장합니다.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np


def as_bool(value) -> bool:
    """CSV/JSON에서 읽은 bool 표현을 보수적으로 변환합니다."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_scaled_image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """NIfTI scaling을 적용한 float32 배열과 이미지를 반환합니다."""
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"3차원 NIfTI만 지원합니다: shape={data.shape}, path={path}")
    return image, data


def decode_signed12(values: np.ndarray) -> np.ndarray:
    """unsigned처럼 읽힌 12-bit two's-complement 후보를 signed 값으로 복원합니다."""
    array = np.asarray(values, dtype=np.float32)
    return np.where(array > 2047, array - 4096, array).astype(np.float32, copy=False)


def image_with_data(data: np.ndarray, affine: np.ndarray, dtype) -> nib.Nifti1Image:
    """qform/sform을 동일한 affine로 정규화한 NIfTI 이미지를 만듭니다."""
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), np.asarray(affine, dtype=np.float64))
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    image.set_data_dtype(dtype)
    return image


def align_mask_to_ct_grid(
    ct_image: nib.Nifti1Image,
    mask_image: nib.Nifti1Image,
    mask_data: np.ndarray,
    *,
    affine_action: str | None,
    tolerance: float,
) -> nib.Nifti1Image:
    """mask를 CT grid에 맞춥니다.

    실제 affine 불일치에는 명시적으로 승인된 COPY_CT_AFFINE_TO_MASK 또는
    RESAMPLE_MASK_TO_CT만 허용합니다.
    """
    same_shape = tuple(ct_image.shape) == tuple(mask_image.shape)
    same_affine = np.allclose(ct_image.affine, mask_image.affine, atol=tolerance, rtol=0)
    source = image_with_data(mask_data, mask_image.affine, np.float32)

    if same_shape and same_affine:
        return image_with_data(mask_data, ct_image.affine, np.float32)
    if affine_action == "COPY_CT_AFFINE_TO_MASK":
        if not same_shape:
            raise ValueError("COPY_CT_AFFINE_TO_MASK는 CT-mask shape가 같을 때만 허용됩니다.")
        return image_with_data(mask_data, ct_image.affine, np.float32)
    if affine_action == "RESAMPLE_MASK_TO_CT":
        return resample_from_to(source, (ct_image.shape, ct_image.affine), order=0, mode="constant", cval=0)
    raise RuntimeError(
        "CT-mask affine가 다르지만 승인된 조치가 없습니다. "
        "04_case_review_registry.csv에서 조치를 승인하세요."
    )


def preprocess_pair(
    image_path: Path,
    mask_path: Path,
    output_image_path: Path,
    output_mask_path: Path,
    *,
    target_spacing: tuple[float, float, float],
    hu_clip: tuple[float, float],
    affine_tolerance: float,
    intensity_action: str | None = None,
    affine_action: str | None = None,
    overwrite: bool = False,
) -> dict:
    """CT-mask 한 쌍을 RAS, 공통 spacing, binary mask로 변환해 저장합니다."""
    if output_image_path.exists() or output_mask_path.exists():
        if not overwrite:
            if output_image_path.exists() and output_mask_path.exists():
                return {"status": "EXISTS", "image_path": str(output_image_path), "mask_path": str(output_mask_path)}
            raise FileExistsError("CT와 mask 중 하나만 존재합니다. 출력 상태를 확인하세요.")

    ct_image, ct_data = load_scaled_image(image_path)
    mask_image, mask_data = load_scaled_image(mask_path)

    if intensity_action == "APPLY_SIGNED12":
        ct_data = decode_signed12(ct_data)
    elif intensity_action not in {None, "", "KEEP_CONFIRMED"}:
        raise ValueError(f"지원하지 않는 intensity 조치: {intensity_action}")

    aligned_mask = align_mask_to_ct_grid(
        ct_image,
        mask_image,
        mask_data,
        affine_action=affine_action,
        tolerance=affine_tolerance,
    )

    ct_work = image_with_data(ct_data, ct_image.affine, np.float32)
    mask_work = image_with_data(aligned_mask.get_fdata(dtype=np.float32), aligned_mask.affine, np.float32)
    ct_ras = nib.as_closest_canonical(ct_work)
    mask_ras = nib.as_closest_canonical(mask_work)

    # CT grid를 먼저 확정하고 mask를 그 grid에 nearest-neighbor로 맞춥니다.
    ct_resampled = resample_to_output(
        ct_ras,
        voxel_sizes=target_spacing,
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    mask_resampled = resample_from_to(
        mask_ras,
        (ct_resampled.shape, ct_resampled.affine),
        order=0,
        mode="constant",
        cval=0,
    )

    low, high = hu_clip
    ct_output = np.clip(ct_resampled.get_fdata(dtype=np.float32), low, high).astype(np.float32)
    mask_output = (mask_resampled.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    ct_final = image_with_data(ct_output, ct_resampled.affine, np.float32)
    mask_final = image_with_data(mask_output, ct_resampled.affine, np.uint8)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(ct_final, str(output_image_path))
    nib.save(mask_final, str(output_mask_path))

    return {
        "status": "OK",
        "image_path": str(output_image_path),
        "mask_path": str(output_mask_path),
        "shape": "x".join(map(str, ct_final.shape)),
        "positive_voxels": int(mask_output.sum()),
        "ct_min": float(ct_output.min()),
        "ct_max": float(ct_output.max()),
    }


def prepare_nnunet_pair(
    image_path: Path,
    mask_path: Path,
    output_image_path: Path,
    output_mask_path: Path,
    *,
    affine_tolerance: float,
    intensity_action: str | None = None,
    affine_action: str | None = None,
    overwrite: bool = False,
) -> dict:
    """nnU-Net raw 입력용 최소 교정 NIfTI를 생성합니다.

    승인된 강도/affine 예외와 binary label 변환만 적용하고, 축은 데이터 손실 없는
    RAS 재배열만 수행합니다. spacing resampling, HU clipping, normalization은
    nnU-Net의 fingerprint/planning/preprocessing에 맡깁니다.
    """
    if output_image_path.exists() or output_mask_path.exists():
        if not overwrite:
            if output_image_path.exists() and output_mask_path.exists():
                return {"status": "EXISTS", "image_path": str(output_image_path), "mask_path": str(output_mask_path)}
            raise FileExistsError("CT와 mask 중 하나만 존재합니다. 출력 상태를 확인하세요.")

    ct_image, ct_data = load_scaled_image(image_path)
    mask_image, mask_data = load_scaled_image(mask_path)

    if intensity_action == "APPLY_SIGNED12":
        ct_data = decode_signed12(ct_data)
    elif intensity_action not in {None, "", "KEEP_CONFIRMED"}:
        raise ValueError(f"지원하지 않는 intensity 조치: {intensity_action}")

    aligned_mask = align_mask_to_ct_grid(
        ct_image,
        mask_image,
        mask_data,
        affine_action=affine_action,
        tolerance=affine_tolerance,
    )
    ct_work = image_with_data(ct_data, ct_image.affine, np.float32)
    mask_work = image_with_data(aligned_mask.get_fdata(dtype=np.float32), aligned_mask.affine, np.float32)
    ct_ras = nib.as_closest_canonical(ct_work)
    mask_ras = nib.as_closest_canonical(mask_work)

    if tuple(ct_ras.shape) != tuple(mask_ras.shape) or not np.allclose(
        ct_ras.affine, mask_ras.affine, atol=affine_tolerance, rtol=0
    ):
        raise RuntimeError("최소 교정 후 CT-mask grid가 일치하지 않습니다.")

    ct_output = np.asarray(ct_ras.get_fdata(dtype=np.float32), dtype=np.float32)
    mask_output = (mask_ras.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    ct_final = image_with_data(ct_output, ct_ras.affine, np.float32)
    mask_final = image_with_data(mask_output, ct_ras.affine, np.uint8)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(ct_final, str(output_image_path))
    nib.save(mask_final, str(output_mask_path))

    spacing = tuple(float(x) for x in ct_final.header.get_zooms()[:3])
    return {
        "status": "OK",
        "image_path": str(output_image_path),
        "mask_path": str(output_mask_path),
        "shape": "x".join(map(str, ct_final.shape)),
        "spacing_mm": "x".join(f"{x:.6g}" for x in spacing),
        "positive_voxels": int(mask_output.sum()),
        "ct_min": float(ct_output.min()),
        "ct_max": float(ct_output.max()),
    }


def voxel_volume_mm3(image: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(np.asarray(image.affine, dtype=np.float64)[:3, :3])))


def binary_mask_volume_mm3(image: nib.Nifti1Image) -> float:
    data = image.get_fdata(dtype=np.float32)
    return float((data > 0).sum()) * voxel_volume_mm3(image)


def dice_score(reference: np.ndarray, prediction: np.ndarray) -> float:
    a = np.asarray(reference) > 0
    b = np.asarray(prediction) > 0
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)
