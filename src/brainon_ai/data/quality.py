"""CT/mask 품질 검증용 공통 함수.

원본 파일은 항상 읽기 전용으로 열며, 결과는 프로젝트의 results 폴더에만 씁니다.
NIfTI-1 single-file(.nii/.nii.gz)을 투명하게 살펴볼 수 있도록 핵심 헤더와
스케일링, qform/sform affine 계산을 작은 순수 Python 구현으로 제공합니다.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


NIFTI_DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    32: "c8",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
    1024: "i8",
    1280: "u8",
}


def nifti_stem(path: Path) -> str:
    """`.nii`와 `.nii.gz`를 모두 제거한 사례 ID를 반환합니다."""
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def list_nifti(folder: Path) -> list[Path]:
    """폴더 안의 NIfTI 파일을 사례 ID 순으로 정렬합니다."""
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))],
        key=lambda p: nifti_stem(p),
    )


def discover_datasets(data_root: Path) -> dict[str, dict[str, Path]]:
    """이 연습 데이터의 두 source 위치를 확인합니다."""
    datasets = {
        "BHSD": {
            "images": data_root / "BHSD" / "images",
            "masks": data_root / "BHSD" / "ground truths",
        },
        "PhysioNet_CT_ICH": {
            "images": data_root / "PhysioNet_CT_ICH" / "ct_scans",
            "masks": data_root / "PhysioNet_CT_ICH" / "masks",
        },
    }
    missing = [str(p) for item in datasets.values() for p in item.values() if not p.is_dir()]
    if missing:
        raise FileNotFoundError("필수 폴더가 없습니다:\n" + "\n".join(missing))
    return datasets


def pair_nifti(image_dir: Path, mask_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    """파일명 stem을 기준으로 CT-mask를 1:1 대응시킵니다."""
    images = {nifti_stem(p): p for p in list_nifti(image_dir)}
    masks = {nifti_stem(p): p for p in list_nifti(mask_dir)}
    common = sorted(images.keys() & masks.keys())
    pairs = [(case_id, images[case_id], masks[case_id]) for case_id in common]
    return pairs, sorted(images.keys() - masks.keys()), sorted(masks.keys() - images.keys())


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.name.lower().endswith(".gz") else path.open("rb")


def _unpack(fmt: str, raw: bytes, offset: int):
    return struct.unpack_from(fmt, raw, offset)


def read_nifti_header(path: Path) -> dict:
    """NIfTI-1 헤더를 읽고 분석에 필요한 필드만 반환합니다."""
    with _open_binary(path) as f:
        raw = f.read(348)
    if len(raw) < 348:
        raise ValueError(f"NIfTI 헤더가 너무 짧습니다: {path}")

    little = _unpack("<i", raw, 0)[0]
    big = _unpack(">i", raw, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"NIfTI-1 sizeof_hdr=348을 찾지 못했습니다: {path}")

    dim = _unpack(endian + "8h", raw, 40)
    ndim = int(dim[0])
    if not 1 <= ndim <= 7:
        raise ValueError(f"잘못된 NIfTI 차원 수 {ndim}: {path}")
    shape = tuple(int(x) for x in dim[1 : ndim + 1])
    datatype = int(_unpack(endian + "h", raw, 70)[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"지원하지 않는 NIfTI datatype={datatype}: {path}")

    pixdim = tuple(float(x) for x in _unpack(endian + "8f", raw, 76))
    slope = float(_unpack(endian + "f", raw, 112)[0])
    inter = float(_unpack(endian + "f", raw, 116)[0])
    if slope == 0 or not math.isfinite(slope):
        slope = 1.0
    if not math.isfinite(inter):
        inter = 0.0

    return {
        "path": str(path),
        "endian": endian,
        "shape": shape,
        "datatype": datatype,
        "dtype": endian + NIFTI_DTYPES[datatype],
        "bitpix": int(_unpack(endian + "h", raw, 72)[0]),
        "pixdim": pixdim,
        "vox_offset": max(0, int(round(float(_unpack(endian + "f", raw, 108)[0])))),
        "scl_slope": slope,
        "scl_inter": inter,
        "qform_code": int(_unpack(endian + "h", raw, 252)[0]),
        "sform_code": int(_unpack(endian + "h", raw, 254)[0]),
        "quatern": tuple(float(x) for x in _unpack(endian + "3f", raw, 256)),
        "qoffset": tuple(float(x) for x in _unpack(endian + "3f", raw, 268)),
        "srow_x": tuple(float(x) for x in _unpack(endian + "4f", raw, 280)),
        "srow_y": tuple(float(x) for x in _unpack(endian + "4f", raw, 296)),
        "srow_z": tuple(float(x) for x in _unpack(endian + "4f", raw, 312)),
        "magic": raw[344:348].decode("ascii", errors="replace").rstrip("\x00"),
    }


def read_nifti_data(path: Path, header: dict | None = None, scaled: bool = True) -> np.ndarray:
    """NIfTI voxel 배열을 x,y,z 순서로 읽습니다.

    NIfTI는 첫 번째 축이 가장 빠르게 저장되므로 Fortran order로 reshape합니다.
    scaled=True이면 `stored_value * scl_slope + scl_inter`를 적용합니다.
    """
    header = header or read_nifti_header(path)
    count = int(np.prod(header["shape"], dtype=np.int64))
    dtype = np.dtype(header["dtype"])
    expected_bytes = count * dtype.itemsize
    with _open_binary(path) as f:
        f.seek(header["vox_offset"])
        raw = f.read(expected_bytes)
    if len(raw) != expected_bytes:
        raise ValueError(f"voxel byte 수 불일치: expected={expected_bytes}, actual={len(raw)}, {path}")
    arr = np.frombuffer(raw, dtype=dtype, count=count).reshape(header["shape"], order="F")
    if scaled and (header["scl_slope"] != 1.0 or header["scl_inter"] != 0.0):
        arr = arr.astype(np.float32) * header["scl_slope"] + header["scl_inter"]
    return arr


def qform_affine(header: dict) -> np.ndarray:
    """NIfTI quaternion과 pixdim으로 4x4 qform affine을 계산합니다."""
    b, c, d = header["quatern"]
    a2 = 1.0 - (b * b + c * c + d * d)
    if a2 < 1e-7:
        norm = math.sqrt(b * b + c * c + d * d)
        b, c, d = b / norm, c / norm, d / norm
        a = 0.0
    else:
        a = math.sqrt(a2)
    rotation = np.array(
        [
            [a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c],
            [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
            [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b],
        ],
        dtype=np.float64,
    )
    dx, dy, dz = (abs(float(x)) or 1.0 for x in header["pixdim"][1:4])
    qfac = -1.0 if header["pixdim"][0] < 0 else 1.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ np.diag([dx, dy, dz * qfac])
    affine[:3, 3] = np.asarray(header["qoffset"], dtype=np.float64)
    return affine


def effective_affine(header: dict) -> tuple[np.ndarray, str]:
    """일반 로더 관례대로 sform > qform > pixdim 순서로 affine을 선택합니다."""
    if header["sform_code"] > 0:
        affine = np.array(
            [header["srow_x"], header["srow_y"], header["srow_z"], (0.0, 0.0, 0.0, 1.0)],
            dtype=np.float64,
        )
        return affine, "sform"
    if header["qform_code"] > 0:
        return qform_affine(header), "qform"
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = [abs(float(x)) or 1.0 for x in header["pixdim"][1:4]]
    return affine, "fallback_pixdim"


def affine_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(x) for x in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))


def affine_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def sampled_percentiles(array: np.ndarray, percentiles: Iterable[float], max_samples: int = 300_000) -> list[float]:
    """메모리를 제한하면서 재현 가능한 간격 표본으로 강도 분위수를 계산합니다."""
    flat = np.asarray(array).reshape(-1)
    step = max(1, int(math.ceil(flat.size / max_samples)))
    sample = flat[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return [float("nan") for _ in percentiles]
    return [float(x) for x in np.percentile(sample, list(percentiles))]


def array_summary(array: np.ndarray, percentiles=(1, 50, 99)) -> dict:
    finite = np.isfinite(array)
    values = array[finite]
    qs = sampled_percentiles(values, percentiles)
    return {
        "finite_fraction": float(finite.mean()),
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        **{f"p{int(q):02d}": v for q, v in zip(percentiles, qs)},
    }


def normalize_ct_slice(slice2d: np.ndarray, low: float = -100.0, high: float = 200.0) -> np.ndarray:
    clipped = np.clip(np.asarray(slice2d, dtype=np.float32), low, high)
    return np.round((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def save_overlay(ct: np.ndarray, mask: np.ndarray, output_path: Path, alpha: float = 0.42) -> Path:
    """출혈이 가장 많은 axial slice를 빨간색 overlay PNG로 저장합니다."""
    if ct.ndim < 3 or mask.ndim < 3:
        raise ValueError("overlay에는 3차원 CT와 mask가 필요합니다")
    per_slice = (mask > 0).sum(axis=(0, 1))
    z = int(np.argmax(per_slice)) if per_slice.max() > 0 else ct.shape[2] // 2
    gray = normalize_ct_slice(ct[:, :, z])
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    positive = mask[:, :, z] > 0
    rgb[positive, 0] = (1 - alpha) * rgb[positive, 0] + alpha * 255
    rgb[positive, 1] *= 1 - alpha
    rgb[positive, 2] *= 1 - alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 배열은 (x, y, channel), 이미지는 (row=y, column=x, channel)를 사용합니다.
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2)).save(output_path)
    return output_path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(manifest_path: Path) -> list[dict]:
    """`SHA256SUMS.txt`의 상대 경로와 해시를 직접 검증합니다."""
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ").replace("/", "\\")
        target = manifest_path.parent / Path(relative)
        actual = sha256_file(target) if target.is_file() else None
        rows.append({
            "relative_path": relative,
            "exists": target.is_file(),
            "expected_sha256": expected.lower(),
            "actual_sha256": actual,
            "match": actual == expected.lower() if actual else False,
        })
    return rows


def bhsd_group_id(case_id: str) -> str:
    """BHSD 이름의 첫 익명 ID를 보수적인 split 그룹으로 사용합니다."""
    match = re.match(r"(ID_[0-9a-fA-F]+)", case_id)
    return match.group(1) if match else case_id


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def json_safe_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    return value


def as_bool(value) -> bool:
    """CSV/JSON에서 읽은 bool 표현을 보수적으로 변환합니다."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
