from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from brainon_ai.models.aneurysm.segresnet.checkpointing import (
    atomic_json_save,
    atomic_torch_save,
    config_sha256,
    load_checkpoint,
    sha256_file,
    validate_checkpoint,
)
from brainon_ai.models.aneurysm.segresnet.config import (
    resolve_repository_path,
)
from brainon_ai.models.aneurysm.segresnet.data import (
    build_full_volume_transform,
    load_manifest,
    records_for_split,
)
from brainon_ai.models.aneurysm.segresnet.metrics import (
    patient_metrics,
    summarize_final_validation,
)
from brainon_ai.models.aneurysm.segresnet.network import (
    build_model,
)


REQUIRED_OUTPUTS = (
    "per_patient_metrics.csv",
    "per_class_metrics.csv",
    "summary.json",
)


def _json_safe(value):
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return (
            number
            if math.isfinite(number)
            else None
        )

    return value


def _atomic_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temporary,
        path,
    )


def _save_prediction(
    prediction: torch.Tensor,
    reference_path: str,
    path: Path,
) -> None:
    reference = nib.load(reference_path)

    array = (
        prediction.detach()
        .to(
            device="cpu",
            dtype=torch.uint8,
        )
        .numpy()
    )

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            array,
            reference.affine,
            header,
        ),
        str(temporary),
    )

    os.replace(
        temporary,
        path,
    )


def _new_state(
    config: dict,
    manifest_hash: str,
    checkpoint_hash: str,
    num_classes: int,
) -> dict:
    return {
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "completed_case_ids": [],
        "per_patient": [],
        "tp": [0] * num_classes,
        "fp": [0] * num_classes,
        "fn": [0] * num_classes,
    }


def _validate_state(
    state: dict,
    config: dict,
    manifest_hash: str,
    checkpoint_hash: str,
) -> None:
    expected = {
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
    }

    for key, expected_value in expected.items():
        if state.get(key) != expected_value:
            raise RuntimeError(
                "Final validation resume state has "
                f"a different {key}"
            )


def _validate_complete(
    complete_path: Path,
    final_dir: Path,
    config: dict,
    manifest_hash: str,
    checkpoint_hash: str,
    expected_cases: int,
) -> None:
    try:
        payload = json.loads(
            complete_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise RuntimeError(
            "COMPLETE.json cannot be read"
        ) from error

    expected_values = {
        "experiment_id": "A-S00",
        "status": "complete",
        "validation_cases": expected_cases,
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
    }

    for key, expected_value in (
        expected_values.items()
    ):
        if payload.get(key) != expected_value:
            raise RuntimeError(
                "COMPLETE.json contract mismatch: "
                f"{key}"
            )

    stored_hashes = payload.get(
        "required_output_sha256"
    )

    if not isinstance(stored_hashes, dict):
        raise RuntimeError(
            "COMPLETE.json has no output hash mapping"
        )

    for filename in REQUIRED_OUTPUTS:
        output_path = final_dir / filename

        if not output_path.is_file():
            raise RuntimeError(
                "COMPLETE.json exists but required "
                f"output is missing: {output_path}"
            )

        actual_hash = sha256_file(
            output_path
        )

        if stored_hashes.get(filename) != actual_hash:
            raise RuntimeError(
                "Completed output hash mismatch: "
                f"{filename}"
            )


def _format_seconds(
    seconds: float,
) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(
        seconds,
        3600,
    )
    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def final_validation(
    config: dict,
    run_dir: Path,
) -> Path:
    final_dir = (
        run_dir / "final_validation"
    )
    complete_path = (
        final_dir / "COMPLETE.json"
    )

    manifest = load_manifest(config)

    validation_records = records_for_split(
        manifest,
        "validation",
    )

    expected_cases = int(
        config["validation"][
            "patient_full_volume"
        ]["expected_cases"]
    )

    if len(validation_records) != expected_cases:
        raise RuntimeError(
            "Final validation expected "
            f"{expected_cases} cases, got "
            f"{len(validation_records)}"
        )

    manifest_path = resolve_repository_path(
        config["data"]["manifest_path"]
    )
    manifest_hash = sha256_file(
        manifest_path
    )

    checkpoint_dir = (
        run_dir / "checkpoints"
    )
    best_checkpoint_path = (
        checkpoint_dir
        / str(
            config["validation"][
                "patient_full_volume"
            ]["checkpoint"]
        )
    )
    final_checkpoint_path = (
        checkpoint_dir
        / "checkpoint_final.pth"
    )

    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Final validation requires "
            f"{best_checkpoint_path}"
        )

    if not final_checkpoint_path.is_file():
        raise RuntimeError(
            "Training has not produced "
            "checkpoint_final.pth"
        )

    checkpoint_hash = sha256_file(
        best_checkpoint_path
    )

    # 학습 종료 상태는 CPU에서 먼저 검증합니다.
    training_final = load_checkpoint(
        final_checkpoint_path,
        torch.device("cpu"),
    )

    validate_checkpoint(
        training_final,
        config,
        manifest_hash,
    )

    training_phase = training_final.get(
        "progress",
        {},
    ).get("phase")

    if training_phase not in {
        "final_validation",
        "complete",
    }:
        raise RuntimeError(
            "Full-volume validation cannot start "
            "before training finishes. "
            f"Current phase: {training_phase}"
        )

    # 이미 완료된 경우 추론하지 않고 결과 무결성만 확인
    if complete_path.is_file():
        _validate_complete(
            complete_path,
            final_dir,
            config,
            manifest_hash,
            checkpoint_hash,
            expected_cases,
        )
        print(
            "A-S00 ALREADY COMPLETE:",
            complete_path,
        )
        return final_dir

    device = torch.device(
        str(config["experiment"]["device"])
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Final validation requires CUDA"
        )

    best_checkpoint = load_checkpoint(
        best_checkpoint_path,
        device,
    )

    validate_checkpoint(
        best_checkpoint,
        config,
        manifest_hash,
    )

    model = build_model(config).to(device)
    model.load_state_dict(
        best_checkpoint["model_state"]
    )
    model.eval()

    num_classes = int(
        config["model"]["out_channels"]
    )

    state_path = (
        final_dir / "resume_state.pth"
    )

    if state_path.is_file():
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        _validate_state(
            state,
            config,
            manifest_hash,
            checkpoint_hash,
        )
        mode = "FINAL PATIENT RESUME"
    else:
        state = _new_state(
            config,
            manifest_hash,
            checkpoint_hash,
            num_classes,
        )
        mode = "FINAL PATIENT NEW"

    completed_ids = list(
        state["completed_case_ids"]
    )

    if len(completed_ids) > expected_cases:
        raise RuntimeError(
            "Final validation resume state "
            "contains too many patients"
        )

    expected_prefix = [
        record["case_id"]
        for record in validation_records[
            : len(completed_ids)
        ]
    ]

    if completed_ids != expected_prefix:
        raise RuntimeError(
            "Completed final-validation cases "
            "are not the expected sorted prefix"
        )

    if len(state["per_patient"]) != len(
        completed_ids
    ):
        raise RuntimeError(
            "Per-patient rows and completed case "
            "count differ"
        )

    print("[EXPENSIVE FULL-VOLUME]")
    print("Mode                    :", mode)
    print("During every epoch      : NO")
    print("After all training      : YES")
    print(
        "Checkpoint              :",
        "checkpoint_best.pth",
    )
    print(
        "Validation patients     :",
        expected_cases,
    )
    print(
        "Completed before start  :",
        len(completed_ids),
    )

    dataset = Dataset(
        validation_records,
        transform=build_full_volume_transform(
            config
        ),
    )

    roi_size = tuple(
        int(value)
        for value in config["inference"][
            "roi_size"
        ]
    )
    sw_batch_size = int(
        config["inference"][
            "sliding_window_batch_size"
        ]
    )
    overlap = float(
        config["inference"]["overlap"]
    )
    blend_mode = str(
        config["inference"]["blend_mode"]
    )
    amp_enabled = bool(
        config["training"]["amp"]
        and device.type == "cuda"
    )

    initial_completed = len(completed_ids)
    started = monotonic()

    for index in range(
        initial_completed,
        len(validation_records),
    ):
        record = validation_records[index]
        item = dataset[index]

        image = item["image"].unsqueeze(0)
        target = item["label"][0]

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ),
        ):
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode=blend_mode,
                sw_device=device,
                device=torch.device("cpu"),
            )

        prediction = torch.argmax(
            logits,
            dim=1,
        )[0]

        row, case_tp, case_fp, case_fn = (
            patient_metrics(
                prediction,
                target,
                num_classes=num_classes,
            )
        )

        row.update(
            {
                "case_id": record["case_id"],
                "image_path": record[
                    "image_path"
                ],
                "label_path": record[
                    "label_path"
                ],
            }
        )

        if bool(
            config["inference"][
                "save_predictions"
            ]
        ):
            _save_prediction(
                prediction,
                record["image_path"],
                final_dir
                / "predictions"
                / (
                    f"{record['case_id']}"
                    ".nii.gz"
                ),
            )

        state["completed_case_ids"].append(
            record["case_id"]
        )
        state["per_patient"].append(row)

        state["tp"] = (
            np.asarray(
                state["tp"],
                dtype=np.int64,
            )
            + case_tp
        ).tolist()

        state["fp"] = (
            np.asarray(
                state["fp"],
                dtype=np.int64,
            )
            + case_fp
        ).tolist()

        state["fn"] = (
            np.asarray(
                state["fn"],
                dtype=np.int64,
            )
            + case_fn
        ).tolist()

        state["updated_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # 환자 1명 완료 직후 즉시 저장
        atomic_torch_save(
            state,
            state_path,
        )

        _atomic_csv(
            pd.DataFrame(
                state["per_patient"]
            ),
            final_dir
            / "per_patient_metrics.csv",
        )

        elapsed = monotonic() - started
        processed_now = (
            index - initial_completed + 1
        )
        remaining = (
            len(validation_records)
            - index
            - 1
        )
        eta = (
            elapsed
            / processed_now
            * remaining
            if processed_now
            else 0.0
        )

        running_mean = pd.Series(
            [
                entry["patient_dice"]
                for entry
                in state["per_patient"]
            ],
            dtype=float,
        ).mean(skipna=True)

        print(
            "[FINAL-VAL] "
            f"{index + 1}/"
            f"{len(validation_records)}"
        )
        print(
            "Case ID            :",
            record["case_id"],
        )
        print(
            "Patient Dice       :",
            row["patient_dice"],
        )
        print(
            "Patient Macro Dice :",
            row["patient_macro_dice"],
        )
        print(
            "Precision          :",
            row["precision"],
        )
        print(
            "Recall             :",
            row["recall"],
        )
        print(
            "Running Mean Dice  :",
            float(running_mean),
        )
        print(
            "ETA                 :",
            _format_seconds(eta),
            flush=True,
        )

        del image
        del target
        del logits
        del prediction

    if len(state["completed_case_ids"]) != (
        expected_cases
    ):
        raise RuntimeError(
            "Final validation ended before all "
            "patients were completed"
        )

    tp = np.asarray(
        state["tp"],
        dtype=np.int64,
    )
    fp = np.asarray(
        state["fp"],
        dtype=np.int64,
    )
    fn = np.asarray(
        state["fn"],
        dtype=np.int64,
    )

    summary, per_class = (
        summarize_final_validation(
            state["per_patient"],
            tp,
            fp,
            fn,
        )
    )

    summary.update(
        {
            "experiment_id": "A-S00",
            "config_sha256": config_sha256(
                config
            ),
            "checkpoint": str(
                best_checkpoint_path
            ),
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "manifest_sha256": (
                manifest_hash
            ),
            "evaluation_execution": (
                "single_prediction_per_patient"
            ),
        }
    )

    _atomic_csv(
        pd.DataFrame(
            state["per_patient"]
        ),
        final_dir
        / "per_patient_metrics.csv",
    )

    _atomic_csv(
        per_class,
        final_dir
        / "per_class_metrics.csv",
    )

    atomic_json_save(
        _json_safe(summary),
        final_dir / "summary.json",
    )

    output_hashes = {
        filename: sha256_file(
            final_dir / filename
        )
        for filename in REQUIRED_OUTPUTS
    }

    # 세 필수 결과가 모두 완성된 뒤 마지막으로 작성
    atomic_json_save(
        {
            "experiment_id": "A-S00",
            "status": "complete",
            "completed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "validation_cases": len(
                state["completed_case_ids"]
            ),
            "config_sha256": config_sha256(
                config
            ),
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "manifest_sha256": (
                manifest_hash
            ),
            "required_outputs": list(
                REQUIRED_OUTPUTS
            ),
            "required_output_sha256": (
                output_hashes
            ),
        },
        complete_path,
    )

    _validate_complete(
        complete_path,
        final_dir,
        config,
        manifest_hash,
        checkpoint_hash,
        expected_cases,
    )

    return final_dir
