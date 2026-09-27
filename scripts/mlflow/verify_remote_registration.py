#!/usr/bin/env python3
"""Verify COMPLETE inventory records against remote MLflow tracking/registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow
from mlflow import MlflowClient


MODEL_NAMES = {
    "CEREBRAL_ANEURYSM_MRA": "cerebral-aneurysm-mra",
    "HEMORRHAGIC_STROKE_CT": "hemorrhagic-stroke-ct",
    "ISCHEMIC_STROKE_DWI_ADC": "ischemic-stroke-mri",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/all_experiments_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlflow_remote_verification.json"))
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    records = inventory.get("experiments", inventory) if isinstance(inventory, dict) else inventory
    expected = [record for record in records if record.get("status") == "COMPLETE"]
    client = MlflowClient()
    details = {}
    all_run_ids: list[str] = []
    all_version_ids: list[str] = []
    missing_runs: list[str] = []
    missing_versions: list[str] = []
    bad_run_status: list[dict] = []
    bad_version_status: list[dict] = []

    for analysis_type, model_name in MODEL_NAMES.items():
        wanted = sorted(r["experiment_id"] for r in expected if r.get("analysis_type") == analysis_type)
        experiment = mlflow.get_experiment_by_name(model_name)
        runs = [] if experiment is None else mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            output_format="list",
            max_results=5000,
        )
        run_by_id = {}
        for run in runs:
            experiment_id = run.data.tags.get("experiment_id")
            if experiment_id in wanted:
                run_by_id.setdefault(experiment_id, []).append(run)
                all_run_ids.append(run.info.run_id)
                if run.info.status != "FINISHED":
                    bad_run_status.append({"experiment_id": experiment_id, "status": run.info.status})

        versions = list(client.search_model_versions(f"name='{model_name}'", max_results=5000))
        version_by_id = {}
        for version in versions:
            experiment_id = (version.tags or {}).get("experiment_id")
            if experiment_id in wanted:
                version_by_id.setdefault(experiment_id, []).append(version)
                all_version_ids.append(f"{model_name}:{version.version}")
                status = str(version.status)
                if status not in {"READY", "PENDING_REGISTRATION"}:
                    bad_version_status.append({
                        "experiment_id": experiment_id,
                        "version": version.version,
                        "status": status,
                    })

        missing_runs.extend(sorted(set(wanted) - set(run_by_id)))
        missing_versions.extend(sorted(set(wanted) - set(version_by_id)))
        details[model_name] = {
            "expected": len(wanted),
            "matched_run_experiment_ids": len(run_by_id),
            "matched_runs": sum(len(v) for v in run_by_id.values()),
            "matched_version_experiment_ids": len(version_by_id),
            "matched_versions": sum(len(v) for v in version_by_id.values()),
            "duplicate_run_experiment_ids": sorted(k for k, v in run_by_id.items() if len(v) > 1),
            "duplicate_version_experiment_ids": sorted(k for k, v in version_by_id.items() if len(v) > 1),
            "version_statuses": dict(Counter(str(v.status) for values in version_by_id.values() for v in values)),
        }

    report = {
        "tracking_uri": mlflow.get_tracking_uri(),
        "expected_complete": len(expected),
        "unique_matched_runs": len(set(all_run_ids)),
        "unique_matched_versions": len(set(all_version_ids)),
        "missing_runs": missing_runs,
        "missing_versions": missing_versions,
        "bad_run_status": bad_run_status,
        "bad_version_status": bad_version_status,
        "models": details,
        "passed": not (missing_runs or missing_versions or bad_run_status or bad_version_status)
        and len(set(all_run_ids)) == len(expected)
        and len(set(all_version_ids)) == len(expected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
