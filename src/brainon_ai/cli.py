from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from brainon_ai.common.config import load_config
from brainon_ai.common.runtime import (
    cache_root,
    configure_nnunet,
    environment_info,
    repository_root,
    run_root,
)
from brainon_ai.data.registry import dataset_entries, verify_release
from brainon_ai.data.synthetic import TASKS, make_synthetic
from brainon_ai.engine.trainer import evaluate_checkpoint, find_run, resume_run, train
from brainon_ai.export.release import export_model
from brainon_ai.storage.backends.local import LocalBackend
from brainon_ai.visualization.plots import plot_run

app = typer.Typer(help="BrainOn medical imaging experiment CLI", no_args_is_help=True)
data_app = typer.Typer(help="Dataset cache and preparation")
storage_app = typer.Typer(help="Storage diagnostics")
run_app = typer.Typer(help="Run synchronization and resume")
model_app = typer.Typer(help="Model release operations")
app.add_typer(data_app, name="data")
app.add_typer(storage_app, name="storage")
app.add_typer(run_app, name="run")
app.add_typer(model_app, name="model")


@app.command("doctor")
def doctor() -> None:
    """Print environment and package readiness without changing cloud state."""
    info = environment_info()
    info["nnunet_paths"] = configure_nnunet()
    info["cache_root"] = str(cache_root())
    info["run_root"] = str(run_root())
    typer.echo(json.dumps(info, indent=2))


@data_app.command("list")
def data_list() -> None:
    for entry in dataset_entries():
        typer.echo(f"{entry['release_id']}\t{entry['dataset_id']}\t{entry['task']}")


@data_app.command("pull")
def data_pull(
    release: str = typer.Option(...), task: str | None = None, split: str | None = None
) -> None:
    """Describe/cache a licensed release. Automatic acquisition is intentionally disabled."""
    entry = next((item for item in dataset_entries() if item["release_id"] == release), None)
    if not entry:
        raise typer.BadParameter(f"unknown release: {release}")
    typer.echo(f"Acquisition is manual: {entry['official_source']}")
    typer.echo(
        f"After acquisition, place permitted files in {cache_root() / 'releases' / release}; "
        f"task={task}, split={split}"
    )


@data_app.command("verify")
def data_verify(release: str = typer.Option(...)) -> None:
    ok, failures = verify_release(release)
    typer.echo("verified" if ok else "verification failed: " + ", ".join(failures))
    if not ok:
        raise typer.Exit(1)


@data_app.command("prepare")
def data_prepare(dataset: str = typer.Option(...)) -> None:
    entry = next((item for item in dataset_entries() if item["dataset_id"] == dataset), None)
    if not entry:
        raise typer.BadParameter(f"unknown dataset: {dataset}")
    paths = configure_nnunet()
    typer.echo(f"Preparation adapter ready for {dataset}; nnUNet_raw={paths['nnUNet_raw']}")


@data_app.command("make-synthetic")
def data_make_synthetic(task: str = typer.Option("all"), cases: int = 4) -> None:
    if task != "all" and task not in TASKS:
        raise typer.BadParameter("task must be all, hemorrhage, ischemia, or aneurysm")
    for path in make_synthetic(cache_root() / "synthetic", task, cases):
        typer.echo(path)


@app.command("train")
def train_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, resolve_path=True),
) -> None:
    """Train from YAML and persist a fully resolved run."""
    typer.echo(train(load_config(config)))


@app.command("evaluate")
def evaluate(
    config: Path = typer.Option(..., exists=True), checkpoint: Path = typer.Option(..., exists=True)
) -> None:
    typer.echo(json.dumps(evaluate_checkpoint(load_config(config), checkpoint), indent=2))


@app.command("predict")
def predict(
    config: Path = typer.Option(..., exists=True), checkpoint: Path = typer.Option(..., exists=True)
) -> None:
    metrics = evaluate_checkpoint(load_config(config), checkpoint)
    typer.echo(
        json.dumps({"status": "prediction smoke pass", "validation_metrics": metrics}, indent=2)
    )


@app.command("visualize")
def visualize(run_id: str = typer.Option(...)) -> None:
    for output in plot_run(find_run(run_id)):
        typer.echo(output)


@app.command("smoke-test")
def smoke_test(task: str = typer.Option("all")) -> None:
    selected = TASKS if task == "all" else {task: TASKS[task]}
    make_synthetic(cache_root() / "synthetic", task)
    for name in selected:
        path = (
            repository_root()
            / "configs/experiments"
            / name
            / (
                {"hemorrhage": "H-S00", "ischemia": "I-S00", "aneurysm": "A-S00"}[name]
                + "-smoke.yaml"
            )
        )
        directory = train(load_config(path))
        typer.echo(f"{name}: {directory.name}")


@storage_app.command("doctor")
def storage_doctor(backend: str = typer.Option("local")) -> None:
    if backend == "local":
        probe = cache_root() / "storage-doctor.txt"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        typer.echo(json.dumps({"backend": "file", "status": "ok", "uri": probe.as_uri()}))
        return
    if backend == "gcs":
        from brainon_ai.storage.backends.gcs import GCSBackend

        GCSBackend()
        typer.echo("GCS credentials and client are available")
        return
    raise typer.BadParameter("backend must be local or gcs")


@run_app.command("sync")
def run_sync(
    run_id: str = typer.Option(...), destination: str | None = None, dry_run: bool = True
) -> None:
    source = find_run(run_id)
    destination = destination or os.getenv("BRAINON_EXPERIMENT_URI", "")
    if not destination:
        raise typer.BadParameter("set --destination or BRAINON_EXPERIMENT_URI")
    if destination.startswith("gs://"):
        from brainon_ai.storage.backends.gcs import GCSBackend

        backend = GCSBackend()
        for path in source.rglob("*"):
            if path.is_file():
                backend.upload(
                    path, f"{destination.rstrip('/')}/runs/{path.relative_to(source)}", dry_run
                )
    else:
        LocalBackend().sync(source, destination, dry_run)
    typer.echo(f"sync {'planned' if dry_run else 'completed'}")


@run_app.command("resume")
def run_resume(run_id: str = typer.Option(...)) -> None:
    typer.echo(resume_run(run_id))


@model_app.command("export")
def model_export(run_id: str = typer.Option(...), version: str = typer.Option(...)) -> None:
    typer.echo(export_model(run_id, version))


@model_app.command("pull")
def model_pull(
    task: str = typer.Option(...),
    version: str = typer.Option(...),
    destination: Path | None = None,
) -> None:
    base = os.getenv("BRAINON_MODEL_URI", "")
    if not base:
        raise typer.BadParameter("BRAINON_MODEL_URI is required")
    destination = destination or cache_root() / "models" / task / version
    destination.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Model manifest: {base}/releases/{task}/{version}/manifest.json; "
        f"destination: {destination}"
    )


if __name__ == "__main__":
    app()
