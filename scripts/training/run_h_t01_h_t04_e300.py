#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

EXPERIMENTS = ("H-T01-E300", "H-T02-E300", "H-T03-E300", "H-T04-E300")

def now(): return datetime.now(timezone.utc).isoformat()
def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(temporary, path)
def read_status(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    results = repository / "runs/nnunet"
    batch = repository / "runs/batches/H-T01_H-T04_E300"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "nnUNet_n_proc_DA": "0", "nnUNet_def_n_proc": "1"})
    env["PYTHONPATH"] = os.pathsep.join([str(repository / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    prepare = [sys.executable, str(repository / "scripts/preprocessing/prepare_h_t01_h_t04_e300_plans.py"), "--preprocessed-root", str(args.data_root.resolve() / "nnUNet_preprocessed")]
    if args.dry_run: print(" ".join(prepare))
    else: subprocess.run(prepare, check=True, env=env)
    commands = {}
    for name in EXPERIMENTS:
        config = repository / f"configs/experiments/hemorrhage/tuning/{name}.yaml"
        command = [sys.executable, str(repository / "scripts/training/run_nnunet_continuation.py"), "--config", str(config), "--data-root", str(args.data_root.resolve()), "--results-root", str(results)]
        if args.resume: command.append("--resume")
        if args.dry_run: command.append("--dry-run")
        commands[name] = command
    validation = [sys.executable, str(repository / "scripts/validation/evaluate_h_tuning_e300_neighborhood.py"), "--data-root", str(args.data_root.resolve()), "--radius", "4"]
    if args.dry_run:
        for name, command in commands.items(): print(name + ": " + " ".join(command))
        print("Top-3 neighborhood validation: " + " ".join(validation))
        return 0
    previous = read_status(batch / "batch_status.json") if args.resume else {}
    status = {"status": "running", "started_at": previous.get("started_at", now()), "current": None, "experiments": previous.get("experiments", {})}
    write_status(batch / "batch_status.json", status)
    for name, command in commands.items():
        old = status["experiments"].get(name, {})
        if args.resume and old.get("status") == "completed":
            print(f"[skip completed] {name}", flush=True)
            continue
        status["current"] = name
        status["experiments"][name] = {"status": "running", "started_at": now()}
        write_status(batch / "batch_status.json", status)
        with (batch / f"{name}.log").open("a", encoding="utf-8") as log:
            code = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True).wait()
        status["experiments"][name].update({"status": "completed" if code == 0 else "failed", "finished_at": now(), "exit_code": code})
        write_status(batch / "batch_status.json", status)
        if code:
            status.update({"status": "failed", "current": None, "finished_at": now()})
            write_status(batch / "batch_status.json", status)
            return code
    status.update({"current": "Top-3 neighborhood validation", "phase": "full_validation"})
    write_status(batch / "batch_status.json", status)
    with (batch / "top3_neighborhood_validation.log").open("a", encoding="utf-8") as log:
        code = subprocess.Popen(validation, env=env, stdout=log, stderr=subprocess.STDOUT, text=True).wait()
    if code:
        status.update({"status": "failed", "current": None, "finished_at": now(), "validation_exit_code": code})
        write_status(batch / "batch_status.json", status)
        return code
    status.update({"status": "completed", "current": None, "phase": "completed", "finished_at": now()})
    write_status(batch / "batch_status.json", status)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
