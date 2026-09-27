from __future__ import annotations
from pathlib import Path
import hashlib
import yaml

def load_registry(path:str|Path)->dict:
    p=Path(path); payload=yaml.safe_load(p.read_text(encoding="utf-8")) or {}; models=payload.get("models",{})
    if not models: raise ValueError(f"no models in {p}")
    return models

def resolve_model(model_id:str, path:str|Path, repo:Path)->dict:
    models=load_registry(path)
    if model_id not in models: raise KeyError(f"unknown model ID {model_id}; choices={sorted(models)}")
    m=dict(models[model_id]); m["model_id"]=model_id
    for key in ("prediction_dir","label_dir","metrics_csv","model_folder","raw_images","split_file","checkpoint","config","data_root","results_root","data_root"):
        if m.get(key):
            q=Path(m[key]); m[key]=str(q if q.is_absolute() else (repo/q).resolve())
    m["tta_prediction_dirs"]=[str((repo/Path(x)).resolve()) if not Path(x).is_absolute() else x for x in m.get("tta_prediction_dirs",[])]
    checkpoint = Path(m["checkpoint"]) if m.get("checkpoint") else (Path(m["model_folder"]) / "fold_0" / m.get("checkpoint_name", "")) if m.get("model_folder") else None
    base_version = str(m.get("version", "unversioned"))
    component_paths = [Path(x) if Path(x).is_absolute() else (repo / x).resolve() for x in m.get("component_checkpoints", [])]
    if component_paths:
        missing = [str(x) for x in component_paths if not x.is_file()]
        if missing:
            m["checkpoint_sha256"] = None
            m["checkpoint_resolved"] = [str(x) for x in component_paths]
        else:
            digest = hashlib.sha256("".join(hashlib.sha256(x.read_bytes()).hexdigest() for x in component_paths).encode()).hexdigest()
            m["checkpoint_sha256"] = digest
            m["checkpoint_resolved"] = [str(x) for x in component_paths]
            m["version"] = f"{base_version}-{digest[:12]}"
    elif checkpoint is not None and checkpoint.is_file():
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        m["checkpoint_sha256"] = digest
        m["checkpoint_resolved"] = str(checkpoint)
        m["version"] = f"{base_version}-{digest[:12]}"
    else:
        m["checkpoint_sha256"] = None
    return m
