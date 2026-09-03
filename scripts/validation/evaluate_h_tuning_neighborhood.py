#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.run.run_training import get_trainer_from_args

EXPERIMENTS=("H-T01","H-T02","H-T03","H-T04")
def positive_macro(summary):
    values=[]
    for item in summary["metric_per_case"]:
        metric=item["metrics"]["1"]
        if float(metric["n_ref"])>0 and np.isfinite(float(metric["Dice"])):values.append(float(metric["Dice"]))
    return float(np.mean(values)),len(values)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--data-root",required=True,type=Path);parser.add_argument("--radius",type=int,default=4);args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2];results=repository/"runs/nnunet";configure_paths(args.data_root,results)
    for name in EXPERIMENTS:
        config_path=repository/f"configs/experiments/hemorrhage/tuning/{name}.yaml";config=yaml.safe_load(config_path.read_text());e,p=config["experiment"],config["plans"]
        fold=results/f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"/f"nnUNetTrainerICHBaseline__{p['output_identifier']}__{e['configuration']}"/f"fold_{e['fold']}"
        history_path=fold/"full_validation_history.json";history=json.loads(history_path.read_text())
        top_k = int(config.get("full_validation", {}).get("keep_top_k", 3))
        anchors = [
            int(item["epoch"])
            for item in sorted(
                history,
                key=lambda item: float(item["macro_dice_positive"]),
                reverse=True,
            )[:top_k]
        ]
        max_epochs = int(config["trainer"]["epochs"])
        epochs = sorted({
            epoch
            for anchor in anchors
            for epoch in range(max(1, anchor - args.radius), min(max_epochs, anchor + args.radius) + 1)
        })
        print(f"{name} Top-{top_k} anchors: {anchors}; evaluating {len(epochs)} checkpoints", flush=True)
        set_config_environment(config_path)
        trainer=get_trainer_from_args(str(e["dataset_id"]),e["configuration"],int(e["fold"]),"nnUNetTrainerICHBaseline",p["output_identifier"],False,device=torch.device("cuda"))
        known={int(item["epoch"]):item for item in history}
        for epoch in epochs:
            checkpoint=fold/"epoch_checkpoints"/f"checkpoint_epoch_{epoch:04d}.pth"
            if not checkpoint.is_file():
                print(f"{name} epoch {epoch}: checkpoint not available; skipping", flush=True)
                continue
            destination=fold/"full_validation"/f"epoch_{epoch:04d}";summary_path=destination/"summary.json"
            if not summary_path.is_file():
                trainer.load_checkpoint(checkpoint);transient=fold/"validation"
                if transient.exists():shutil.rmtree(transient)
                trainer.perform_actual_validation(False);destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():shutil.rmtree(destination)
                shutil.move(str(transient),str(destination))
            macro,count=positive_macro(json.loads(summary_path.read_text()))
            known[epoch]={"epoch":epoch,"macro_dice_positive":macro,"positive_cases":count,"summary":str(summary_path),"checkpoint":str(checkpoint)}
            print(name,epoch,f"{macro:.6f}",flush=True)
        history=[known[key] for key in sorted(known)];history_path.write_text(json.dumps(history,ensure_ascii=False,indent=2)+chr(10))
        ranked = sorted(history, key=lambda x: float(x["macro_dice_positive"]), reverse=True)
        best = ranked[0]
        shutil.copy2(best["checkpoint"],fold/"checkpoint_best_full_macro.pth")
        (fold/"best_full_macro.json").write_text(json.dumps(best,ensure_ascii=False,indent=2)+chr(10))
        final_anchors = [int(item["epoch"]) for item in ranked[:top_k]]
        keep = {
            epoch
            for anchor in final_anchors
            for epoch in range(max(1, anchor - args.radius), min(max_epochs, anchor + args.radius) + 1)
        }
        for checkpoint in (fold / "epoch_checkpoints").glob("checkpoint_epoch_*.pth"):
            if int(checkpoint.stem.rsplit("_", 1)[1]) not in keep:
                checkpoint.unlink()
        print(f"{name} final Top-{top_k}: {final_anchors}", flush=True)
        del trainer;torch.cuda.empty_cache()
    return 0
if __name__=="__main__":raise SystemExit(main())
