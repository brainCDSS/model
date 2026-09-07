#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch,yaml
from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.run.run_training import get_trainer_from_args

def positive_macro(summary):
 values=[]
 for item in summary["metric_per_case"]:
  metric=item["metrics"]["1"]
  if float(metric["n_ref"])>0 and np.isfinite(float(metric["Dice"])):values.append(float(metric["Dice"]))
 if not values:raise RuntimeError("positive validation case가 없습니다")
 return float(np.mean(values)),len(values)

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True,type=Path);p.add_argument("--data-root",required=True,type=Path);p.add_argument("--results-root",required=True,type=Path);a=p.parse_args()
 config_path=a.config.resolve();c=yaml.safe_load(config_path.read_text());e,q=c["experiment"],c["plans"];configure_paths(a.data_root,a.results_root);set_config_environment(config_path)
 fold=a.results_root.resolve()/f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"/f"nnUNetTrainerICHBaseline__{q['output_identifier']}__{e['configuration']}"/f"fold_{e['fold']}";hp=fold/"full_validation_history.json"
 if not hp.is_file():raise FileNotFoundError(hp)
 history=json.loads(hp.read_text());k=int(c["full_validation"].get("keep_top_k",3));r=int(c["full_validation"].get("neighborhood_radius",4));anchors=[int(x["epoch"]) for x in sorted(history,key=lambda x:float(x["macro_dice_positive"]),reverse=True)[:k]];epochs=sorted({x for anchor in anchors for x in range(max(1,anchor-r),min(int(c["trainer"]["epochs"]),anchor+r)+1)})
 print(f"fold {e['fold']} Top-{k} anchors={anchors}; evaluating {len(epochs)} checkpoints",flush=True)
 trainer=get_trainer_from_args(str(e["dataset_id"]),e["configuration"],int(e["fold"]),"nnUNetTrainerICHBaseline",q["output_identifier"],False,device=torch.device("cuda"));known={int(x["epoch"]):x for x in history}
 for epoch in epochs:
  ck=fold/"epoch_checkpoints"/f"checkpoint_epoch_{epoch:04d}.pth"
  if not ck.is_file():print(f"epoch {epoch}: checkpoint unavailable",flush=True);continue
  dst=fold/"full_validation"/f"epoch_{epoch:04d}";summary=dst/"summary.json"
  if not summary.is_file():
   trainer.load_checkpoint(ck);transient=fold/"validation"
   if transient.exists():shutil.rmtree(transient)
   trainer.perform_actual_validation(False);dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists():shutil.rmtree(dst)
   shutil.move(str(transient),str(dst))
  macro,count=positive_macro(json.loads(summary.read_text()));known[epoch]={"epoch":epoch,"macro_dice_positive":macro,"positive_cases":count,"summary":str(summary),"checkpoint":str(ck)};print(epoch,f"{macro:.6f}",flush=True)
 history=[known[x] for x in sorted(known)];hp.write_text(json.dumps(history,ensure_ascii=False,indent=2)+"\n");ranked=sorted(history,key=lambda x:float(x["macro_dice_positive"]),reverse=True);best=ranked[0];shutil.copy2(best["checkpoint"],fold/"checkpoint_best_full_macro.pth");(fold/"best_full_macro.json").write_text(json.dumps(best,ensure_ascii=False,indent=2)+"\n")
 final=[int(x["epoch"]) for x in ranked[:k]];keep={x for anchor in final for x in range(max(1,anchor-r),min(int(c["trainer"]["epochs"]),anchor+r)+1)}
 for ck in (fold/"epoch_checkpoints").glob("checkpoint_epoch_*.pth"):
  if int(ck.stem.rsplit("_",1)[1]) not in keep:ck.unlink()
 print(f"fold {e['fold']} final best={best['epoch']} macro={best['macro_dice_positive']:.6f} Top-{k}={final}");return 0
if __name__=="__main__":raise SystemExit(main())
