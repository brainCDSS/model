#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
import torch,yaml
from brainon_ai.nnunet.config import set_config_environment
from nnunetv2.run.run_training import get_trainer_from_args

def positive_macro(summary):
    values=[]
    for item in summary.get("metric_per_case",[]):
        metric=item["metrics"]["1"]; value=float(metric["Dice"])
        if float(metric.get("n_ref",0))>0 and np.isfinite(value): values.append(value)
    if not values: raise RuntimeError("positive validation case가 없습니다")
    return float(np.mean(values)),len(values)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True,type=Path); p.add_argument("--data-root",required=True,type=Path); p.add_argument("--results-root",required=True,type=Path); a=p.parse_args()
    cp=set_config_environment(a.config); c=yaml.safe_load(cp.read_text(encoding="utf-8")); e,q=c["experiment"],c["plans"]
    dataset=f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"; fold=a.results_root.resolve()/dataset/f"nnUNetTrainerAneurysmBinaryBaseline__{q['output_identifier']}__{e['configuration']}"/f"fold_{e['fold']}"
    hp=fold/"full_validation_history.json"; history=json.loads(hp.read_text(encoding="utf-8")); policy=c["full_validation"]; k,r=int(policy["keep_top_k"]),int(policy["neighborhood_radius"])
    anchors=[int(x["epoch"]) for x in sorted(history,key=lambda x:float(x["macro_dice_positive"]),reverse=True)[:k]]
    epochs=sorted({x for anchor in anchors for x in range(max(1,anchor-r),min(int(c["trainer"]["epochs"]),anchor+r)+1)})
    print(f"A-B01 Top-{k} anchors={anchors}; neighborhood={epochs}",flush=True)
    os.environ.update(nnUNet_raw=str(a.data_root.resolve()/"nnUNet_raw"),nnUNet_preprocessed=str(a.data_root.resolve()/"nnUNet_preprocessed"),nnUNet_results=str(a.results_root.resolve()))
    trainer=get_trainer_from_args(str(e["dataset_id"]),e["configuration"],int(e["fold"]),"nnUNetTrainerAneurysmBinaryBaseline",q["output_identifier"],False,device=torch.device("cuda")); known={int(x["epoch"]):x for x in history}
    for epoch in epochs:
        ck=fold/"epoch_checkpoints"/f"checkpoint_epoch_{epoch:04d}.pth"
        if not ck.is_file(): print(f"epoch {epoch}: checkpoint 없음",flush=True); continue
        dst=fold/"full_validation"/f"epoch_{epoch:04d}"; summary=dst/"summary.json"
        if not summary.is_file():
            trainer.load_checkpoint(ck); transient=fold/"validation"
            if transient.exists(): shutil.rmtree(transient)
            trainer.perform_actual_validation(False); dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists(): shutil.rmtree(dst)
            shutil.move(str(transient),str(dst))
        macro,count=positive_macro(json.loads(summary.read_text())); known[epoch]={"epoch":epoch,"macro_dice_positive":macro,"positive_cases":count,"summary":str(summary),"checkpoint":str(ck)}; print(epoch,f"{macro:.6f}",flush=True)
    history=[known[x] for x in sorted(known)]; hp.write_text(json.dumps(history,ensure_ascii=False,indent=2)+"\n"); ranked=sorted(history,key=lambda x:float(x["macro_dice_positive"]),reverse=True); best=ranked[0]
    shutil.copy2(best["checkpoint"],fold/"checkpoint_best_full_macro.pth"); (fold/"best_full_macro.json").write_text(json.dumps(best,ensure_ascii=False,indent=2)+"\n")
    anchors=[int(x["epoch"]) for x in ranked[:k]]; keep={x for anchor in anchors for x in range(max(1,anchor-r),min(int(c["trainer"]["epochs"]),anchor+r)+1)}
    for ck in (fold/"epoch_checkpoints").glob("checkpoint_epoch_*.pth"):
        if int(ck.stem.rsplit("_",1)[1]) not in keep: ck.unlink()
    print(f"최종 best epoch={best['epoch']}, macro Dice={best['macro_dice_positive']:.6f}"); return 0
if __name__=="__main__": raise SystemExit(main())
