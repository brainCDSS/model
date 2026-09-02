from __future__ import annotations
import numpy as np
import torch
from torch import nn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class TverskyCELoss(nn.Module):
    """CE와 foreground Tversky loss의 가중합."""
    def __init__(self, *, alpha, beta, smooth, weight_ce, weight_tversky, include_background, batch_dice, ignore_label):
        super().__init__()
        if alpha <= 0 or beta <= 0: raise ValueError("Tversky alpha/beta는 0보다 커야 합니다.")
        self.alpha,self.beta,self.smooth=float(alpha),float(beta),float(smooth)
        self.weight_ce,self.weight_tversky=float(weight_ce),float(weight_tversky)
        self.include_background,self.batch_dice=bool(include_background),bool(batch_dice)
        self.ignore_label=ignore_label
        self.ce=RobustCrossEntropyLoss(**({"ignore_index":ignore_label} if ignore_label is not None else {}))
    def forward(self, net_output, target):
        labels=target[:,0].long()
        if self.ignore_label is None: valid,safe_labels=torch.ones_like(labels,dtype=torch.bool),labels
        else:
            valid=labels!=self.ignore_label; safe_labels=torch.where(valid,labels,0)
        ce=self.ce(net_output,labels)
        probabilities=torch.softmax(net_output,dim=1)
        one_hot=torch.zeros_like(probabilities).scatter_(1,safe_labels[:,None],1)
        mask=valid[:,None].to(probabilities.dtype)
        probabilities,one_hot=probabilities*mask,one_hot*mask
        if not self.include_background: probabilities,one_hot=probabilities[:,1:],one_hot[:,1:]
        axes=tuple(range(2,net_output.ndim))
        if self.batch_dice: axes=(0,*axes)
        tp=(probabilities*one_hot).sum(axes); fp=(probabilities*(1-one_hot)).sum(axes); fn=((1-probabilities)*one_hot).sum(axes)
        score=(tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        return self.weight_ce*ce-self.weight_tversky*score.mean()

def _deep_supervision(trainer,loss):
    if not trainer.enable_deep_supervision:return loss
    scales=trainer._get_deep_supervision_scales()
    weights=np.asarray([1/(2**i) for i in range(len(scales))],dtype=np.float64)
    weights[-1]=1e-6 if trainer.is_ddp and not trainer._do_i_compile() else 0
    weights/=weights.sum()
    return DeepSupervisionWrapper(loss,weights)

def build_loss(trainer,config):
    name=str(config.get("name","dice_ce")).lower()
    if name=="dice_ce":
        loss=DC_and_CE_loss({"batch_dice":trainer.configuration_manager.batch_dice,"smooth":float(config.get("smooth",1e-5)),"do_bg":bool(config.get("include_background",False)),"ddp":trainer.is_ddp},{},weight_ce=float(config.get("ce_weight",1.0)),weight_dice=float(config.get("dice_weight",1.0)),ignore_label=trainer.label_manager.ignore_label,dice_class=MemoryEfficientSoftDiceLoss)
    elif name=="tversky_ce":
        loss=TverskyCELoss(alpha=float(config.get("alpha",.3)),beta=float(config.get("beta",.7)),smooth=float(config.get("smooth",1e-5)),weight_ce=float(config.get("ce_weight",1.0)),weight_tversky=float(config.get("tversky_weight",1.0)),include_background=bool(config.get("include_background",False)),batch_dice=bool(trainer.configuration_manager.batch_dice),ignore_label=trainer.label_manager.ignore_label)
    else: raise ValueError(f"지원하지 않는 nnU-Net loss: {name}")
    return _deep_supervision(trainer,loss)
