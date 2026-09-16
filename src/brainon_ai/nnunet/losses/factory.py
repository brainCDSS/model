from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch import nn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


def _targets_and_probabilities(net_output, target, include_background, ignore_label):
    labels=target[:,0].long()
    if ignore_label is None:
        valid=torch.ones_like(labels,dtype=torch.bool)
        safe_labels=labels
    else:
        valid=labels!=ignore_label
        safe_labels=torch.where(valid,labels,0)
    probabilities=torch.softmax(net_output,dim=1)
    one_hot=torch.zeros_like(probabilities).scatter_(1,safe_labels[:,None],1)
    mask=valid[:,None].to(probabilities.dtype)
    probabilities,one_hot=probabilities*mask,one_hot*mask
    if not include_background:
        probabilities,one_hot=probabilities[:,1:],one_hot[:,1:]
    return labels,valid,probabilities,one_hot


def _overlap_score(probabilities,one_hot,batch_dice,smooth,alpha=None,beta=None):
    axes=tuple(range(2,probabilities.ndim))
    if batch_dice:axes=(0,*axes)
    tp=(probabilities*one_hot).sum(axes)
    fp=(probabilities*(1-one_hot)).sum(axes)
    fn=((1-probabilities)*one_hot).sum(axes)
    if alpha is None:
        return (2*tp+smooth)/(2*tp+fp+fn+smooth)
    return (tp+smooth)/(tp+alpha*fp+beta*fn+smooth)


def _focal_loss(net_output,labels,valid,gamma,foreground_alpha,background_alpha):
    log_probabilities=torch.log_softmax(net_output,dim=1)
    safe_labels=torch.where(valid,labels,0)
    log_pt=log_probabilities.gather(1,safe_labels[:,None]).squeeze(1)
    pt=log_pt.exp()
    alpha=torch.where(labels>0,float(foreground_alpha),float(background_alpha))
    loss=-alpha*(1-pt).pow(float(gamma))*log_pt
    valid_count=valid.sum().clamp_min(1)
    return (loss*valid).sum()/valid_count


class TverskyCELoss(nn.Module):
    def __init__(self,*,alpha,beta,smooth,weight_ce,weight_tversky,include_background,batch_dice,ignore_label):
        super().__init__()
        if alpha<=0 or beta<=0:raise ValueError("Tversky alpha/beta must be positive")
        self.alpha,self.beta,self.smooth=float(alpha),float(beta),float(smooth)
        self.weight_ce,self.weight_tversky=float(weight_ce),float(weight_tversky)
        self.include_background,self.batch_dice=bool(include_background),bool(batch_dice)
        self.ignore_label=ignore_label
        self.ce=RobustCrossEntropyLoss(**({"ignore_index":ignore_label} if ignore_label is not None else {}))
    def forward(self,net_output,target):
        labels,_,probabilities,one_hot=_targets_and_probabilities(net_output,target,self.include_background,self.ignore_label)
        ce=self.ce(net_output,labels)
        score=_overlap_score(probabilities,one_hot,self.batch_dice,self.smooth,self.alpha,self.beta)
        return self.weight_ce*ce-self.weight_tversky*score.mean()


class RegionFocalLoss(nn.Module):
    def __init__(self,*,region,gamma,foreground_alpha,background_alpha,smooth,region_weight,focal_weight,include_background,batch_dice,ignore_label,tversky_alpha=.3,tversky_beta=.7):
        super().__init__()
        if region not in {"dice","tversky"}:raise ValueError(region)
        if gamma<0:raise ValueError("Focal gamma must be non-negative")
        self.region=region;self.gamma=float(gamma)
        self.foreground_alpha=float(foreground_alpha);self.background_alpha=float(background_alpha)
        self.smooth=float(smooth);self.region_weight=float(region_weight);self.focal_weight=float(focal_weight)
        self.include_background=bool(include_background);self.batch_dice=bool(batch_dice);self.ignore_label=ignore_label
        self.tversky_alpha=float(tversky_alpha);self.tversky_beta=float(tversky_beta)
    def forward(self,net_output,target):
        labels,valid,probabilities,one_hot=_targets_and_probabilities(net_output,target,self.include_background,self.ignore_label)
        if self.region=="dice":
            score=_overlap_score(probabilities,one_hot,self.batch_dice,self.smooth)
        else:
            score=_overlap_score(probabilities,one_hot,self.batch_dice,self.smooth,self.tversky_alpha,self.tversky_beta)
        focal=_focal_loss(net_output,labels,valid,self.gamma,self.foreground_alpha,self.background_alpha)
        return self.focal_weight*focal-self.region_weight*score.mean()


class InstanceCenterLoss(nn.Module):
    """DiceCE plus equal-per-lesion Dice and center detection supervision."""

    def __init__(self, *, base_loss, instance_weight, center_weight, smooth, center_fraction=0.25):
        super().__init__()
        if instance_weight < 0 or center_weight < 0:
            raise ValueError("instance/center weights must be non-negative")
        if not 0 < center_fraction <= 1:
            raise ValueError("center_fraction must be in (0, 1]")
        self.base_loss = base_loss
        self.instance_weight = float(instance_weight)
        self.center_weight = float(center_weight)
        self.smooth = float(smooth)
        self.center_fraction = float(center_fraction)

    def _instance_terms(self, probabilities, target):
        labels = target[:, 0].long()
        foreground = probabilities[:, 1]
        instance_scores, center_scores = [], []
        for batch_index in range(labels.shape[0]):
            target_np = labels[batch_index].detach().cpu().numpy() > 0
            components, count = connected_components(
                target_np, structure=np.ones((3,) * target_np.ndim, dtype=np.uint8)
            )
            prediction = foreground[batch_index]
            for component_id in range(1, count + 1):
                coordinates_np = np.argwhere(components == component_id)
                if not len(coordinates_np):
                    continue
                coordinates = torch.as_tensor(
                    coordinates_np.T, device=foreground.device, dtype=torch.long
                )
                values = prediction[tuple(coordinates)]
                intersection = values.sum()
                instance_scores.append(
                    (2 * intersection + self.smooth)
                    / (intersection + len(coordinates_np) + self.smooth)
                )

                centroid = coordinates_np.mean(axis=0, keepdims=True)
                squared_distance = ((coordinates_np - centroid) ** 2).sum(axis=1)
                center_count = max(1, int(np.ceil(len(coordinates_np) * self.center_fraction)))
                if center_count < len(coordinates_np):
                    center_indices_np = np.argpartition(
                        squared_distance, center_count - 1
                    )[:center_count]
                    center_indices = torch.as_tensor(
                        center_indices_np, device=foreground.device, dtype=torch.long
                    )
                    center_values = values[center_indices]
                else:
                    center_values = values
                center_scores.append(center_values.mean().clamp_min(1e-6))
        zero = foreground.sum() * 0
        instance_loss = 1 - torch.stack(instance_scores).mean() if instance_scores else zero
        center_loss = -torch.log(torch.stack(center_scores)).mean() if center_scores else zero
        return instance_loss, center_loss

    def forward(self, net_output, target):
        base = self.base_loss(net_output, target)
        instance_loss, center_loss = self._instance_terms(
            torch.softmax(net_output, dim=1), target
        )
        return base + self.instance_weight * instance_loss + self.center_weight * center_loss


class InstanceCenterDeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weights):
        super().__init__()
        self.loss = loss
        self.weights = [float(value) for value in weights]

    def forward(self, outputs, targets):
        result = self.weights[0] * self.loss(outputs[0], targets[0])
        for weight, output, target in zip(self.weights[1:], outputs[1:], targets[1:]):
            if weight:
                result = result + weight * self.loss.base_loss(output, target)
        return result


def _deep_supervision(trainer,loss):
    if not trainer.enable_deep_supervision:return loss
    scales=trainer._get_deep_supervision_scales()
    weights=np.asarray([1/(2**i) for i in range(len(scales))],dtype=np.float64)
    weights[-1]=1e-6 if trainer.is_ddp and not trainer._do_i_compile() else 0
    weights/=weights.sum()
    if isinstance(loss, InstanceCenterLoss):
        return InstanceCenterDeepSupervisionWrapper(loss, weights)
    return DeepSupervisionWrapper(loss,weights)


def build_loss(trainer,config):
    name=str(config.get("name","dice_ce")).lower()
    common={"include_background":bool(config.get("include_background",False)),"batch_dice":bool(trainer.configuration_manager.batch_dice),"ignore_label":trainer.label_manager.ignore_label}
    if name in {"dice_ce", "instance_center"}:
        loss=DC_and_CE_loss({"batch_dice":trainer.configuration_manager.batch_dice,"smooth":float(config.get("smooth",1e-5)),"do_bg":common["include_background"],"ddp":trainer.is_ddp},{},weight_ce=float(config.get("ce_weight",1.0)),weight_dice=float(config.get("dice_weight",1.0)),ignore_label=common["ignore_label"],dice_class=MemoryEfficientSoftDiceLoss)
        if name == "instance_center":
            loss=InstanceCenterLoss(base_loss=loss,instance_weight=float(config.get("instance_weight",1.0)),center_weight=float(config.get("center_weight",0.25)),center_fraction=float(config.get("center_fraction",0.25)),smooth=float(config.get("smooth",1e-5)))
    elif name=="tversky_ce":
        loss=TverskyCELoss(alpha=float(config.get("alpha",.3)),beta=float(config.get("beta",.7)),smooth=float(config.get("smooth",1e-5)),weight_ce=float(config.get("ce_weight",1.0)),weight_tversky=float(config.get("tversky_weight",1.0)),**common)
    elif name in {"dice_focal","tversky_focal"}:
        loss=RegionFocalLoss(region="dice" if name=="dice_focal" else "tversky",gamma=float(config.get("gamma",2.0)),foreground_alpha=float(config.get("foreground_alpha",.75)),background_alpha=float(config.get("background_alpha",.25)),smooth=float(config.get("smooth",1e-5)),region_weight=float(config.get("dice_weight" if name=="dice_focal" else "tversky_weight",1.0)),focal_weight=float(config.get("focal_weight",1.0)),tversky_alpha=float(config.get("tversky_alpha",.3)),tversky_beta=float(config.get("tversky_beta",.7)),**common)
    else:raise ValueError(f"unsupported nnU-Net loss: {name}")
    return _deep_supervision(trainer,loss)
