import pytest
import torch
from brainon_ai.nnunet.losses.factory import RegionFocalLoss
@pytest.mark.parametrize("region",["dice","tversky"])
def test_region_focal_backward(region):
    loss=RegionFocalLoss(region=region,gamma=2,foreground_alpha=.75,background_alpha=.25,smooth=1e-5,region_weight=1,focal_weight=1,include_background=False,batch_dice=False,ignore_label=None,tversky_alpha=.3,tversky_beta=.7)
    logits=torch.randn(2,2,4,4,4,requires_grad=True);target=torch.randint(0,2,(2,1,4,4,4))
    value=loss(logits,target);assert torch.isfinite(value);value.backward();assert torch.isfinite(logits.grad).all()
def test_focal_emphasizes_hard_voxel():
    from brainon_ai.nnunet.losses.factory import _focal_loss
    labels=torch.tensor([[[[1]]]]);valid=torch.ones_like(labels,dtype=torch.bool)
    easy=torch.tensor([[[[[0.0]]],[[[4.0]]]]]);hard=torch.tensor([[[[[4.0]]],[[[0.0]]]]])
    assert _focal_loss(hard,labels,valid,2,.75,.25)>_focal_loss(easy,labels,valid,2,.75,.25)
