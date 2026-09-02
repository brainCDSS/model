import numpy as np
import torch
from brainon_ai.nnunet.losses.factory import TverskyCELoss
from brainon_ai.nnunet.sampling.component_patch import component_weight,build_component_centers,choose_component_center
CONFIG={"volume_thresholds_ml":[1,5,20],"component_weights":{"lt_1ml":8,"1_to_5ml":4,"5_to_20ml":2,"ge_20ml":1}}
def test_component_weights():
    assert [component_weight(v,CONFIG) for v in (.5,3,10,25)]==[8,4,2,1]
class Dataset:
    identifiers=["case"]
    def load_case(self,_):
        seg=np.zeros((1,8,8,8),dtype=np.uint8);seg[0,1,1,1]=1;seg[0,5:7,5:7,5:7]=1
        return None,seg,None,{}
def test_component_centers():
    centers,report=build_component_centers(Dataset(),[1,1,1],CONFIG)
    assert report["num_components"]==2
    assert choose_component_center("case",centers).shape==(3,)
def test_tversky_loss_backward():
    loss=TverskyCELoss(alpha=.3,beta=.7,smooth=1e-5,weight_ce=1,weight_tversky=1,include_background=False,batch_dice=False,ignore_label=None)
    logits=torch.randn(2,2,4,4,4,requires_grad=True);target=torch.randint(0,2,(2,1,4,4,4))
    value=loss(logits,target);assert torch.isfinite(value);value.backward();assert logits.grad is not None
