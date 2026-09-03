from pathlib import Path
import numpy as np
from brainon_ai.nnunet.config import load_config
from brainon_ai.nnunet.sampling.small_lesion import build_sampling_probabilities
from brainon_ai.nnunet.sampling.component_patch import build_component_centers

class Dataset:
    identifiers=["negative","tiny","large"]
    def load_case(self,identifier):
        seg=np.zeros((1,20,20,20),dtype=np.uint8)
        if identifier=="tiny":seg[0,2,2,2]=1
        if identifier=="large":seg[0,5:15,5:15,5:15]=1
        return None,seg,None,{}

def test_h_n130_builds_case_and_component_sampling():
    config=load_config(Path("configs/experiments/hemorrhage/H-N130.yaml"))["sampling"]
    probabilities,case_report=build_sampling_probabilities(Dataset(),[1,1,1],config)
    centers,component_report=build_component_centers(Dataset(),[1,1,1],config)
    assert probabilities[1]>probabilities[2]
    assert probabilities[1]>probabilities[0]
    assert "tiny" in centers and "large" in centers
    assert case_report["num_train_cases"]==3
    assert component_report["num_components"]==2
