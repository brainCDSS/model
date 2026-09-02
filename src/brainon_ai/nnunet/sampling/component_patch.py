from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label as connected_components

def component_weight(volume_ml,config):
    t=[float(v) for v in config["volume_thresholds_ml"]]; w=config["component_weights"]
    if volume_ml<t[0]:return float(w["lt_1ml"])
    if volume_ml<t[1]:return float(w["1_to_5ml"])
    if volume_ml<t[2]:return float(w["5_to_20ml"])
    return float(w["ge_20ml"])

def build_component_centers(dataset,spacing,config):
    voxel_volume_ml=float(np.prod(np.asarray(spacing,dtype=float)))/1000.0
    result,records={},[]
    for case_id in dataset.identifiers:
        _,segmentation,_,_=dataset.load_case(case_id)
        components,count=connected_components(np.any(np.asarray(segmentation)>0,axis=0))
        centers,weights=[],[]
        for component_id in range(1,count+1):
            coordinates=np.argwhere(components==component_id); voxels=int(len(coordinates))
            if not voxels:continue
            volume_ml=voxels*voxel_volume_ml; center=coordinates[len(coordinates)//2]; weight=component_weight(volume_ml,config)
            centers.append(center);weights.append(weight)
            records.append({"case_id":str(case_id),"component":component_id,"voxels":voxels,"volume_ml":round(volume_ml,6),"weight":weight,"center_zyx":[int(v) for v in center]})
        if centers:
            probabilities=np.asarray(weights,dtype=np.float64);probabilities/=probabilities.sum()
            result[str(case_id)]=(np.asarray(centers,dtype=np.int64),probabilities)
    return result,{"strategy":"foreground patches centered on connected components; validation unchanged","voxel_volume_ml":voxel_volume_ml,"num_cases_with_components":len(result),"num_components":len(records),"components":records}

def choose_component_center(case_id,centers,rng=np.random):
    item=centers.get(str(case_id))
    if item is None:return None
    coordinates,probabilities=item
    return coordinates[rng.choice(len(coordinates),p=probabilities)]

def write_component_report(path,report):
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".tmp")
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");temporary.replace(destination)


from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class ComponentCenteredDataLoader(nnUNetDataLoader):
    """Foreground oversampling patch를 가중 component 좌표에 맞추는 loader."""

    def __init__(self, *args, component_centers, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_centers = component_centers
        self._active_keys = []
        self._active_position = 0

    def get_indices(self):
        keys = super().get_indices()
        self._active_keys = list(keys)
        self._active_position = 0
        return keys

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        case_id = self._active_keys[self._active_position]
        self._active_position += 1
        center = choose_component_center(case_id, self.component_centers) if force_fg else None
        if center is None:
            return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)

        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for axis in range(dim):
            if need_to_pad[axis] + data_shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - data_shape[axis]
        lower_bounds = [-need_to_pad[axis] // 2 for axis in range(dim)]
        upper_bounds = [
            data_shape[axis] + need_to_pad[axis] // 2 + need_to_pad[axis] % 2
            - self.patch_size[axis] for axis in range(dim)
        ]
        bbox_lbs = [
            min(upper_bounds[axis], max(lower_bounds[axis],
                int(center[axis]) - self.patch_size[axis] // 2))
            for axis in range(dim)
        ]
        return bbox_lbs, [bbox_lbs[axis] + self.patch_size[axis] for axis in range(dim)]
