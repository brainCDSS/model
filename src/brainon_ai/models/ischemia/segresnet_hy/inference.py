"""Full-volume inference used by I-S00 training and evaluation."""

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from brainon_ai.models.ischemia.segresnet_hy.metrics import binary_dice


def infer_logits(model, image, roi_size, sw_batch_size, overlap, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        output = sliding_window_inference(
            image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        return output[0] if isinstance(output, (list, tuple)) else output


@torch.no_grad()
def validate(model, loader, device, roi_size, sw_batch_size, overlap, threshold, amp):
    model.eval()
    scores = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = infer_logits(model, image, roi_size, sw_batch_size, overlap, amp)
        scores.append(binary_dice(logits, label, threshold))
    return float(np.mean(scores))


@torch.no_grad()
def infer_probability(model, image, roi_size, sw_batch_size, overlap, amp):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits = sliding_window_inference(
            image, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, overlap=overlap,
        )
    return torch.sigmoid(logits)


@torch.no_grad()
def infer_binary(model, image, roi_size, sw_batch_size, overlap, threshold, amp):
    return infer_probability(model, image, roi_size, sw_batch_size, overlap, amp) >= threshold
