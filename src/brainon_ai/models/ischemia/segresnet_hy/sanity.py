"""Forward, shape, and loss sanity check from the original training entry point."""

import torch


def run_sanity(model, loader, loss_fn, device, amp):
    model.train()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    label = batch["label"].to(device)

    print("Image shape:", tuple(image.shape))
    print("Label shape:", tuple(label.shape))

    with torch.cuda.amp.autocast(enabled=amp):
        output = model(image)
        loss = loss_fn(output, label)

    output_shapes = [tuple(item.shape) for item in output] if isinstance(output, (list, tuple)) else [tuple(output.shape)]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Sanity loss is non-finite: {loss.item()}")
    print("Output shapes:", output_shapes)
    print("Loss:", float(loss.item()))
    print("✓ SegResNet sanity check 통과")
    return output, loss
