from typing import Optional

import torch
from torch.optim import Optimizer


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    prev_lr: float,
    loss: float,
    filename: str,
):
    print("=> Saving checkpoint")

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "prev_lr": prev_lr,
        "epoch": epoch,
        "loss": loss,
    }

    torch.save(state, filename)


def load_checkpoint(
    device: torch.device,
    model: torch.nn.Module,
    filename: str,
    optimizer: Optional[Optimizer] = None,
):
    print("=> Loading checkpoint")

    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    prev_lr = checkpoint.get("prev_lr", 3e-05)
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return epoch, prev_lr, loss
