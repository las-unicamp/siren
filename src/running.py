from dataclasses import dataclass
from typing import Any, Tuple, TypedDict

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio

from src.datasets import DatasetReturnItems
from src.dtos import RunnerReturnItems
from src.metrics import RelativeResidualError
from src.tracking import NetworkTracker


class TrainingConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    device: torch.device
    loss_fn: torch.nn.Module


@dataclass
class TrainingMetrics:
    psnr_metric = PeakSignalNoiseRatio()
    rre_metric = RelativeResidualError()
    mae_metric = MeanAbsoluteError()


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        config: TrainingConfig,
        metrics: TrainingMetrics,
    ):
        self.epoch = 0
        self.loader = loader
        self.model = model
        self.optimizer = config["optimizer"]
        self.loss_fn = config["loss_fn"]
        self.device = config["device"]

        # Send to device
        self.model = self.model.to(device=self.device)
        self.psnr_metric = metrics.psnr_metric.to(device=self.device)
        self.mae_metric = metrics.mae_metric.to(device=self.device)
        self.loss_fn = self.loss_fn.to(device=self.device)

    def run(self, tracker: NetworkTracker) -> RunnerReturnItems:
        num_batches = len(self.loader)

        epoch_loss = 0.0

        self.model.train()

        for batch_index, data in enumerate(self.loader):
            data: DatasetReturnItems

            inputs = data["coords"]
            targets = data["derivatives"]

            predictions = self.model(inputs)
            loss, derivatives = self.loss_fn(predictions, targets, inputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.psnr_metric.forward(targets, derivatives)
            self.mae_metric.forward(targets, derivatives)

            if tracker.should_save_intermediary_data():
                tracker.add_batch_data("coordinates", inputs)
                tracker.add_batch_data("predictions", predictions)
                tracker.add_batch_data("fitted_derivatives", derivatives)

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / num_batches
        epoch_psnr = self.psnr_metric.compute()
        epoch_mae = self.mae_metric.compute()

        self.psnr_metric.reset()
        self.mae_metric.reset()

        self.epoch += 1

        return RunnerReturnItems(
            epoch_loss=epoch_loss,
            epoch_psnr=epoch_psnr,
            epoch_mae=epoch_mae,
        )


def run_epoch(runner: Runner, tracker: NetworkTracker) -> Tuple[float, float, float]:
    results = runner.run(tracker)

    tracker.add_epoch_metric("loss", results["epoch_loss"], runner.epoch)
    tracker.add_epoch_metric("psnr", results["epoch_psnr"], runner.epoch)
    tracker.add_epoch_metric("mae", results["epoch_mae"], runner.epoch)

    return (
        results["epoch_loss"],
        results["epoch_psnr"],
        results["epoch_mae"],
    )
