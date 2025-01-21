from dataclasses import dataclass
from typing import Any, Literal, Protocol, Tuple, TypedDict

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio

from src.datasets import DatasetReturnItems
from src.dtos import RunnerReturnItems
from src.losses import FitGradients, FitLaplacian
from src.metrics import RelativeResidualError
from src.my_types import TensorFloatN, TensorFloatNx2, TensorFloatNx3
from src.tracking import NetworkTracker


class TrainingPrecisionStrategy(Protocol):
    def forward_batch(
        self, model: torch.nn.Module, inputs: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatN:
        """Implements forward pass for a batch"""

    def backward_batch(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """Implements backward pass for a batch"""


class MixedPrecisionStrategy:
    def __init__(self, dtype=torch.float16):
        self.scaler = torch.amp.GradScaler()
        self.dtype = dtype  # Default to float16 for mixed precision

    def forward_batch(
        self, model: torch.nn.Module, inputs: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatN:
        device = next(model.parameters()).device
        device_type = "cuda" if device.type == "cuda" else "cpu"

        with torch.autocast(
            device_type=device_type,
            dtype=self.dtype,
            cache_enabled=True,
            enabled=True,
        ):
            predictions = model(inputs)
        return predictions

    def backward_batch(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


class StandardPrecisionStrategy:
    def forward_batch(
        self, model: torch.nn.Module, inputs: TensorFloatNx2 | TensorFloatNx3
    ) -> TensorFloatN:
        return model(inputs)

    def backward_batch(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TrainingConfig(TypedDict):
    fit_option: Literal["gradients, laplacian"]
    optimizer: torch.optim.Optimizer
    device: torch.device
    precision_strategy: TrainingPrecisionStrategy


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
        self._validate_config(config)

        self.epoch = 0
        self.loader = loader
        self.model = model
        self.optimizer = config["optimizer"]
        is_laplacian = config["fit_option"] == "laplacian"
        self.loss_fn = FitLaplacian() if is_laplacian else FitGradients()
        self.device = config["device"]
        self.precision_strategy = config["precision_strategy"]

        # Send to device
        self.model = self.model.to(device=self.device)
        self.psnr_metric = metrics.psnr_metric.to(device=self.device)
        self.mae_metric = metrics.mae_metric.to(device=self.device)
        self.loss_fn = self.loss_fn.to(device=self.device)

    @staticmethod
    def _validate_config(config: TrainingConfig):
        """Validates the provided training configuration."""
        if not isinstance(
            config["precision_strategy"],
            (MixedPrecisionStrategy, StandardPrecisionStrategy),
        ):
            raise TypeError(
                f"Invalid strategy type. "
                f"Expected MixedPrecisionStrategy or StandardPrecisionStrategy, "
                f"but got {type(config['precision_strategy']).__name__}."
            )

    def run(self, tracker: NetworkTracker) -> RunnerReturnItems:
        num_batches = len(self.loader)

        epoch_loss = 0.0

        self.model.train()

        for batch_index, data in enumerate(self.loader):
            data: DatasetReturnItems

            inputs = data["coords"].squeeze()
            targets = data["derivatives"].squeeze()

            predictions = self.precision_strategy.forward_batch(self.model, inputs)
            loss, derivatives = self.loss_fn(predictions, targets, inputs)

            self.optimizer.zero_grad()
            self.precision_strategy.backward_batch(loss, self.optimizer)

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
