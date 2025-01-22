# ruff: noqa: N806

import numpy as np
import scipy
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from src.datasets import DerivativesDataset
from src.dtos import DatasetReturnItems, TrainingData
from src.losses import FiniteDifferenceConfig, FitGradients, FitLaplacian
from src.model import SIREN
from src.running import Runner, TrainingMetrics, run_epoch
from src.vector_ops import (
    AutogradDerivativesStrategy,
    FiniteDifferenceDerivativesStrategy,
)

torch.set_default_dtype(torch.float32)


def pressure_field(x, y, t, rho, nu):
    """
    Compute the analytical pressure field of the Taylor-Green vortex.
    """
    F = np.exp(-2 * nu * t)
    return (rho / 4) * (np.cos(2 * x) + np.cos(2 * y)) * F**2


def pressure_gradient(x, y, t, rho, nu):
    F = np.exp(-2 * nu * t)
    dpdx = -rho / 2 * np.sin(2 * x) * F**2
    dpdy = -rho / 2 * np.sin(2 * y) * F**2
    return dpdx, dpdy


def pressure_laplacian(x, y, t, rho, nu):
    F = np.exp(-2 * nu * t)
    return -rho * (np.cos(2 * x) + np.cos(2 * y)) * F**2


def create_tgv_training_data(
    grid_size=100, t=0.5, rho=1.0, nu=0.01, include_laplacian=True
) -> TrainingData:
    """
    Generate Taylor-Green vortex training data.
    """
    x = np.linspace(-np.pi, np.pi, grid_size)
    y = np.linspace(-np.pi, np.pi, grid_size)
    x, y = np.meshgrid(x, y)

    if include_laplacian:
        laplacian = pressure_laplacian(x, y, t, rho, nu)
        mask = np.zeros_like(laplacian, dtype=bool)
        return TrainingData(
            coordinates=np.stack((x.ravel(), y.ravel()), axis=-1),
            laplacian=laplacian,
            mask=mask,
        )
    else:
        gradient_x, gradient_y = pressure_gradient(x, y, t, rho, nu)
        mask = np.zeros_like(gradient_x, dtype=bool)
        return TrainingData(
            coordinates=np.stack((x.ravel(), y.ravel()), axis=-1),
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            mask=mask,
        )


class MockTracker:
    def add_batch_metric(self, name: str, value: float, step: int):
        pass

    def add_batch_data(self, key: str, data):
        pass

    def add_epoch_metric(self, name: str, value: float, step: int):
        pass

    def should_save_intermediary_data(self) -> bool:
        return False

    def get_batch_data(self):
        return {}

    def save_epoch_data(self, name: str, step: int):
        pass


def custom_collate_fn(batch):
    """Custom collate function to remove the leading batch dimension."""
    data: DatasetReturnItems
    data = batch[0]  # Get the first (and only) item in the batch

    # Now, remove the leading dimension (which is always 1)
    data["coords"] = data["coords"].squeeze(0)
    data["derivatives"] = data["derivatives"].squeeze(0)

    return data


def test_tgv_gradients_integration_autograd():
    """
    Integration test for SIREN model with Taylor-Green vortex data.
    Runs optimization for 2000 epochs and compares the predicted field
    to the true TGV pressure field.
    """
    training_data = create_tgv_training_data(
        grid_size=100, t=0.5, rho=1.0, nu=0.01, include_laplacian=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DerivativesDataset(training_data=training_data, device=device)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
    )

    model = SIREN(
        hidden_features=64,
        hidden_layers=3,
        first_omega=5.0,
        hidden_omega=10.0,
    ).to(device)

    derivatives_strategy = AutogradDerivativesStrategy()

    loss_fn = FitGradients(derivatives_strategy, finite_diff_config=None)

    optimizer = Adam(model.parameters(), lr=3e-5)

    tracker = MockTracker()
    config = {
        "device": device,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    runner = Runner(dataloader, model, config, TrainingMetrics())

    num_epochs = 2000

    for epoch in range(num_epochs):
        epoch_loss, epoch_psnr, epoch_mae = run_epoch(runner, tracker)

    assert epoch_loss < 1e-4, f"Epoch loss {epoch_loss} is higher than expected"
    assert epoch_psnr > 45, f"Epoch PSNR {epoch_psnr} is lower than expected"
    assert epoch_mae < 1e-2, f"Epoch MAE {epoch_mae} is higher than expected"

    # Check if the model predictions are close enough to the TGV pressure field
    model.eval()
    with torch.no_grad():
        predictions = model(dataset[0]["coords"].to(device)).cpu().numpy()

    predictions_field = predictions.reshape(
        100, 100
    )  # Reshape predictions to match grid

    # Compute the true pressure field
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    x, y = np.meshgrid(x, y)
    true_pressure_field = pressure_field(x, y, t=0.5, rho=1.0, nu=0.01)

    # Normalize by subtracting the mean to account for integration constant
    predictions_field -= predictions_field.mean()
    true_pressure_field -= true_pressure_field.mean()

    # Compare model output to true TGV pressure field (mean squared error tolerance)
    mse = np.mean((predictions_field - true_pressure_field) ** 2)
    assert (
        mse < 1e-4
    ), f"MSE between predicted and true pressure field is too high: {mse}"

    scipy.io.savemat(
        "test_tgv_gradients_integration_autograd.mat",
        {
            "prediction": predictions_field,
            "ground_true": true_pressure_field,
        },
    )


def test_tgv_gradients_integration_finite_diff():
    """
    Integration test for SIREN model with Taylor-Green vortex data.
    Runs optimization for 2000 epochs and compares the predicted field
    to the true TGV pressure field.
    """
    training_data = create_tgv_training_data(
        grid_size=100, t=0.5, rho=1.0, nu=0.01, include_laplacian=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DerivativesDataset(training_data=training_data, device=device)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
    )

    model = SIREN(
        hidden_features=64,
        hidden_layers=3,
        first_omega=5.0,
        hidden_omega=10.0,
    ).to(device)

    derivatives_strategy = FiniteDifferenceDerivativesStrategy()
    finite_diff_config = FiniteDifferenceConfig(model=model, delta=1e-5)

    loss_fn = FitGradients(derivatives_strategy, finite_diff_config=finite_diff_config)

    optimizer = Adam(model.parameters(), lr=3e-5)

    tracker = MockTracker()
    config = {
        "device": device,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    runner = Runner(dataloader, model, config, TrainingMetrics())

    num_epochs = 2000

    for epoch in range(num_epochs):
        epoch_loss, epoch_psnr, epoch_mae = run_epoch(runner, tracker)

    assert epoch_loss < 1e-4, f"Epoch loss {epoch_loss} is higher than expected"
    assert epoch_psnr > 40, f"Epoch PSNR {epoch_psnr} is lower than expected"
    assert epoch_mae < 1e-2, f"Epoch MAE {epoch_mae} is higher than expected"

    # Check if the model predictions are close enough to the TGV pressure field
    model.eval()
    with torch.no_grad():
        predictions = model(dataset[0]["coords"].to(device)).cpu().numpy()

    predictions_field = predictions.reshape(
        100, 100
    )  # Reshape predictions to match grid

    # Compute the true pressure field
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    x, y = np.meshgrid(x, y)
    true_pressure_field = pressure_field(x, y, t=0.5, rho=1.0, nu=0.01)

    # Normalize by subtracting the mean to account for integration constant
    predictions_field -= predictions_field.mean()
    true_pressure_field -= true_pressure_field.mean()

    # Compare model output to true TGV pressure field (mean squared error tolerance)
    mse = np.mean((predictions_field - true_pressure_field) ** 2)
    assert (
        mse < 1e-4
    ), f"MSE between predicted and true pressure field is too high: {mse}"

    scipy.io.savemat(
        "test_tgv_gradients_integration_finite_diff.mat",
        {
            "prediction": predictions_field,
            "ground_true": true_pressure_field,
        },
    )


def test_tgv_laplacian_integration_autograd():
    """
    Integration test for SIREN model with Taylor-Green vortex data.
    Runs optimization for 2000 epochs and compares the predicted field
    to the true TGV pressure field.
    """
    training_data = create_tgv_training_data(
        grid_size=100, t=0.5, rho=1.0, nu=0.01, include_laplacian=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DerivativesDataset(training_data=training_data, device=device)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
    )

    model = SIREN(
        hidden_features=32,
        hidden_layers=5,
        first_omega=1.0,
        hidden_omega=30.0,
    ).to(device)

    derivatives_strategy = AutogradDerivativesStrategy()

    loss_fn = FitLaplacian(derivatives_strategy, finite_diff_config=None)

    optimizer = Adam(model.parameters(), lr=3e-5)

    tracker = MockTracker()
    config = {
        "device": device,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    runner = Runner(dataloader, model, config, TrainingMetrics())

    num_epochs = 2000

    for epoch in range(num_epochs):
        epoch_loss, epoch_psnr, epoch_mae = run_epoch(runner, tracker)

    assert epoch_loss < 1e-4, f"Epoch loss {epoch_loss} is higher than expected"
    assert epoch_psnr > 45, f"Epoch PSNR {epoch_psnr} is lower than expected"
    assert epoch_mae < 1e-2, f"Epoch MAE {epoch_mae} is higher than expected"

    # Check if the model predictions are close enough to the TGV pressure field
    model.eval()
    with torch.no_grad():
        predictions = model(dataset[0]["coords"].to(device)).cpu().numpy()

    predictions_field = predictions.reshape(
        100, 100
    )  # Reshape predictions to match grid

    # Compute the true pressure field
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    x, y = np.meshgrid(x, y)
    true_pressure_field = pressure_field(x, y, t=0.5, rho=1.0, nu=0.01)

    # Normalize by subtracting the mean to account for integration constant
    predictions_field -= predictions_field.mean()
    true_pressure_field -= true_pressure_field.mean()

    # Compare model output to true TGV pressure field (mean squared error tolerance)
    # NOTICE: the tolerance is high because the laplacian solution has some
    # artifacts along domain boundaries
    mse = np.mean((predictions_field - true_pressure_field) ** 2)
    assert (
        mse < 1e-2
    ), f"MSE between predicted and true pressure field is too high: {mse}"

    scipy.io.savemat(
        "test_tgv_laplacian_integration_autograd.mat",
        {
            "prediction": predictions_field,
            "ground_true": true_pressure_field,
        },
    )


def test_tgv_laplacian_integration_finite_diff():
    """
    Integration test for SIREN model with Taylor-Green vortex data.
    Runs optimization for 2000 epochs and compares the predicted field
    to the true TGV pressure field.
    """
    training_data = create_tgv_training_data(
        grid_size=100, t=0.5, rho=1.0, nu=0.01, include_laplacian=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DerivativesDataset(training_data=training_data, device=device)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
    )

    model = SIREN(
        hidden_features=32,
        hidden_layers=5,
        first_omega=1.0,
        hidden_omega=30.0,
    ).to(device)

    derivatives_strategy = FiniteDifferenceDerivativesStrategy()
    finite_diff_config = FiniteDifferenceConfig(model=model, delta=1e-2)

    loss_fn = FitLaplacian(derivatives_strategy, finite_diff_config=finite_diff_config)

    optimizer = Adam(model.parameters(), lr=3e-5)

    tracker = MockTracker()
    config = {
        "device": device,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    runner = Runner(dataloader, model, config, TrainingMetrics())

    num_epochs = 2000

    for epoch in range(num_epochs):
        epoch_loss, epoch_psnr, epoch_mae = run_epoch(runner, tracker)

    assert epoch_loss < 1e-4, f"Epoch loss {epoch_loss} is higher than expected"
    assert epoch_psnr > 45, f"Epoch PSNR {epoch_psnr} is lower than expected"
    assert epoch_mae < 1e-2, f"Epoch MAE {epoch_mae} is higher than expected"

    # Check if the model predictions are close enough to the TGV pressure field
    model.eval()
    with torch.no_grad():
        predictions = model(dataset[0]["coords"].to(device)).cpu().numpy()

    predictions_field = predictions.reshape(
        100, 100
    )  # Reshape predictions to match grid

    # Compute the true pressure field
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    x, y = np.meshgrid(x, y)
    true_pressure_field = pressure_field(x, y, t=0.5, rho=1.0, nu=0.01)

    # Normalize by subtracting the mean to account for integration constant
    predictions_field -= predictions_field.mean()
    true_pressure_field -= true_pressure_field.mean()

    # Compare model output to true TGV pressure field (mean squared error tolerance)
    # NOTICE: the tolerance is high because the laplacian solution has some
    # artifacts along domain boundaries
    mse = np.mean((predictions_field - true_pressure_field) ** 2)
    assert (
        mse < 1e-2
    ), f"MSE between predicted and true pressure field is too high: {mse}"

    scipy.io.savemat(
        "test_tgv_laplacian_integration_finite_diff.mat",
        {
            "prediction": predictions_field,
            "ground_true": true_pressure_field,
        },
    )
