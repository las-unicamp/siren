from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer

from src.checkpoint import load_checkpoint, save_checkpoint


# Mock model fixture
@pytest.fixture
def mock_model() -> Module:
    """Fixture to mock a model."""
    model = MagicMock(spec=Module)  # Mock nn.Module class

    # Now explicitly mock `state_dict` and `parameters` as methods.
    model.state_dict.return_value = {"param1": torch.tensor([1.0, 2.0])}
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

    return model


# Mock optimizer fixture
@pytest.fixture
def mock_optimizer(mock_model: Module) -> Optimizer:
    """Fixture to mock an optimizer."""
    optimizer = Adam(mock_model.parameters(), lr=0.001)

    # Mock the `state_dict` method properly
    mock_state_dict = MagicMock()
    mock_state_dict.return_value = {
        "param_groups": [
            {
                "params": [torch.tensor([1.0, 2.0])],  # Mock the params in the group
                "lr": 0.001,
            }
        ],
        "state": {  # Mocking the state dictionary per parameter
            torch.tensor([1.0, 2.0]): {
                "exp_avg": torch.tensor([0.1, 0.2]),  # Mock momentum
                "exp_avg_sq": torch.tensor([0.01, 0.02]),  # Mock squared gradient
            }
        },
    }

    # Assign the mocked `state_dict` method to the optimizer
    optimizer.state_dict = mock_state_dict

    return optimizer


# Checkpoint file fixture
@pytest.fixture
def checkpoint_filename() -> Generator[str, None, None]:
    """Fixture to return a temporary checkpoint filename with teardown."""
    filename = "test_checkpoint.pth.tar"

    yield filename  # Provide the filename to the test

    # Teardown: Delete the file after the test
    checkpoint_path = Path(filename)
    if checkpoint_path.exists():
        checkpoint_path.unlink()  # Remove the file


# Test saving checkpoint
def test_save_checkpoint(
    mock_model: Module, mock_optimizer: Optimizer, checkpoint_filename: str
) -> None:
    """Test saving a checkpoint."""
    epoch = 10
    prev_lr = 1e-4
    loss = 0.25

    save_checkpoint(
        mock_model,
        mock_optimizer,
        epoch,
        prev_lr,
        loss,
        filename=checkpoint_filename,
    )

    # Check that the file was created
    assert Path(checkpoint_filename).exists(), "Checkpoint file should be saved."

    # Check that saved state contains expected fields
    checkpoint = torch.load(checkpoint_filename, weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["epoch"] == epoch
    assert checkpoint["prev_lr"] == prev_lr
    assert checkpoint["loss"] == loss


# Test loading checkpoint
def test_load_checkpoint(
    mock_model: Module, mock_optimizer: Optimizer, checkpoint_filename: str
) -> None:
    """Test loading a checkpoint."""
    epoch = 10
    prev_lr = 1e-4
    loss = 0.25

    # Save a checkpoint first
    save_checkpoint(
        mock_model,
        mock_optimizer,
        epoch,
        prev_lr,
        loss,
        filename=checkpoint_filename,
    )

    # Now test loading the checkpoint
    loaded_epoch, loaded_prev_lr, loaded_loss = load_checkpoint(
        device="cpu",
        model=mock_model,
        filename=checkpoint_filename,
        optimizer=mock_optimizer,
    )

    # Verify the loaded values match what was saved
    assert loaded_epoch == epoch
    assert loaded_prev_lr == prev_lr
    assert loaded_loss == loss


# Test loading checkpoint with default values
def test_load_checkpoint_with_default_values(
    mock_model: Module, mock_optimizer: Optimizer, checkpoint_filename: str
) -> None:
    """Test loading a checkpoint with missing default values."""
    epoch = 10
    prev_lr = 1e-4
    loss = 0.25

    # Save a checkpoint with missing `relative_residual_error`
    save_checkpoint(
        mock_model,
        mock_optimizer,
        epoch,
        prev_lr,
        loss,
        filename=checkpoint_filename,
    )

    # Load checkpoint and check defaults
    loaded_epoch, loaded_prev_lr, loaded_loss = load_checkpoint(
        device="cpu",
        model=mock_model,
        filename=checkpoint_filename,
        optimizer=mock_optimizer,
    )

    assert loaded_epoch == epoch
    assert loaded_prev_lr == prev_lr
    assert loaded_loss == loss
