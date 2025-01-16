import pytest

from src.early_stopping import EarlyStopping


@pytest.fixture
def early_stopping():
    return EarlyStopping(patience=3, delta=0.1)


def test_initialization(early_stopping):
    """Test the initialization of the EarlyStopping class."""
    assert early_stopping.patience == 3
    assert early_stopping.delta == 0.1
    assert early_stopping.counter == 0
    assert early_stopping.best_score is None
    assert early_stopping.stop is False


def test_no_early_stopping_when_loss_improving(early_stopping):
    """Test that early stopping is not triggered when validation loss is improving."""
    # Simulate loss values improving
    early_stopping(0.5)  # first epoch
    assert early_stopping.stop is False
    early_stopping(0.4)  # second epoch, improvement
    assert early_stopping.stop is False
    assert early_stopping.counter == 0
    early_stopping(0.3)  # third epoch, further improvement
    assert early_stopping.stop is False
    assert early_stopping.counter == 0


def test_early_stopping_triggered(early_stopping):
    """
    Test that early stopping is triggered after patience epochs without
    improvement.
    """
    early_stopping(0.5)  # first epoch (best score should be -0.5)
    early_stopping(0.6)  # second epoch (no improvement)
    early_stopping(0.7)  # third epoch (no improvement)
    early_stopping(0.8)  # fourth epoch (no improvement)

    # Now check if stop is triggered
    assert early_stopping.stop is True
    assert (
        early_stopping.counter == 3
    )  # It should have stopped after 4 epochs without improvement


def test_early_stopping_with_delta(early_stopping):
    """Test early stopping with a delta value."""

    # First epoch: improvement (best score should be -0.5)
    early_stopping(0.5)  # score is -0.5 (initial best)

    # Second epoch: no improvement, so counter increments
    early_stopping(0.6)  # counter should be 1

    # Third epoch: no improvement, so counter increments
    early_stopping(0.7)  # counter should be 2

    # Fourth epoch: improvement larger than delta (so counter resets)
    early_stopping(0.8)  # counter should reset to 0

    # Fifth epoch: no improvement, counter increments
    early_stopping(0.75)  # counter should be 1

    # Sixth epoch: no improvement, counter increments
    early_stopping(0.74)  # counter should be 2

    # Seventh epoch: no improvement, counter increments
    early_stopping(0.73)  # counter should be 3

    # Check if early stopping is triggered
    assert early_stopping.stop is True  # After 3 epochs without improvement
    assert early_stopping.counter == 3  # Patience is 3, so counter should be 3
