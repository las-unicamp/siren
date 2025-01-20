import unittest
from unittest.mock import MagicMock

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.running import (
    Runner,
    StandardPrecisionStrategy,
    TrainingConfig,
    TrainingMetrics,
    run_epoch,
)
from src.tracking import NetworkTracker


class TestRunner(unittest.TestCase):
    def setUp(self):
        # Mocking DataLoader and NetworkTracker
        self.mock_loader = MagicMock(spec=DataLoader)
        self.mock_tracker = MagicMock(spec=NetworkTracker)

        # Create mock dataset and data items
        self.mock_data = {
            "coords": torch.rand(10, 2).requires_grad_(True),
            "derivatives": torch.rand(10, 2),  # fit gradients
            "mask": torch.ones(10),
        }
        self.mock_loader.__len__.return_value = 1  # single batch
        self.mock_loader.__iter__.return_value = [self.mock_data]

        # Model and optimizer setup
        self.model = torch.nn.Linear(2, 1)  # Simple model for testing
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        # Mock training configuration
        config = TrainingConfig(
            fit_option="gradients",
            optimizer=self.optimizer,
            device=torch.device("cpu"),
            strategy=StandardPrecisionStrategy(),  # You can choose any strategy
        )

        # Metrics setup
        metrics = TrainingMetrics()

        # Initialize the runner
        self.runner = Runner(self.mock_loader, self.model, config, metrics)

    def test_run_epoch(self):
        # Simulate running an epoch and check the results
        epoch_loss, epoch_psnr, epoch_mae = run_epoch(self.runner, self.mock_tracker)

        # Check if the correct epoch metrics were added
        self.mock_tracker.add_epoch_metric.assert_any_call(
            "loss", epoch_loss, self.runner.epoch
        )
        self.mock_tracker.add_epoch_metric.assert_any_call(
            "psnr", epoch_psnr, self.runner.epoch
        )
        self.mock_tracker.add_epoch_metric.assert_any_call(
            "mae", epoch_mae, self.runner.epoch
        )

    def test_batch_data_tracking(self):
        # Save the initial model state
        initial_state_dict = self.runner.model.state_dict()

        # Run a single batch to test if batch data is tracked
        self.runner.run(self.mock_tracker)

        # Restore the initial model state
        self.runner.model.load_state_dict(initial_state_dict)

        # Check if the batch data tracking functions are called correctly
        # Mock will run a shallow equality check for the coordinates argument, so this
        # approach will work: Mock will certify that the method was called with the
        # correct argument.
        self.mock_tracker.add_batch_data.assert_any_call(
            "coordinates", self.mock_data["coords"]
        )

        # Generate the actual predictions from the model
        predictions_tensor = self.runner.strategy.forward_batch(
            self.runner.model, self.mock_data["coords"]
        )

        # When dealing with the predictions, predictions_tensor is computed after the
        # run method executes, so it is NOT the same object in memory. Hence, the
        # approach we used for the coordinates won't work... Instead, we verify if
        # the method was called, and then we verify if the values for prediction
        # are close enough.. so it is gonna be 2 checks in this case

        # Extract the "predictions" call arguments
        for call in self.mock_tracker.add_batch_data.call_args_list:
            if call[0][0] == "predictions":  # Check the first argument ("predictions")
                tracked_predictions = call[0][1]  # Second argument (tensor)
                break
        else:
            self.fail("add_batch_data was not called with 'predictions'.")

        print(f"Tracked Predictions: {tracked_predictions}")
        print(f"Generated Predictions: {predictions_tensor}")
        diff = tracked_predictions - predictions_tensor
        print(f"Difference: {diff}")

        # Compare the predicted tensors explicitly
        self.assertTrue(
            torch.allclose(tracked_predictions, predictions_tensor, atol=1e-2),
            f"Expected predictions tensor: {predictions_tensor}, "
            f"but got: {tracked_predictions}",
        )

    def test_loss_computation(self):
        # Run the runner for one batch and check the loss value
        results = self.runner.run(self.mock_tracker)

        # Ensure the loss is computed (we can't check the exact value here,
        # but we can ensure it's a float)
        self.assertIsInstance(results["epoch_loss"], float)

    def test_metrics_computation(self):
        # Run the runner and verify if PSNR and MAE metrics are computed
        results = self.runner.run(self.mock_tracker)

        # Check if PSNR and MAE are PyTorch tensors
        self.assertIsInstance(
            results["epoch_psnr"], torch.Tensor, "PSNR should be a tensor"
        )
        self.assertIsInstance(
            results["epoch_mae"], torch.Tensor, "MAE should be a tensor"
        )

        # Validate that the tensors are scalar tensors of float type
        self.assertEqual(
            results["epoch_psnr"].ndim, 0, "PSNR should be a scalar tensor"
        )
        self.assertEqual(results["epoch_mae"].ndim, 0, "MAE should be a scalar tensor")
        self.assertEqual(
            results["epoch_psnr"].dtype,
            torch.float,
            "PSNR should have dtype torch.float",
        )
        self.assertEqual(
            results["epoch_mae"].dtype, torch.float, "MAE should have dtype torch.float"
        )

    def test_invalid_strategy(self):
        # Try to initialize with an invalid strategy
        invalid_config = TrainingConfig(
            fit_option="gradients",
            optimizer=self.optimizer,
            device=torch.device("cpu"),
            strategy=None,  # Invalid strategy
        )
        with self.assertRaises(TypeError):
            Runner(self.mock_loader, self.model, invalid_config, TrainingMetrics())


if __name__ == "__main__":
    unittest.main()
