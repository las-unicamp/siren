import shutil
import unittest
from pathlib import Path

import torch

from src.tensorboard_tracker import TensorboardTracker


class TestTensorboardTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_log_dir = "test_logs"
        self.tracker = TensorboardTracker(log_dir=self.test_log_dir)

    def tearDown(self):
        # Cleanup the temporary directory after tests
        if Path(self.test_log_dir).exists():
            shutil.rmtree(self.test_log_dir)

    def test_log_dir_creation(self):
        # Ensure the log directory is created
        self.assertTrue(Path(self.test_log_dir).exists())
        self.assertTrue(Path(self.tracker.directory).exists())

    def test_add_batch_data(self):
        coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        data = torch.tensor([0.9, 0.8])

        self.tracker.add_batch_data("coordinates", coords)
        self.tracker.add_batch_data("data", data)

        # Verify the batch data is updated
        torch.testing.assert_close(self.tracker.get_batch_data()["coordinates"], coords)
        torch.testing.assert_close(self.tracker.get_batch_data()["data"], data)

    def test_batch_data_flush(self):
        coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        data = torch.tensor([0.9, 0.8])

        self.tracker.add_batch_data("coordinates", coords)
        self.tracker.add_batch_data("data", data)
        self.tracker.flush()

        # Verify the batch data is reset
        self.assertEqual(len(self.tracker.get_batch_data()), 0)

    def test_add_batch_metric(self):
        # Add a batch metric and ensure no errors are raised
        self.tracker.add_batch_metric(name="accuracy", value=0.85, step=1)
        self.tracker.flush()  # Flush to finalize any logs

    def test_add_epoch_metric(self):
        # Add an epoch metric and ensure no errors are raised
        self.tracker.add_epoch_metric(name="loss", value=0.15, step=1)
        self.tracker.flush()  # Flush to finalize any logs

    def test_save_epoch_data(self):
        coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        prediction = torch.tensor([0.9, 0.8])
        step = 1
        filename = f"predictions_{step}.mat"

        # Add batch data
        self.tracker.add_batch_data("coordinates", coords)
        self.tracker.add_batch_data("data", prediction)

        # Save epoch data (no need to pass batch_data explicitly)
        self.tracker.save_epoch_data(name="predictions", step=step)

        # Verify the .mat file is created
        saved_file_path = Path(self.tracker.directory) / filename
        self.assertTrue(saved_file_path.exists())

    def test_invalid_log_dir(self):
        with self.assertRaises(NotADirectoryError):
            TensorboardTracker(log_dir="invalid_dir", create=False)


if __name__ == "__main__":
    unittest.main()
