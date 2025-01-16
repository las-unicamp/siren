import unittest

import numpy as np
import torch

from src.datasets import (
    DerivativesPixelDataset,
    DerivativesPixelDatasetBatches,
    has_gradients,
    has_laplacian,
    process_coordinates,
    process_gradients,
    process_laplacian,
    process_mask,
)


class TestDatasetProcessingFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test data for the utility functions."""
        self.device = torch.device("cpu")
        self.coordinates = np.random.rand(100, 2).astype(np.float32)
        self.mask = np.random.randint(0, 2, (10, 10)).astype(bool)
        self.laplacian = np.random.rand(10, 10).astype(np.float32)
        self.gradients = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

    def test_process_coordinates(self):
        """Test processing of coordinates."""
        coords_tensor = process_coordinates(self.coordinates, self.device)
        self.assertIsInstance(coords_tensor, torch.Tensor)
        self.assertEqual(coords_tensor.shape, (100, 2))

    def test_process_mask(self):
        """Test processing of mask."""
        mask_tensor = process_mask(self.mask)
        self.assertIsInstance(mask_tensor, torch.Tensor)
        self.assertEqual(mask_tensor.shape, (100,))

    def test_process_laplacian(self):
        """Test processing of Laplacian."""
        laplacian_tensor = process_laplacian(self.laplacian, self.device)
        self.assertIsInstance(laplacian_tensor, torch.Tensor)
        self.assertEqual(laplacian_tensor.shape, (100, 1))

    def test_process_gradients(self):
        """Test processing of gradients."""
        gradients_tensor = process_gradients(self.gradients, self.device)
        self.assertIsInstance(gradients_tensor, torch.Tensor)
        self.assertEqual(gradients_tensor.shape, (100, 2))

    def test_has_laplacian(self):
        """Test detection of Laplacian in training data."""
        training_data = {"laplacian": self.laplacian}
        self.assertTrue(has_laplacian(training_data))
        self.assertFalse(has_laplacian({}))

    def test_has_gradients(self):
        """Test detection of gradients in training data."""
        training_data = {
            "gradient_x": self.gradients[0],
            "gradient_y": self.gradients[1],
        }
        self.assertTrue(has_gradients(training_data))
        self.assertFalse(has_gradients({}))


class TestDerivativesPixelDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data for the datasets."""
        self.device = torch.device("cpu")
        self.training_data_laplacian = {
            "coordinates": np.random.rand(100, 2).astype(np.float32),
            "mask": np.zeros((10, 10)).astype(bool),
            "laplacian": np.random.rand(10, 10).astype(np.float32),
        }
        self.training_data_gradients = {
            "coordinates": np.random.rand(100, 2).astype(np.float32),
            "mask": np.zeros((10, 10)).astype(bool),
            "gradient_x": np.random.rand(10, 10).astype(np.float32),
            "gradient_y": np.random.rand(10, 10).astype(np.float32),
        }

    def test_dataset_with_laplacian(self):
        """Test dataset initialization with Laplacian data."""
        dataset = DerivativesPixelDataset(self.training_data_laplacian, self.device)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        assert isinstance(
            sample["coords"], torch.Tensor
        ), "'coords' should be of type torch.Tensor"
        assert sample["coords"].ndimension() == 2, "'coords' should have 2 dimensions"
        assert sample["coords"].shape[0] == 100, "'coords' batch dimension should be 1"

        assert isinstance(
            sample["derivatives"], torch.Tensor
        ), "'derivatives' should be of type torch.Tensor"
        assert (
            sample["derivatives"].ndimension() == 2
        ), "'derivatives' should have 2 dimensions"
        assert (
            sample["derivatives"].shape[0] == 100
        ), "'derivatives' batch dimension should be 100"
        assert sample["derivatives"].shape[1] == 1, "'derivatives' channels should be 1"

        assert isinstance(
            sample["mask"], torch.Tensor
        ), "'mask' should be of type torch.Tensor"
        assert sample["mask"].ndimension() == 1, "'mask' should have 1 dimension"
        assert sample["mask"].shape[0] == 100, "'mask' batch dimension should be 100"
        assert sample["mask"].dtype == torch.bool, "'mask' should be of dtype bool"

        self.assertEqual(sample["coords"].shape, (100, 2))
        self.assertEqual(sample["derivatives"].shape, (100, 1))

    def test_dataset_with_gradients(self):
        """Test dataset initialization with gradient data."""
        dataset = DerivativesPixelDataset(self.training_data_gradients, self.device)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        assert isinstance(
            sample["coords"], torch.Tensor
        ), "'coords' should be of type torch.Tensor"
        assert sample["coords"].ndimension() == 2, "'coords' should have 2 dimensions"
        assert sample["coords"].shape[0] == 100, "'coords' batch dimension should be 1"

        assert isinstance(
            sample["derivatives"], torch.Tensor
        ), "'derivatives' should be of type torch.Tensor"
        assert (
            sample["derivatives"].ndimension() == 2
        ), "'derivatives' should have 2 dimensions"
        assert (
            sample["derivatives"].shape[0] == 100
        ), "'derivatives' batch dimension should be 100"
        assert sample["derivatives"].shape[1] == 2, "'derivatives' channels should be 2"

        assert isinstance(
            sample["mask"], torch.Tensor
        ), "'mask' should be of type torch.Tensor"
        assert sample["mask"].ndimension() == 1, "'mask' should have 1 dimension"
        assert sample["mask"].shape[0] == 100, "'mask' batch dimension should be 100"
        assert sample["mask"].dtype == torch.bool, "'mask' should be of dtype bool"

        self.assertEqual(sample["coords"].shape, (100, 2))
        self.assertEqual(sample["derivatives"].shape, (100, 2))

    def test_dataset_batches(self):
        """Test batch-wise access with DerivativesPixelDatasetBatches."""
        dataset = DerivativesPixelDatasetBatches(
            self.training_data_gradients, self.device
        )
        self.assertEqual(len(dataset), 100)
        sample = dataset[0]
        self.assertEqual(sample["coords"].shape, (2,))
        self.assertEqual(sample["derivatives"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
