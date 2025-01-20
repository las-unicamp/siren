import unittest

import numpy as np
import torch

from src.datasets import (
    DerivativesDataset,
    DerivativesDatasetBatches,
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

        # 2D data
        self.coordinates_2d = np.random.rand(100, 2).astype(np.float32)
        self.mask = np.random.randint(0, 2, (10, 10)).astype(bool)
        self.laplacian = np.random.rand(10, 10).astype(np.float32)
        self.gradients_2d = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

        # 3D data
        self.coordinates_3d = np.random.rand(100, 3).astype(np.float32)
        self.gradients_3d = [
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
            np.random.rand(10, 10).astype(np.float32),
        ]

    def test_process_coordinates(self):
        """Test processing of 2D coordinates."""
        coords_tensor = process_coordinates(self.coordinates_2d, self.device)
        self.assertIsInstance(coords_tensor, torch.Tensor)
        self.assertEqual(coords_tensor.shape, (100, 2))

    def test_process_coordinates_3d(self):
        """Test processing of 3D coordinates."""
        coords_tensor = process_coordinates(self.coordinates_3d, self.device)
        self.assertIsInstance(coords_tensor, torch.Tensor)
        self.assertEqual(coords_tensor.shape, (100, 3))

    def test_process_mask(self):
        """Test processing of mask (boolean type)."""
        mask_tensor = process_mask(self.mask)
        self.assertIsInstance(mask_tensor, torch.Tensor)
        self.assertEqual(mask_tensor.shape, (100,))
        self.assertTrue(mask_tensor.dtype == torch.bool)

    def test_process_mask_uint(self):
        """Test processing of mask (uint8 type), should convert to boolean."""
        mask_uint = np.random.randint(0, 2, (10, 10)).astype(np.uint8)
        mask_tensor = process_mask(mask_uint)
        self.assertIsInstance(mask_tensor, torch.Tensor)
        self.assertEqual(mask_tensor.shape, (100,))
        self.assertTrue(mask_tensor.dtype == torch.bool)
        # Ensure it only contains 0s and 1s:
        self.assertTrue((mask_tensor == mask_tensor.int()).all())

    def test_process_laplacian(self):
        """Test processing of Laplacian."""
        laplacian_tensor = process_laplacian(self.laplacian, self.device)
        self.assertIsInstance(laplacian_tensor, torch.Tensor)
        self.assertEqual(laplacian_tensor.shape, (100, 1))

    def test_process_gradients(self):
        """Test processing of 2D gradients."""
        gradients_tensor = process_gradients(self.gradients_2d, self.device)
        self.assertIsInstance(gradients_tensor, torch.Tensor)
        self.assertEqual(gradients_tensor.shape, (100, 2))

    def test_process_gradients_3d(self):
        """Test processing of 3D gradients."""
        gradients_tensor = process_gradients(self.gradients_3d, self.device)
        self.assertIsInstance(gradients_tensor, torch.Tensor)
        self.assertEqual(gradients_tensor.shape, (100, 3))

    def test_has_laplacian(self):
        """Test detection of Laplacian in training data."""
        training_data = {"laplacian": self.laplacian}
        self.assertTrue(has_laplacian(training_data))
        self.assertFalse(has_laplacian({}))

    def test_has_gradients(self):
        """Test detection of gradients in training data."""
        training_data = {
            "gradient_x": self.gradients_2d[0],
            "gradient_y": self.gradients_2d[1],
        }
        self.assertTrue(has_gradients(training_data))
        self.assertFalse(has_gradients({}))


class TestDerivativesDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data for the datasets."""
        self.device = torch.device("cpu")
        # Coordinates for testing
        self.coordinates_2d = np.random.rand(100, 2).astype(np.float32)
        self.coordinates_3d = np.random.rand(100, 3).astype(np.float32)

        # Masks and Laplacians for testing
        self.mask = np.zeros((100,)).astype(bool)
        self.laplacian = np.random.rand(100, 1).astype(np.float32)

        # Gradients for 2D and 3D
        self.gradients_2d = [
            np.random.rand(100, 1).astype(np.float32),  # gradient_x
            np.random.rand(100, 1).astype(np.float32),  # gradient_y
        ]
        self.gradients_3d = [
            np.random.rand(100, 1).astype(np.float32),  # gradient_x
            np.random.rand(100, 1).astype(np.float32),  # gradient_y
            np.random.rand(100, 1).astype(np.float32),  # gradient_z
        ]
        self.invalid_gradients_3d = [
            np.random.rand(100, 1).astype(np.float32),  # gradient_x
            np.random.rand(100, 1).astype(np.float32),  # gradient_y
            None,  # Missing gradient_z for 3D coordinates
        ]
        self.invalid_gradients_2d = [
            np.random.rand(100, 1).astype(np.float32),  # gradient_x
            np.random.rand(100, 1).astype(np.float32),  # gradient_y
            np.random.rand(100, 1).astype(np.float32),  # gradient_z (should not exist)
        ]

    def test_dataset_with_laplacian(self):
        """Test dataset initialization with Laplacian data."""
        training_data = {
            "coordinates": self.coordinates_2d,
            "mask": self.mask,
            "laplacian": self.laplacian,
        }
        dataset = DerivativesDataset(training_data, self.device)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        assert isinstance(sample["coords"], torch.Tensor)
        assert sample["coords"].shape == (100, 2)
        assert isinstance(sample["derivatives"], torch.Tensor)
        assert sample["derivatives"].shape == (100, 1)

    def test_dataset_with_valid_gradients_2d(self):
        """Test dataset initialization with valid 2D gradient data."""
        training_data = {
            "coordinates": self.coordinates_2d,
            "mask": self.mask,
            "gradient_x": self.gradients_2d[0],
            "gradient_y": self.gradients_2d[1],
        }
        dataset = DerivativesDataset(training_data, self.device)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        assert isinstance(sample["coords"], torch.Tensor)
        assert sample["coords"].shape == (100, 2)
        assert isinstance(sample["derivatives"], torch.Tensor)
        assert sample["derivatives"].shape == (100, 2)

    def test_dataset_with_invalid_gradients_2d(self):
        """Test dataset raises error when 3D gradient_z is provided for 2D coords."""
        training_data = {
            "coordinates": self.coordinates_2d,
            "mask": self.mask,
            "gradient_x": self.gradients_2d[0],
            "gradient_y": self.gradients_2d[1],
            "gradient_z": self.gradients_2d[
                0
            ],  # Invalid, 2D coordinates should not have a z-gradient
        }
        with self.assertRaises(ValueError):
            DerivativesDataset(training_data, self.device)

    def test_dataset_with_valid_gradients_3d(self):
        """Test dataset initialization with valid 3D gradient data."""
        training_data = {
            "coordinates": self.coordinates_3d,
            "mask": self.mask,
            "gradient_x": self.gradients_3d[0],
            "gradient_y": self.gradients_3d[1],
            "gradient_z": self.gradients_3d[2],
        }
        dataset = DerivativesDataset(training_data, self.device)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        assert isinstance(sample["coords"], torch.Tensor)
        assert sample["coords"].shape == (100, 3)
        assert isinstance(sample["derivatives"], torch.Tensor)
        assert sample["derivatives"].shape == (100, 3)

    def test_dataset_with_invalid_gradients_3d(self):
        """Test dataset raises error when z-gradient is missing for 3D coordinates."""
        training_data = {
            "coordinates": self.coordinates_3d,
            "mask": self.mask,
            "gradient_x": self.gradients_3d[0],
            "gradient_y": self.gradients_3d[1],
            "gradient_z": None,  # Missing z-gradient for 3D coordinates
        }
        with self.assertRaises(ValueError):
            DerivativesDataset(training_data, self.device)

    def test_dataset_batches(self):
        """Test batch-wise access with DerivativesDatasetBatches."""
        training_data = {
            "coordinates": self.coordinates_2d,
            "mask": self.mask,
            "gradient_x": self.gradients_2d[0],
            "gradient_y": self.gradients_2d[1],
        }
        dataset = DerivativesDatasetBatches(training_data, self.device)
        self.assertEqual(len(dataset), 100)
        sample = dataset[0]
        self.assertEqual(sample["coords"].shape, (2,))
        self.assertEqual(sample["derivatives"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
