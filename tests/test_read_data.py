import os
import unittest

import numpy as np
import scipy.io

from src.read_data import read_data_from_matfile


class TestReadDataFromMatfile(unittest.TestCase):
    def setUp(self):
        """Set up temporary files and data for testing."""
        self.valid_data = {
            "coordinates": np.random.rand(10, 2),
            "mask": np.random.randint(0, 2, size=(10, 10), dtype=bool),
            "laplacian": np.random.rand(10, 10),
        }
        self.valid_filename = "valid_data.mat"
        scipy.io.savemat(self.valid_filename, self.valid_data)

        self.invalid_data = {
            "coordinates": np.random.rand(10, 2),
            # Missing "mask"
        }
        self.invalid_filename = "invalid_data.mat"
        scipy.io.savemat(self.invalid_filename, self.invalid_data)

    def tearDown(self):
        """Clean up temporary files after testing."""
        if os.path.exists(self.valid_filename):
            os.remove(self.valid_filename)
        if os.path.exists(self.invalid_filename):
            os.remove(self.invalid_filename)

    def test_read_valid_file(self):
        """Test reading a valid .mat file."""
        data = read_data_from_matfile(self.valid_filename)
        self.assertIsInstance(data, dict, "Output is not a dictionary.")
        self.assertIn("coordinates", data, "Missing 'coordinates' in output.")
        self.assertIn("mask", data, "Missing 'mask' in output.")
        self.assertIn("laplacian", data, "Missing 'laplacian' in output.")

    def test_missing_file(self):
        """Test behavior when the file is missing."""
        with self.assertRaises(FileNotFoundError):
            read_data_from_matfile("nonexistent.mat")

    def test_missing_required_keys(self):
        """Test behavior when required keys are missing."""
        with self.assertRaises(KeyError):
            read_data_from_matfile(self.invalid_filename)

    def test_missing_derivative_keys(self):
        """Test behavior when neither 'laplacian' nor gradients are present."""
        data = {
            "coordinates": np.random.rand(10, 2),
            "mask": np.random.randint(0, 2, size=(10, 10), dtype=bool),
        }
        filename = "no_derivatives.mat"
        scipy.io.savemat(filename, data)

        try:
            with self.assertRaises(ValueError):
                read_data_from_matfile(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_gradient_instead_of_laplacian(self):
        """Test reading a file with gradients instead of a laplacian."""
        data = {
            "coordinates": np.random.rand(10, 2),
            "mask": np.random.randint(0, 2, size=(10, 10), dtype=bool),
            "gradient_x": np.random.rand(10, 10),
            "gradient_y": np.random.rand(10, 10),
        }
        filename = "gradient_data.mat"
        scipy.io.savemat(filename, data)

        try:
            training_data = read_data_from_matfile(filename)
            self.assertIn(
                "gradient_x", training_data, "Missing 'gradient_x' in output."
            )
            self.assertIn(
                "gradient_y", training_data, "Missing 'gradient_y' in output."
            )
            self.assertNotIn(
                "laplacian", training_data, "Unexpected 'laplacian' in output."
            )
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    unittest.main()
