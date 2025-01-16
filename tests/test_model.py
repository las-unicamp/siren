import unittest

import torch
from torch import nn

from src.model import SIREN, SineLayer


class TestSineLayer(unittest.TestCase):
    def test_forward_pass(self):
        """Test the forward pass of SineLayer."""
        layer = SineLayer(in_features=2, out_features=5, omega=30)
        x = torch.randn(10, 2)  # 10 samples, 2 features
        output = layer(x)

        self.assertEqual(output.shape, (10, 5), "Output shape mismatch.")
        self.assertTrue(
            torch.all(output <= 1) and torch.all(output >= -1),
            "Output values not in range [-1, 1].",
        )

    def test_custom_initialization(self):
        """Test custom weight initialization."""

        def custom_init(weight):
            nn.init.constant_(weight, 0.5)

        layer = SineLayer(
            in_features=2, out_features=5, custom_init_function=custom_init
        )
        self.assertTrue(
            torch.all(layer.linear.weight == 0.5), "Custom initialization failed."
        )


class TestSIREN(unittest.TestCase):
    def setUp(self):
        """Set up a default SIREN network for testing."""
        self.siren = SIREN(
            hidden_features=16,
            hidden_layers=3,
            first_omega=30,
            hidden_omega=30,
        )

    def test_forward_pass(self):
        """Test the forward pass of SIREN."""
        x = torch.randn(20, 2)  # 20 samples, 2 features (2D coordinates)
        output = self.siren(x)

        self.assertEqual(output.shape, (20, 1), "Output shape mismatch.")

    def test_initialization(self):
        """Test that custom initialization propagates correctly."""

        def custom_init(weight):
            nn.init.constant_(weight, 1.0)

        siren = SIREN(
            hidden_features=16,
            hidden_layers=3,
            first_omega=30,
            hidden_omega=30,
            custom_init_function=custom_init,
        )

        for layer in siren.net:
            if isinstance(layer, nn.Linear):
                self.assertTrue(
                    torch.all(layer.weight == 1.0), "Custom initialization failed."
                )

    def test_hidden_layer_count(self):
        """Test the correct number of hidden layers."""
        hidden_layers = 3
        siren = SIREN(
            hidden_features=16,
            hidden_layers=hidden_layers,
            first_omega=30,
            hidden_omega=30,
        )

        sine_layers = [layer for layer in siren.net if isinstance(layer, SineLayer)]
        self.assertEqual(
            len(sine_layers), hidden_layers + 1, "Incorrect number of SineLayers."
        )

    def test_final_linear_layer(self):
        """Test the final layer of SIREN."""
        final_layer = self.siren.net[-1]
        self.assertIsInstance(
            final_layer, nn.Linear, "Final layer is not a Linear layer."
        )
        self.assertEqual(
            final_layer.out_features, 1, "Final Linear layer output features mismatch."
        )


if __name__ == "__main__":
    unittest.main()
