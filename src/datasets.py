from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dtos import DatasetReturnItems, TrainingData
from src.my_types import (
    ArrayBoolNxN,
    ArrayFloat32Nx2,
    ArrayFloat32Nx3,
    ArrayFloat32NxN,
    TensorBoolN,
    TensorFloatNx1,
    TensorFloatNx2,
    TensorFloatNx3,
)


class DerivativesDataset(Dataset):
    """Dataset yielding coordinates, derivatives and the (integrated) image.

    Parameters
    ----------
    training_data : TrainingData
        Dictionary containing all the data used for training

    device : torch.device
        Inform the device where training will occur (`cuda` or `cpu`).


    Attributes
    ----------
    coordinates : TensorFloatNx2 or TensorFloatNx3
        Coordinates of the dataset.

    derivatives : TensorFloatNx1, TensorFloatNx2 or TensorFloatNx3
        Laplacian or gradient derivatives.

    mask : TensorBoolN
        Boolean mask used to filter valid data.

    len : int
        Number of valid coordinates after applying the mask.
    """

    def __init__(
        self,
        training_data: TrainingData,
        device: torch.device,
    ):
        if has_laplacian(training_data):
            self.derivatives = process_laplacian(
                laplacian=training_data["laplacian"],
                device=device,
            )

        elif has_gradients(training_data):
            gradients = [
                training_data["gradient_x"],
                training_data["gradient_y"],
                training_data.get("gradient_z", None),  # Optional z-gradient
            ]

            # Check that the number of gradients matches the coordinates' dimensionality
            coordinates = training_data["coordinates"]
            if coordinates.shape[1] == 3:  # 3D coordinates
                if gradients[2] is None:
                    raise ValueError(
                        "For 3D coordinates, the z-gradient ('gradient_z') is required."
                    )
            elif coordinates.shape[1] == 2:  # 2D coordinates
                if gradients[2] is not None:
                    raise ValueError(
                        "For 2D coordinates, the z-gradient ('gradient_z') "
                        "should not be provided."
                    )

            self.derivatives = process_gradients(gradients=gradients, device=device)
        else:
            raise ReferenceError(
                "Derivative data is missing. "
                "Ensure that the dataset includes either 'laplacian' or both"
                "'gradient_x' and 'gradient_y'."
            )

        # Process coordinates
        self.coordinates = process_coordinates(
            training_data["coordinates"], device=device
        )

        # Verify the shape of coordinates (2D or 3D)
        if self.coordinates.size(1) not in [2, 3]:
            raise ValueError(
                f"Coordinates must have shape (N, 2) or (N, 3), "
                f"but got {self.coordinates.shape}."
            )

        # Process mask
        self.mask = process_mask(training_data["mask"])

        # Apply the mask to coordinates and derivatives
        valid_indices = ~self.mask
        self.coordinates = self.coordinates[valid_indices]
        self.derivatives = self.derivatives[valid_indices]

        # Set dataset length
        self.len = len(self.coordinates)

    def __len__(self):
        """Returns the dataset length (all data at once)."""
        return 1

    def __getitem__(self, idx: int) -> DatasetReturnItems:
        """Get all relevant data for a single coordinate."""

        return DatasetReturnItems(
            coords=self.coordinates.requires_grad_(True),
            derivatives=self.derivatives,
        )


class DerivativesDatasetBatches(DerivativesDataset):
    """Dataset extending the parent class to allow batch training."""

    def __len__(self):
        """Returns the number of valid coordinates."""
        return self.len

    def __getitem__(self, idx: int) -> DatasetReturnItems:
        """Get all relevant data for a single coordinate."""

        return DatasetReturnItems(
            coords=self.coordinates[idx].requires_grad_(True),
            derivatives=self.derivatives[idx],
        )


def process_coordinates(
    coordinates: ArrayFloat32Nx2 | ArrayFloat32Nx3,
    device: torch.device,
) -> TensorFloatNx2 | TensorFloatNx3:
    """Processes and converts coordinates to a PyTorch tensor."""
    return torch.from_numpy(coordinates).to(device=device, dtype=torch.float)


def process_mask(mask: ArrayBoolNxN) -> TensorBoolN:
    """Flattens and converts the mask to a PyTorch tensor."""
    if mask.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        mask = mask.astype(bool)

    mask = mask.ravel()
    return torch.from_numpy(mask)


def process_laplacian(
    laplacian: ArrayFloat32NxN,
    device: torch.device,
) -> TensorFloatNx1:
    """Processes the Laplacian into a PyTorch tensor."""
    laplacian = (
        torch.from_numpy(laplacian).to(device=device, dtype=torch.float).reshape(-1, 1)
    )
    return laplacian


def process_gradients(
    gradients: List[ArrayFloat32NxN],
    device: torch.device,
) -> TensorFloatNx2 | TensorFloatNx3:
    """Processes gradients into a PyTorch tensor, considering missing gradients."""

    # Filter out None values from the gradients list
    valid_gradients = [grad for grad in gradients if grad is not None]

    # Ensure that we have either 2 or 3 gradients
    if len(valid_gradients) not in [2, 3]:
        raise ValueError(f"Expected 2 or 3 gradients, but got {len(valid_gradients)}.")

    # Stack the valid gradients along the last dimension
    grads = np.stack(valid_gradients, axis=-1)

    # Convert to PyTorch tensor and move to the correct device
    grads = torch.from_numpy(grads).to(device=device, dtype=torch.float).contiguous()

    # Return the reshaped tensor based on the number of gradients
    if grads.shape[-1] == 2:
        grads = grads.view(-1, 2)
    elif grads.shape[-1] == 3:
        grads = grads.view(-1, 3)

    return grads


def has_laplacian(data: TrainingData) -> bool:
    """Checks if the dataset includes Laplacian data."""
    return "laplacian" in data


def has_gradients(data: TrainingData) -> bool:
    """Checks if the dataset includes gradient data."""
    return "gradient_x" in data and "gradient_y" in data
