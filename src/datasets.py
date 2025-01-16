from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dtos import DatasetReturnItems, TrainingData
from src.my_types import (
    ArrayBoolNxN,
    ArrayFloat32Nx2,
    ArrayFloat32NxN,
    TensorBoolN,
    TensorFloatNx1,
    TensorFloatNx2,
)


class DerivativesPixelDataset(Dataset):
    """Dataset yielding coordinates, derivatives and the (integrated) image.

    Parameters
    ----------
    training_data : TrainingData
        Dictionary containing all the data used for training

    device : torch.device
        Inform the device where training will occur (`cuda` or `cpu`).


    Attributes
    ----------
    coordinates : TensorFloatNx2
        Normalized coordinates of the dataset.

    derivatives : TensorFloatNx1 or TensorFloatNx2
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
            self.derivatives = process_gradients(
                gradients=[
                    training_data["gradient_x"],
                    training_data["gradient_y"],
                    # training_data["gradient_z"],
                ],
                device=device,
            )
        else:
            raise ReferenceError(
                "Derivative data is missing. "
                "Ensure that the dataset includes either 'laplacian' or both"
                "'gradient_x' and 'gradient_y'."
            )

        # Process coordinates and mask
        self.coordinates = process_coordinates(
            training_data["coordinates"], device=device
        )
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
            mask=self.mask,
        )


class DerivativesPixelDatasetBatches(DerivativesPixelDataset):
    """Dataset extending the parent class to allow batch training."""

    def __len__(self):
        """Returns the number of valid coordinates."""
        return self.len

    def __getitem__(self, idx: int) -> DatasetReturnItems:
        """Get all relevant data for a single coordinate."""

        return DatasetReturnItems(
            coords=self.coordinates[idx].requires_grad_(True),
            derivatives=self.derivatives[idx],
            mask=self.mask[idx],
        )


def process_coordinates(
    coordinates: ArrayFloat32Nx2,
    device: torch.device,
) -> TensorFloatNx2:
    """Processes and converts coordinates to a PyTorch tensor."""
    return torch.from_numpy(coordinates).to(device=device, dtype=torch.float)


def process_mask(mask: ArrayBoolNxN) -> TensorBoolN:
    """Flattens and converts the mask to a PyTorch tensor."""
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
) -> TensorFloatNx2:
    """Processes gradients into a PyTorch tensor."""
    grads = np.stack([gradients[0], gradients[1]], axis=-1)
    # grads = np.stack([gradients[0], gradients[1], gradients[2]], axis=-1)
    grads = (
        torch.from_numpy(grads)
        .to(device=device, dtype=torch.float)
        .contiguous()
        .view(-1, 2)
        # .view(-1, 3)
    )
    return grads


def has_laplacian(data: TrainingData) -> bool:
    """Checks if the dataset includes Laplacian data."""
    return "laplacian" in data


def has_gradients(data: TrainingData) -> bool:
    """Checks if the dataset includes gradient data."""
    return "gradient_x" in data and "gradient_y" in data
