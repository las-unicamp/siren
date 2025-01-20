"""This module stores all data transfer objects (DTOs) used throughout the project"""

from typing import TypedDict

from typing_extensions import NotRequired, Required

from src.my_types import (
    ArrayBoolNxN,
    ArrayFloat32Nx2,
    ArrayFloat32Nx3,
    ArrayFloat32NxN,
    TensorFloatN,
    TensorFloatNx1,
    TensorFloatNx2,
    TensorFloatNx3,
    TensorScalar,
)


class DatasetReturnItems(TypedDict):
    coords: TensorFloatN
    derivatives: TensorFloatNx1 | TensorFloatNx2 | TensorFloatNx3


class TrainingData(TypedDict):
    coordinates: Required[ArrayFloat32Nx2 | ArrayFloat32Nx3]
    laplacian: NotRequired[ArrayFloat32NxN]
    gradient_x: NotRequired[ArrayFloat32NxN]
    gradient_y: NotRequired[ArrayFloat32NxN]
    mask: Required[ArrayBoolNxN]


class RunnerReturnItems(TypedDict):
    epoch_loss: TensorScalar
    epoch_psnr: TensorScalar
    epoch_mae: TensorScalar
