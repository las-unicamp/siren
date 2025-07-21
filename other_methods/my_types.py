from typing import NotRequired, TypedDict

from nptyping import Bool, Float, NDArray, Shape

Array1Dfloat = NDArray[Shape["*"], Float]
Array2Dfloat = NDArray[Shape["*, *"], Float]
Array2Dbool = NDArray[Shape["*, *"], Bool]

Coords2D = NDArray[Shape["*, 2"], Float]
Coords3D = NDArray[Shape["*, 3"], Float]


class SolverReturn(TypedDict):
    pressure: Array1Dfloat | Array2Dfloat
    pressure_coordinates: Array2Dfloat
    # data_coordinates: for unstructured grids, where p is at centroid of cell
    data_coordinates: NotRequired[Array2Dfloat]
    number_boundary_elements: NotRequired[int]


class Source(TypedDict):
    ground_truth: Array1Dfloat | Array2Dfloat
    coordinates: Coords2D
    gradient_x: Array1Dfloat | Array2Dfloat
    gradient_y: Array1Dfloat | Array2Dfloat
    delta: float
    shape: tuple[int, int]


class ProductionData(TypedDict):
    ground_truth: Array1Dfloat | Array2Dfloat
    prediction: Array1Dfloat | Array2Dfloat
    pressure_coordinates: Array2Dfloat
    data_coordinates: NotRequired[Array2Dfloat]
    number_boundary_elements: NotRequired[int]
    relative_residual_norm: float
    relative_mean_squared_error: float
    relative_mean_absolute_error: float
