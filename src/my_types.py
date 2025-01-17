from nptyping import Bool, Float32, NDArray, Shape
from torchtyping import TensorType

ArrayFloat32Nx2 = NDArray[Shape["*, 2"], Float32]
ArrayFloat32Nx3 = NDArray[Shape["*, 3"], Float32]
ArrayFloat32NxN = NDArray[Shape["*, *"], Float32]
ArrayBoolNxN = NDArray[Shape["*, *"], Bool]


TensorScalar = TensorType[float]
TensorBoolN = TensorType["batch":1, "pixels":-1, bool]
TensorFloatN = TensorType["batch":1, "pixels":-1, float]
TensorFloatNx1 = TensorType["batch":1, "pixels":-1, "channels":1, float]
TensorFloatNx2 = TensorType["batch":1, "pixels":-1, "channels":2, float]
TensorFloatNx3 = TensorType["batch":1, "pixels":-1, "channels":3, float]
TensorFloatNxN = TensorType["batch":1, "height":-1, "width":-1, float]
