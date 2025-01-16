from nptyping import Bool, Float32, NDArray, Shape, UInt8
from torchtyping import TensorType

ArrayUintNx2 = NDArray[Shape["*, 2"], UInt8]
ArrayUInt8NxN = NDArray[Shape["*, *"], UInt8]
ArrayFloat32Nx2 = NDArray[Shape["*, 2"], Float32]
ArrayFloat32NxN = NDArray[Shape["*, *"], Float32]
ArrayFloat32NxNx2 = NDArray[Shape["*, *, 2"], Float32]
ArrayBoolNxN = NDArray[Shape["*, *"], Bool]


TensorScalar = TensorType[float]
TensorBoolN = TensorType["batch":1, "pixels":-1, bool]
TensorFloatN = TensorType["batch":1, "pixels":-1, float]
TensorFloatNx1 = TensorType["batch":1, "pixels":-1, "channels":1, float]
TensorFloatNx2 = TensorType["batch":1, "pixels":-1, "channels":2, float]
TensorFloatNxN = TensorType["batch":1, "height":-1, "width":-1, float]
