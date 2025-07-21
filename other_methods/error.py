import numpy as np
from numpy.typing import NDArray


def zero_mean(array: NDArray) -> NDArray:
    return array - np.mean(array)


def standardize(array: NDArray) -> NDArray:
    return (array - array.mean()) / array.std()


def relative_mean_squared_error(prediction: NDArray, ground_truth: NDArray) -> float:
    magnitude_gt = abs(ground_truth.max() - ground_truth.min())
    return np.mean(np.subtract(ground_truth, prediction) ** 2) / magnitude_gt.max()


def relative_mean_absolute_error(prediction: NDArray, ground_truth: NDArray) -> float:
    magnitude_gt = abs(ground_truth.max() - ground_truth.min())
    return (
        np.absolute(np.subtract(ground_truth, prediction)).mean() / magnitude_gt.max()
    )


def relative_difference(prediction: NDArray, ground_truth: NDArray) -> NDArray:
    magnitude_gt = abs(ground_truth.max() - ground_truth.min())
    return (abs(ground_truth - prediction)) / magnitude_gt.max()


def relative_residual_norm(prediction: NDArray, ground_truth: NDArray) -> float:
    return np.linalg.norm(ground_truth - prediction) / np.linalg.norm(ground_truth)


def manhattan_norm(prediction: NDArray, ground_truth: NDArray) -> float:
    """
    Manhattan norm (the sum of the absolute values) is a measure of how much
    the image is off. Here, we give the result divided by the number of pixels.
    Also, the image of the difference is normalized between [0-1], so that the
    final output can be interpreted as the percentage of deviation of the images
    """
    difference = abs(ground_truth - prediction)
    rescaled_difference = difference / 255.0
    return np.sum(rescaled_difference) / difference.size


def zero_norm(prediction: NDArray, ground_truth: NDArray):
    """
    Zero norm (the number of elements not equal to zero) tell how many pixels
    differ. Here, we give the result divided by the number of pixels. This means
    that this metric shows the percentage of pixels of the image that differ.
    """
    difference = abs(ground_truth - prediction)
    return np.linalg.norm(difference.ravel(), ord=0) / difference.size
