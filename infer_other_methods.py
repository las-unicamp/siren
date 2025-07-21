import matplotlib.pyplot as plt
import numpy as np
import scipy

from other_methods.my_types import Source
from other_methods.params import args


def parse_source(filename) -> Source:
    source = scipy.io.loadmat(filename)

    for key, val in source.items():
        if key.startswith("_"):
            continue
        source[key] = val.squeeze()

    return source


def zero_mean(array):
    return array - np.mean(array)


def relative_mean_absolute_error(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    magnitude_gt = abs(ground_truth.max() - ground_truth.min())
    return (
        np.absolute(np.subtract(ground_truth, prediction)).mean() / magnitude_gt.max()
    )


def main():
    ground_truth = zero_mean(scipy.io.loadmat(args.input_filename)["gt"])
    vmax = ground_truth.max()
    vmin = ground_truth.min()

    prediction = zero_mean(scipy.io.loadmat(args.output_filename)["prediction"])

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, vmax=vmax, vmin=vmin, cmap="jet")
    plt.title("Ground Truth")
    plt.subplot(1, 2, 2)
    plt.imshow(prediction, vmax=vmax, vmin=vmin, cmap="jet")
    plt.title("Prediction")
    plt.savefig(f"{args.experiment_name}.png")

    mae = relative_mean_absolute_error(prediction, ground_truth)

    # Append MAE value to mae.txt
    with open("mae.txt", "a") as f:
        f.write(f"{mae}\n")


if __name__ == "__main__":
    main()
