import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from src.checkpoint import load_checkpoint
from src.datasets import process_coordinates
from src.hyperparameters import args
from src.model import SIREN
from src.read_data import read_data_from_matfile

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def zero_mean(array):
    try:
        return array - np.mean(array)
    except TypeError:
        return array - torch.mean(array)


def relative_mean_absolute_error(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    magnitude_gt = abs(ground_truth.max() - ground_truth.min())
    return (
        np.absolute(np.subtract(ground_truth, prediction)).mean() / magnitude_gt.max()
    )


def main():
    device = "cpu"
    print(f"Running on {device} - CUDA, if available, was deactivated for this script")

    training_data = read_data_from_matfile(args.input_filename)

    ground_truth = zero_mean(scipy.io.loadmat(args.input_filename)["gt"])
    vmax = ground_truth.max()
    vmin = ground_truth.min()

    coordinates = process_coordinates(training_data["coordinates"], device)

    model = SIREN(
        hidden_features=args.num_nodes_per_layer,
        hidden_layers=args.num_hidden_layers,
        first_omega=args.first_layer_omega,
        hidden_omega=args.hidden_layer_omega,
    )
    load_checkpoint(
        model=model, device=device, filename=args.checkpoint_file_name_on_save
    )
    model.eval()

    with torch.no_grad():
        prediction = model(coordinates)

    prediction = zero_mean(prediction)

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth, vmax=vmax, vmin=vmin, cmap="jet")
    plt.title("Ground Truth")
    plt.subplot(1, 2, 2)
    plt.imshow(
        prediction.numpy().reshape(100, 100),
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    plt.title("Prediction")
    plt.savefig(f"{args.experiment_name}.png")

    mae = relative_mean_absolute_error(prediction.reshape(100, 100), ground_truth)

    # Append MAE value to mae.txt
    with open("mae.txt", "a") as f:
        f.write(f"{mae}\n")


if __name__ == "__main__":
    main()
