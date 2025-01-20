from dataclasses import dataclass

import configargparse


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    config_filepath: any

    # logger parameters
    logging_root: str
    experiment_name: str
    epochs_until_summary: int
    save_intermediary_outputs: int

    # input parameters
    fit: str
    input_filename: str

    # network architecture parameters
    num_hidden_layers: int
    num_nodes_per_layer: int
    first_layer_omega: float
    hidden_layer_omega: float

    # training parameters
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    use_autocast: bool

    # checkpoint parameters
    epochs_until_checkpoint: int
    load_checkpoint: str
    checkpoint_file_name_on_save: str


parser = configargparse.ArgumentParser()


parser.add(
    "-c",
    "--config_filepath",
    required=False,
    is_config_file=True,
    help="Path to config file.",
)


# logger parameters
parser.add_argument(
    "--logging_root", type=str, default="./logs", help="Root for logging"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of subdirectory in logging_root where summaries and checkpoints"
    "will be saved.",
)
parser.add_argument(
    "--epochs_until_summary",
    type=int,
    default=200,
    help="Number of epochs until tensorboard summary is saved. default=1,000",
)
parser.add_argument(
    "--save_intermediary_outputs",
    type=bool,
    default=True,
    help="If intermediary results should be saved. default=True",
)


# input parameters
parser.add_argument(
    "--fit",
    type=str,
    choices=["gradients", "laplacian"],
    default="gradients",
    help="Whether training will fit the gradient or the laplacian. default='gradient'",
)
parser.add_argument(
    "--input_filename",
    type=str,
    help="Name of the .mat file containing the input data (it must have the following"
    "column headings: `coords_x`, `coords_y`, `grad_x`, `grad_y`, `mask` ",
)


# network architecture parameters
parser.add_argument(
    "--num_hidden_layers",
    type=int,
    default=3,
    help="Number of hidden layers. default=3",
)
parser.add_argument(
    "--num_nodes_per_layer",
    type=int,
    default=256,
    help="Number of nodes per layer. default=256",
)
parser.add_argument(
    "--first_layer_omega",
    type=float,
    default=10,
    help="Frequency scaling of the first layer. default=10",
)
parser.add_argument(
    "--hidden_layer_omega",
    type=float,
    default=30,
    help="Frequency scaling of the hidden layers. default=30",
)


# training parameters
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="learning rate. default=1e-4"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10_000,
    help="Number of epochs to train for. default=10,000",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Make sure that the batch size is not greater than the total number of pixels"
    "of the image. default=None",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers. default=0"
)
parser.add_argument(
    "--use_autocast",
    type=bool,
    default=False,
    help="Use mixed precision training. default=False",
)


# checkpoint parameters
parser.add_argument(
    "--epochs_until_checkpoint",
    type=int,
    default=1_000,
    help="Number of epochs until checkpoint is saved. default=1,000",
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Name of the checkpoint file to continue training from a given point"
    "or make inference. default=None (fresh start)",
)
parser.add_argument(
    "--checkpoint_file_name_on_save",
    type=str,
    default="my_checkpoint.pth.tar",
    help="Name of checkpoint file to be saved. default=my_checkpoint.pth.tar",
)


args = MyProgramArgs(**vars(parser.parse_args()))
