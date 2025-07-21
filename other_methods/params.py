from dataclasses import dataclass

import configargparse


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    experiment_name: str
    input_filename: str
    output_filename: str
    mesh: str


parser = configargparse.ArgumentParser(
    description="Parameters to run the OS-MODI solver",
    default_config_files=["params_other_methods.yaml"],
)

parser.add_argument(
    "-c",
    "--config",
    is_config_file=True,
    help="Path to configuration file in YAML format",
)

parser.add_argument(
    "--experiment_name",
    type=str,
    default="experiment",
    help="Name of the experiment",
)
parser.add_argument(
    "--input_filename",
    type=str,
    help="Name of the .mat file containing the input data (it must have the following"
    "column headings: `coords_x`, `coords_y`, `grad_x`, `grad_y`, `mask` ",
)
parser.add_argument(
    "--output_filename",
    type=str,
    help="Name of the .mat file containing the output data",
)
parser.add_argument(
    "--mesh",
    type=str,
    choices=["structured", "unstructured"],
    default="structured",
    help="Whether mesh is structured (PIV) or unstructured (LPT). default='structured'",
)


raw_args = vars(parser.parse_args())
raw_args.pop("config", None)

args = MyProgramArgs(**raw_args)
