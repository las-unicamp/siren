from dataclasses import dataclass

import configargparse


@dataclass
class SourceArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    input_jhtdb_pressure_filename: str
    output_filename: str
    mesh: str
    num_random_points: int
    size_uniform_mesh: int


parser = configargparse.ArgumentParser(
    description="Parameters to generate the source term from JHTDB dataset",
    default_config_files=["params.yaml"],
)

parser.add_argument(
    "-c",
    "--config",
    is_config_file=True,
    help="Path to configuration file in YAML format",
)

parser.add_argument(
    "--input_jhtdb_pressure_filename",
    type=str,
    help="Name of the JHTDB file containing the pressure data in hdf5 format",
)
parser.add_argument(
    "--output_filename",
    type=str,
    help="Name of the .mat file containing the output data",
)
parser.add_argument(
    "--mesh",
    type=str,
    choices=["uniform", "random", "cartesian"],
    default="uniform",
    help="Whether mesh is 'uniform', 'random' or 'cartesian'. default='uniform'",
)
parser.add_argument(
    "--num_random_points",
    type=int,
    required=False,
    help="Number of points of the random mesh",
)
parser.add_argument(
    "--size_uniform_mesh",
    type=int,
    required=False,
    help="Number of points along the x- or y-axis of a regular squared grid.",
)

raw_args = vars(parser.parse_args())
raw_args.pop("config", None)

source_args = SourceArgs(**raw_args)
