import os

import scipy

from other_methods.gfi.gfi import green
from other_methods.gfi.gfi_unstructured import green_unstructured
from other_methods.my_types import Source
from other_methods.params import args


def parse_source(filename) -> Source:
    source = scipy.io.loadmat(filename)

    for key, val in source.items():
        if key.startswith("_"):
            continue
        source[key] = val.squeeze()

    return source


def main():
    source: Source
    source = parse_source(args.input_filename)

    if args.mesh == "structured":
        prediction = green(source)["pressure"]
    elif args.mesh == "unstructured":
        prediction = green_unstructured(source)["pressure"]
    else:
        raise ValueError("Mesh must be either structured or unstructured")

    scipy.io.savemat(
        os.path.join(args.output_filename),
        {
            "prediction": prediction,
        },
    )


if __name__ == "__main__":
    main()
