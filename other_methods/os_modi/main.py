import os

import scipy

from other_methods.my_types import Source
from other_methods.os_modi.os_modi import os_modi
from other_methods.os_modi.os_modi_unstructured import os_modi_unstructured
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
        prediction = os_modi(source)["pressure"]
    elif args.mesh == "unstructured":
        prediction = os_modi_unstructured(source)["pressure"]
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
