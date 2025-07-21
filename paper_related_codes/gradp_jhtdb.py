"""
This script takes the gradient of the original pressure field from the JHTDB.
Here, we do NOT consider any noise in the solution. This simplifies the evaluation
of the source term: precisely, we can simply take the gradient of the JHTDB
pressure field. This means that the source terms generated here are used for the
mesh study in our manuscript.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy

from other_methods.my_types import Source
from source.gradient import gradient_6th_compact_2d
from source.gradp_jhtdb_params import source_args
from source.mesh import generate_random_points, generate_uniform_points


def main() -> Source:
    f = h5py.File(source_args.input_jhtdb_pressure_filename, "r")

    xcoor = f["xcoor"]
    ycoor = f["ycoor"]
    pressure = f["Pressure_0001"]
    pressure = np.array(pressure).squeeze()

    # select a subdomain
    size = 100
    pressure = pressure[-size:, -size:]
    xcoor = xcoor[-size:]
    ycoor = ycoor[-size:]

    delta = abs(xcoor[1] - xcoor[0])

    if source_args.mesh == "CARTESIAN":
        dpdx, dpdy = gradient_6th_compact_2d(pressure, delta)

        return {
            "ground_truth": pressure,
            "gradient_x": dpdx,
            "gradient_y": -dpdy,
            "delta": delta,
            "shape": pressure.shape,
        }

    else:
        dpdy, dpdx = gradient_6th_compact_2d(pressure, delta)

    # yy, xx = np.meshgrid(xcoor, ycoor)
    # original_points = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    if source_args.mesh == "RANDOM":
        new_points = generate_random_points(xcoor, ycoor, source_args.num_random_points)
        # new_points = generate_random_points_with_nice_boundaries(
        #     xcoor, ycoor, source_args.num_random_points
        # )
    elif source_args.mesh == "UNIFORM":
        new_points = generate_uniform_points(
            xcoor, ycoor, source_args.size_uniform_mesh
        )
    else:
        raise ValueError("Choose between RANDOM or UNIFORM")

    interp = scipy.interpolate.RegularGridInterpolator((xcoor, ycoor), pressure)
    new_pressure = interp(new_points)

    interp = scipy.interpolate.RegularGridInterpolator((xcoor, ycoor), dpdx)
    new_dpdx = interp(new_points)

    interp = scipy.interpolate.RegularGridInterpolator((xcoor, ycoor), dpdy)
    new_dpdy = interp(new_points)

    source: Source
    source = {
        "ground_truth": new_pressure,
        "coordinates": new_points,
        "gradient_x": new_dpdx,
        "gradient_y": new_dpdy,
        "delta": delta,
        "shape": pressure.shape,
    }
    scipy.io.savemat(source_args.output_filename, source)

    plt.scatter(
        source["coordinates"][:, 0], source["coordinates"][:, 1], s=1, color="k"
    )
    plt.gca().set_aspect("equal")
    plt.show()

    # plt.tricontourf(
    #     source["coordinates"][:, 0],
    #     source["coordinates"][:, 1],
    #     source["ground_truth"],
    #     cmap="jet",
    #     levels=256,
    # )
    # plt.gca().set_aspect("equal")
    # plt.show()


# def add_noise_to_pressure(
#     pressure,
#     amplitude: float,
#     plane_idx: int,
#     plane: Literal["xy", "yz", "xz"],
#     seed: int = 0,
# ):
#     np.random.seed(seed)  # reproducible noise

#     match plane:
#         case "xy":
#             plane_of_interest = pressure[:, :, plane_idx]
#         case "yz":
#             plane_of_interest = pressure[plane_idx, :, :]
#         case "xz":
#             plane_of_interest = pressure[:, plane_idx, :]
#         case _:
#             raise ValueError("plane must be either x, y or z")

#     magnitude = np.linalg.norm(plane_of_interest, axis=-1)
#     vmax = magnitude.max()

#     noise = np.random.normal(0, 0.5 * vmax * amplitude, size=pressure.shape)

#     return pressure + noise


if __name__ == "__main__":
    main()
