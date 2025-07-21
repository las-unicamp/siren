"""
This script generates the source term of the Taylor Green Vortex (TGV) problem.
It is based on the paper from Charonko et al. (2010) "Assessment of pressure
field calculations from particle image velocimetry measurements", Eqs. (10) and
(11). However, we are working in a non-dimensional system.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.animation import FuncAnimation

from other_methods.my_types import Source

TIME = 0.03


def taylor_green_vortex(
    radius: float, time: float, reynolds: float = 1
) -> tuple[float, float]:
    """
    Non-dimensional TGV in cylindrical coordinates
    Charonko (2010) uses Re = 1
    """
    exp_arg = -(radius**2) / (4 * time)
    u_theta = radius / (8 * np.pi * time**2) * np.exp(exp_arg)
    p = -1.0 / (64 * np.pi**2 * time**3 * reynolds**2) * np.exp(2 * exp_arg)
    dpdr = radius / (64 * np.pi**2 * time**4 * reynolds**2) * np.exp(2 * exp_arg)
    return u_theta, p, dpdr


def cartesian_to_cylindrical(x: float, y: float) -> tuple[float, float]:
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return radius, theta


def cylindrical_to_cartesian(radius: float, theta: float) -> tuple[float, float]:
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def viz_movie():
    xl = np.linspace(-0.5, 0.5, 100)
    yl = np.linspace(-0.5, 0.5, 100)
    x, y = np.meshgrid(xl, yl)
    radius, _ = cartesian_to_cylindrical(x, y)

    initial_time = TIME
    dt = 0.0001

    u_theta_initial, p_initial, dpdr_initial = taylor_green_vortex(
        radius, time=initial_time
    )

    fig = plt.figure()

    im = plt.imshow(p_initial, cmap="jet")  # make the plot we want to animate

    plt.gca().set_axis_off()
    plt.title(f"Time: {initial_time}")

    def animate(i, radius, initial_time, dt):
        time = initial_time + i * dt
        u_theta, p, dpdr = taylor_green_vortex(radius, time=time)

        im.set_array(p)

        plt.gca().set_title(f"Time: {time:0.3f}")

    num_seconds = 10
    fps = 60

    anim = FuncAnimation(
        fig,
        partial(animate, radius=radius, initial_time=initial_time, dt=dt),
        frames=num_seconds * fps,
        interval=1000 / fps,  # in ms
    )

    plt.show()


def main():
    xl = np.linspace(-0.5, 0.5, 100)
    yl = np.linspace(-0.5, 0.5, 100)
    x, y = np.meshgrid(xl, yl)
    radius, theta = cartesian_to_cylindrical(x, y)

    _, p, dpdr = taylor_green_vortex(radius, time=TIME)

    dpdx, dpdy = cylindrical_to_cartesian(dpdr, theta)

    source: Source
    source = {
        "coordinates": np.stack((x.ravel(), y.ravel()), axis=-1),
        "delta": abs(xl[1] - xl[0]),
        "gradient_x": dpdx,
        "gradient_y": dpdy,
        "shape": x.shape,
        "ground_truth": p,
    }

    # plt.imshow(p, cmap="jet")
    plt.contourf(x, y, p, cmap="jet")
    plt.show()

    # plt.imshow(dpdx, cmap="jet")
    plt.contourf(x, y, dpdx, cmap="jet")
    plt.show()

    # plt.imshow(dpdy, cmap="jet")
    plt.contourf(x, y, dpdy, cmap="jet")
    plt.show()

    # plt.imshow(x, cmap="jet")
    plt.contourf(x, y, x, cmap="jet")
    plt.show()

    # plt.imshow(y, cmap="jet")
    plt.contourf(x, y, y, cmap="jet")
    plt.show()

    scipy.io.savemat("source_taylor_green_vortex.mat", source)


if __name__ == "__main__":
    main()
    # viz_movie()
