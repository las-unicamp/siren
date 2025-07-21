"""
This script creates the source term (rhs of grad.p in NS equations) to be used
for pressure reconstruction.
Here, we include noise in the velocity field to mimic what happens in a PIV
setting. This error is included in the original mesh from JHTDB simulation.
Then, we interpolate the result to new points, to be used by the pressure
reconstruction solver (these "new" points can be distributed randomically
or in a uniform fashion).
"""

import os
from typing import Literal

import h5py
import matplotlib.pylab as plt
import numpy as np
import scipy

from other_methods.my_types import Source
from source.gradient import gradient_of_scalar, gradient_of_vector
from source.mesh import generate_uniform_points

JHDB_PRESSURE_FILENAME = "jhtdb_isotropic1024coarse_3D_pressure_1024_t39-41.h5"
JHDB_VELOCITY_FILENAME = "jhtdb_isotropic1024coarse_3D_velocity_1024_t39-41.h5"


def advective_term(velocity, h):
    dv1dx, dv1dy, dv1dz, dv2dx, dv2dy, dv2dz, dv3dx, dv3dy, dv3dz = gradient_of_vector(
        velocity, h
    )
    component1 = (
        velocity[:, :, :, 0] * dv1dx
        + velocity[:, :, :, 1] * dv1dy
        + velocity[:, :, :, 2] * dv1dz
    )
    component2 = (
        velocity[:, :, :, 0] * dv2dx
        + velocity[:, :, :, 1] * dv2dy
        + velocity[:, :, :, 2] * dv2dz
    )
    component3 = (
        velocity[:, :, :, 0] * dv3dx
        + velocity[:, :, :, 1] * dv3dy
        + velocity[:, :, :, 2] * dv3dz
    )

    return np.stack((component1, component2, component3), axis=-1)


def compute_material_derivative(velocity_previous, velocity, velocity_next, h, dt):
    # temporal_term = (velocity_next - velocity) / dt
    temporal_term = (velocity_next - velocity_previous) / (2 * dt)
    return temporal_term + advective_term(velocity, h)


def random_unit_vector_3d(size=None):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    See http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """
    phi = np.random.uniform(0, np.pi * 2, size=size)
    costheta = np.random.uniform(-1, 1, size=size)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def add_noise_to(
    velocity,
    amplitude: float,
    plane_idx: int,
    plane: Literal["xy", "yz", "xz"],
    seed: int = 0,
):
    np.random.seed(seed)  # reproducible noise

    match plane:
        case "xy":
            plane_of_interest = velocity[:, :, plane_idx]
        case "yz":
            plane_of_interest = velocity[plane_idx, :, :]
        case "xz":
            plane_of_interest = velocity[:, plane_idx, :]
        case _:
            raise ValueError("plane must be either x, y or z")

    magnitude = np.linalg.norm(plane_of_interest, axis=-1)
    vmax = magnitude.max()

    size = velocity.shape[:-1]  # spatial information only
    noise = np.stack(random_unit_vector_3d(size), axis=-1)
    noise_amplitude = np.random.normal(0, 0.5 * vmax * amplitude, size=size)

    # scale unit noise vector with its corresponding amplitude
    for i in range(3):
        noise[:, :, :, i] *= noise_amplitude

    return velocity + noise


def select_chunk_in_z(original_data, num_points: int):
    return original_data[:, :, :num_points, :]


def interpolate_data(original_data, xcoor, ycoor, zcoor, num_points):
    def run_interp(new_points, original_data, coord1, coord2):
        interp = scipy.interpolate.RegularGridInterpolator(
            (coord1, coord2), original_data
        )
        return interp(new_points)

    new_size = round(np.sqrt(num_points))

    new_delta = abs(xcoor[0] - xcoor[-1]) / new_size

    # first interpolate in z
    new_nz = 3
    new_zcoor = np.linspace(zcoor[0], new_delta * (new_nz - 1), new_nz)
    nx, ny, _ = original_data.shape
    new_yz_points = generate_uniform_points(ycoor, new_zcoor, ny, new_nz)

    interp_data_z = np.zeros((nx, ny, new_nz))

    for i in range(nx):
        interp_data_z[i, :, :] = run_interp(
            new_yz_points, original_data[i, :, :], ycoor, zcoor
        ).reshape(ny, new_nz)

    # now, interpolate to new points along x-y plane
    new_points = generate_uniform_points(xcoor, ycoor, new_size, new_size)

    interp_data = np.zeros((new_size, new_size, new_nz))

    for i in range(new_nz):
        interp_data[:, :, i] = run_interp(
            new_points, interp_data_z[:, :, i], xcoor, ycoor
        ).reshape(new_size, new_size)

    return interp_data, new_points


class JHTDB:
    def __init__(self):
        f_pressure = h5py.File(JHDB_PRESSURE_FILENAME, "r")
        f_velocity = h5py.File(JHDB_VELOCITY_FILENAME, "r")

        self.dt = 0.002

        self.xcoor = f_pressure["xcoor"]
        self.ycoor = f_pressure["ycoor"]
        self.zcoor = f_pressure["zcoor"]

        self.pressure = f_pressure["Pressure_0040"]
        self.velocity_previous = f_velocity["Velocity_0039"]
        self.velocity = f_velocity["Velocity_0040"]
        self.velocity_next = f_velocity["Velocity_0041"]

        self.pressure = self.fix_index_convention(self.pressure)
        self.velocity_previous = self.fix_index_convention(self.velocity_previous)
        self.velocity = self.fix_index_convention(self.velocity)
        self.velocity_next = self.fix_index_convention(self.velocity_next)

        self.delta = abs(self.xcoor[1] - self.xcoor[0])

        num_x = len(self.xcoor)
        num_y = len(self.ycoor)
        self.coordinates = generate_uniform_points(self.xcoor, self.ycoor, num_x, num_y)

    @staticmethod
    def fix_index_convention(array) -> None:
        array = np.transpose(array, axes=[2, 1, 0, 3])

        is_velocity = array.shape[-1] > 1
        if is_velocity:
            vel_x = array[:, :, :, 2]
            vel_y = array[:, :, :, 1]
            vel_z = array[:, :, :, 0]

            array[:, :, :, 0] = vel_x
            array[:, :, :, 1] = vel_y
            array[:, :, :, 2] = vel_z

        return array

    def select_subdomain(self, size: int) -> None:
        self.pressure = self.pressure[:size, :size, :size, :]
        self.velocity_previous = self.velocity_previous[:size, :size, :size, :]
        self.velocity = self.velocity[:size, :size, :size, :]
        self.velocity_next = self.velocity_next[:size, :size, :size, :]
        self.xcoor = self.xcoor[:size]
        self.ycoor = self.ycoor[:size]
        self.zcoor = self.zcoor[:size]

    def interpolate_domain(self, num_points: int):
        new_size = round(np.sqrt(num_points))
        new_nz = 3
        interp_pressure = np.zeros((new_size, new_size, new_nz, 1))
        interp_velocity_previous = np.zeros((new_size, new_size, new_nz, 3))
        interp_velocity = np.zeros((new_size, new_size, new_nz, 3))
        interp_velocity_next = np.zeros((new_size, new_size, new_nz, 3))

        interp_pressure[:, :, :, 0], new_points = interpolate_data(
            self.pressure[:, :, :, 0], self.xcoor, self.ycoor, self.zcoor, num_points
        )

        interp_velocity_previous[:, :, :, 0], _ = interpolate_data(
            self.velocity_previous[:, :, :, 0],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )
        interp_velocity_previous[:, :, :, 1], _ = interpolate_data(
            self.velocity_previous[:, :, :, 1],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )
        interp_velocity_previous[:, :, :, 2], _ = interpolate_data(
            self.velocity_previous[:, :, :, 2],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )

        interp_velocity[:, :, :, 0], _ = interpolate_data(
            self.velocity[:, :, :, 0], self.xcoor, self.ycoor, self.zcoor, num_points
        )
        interp_velocity[:, :, :, 1], _ = interpolate_data(
            self.velocity[:, :, :, 1], self.xcoor, self.ycoor, self.zcoor, num_points
        )
        interp_velocity[:, :, :, 2], _ = interpolate_data(
            self.velocity[:, :, :, 2], self.xcoor, self.ycoor, self.zcoor, num_points
        )

        interp_velocity_next[:, :, :, 0], _ = interpolate_data(
            self.velocity_next[:, :, :, 0],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )
        interp_velocity_next[:, :, :, 1], _ = interpolate_data(
            self.velocity_next[:, :, :, 1],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )
        interp_velocity_next[:, :, :, 2], _ = interpolate_data(
            self.velocity_next[:, :, :, 2],
            self.xcoor,
            self.ycoor,
            self.zcoor,
            num_points,
        )

        self.pressure = interp_pressure
        self.velocity_previous = interp_velocity_previous
        self.velocity = interp_velocity
        self.velocity_next = interp_velocity_next
        self.delta = abs(self.xcoor[0] - self.xcoor[-1]) / new_size
        self.coordinates = new_points


def get_source_term(noise_amplitude: float) -> Source:
    jhtdb = JHTDB()

    source_term = -compute_material_derivative(
        jhtdb.velocity_previous,
        jhtdb.velocity,
        jhtdb.velocity_next,
        jhtdb.delta,
        jhtdb.dt,
    )

    grad_p = np.stack(
        gradient_of_scalar(jhtdb.pressure[:, :, :, 0], jhtdb.delta), axis=-1
    )

    residual = grad_p - source_term

    plane = "xy"
    plane_idx = 1  # in order to use 2nd order central scheme in z-direction

    noisy_vel_previous = add_noise_to(
        jhtdb.velocity_previous, noise_amplitude, plane_idx, plane, seed=0
    )
    noisy_vel = add_noise_to(jhtdb.velocity, noise_amplitude, plane_idx, plane, seed=0)
    noisy_vel_next = add_noise_to(
        jhtdb.velocity_next, noise_amplitude, plane_idx, plane, seed=0
    )

    noisy_source_term = -compute_material_derivative(
        noisy_vel_previous, noisy_vel, noisy_vel_next, jhtdb.delta, jhtdb.dt
    )

    corrected_noisy_source_term = noisy_source_term + residual

    # # Make plot to check if gradients make sense...
    # shape = pressure[:, :, plane_idx, 0].shape
    # x = new_points[:, 0].reshape(shape)
    # y = new_points[:, 1].reshape(shape)
    # plt.contourf(x, y, corrected_noisy_source_term[:, :, plane_idx, 0])
    # plt.show()
    # plt.contourf(x, y, corrected_noisy_source_term[:, :, plane_idx, 1])
    # plt.show()

    vmax = grad_p[:, :, plane_idx, 0].max()
    vmin = grad_p[:, :, plane_idx, 0].min()
    _, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(grad_p[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[1].imshow(source_term[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[2].imshow(residual[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[3].imshow(corrected_noisy_source_term[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[0].set_title("grad p")
    ax[1].set_title("material derivative")
    ax[2].set_title("residual")
    ax[3].set_title("noisy source term with residual")
    plt.show()

    _, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(jhtdb.pressure[:, :, plane_idx])
    ax[1].imshow(source_term[:, :, plane_idx, 0])
    ax[2].imshow(source_term[:, :, plane_idx, 1])
    ax[3].imshow(source_term[:, :, plane_idx, 2])
    ax[0].set_title("pressure in x-y")
    ax[1].set_title("dpdx")
    ax[2].set_title("dpdy")
    ax[3].set_title("dpdz")
    plt.show()

    vmax = noisy_vel[:, :, plane_idx, 0].max()
    vmin = noisy_vel[:, :, plane_idx, 0].min()
    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(jhtdb.velocity[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[1].imshow(noisy_vel[:, :, plane_idx, 0], vmax=vmax, vmin=vmin)
    ax[0].set_title("velocity")
    ax[1].set_title("noisy vel")
    plt.show()

    return {
        "ground_truth": jhtdb.pressure[:, :, plane_idx, 0],
        "coordinates": jhtdb.coordinates,
        "gradient_x": corrected_noisy_source_term[:, :, plane_idx, 0],
        "gradient_y": corrected_noisy_source_term[:, :, plane_idx, 1],
        "delta": jhtdb.delta,
        "shape": jhtdb.pressure[:, :, plane_idx, 0].shape,
    }


def main():
    noise_amplitude_list = np.linspace(0, 0.1, num=11)

    for noise_amplitude in noise_amplitude_list:
        data_dict = get_source_term(noise_amplitude)

        out_filename = f"source_jhtdb_entire_2D_domain_noise_{noise_amplitude:0.2f}.mat"

        scipy.io.savemat(os.path.join("material_derivatives", out_filename), data_dict)


if __name__ == "__main__":
    main()
