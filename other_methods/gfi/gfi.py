import numba as nb
import numpy as np
import scipy

from other_methods.my_types import SolverReturn, Source
from other_methods.time_it import time_it


@nb.njit(nogil=True, parallel=True)
def get_value_at_centroid(values, vertices):
    centroid_value = np.empty(len(vertices))
    for i in nb.prange(len(vertices)):
        centroid_value[i] = np.average(values[vertices[i]])
    return centroid_value


@nb.njit(nogil=True)
def normalized_distance_vectorized(
    observer_position, source_position, problem_dimension: int = 2
):
    difference = observer_position - source_position
    magnitude = np.sqrt(difference[:, 0] ** 2 + difference[:, 1] ** 2)
    r = np.empty_like(difference)
    r[:, 0] = difference[:, 0] / (magnitude**problem_dimension + 1e-10)
    r[:, 1] = difference[:, 1] / (magnitude**problem_dimension + 1e-10)
    return r


@nb.njit(nogil=True)
def normalized_distance(observer_position, source_position, problem_dimension: int = 2):
    difference = observer_position - source_position
    magnitude = np.linalg.norm(difference)
    return difference / (magnitude**problem_dimension + 1e-10)


def get_boundary_element_centroids(points, shape, delta):
    x = points[:, 0].reshape(shape)
    y = points[:, 1].reshape(shape)

    centroids = np.stack((x, y), axis=-1)
    # centroids = points.reshape((*shape, 2))

    half_delta = 0.5 * delta

    # fmt: off
    left = np.stack((centroids[0, :, 0] - half_delta, centroids[0, :, 1]), axis=-1)
    right = np.stack((centroids[-1, :, 0] + half_delta, centroids[-1, :, 1]), axis=-1)
    top = np.stack((centroids[:, -1, 0], centroids[:, -1, 1] + half_delta), axis=-1)
    bottom = np.stack((centroids[:, 0, 0], centroids[:, 0, 1] - half_delta), axis=-1)
    # left = np.stack((centroids[:, 0, 0] - half_delta, centroids[:, 0, 1]), axis=-1)
    # right = np.stack((centroids[:, -1, 0] + half_delta, centroids[:, -1, 1]), axis=-1)
    # top = np.stack((centroids[-1, :, 0], centroids[-1, :, 1] + half_delta), axis=-1)
    # bottom = np.stack((centroids[0, :, 0], centroids[0, :, 1] - half_delta), axis=-1)
    boundary_elements_centers = np.concatenate((left, bottom, right, top))

    left = delta * np.stack((-np.ones(shape[0]), np.zeros(shape[1])), axis=-1)
    right = delta * np.stack((np.ones(shape[0]), np.zeros(shape[1])), axis=-1)
    top = delta * np.stack((np.zeros(shape[0]), np.ones(shape[1])), axis=-1)
    bottom = delta * np.stack((np.zeros(shape[0]), -np.ones(shape[1])), axis=-1)
    boundary_elements_normals = np.concatenate((left, bottom, right, top))
    # fmt: on

    return boundary_elements_centers, boundary_elements_normals


@nb.njit(nogil=True, parallel=True)
def get_inner_pressure(
    grad_x,
    grad_y,
    centroids,
    volumes,
    boundary_elements_centers,
    boundary_elements_normals,
    boundary_pressure,
    num_cells,
    num_boundary_elements,
    problem_dimension: int = 2,
):
    inner_pressure = np.zeros(num_cells)

    for i in nb.prange(num_cells):
        r = normalized_distance_vectorized(centroids[i], centroids)
        dot = r[:, 0] * grad_x + r[:, 1] * grad_y
        inner_pressure[i] = np.sum(dot * volumes)

        for j in nb.prange(num_boundary_elements):
            r = normalized_distance(centroids[i], boundary_elements_centers[j])
            dot = (
                r[0] * boundary_elements_normals[j, 0]
                + r[1] * boundary_elements_normals[j, 1]
            )
            inner_pressure[i] -= dot * boundary_pressure[j]

    factor = 1 / (2 * (problem_dimension - 1) * np.pi)

    return inner_pressure * factor


@nb.njit(nogil=True, parallel=True)
def get_rhs(
    grad_x,
    grad_y,
    centroids,
    volumes,
    boundary_elements_centers,
    num_cells,
    num_boundary_elements,
    problem_dimension: int = 2,
):
    rhs = np.zeros(num_boundary_elements)

    for i in nb.prange(num_boundary_elements):
        for j in nb.prange(num_cells):
            r = normalized_distance(boundary_elements_centers[i], centroids[j])
            dot = r[0] * grad_x[j] + r[1] * grad_y[j]
            rhs[i] += dot * volumes

    factor2 = 1 / ((problem_dimension - 1) * np.pi)

    return rhs * factor2


@time_it
def green(source: Source) -> SolverReturn:
    source["gradient_x"] = source["gradient_x"].ravel()
    source["gradient_y"] = source["gradient_y"].ravel()

    centroids = source["coordinates"]

    boundary_elements_centers, boundary_elements_normals = (
        get_boundary_element_centroids(centroids, source["shape"], source["delta"])
    )

    volumes = source["delta"] ** 2  # Those are areas for 2D (CONSTANT FOR UNIFORM GRID)

    grad_x = source["gradient_x"]
    grad_y = source["gradient_y"]

    num_cells = source["shape"][0] * source["shape"][1]
    num_boundary_elements = len(boundary_elements_centers)
    rhs = np.zeros(num_cells)

    problem_dimension = 2

    factor2 = 1 / ((problem_dimension - 1) * np.pi)

    rhs = get_rhs(
        grad_x,
        grad_y,
        centroids,
        volumes,
        boundary_elements_centers,
        num_cells,
        num_boundary_elements,
    )

    # @nb.njit(nogil=True, parallel=True)
    def operator(p_vec):
        # p_operator = p_vec.copy()  # / factor2
        p_operator = p_vec.astype(np.float64, copy=True)  # / factor2
        for i in nb.prange(num_boundary_elements):
            for j in nb.prange(num_boundary_elements):
                r = normalized_distance(
                    boundary_elements_centers[i], boundary_elements_centers[j]
                )
                dot = (
                    r[0] * boundary_elements_normals[j, 0]
                    + r[1] * boundary_elements_normals[j, 1]
                )
                p_operator += dot * p_vec[j] * factor2

        return p_operator  # * factor2

    def report(relative_residual_norm):
        print(relative_residual_norm)

    matrix = scipy.sparse.linalg.LinearOperator(
        (num_boundary_elements, num_boundary_elements), matvec=operator
    )

    boundary_pressure, info = scipy.sparse.linalg.gmres(
        matrix, rhs, rtol="1e-3", callback=report, callback_type="legacy", maxiter=10
    )

    if info != 0:
        raise RuntimeWarning("Convergence to tolerance not achieved")

    pressure = get_inner_pressure(
        grad_x,
        grad_y,
        centroids,
        volumes,
        boundary_elements_centers,
        boundary_elements_normals,
        boundary_pressure,
        num_cells,
        num_boundary_elements,
    )

    return {
        "pressure": pressure.reshape(source["shape"]),
        "pressure_coordinates": centroids,
        "number_boundary_elements": num_boundary_elements,
    }
