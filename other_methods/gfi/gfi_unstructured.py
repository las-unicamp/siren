import numba as nb
import numpy as np
import scipy
from scipy.spatial.qhull import Delaunay

from other_methods.my_types import SolverReturn, Source
from other_methods.time_it import time_it
from other_methods.triangulation import compute_volumes, get_centroids


def get_value_at_centroid(values, vertices):
    centroid_value = np.empty(len(vertices))
    for i, v in enumerate(vertices):
        centroid_value[i] = np.average(values[v])
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


def get_boundary_element_centroids(tri: Delaunay, points, faces_neighbor):
    indices_cell_boundaries = [i for i, v in enumerate(faces_neighbor) if -1 in v]

    boundary_elements_centers = []
    boundary_elements_normals = []

    for i in indices_cell_boundaries:
        vertices_coordinates = points[tri.simplices[i]]
        indices_boundary_faces = np.where(faces_neighbor[i] == -1)[0]

        deltas = (np.roll(vertices_coordinates, -1, axis=0) - vertices_coordinates)[
            indices_boundary_faces
        ]
        normals = np.stack((deltas[:, 1], -deltas[:, 0]), axis=1)
        # normals = np.stack((deltas[:, 1], -deltas[:, 0]), axis=1) / np.linalg.norm(
        #     deltas, axis=1
        # )

        centers = (
            0.5
            * (np.roll(vertices_coordinates, -1, axis=0) + vertices_coordinates)[
                indices_boundary_faces
            ]
        )

        for center, normal in zip(centers, normals):
            boundary_elements_centers.append(center)
            boundary_elements_normals.append(normal)

    return np.array(boundary_elements_centers), np.array(boundary_elements_normals)


@nb.njit(nogil=True)
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

    for i in range(num_cells):
        r = normalized_distance_vectorized(centroids[i], centroids)
        dot = r[:, 0] * grad_x + r[:, 1] * grad_y
        inner_pressure[i] = np.sum(dot * volumes)

        for j in range(num_boundary_elements):
            r = normalized_distance(centroids[i], boundary_elements_centers[j])
            dot = (
                r[0] * boundary_elements_normals[j, 0]
                + r[1] * boundary_elements_normals[j, 1]
            )
            inner_pressure[i] -= dot * boundary_pressure[j]

    factor = 1 / (2 * (problem_dimension - 1) * np.pi)

    return inner_pressure * factor


# @nb.njit(nogil=True)
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

    for i in range(num_boundary_elements):
        for j in range(num_cells):
            r = normalized_distance(boundary_elements_centers[i], centroids[j])
            dot = r[0] * grad_x[j] + r[1] * grad_y[j]
            rhs[i] += dot * volumes[j]

    factor2 = 1 / ((problem_dimension - 1) * np.pi)

    return rhs * factor2


@time_it
def green_unstructured(source: Source) -> SolverReturn:
    points = source["coordinates"]

    source["gradient_x"] = source["gradient_x"].ravel()
    source["gradient_y"] = source["gradient_y"].ravel()

    tri = Delaunay(points)

    centroids = get_centroids(points[tri.simplices])

    faces_neighbor = np.roll(tri.neighbors, 1, axis=1)

    boundary_elements_centers, boundary_elements_normals = (
        get_boundary_element_centroids(tri, points, faces_neighbor)
    )

    volumes = compute_volumes(points[tri.simplices])  # Those are areas for 2D

    grad_x = get_value_at_centroid(source["gradient_x"], tri.simplices)
    grad_y = get_value_at_centroid(source["gradient_y"], tri.simplices)

    num_cells = len(tri.simplices)
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

    def operator(p_vec):
        p_operator = p_vec.copy()  # / factor2
        for i in range(num_boundary_elements):
            for j in range(num_boundary_elements):
                r = normalized_distance(
                    boundary_elements_centers[i], boundary_elements_centers[j]
                )
                dot = (
                    r[0] * boundary_elements_normals[j, 0]
                    + r[1] * boundary_elements_normals[j, 1]
                )
                p_operator += dot * p_vec[j] * factor2

        return p_operator  # * factor2

    def report(relative_res_norm):
        print(relative_res_norm)

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
        "pressure": pressure,
        "pressure_coordinates": centroids,
        "data_coordinates": points,
        "number_boundary_elements": num_boundary_elements,
    }
