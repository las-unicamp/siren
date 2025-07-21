import numpy as np
import scipy
from scipy.spatial import Delaunay

from other_methods.my_types import SolverReturn, Source
from other_methods.time_it import time_it
from other_methods.triangulation import compute_areas, get_centroids


def get_value_at_centroid(values, vertices):
    centroid_value = np.empty(len(vertices))
    for i, v in enumerate(vertices):
        centroid_value[i] = np.average(values[v])
    return centroid_value


@time_it
def os_modi_unstructured(source: Source) -> SolverReturn:
    points = source["coordinates"]

    tri = Delaunay(points)

    centroids = get_centroids(points[tri.simplices])

    faces_neighbor = np.roll(tri.neighbors, 1, axis=1)

    areas = compute_areas(points[tri.simplices], faces_neighbor)
    internal_areas = np.sum(areas, axis=1)
    relative_areas = areas / internal_areas[:, None]

    grad_x = get_value_at_centroid(source["gradient_x"], tri.simplices)
    grad_y = get_value_at_centroid(source["gradient_y"], tri.simplices)

    # Calculate distances even for -1 neighbors. This value will be multiplied
    # by the null area eventually, so it will not be a problem...
    # deltas = np.empty((len(centroids), 3, 2))
    # for i, n in enumerate(faces_neighbor):
    #     deltas = centroids[n] - centroids[i]

    num_cells = len(tri.simplices)
    rhs = np.zeros(num_cells)

    for i in range(num_cells):
        i_neighbors = faces_neighbor[i]
        deltas = centroids[i_neighbors] - centroids[i]
        # distances = np.sqrt(np.sum(deltas**2, axis=1))
        # j_vecs = deltas / (distances[:, None] + 1e-8)
        x_component = grad_x[i_neighbors] + grad_x[i]
        y_component = grad_y[i_neighbors] + grad_y[i]
        integral = 0.5 * (x_component * deltas[:, 0] + y_component * deltas[:, 1])
        # integral = (
        #     0.5 * (x_component * j_vecs[:, 0] + y_component * j_vecs[:, 1]) * distances
        # )
        # integral = 0.5 * np.einsum(
        #     "ij, ij -> i", np.stack((x_component, y_component), axis=-1), deltas
        # )
        rhs[i] = -np.sum(integral * relative_areas[i])

    def operator(p_vec):
        p_operator = p_vec.copy()
        for i in range(num_cells):
            i_neighbors = faces_neighbor[i]
            p_operator[i] -= np.sum(p_vec[i_neighbors] * relative_areas[i])

        return p_operator

    # def report(relative_res_norm):
    #     print(relative_res_norm)

    def report_cg(sol_vec):
        relative_res_norm = np.linalg.norm(
            rhs.ravel() - matrix @ sol_vec
        ) / np.linalg.norm(rhs)
        print(relative_res_norm)

    matrix = scipy.sparse.linalg.LinearOperator((num_cells, num_cells), matvec=operator)

    pressure, cg_info = scipy.sparse.linalg.cg(
        matrix, rhs, rtol="1e-3", callback=report_cg
    )

    if cg_info != 0:
        raise RuntimeWarning("Convergence to tolerance not achieved")

    return {
        "pressure": pressure,
        "pressure_coordinates": centroids,
        "data_coordinates": points,
    }
