# import matplotlib.pyplot as plt
import numpy as np
import scipy

from other_methods.my_types import SolverReturn, Source
from other_methods.os_modi.bounds import handle_oob
from other_methods.time_it import time_it


@time_it
def os_modi(source: Source) -> SolverReturn:
    oob = handle_oob(source["shape"])

    delta_x = source["delta"]
    delta_y = source["delta"]

    internal_area = (
        (~oob["b_e"]).astype(int) + (~oob["b_w"]).astype(int)
    ) * delta_y + ((~oob["b_n"]).astype(int) + (~oob["b_s"]).astype(int)) * delta_x

    factor_e = (~oob["b_e"]).astype(int) * delta_y / internal_area
    factor_w = (~oob["b_w"]).astype(int) * delta_y / internal_area
    factor_n = (~oob["b_n"]).astype(int) * delta_x / internal_area
    factor_s = (~oob["b_s"]).astype(int) * delta_x / internal_area

    integral = np.zeros(source["shape"], dtype=np.float64)
    integral[:-1, :] = source["gradient_x"][:-1, :] + source["gradient_x"][1:, :]
    # integral[:, :-1] = source["gradient_x"][:, :-1] + source["gradient_x"][:, 1:]
    integral_e = delta_x * factor_e * integral * 0.5

    integral *= 0.0
    integral[1:, :] = source["gradient_x"][:-1, :] + source["gradient_x"][1:, :]
    # integral[:, 1:] = source["gradient_x"][:, :-1] + source["gradient_x"][:, 1:]
    integral_w = -delta_x * factor_w * integral * 0.5

    integral *= 0.0
    integral[:, :-1] = source["gradient_y"][:, 1:] + source["gradient_y"][:, :-1]
    # integral[:-1, :] = source["gradient_y"][1:, :] + source["gradient_y"][:-1, :]
    integral_n = delta_y * factor_n * integral * 0.5

    integral *= 0.0
    integral[:, 1:] = source["gradient_y"][:, 1:] + source["gradient_y"][:, :-1]
    # integral[1:, :] = source["gradient_y"][1:, :] + source["gradient_y"][:-1, :]
    integral_s = -delta_y * factor_s * integral * 0.5

    source_vector = integral_e + integral_w + integral_n + integral_s

    nrows_domain, ncols_domain = source["shape"]
    matrix_size = nrows_domain * ncols_domain

    def operator(p_vec):
        p = p_vec.reshape(source["shape"])

        # p_operator = p.copy()
        p_operator = p.astype(np.float64, copy=True)
        p_operator[:-1, :] -= factor_e[:-1, :] * p[1:, :]
        p_operator[1:, :] -= factor_w[1:, :] * p[:-1, :]
        p_operator[:, :-1] -= factor_n[:, :-1] * p[:, 1:]
        p_operator[:, 1:] -= factor_s[:, 1:] * p[:, :-1]
        # p_operator[:, :-1] -= factor_e[:, :-1] * p[:, 1:]
        # p_operator[:, 1:] -= factor_w[:, 1:] * p[:, :-1]
        # p_operator[:-1, :] -= factor_n[:-1, :] * p[1:, :]
        # p_operator[1:, :] -= factor_s[1:, :] * p[:-1, :]

        # p_operator[0, 0] = p[0, 0]  # Dirichlet BC

        return p_operator.ravel()

    matrix = scipy.sparse.linalg.LinearOperator(
        (matrix_size, matrix_size), matvec=operator
    )

    rhs = -source_vector

    # rhs[0, 0] = 0  # Dirichlet BC

    # def report(sol_vec):
    #     relative_residual_norm = np.linalg.norm(
    #         rhs.ravel() - matrix @ sol_vec
    #     ) / np.linalg.norm(rhs)
    #     residual[0] = relative_residual_norm

    pressure, cg_info = scipy.sparse.linalg.cg(
        matrix,
        rhs.ravel(),
        rtol="1e-5",  # , callback=report
    )

    if cg_info != 0:
        raise RuntimeWarning("Convergence to tolerance not achieved")

    # _, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].imshow(source["ground_truth"], cmap="jet")
    # ax[1].imshow(pressure.reshape(source["shape"]), cmap="jet")
    # ax[0].set_title("Ground truth")
    # ax[1].set_title("Predicted")
    # plt.show()

    return {
        "pressure": pressure.reshape(source["shape"]),
        "pressure_coordinates": source["coordinates"],
    }
