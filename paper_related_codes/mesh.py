import numpy as np

from other_methods.my_types import Array1Dfloat


def generate_uniform_points(
    xcoord: Array1Dfloat,
    ycoord: Array1Dfloat,
    num_points_x: int = 50,
    num_points_y: int = 50,
) -> tuple[Array1Dfloat, Array1Dfloat]:
    """
    Creates a regular 2D mesh inside the domain.
    """
    xl = np.linspace(xcoord[0], xcoord[-1], num_points_x)
    yl = np.linspace(ycoord[0], ycoord[-1], num_points_y)

    x, y = np.meshgrid(xl, yl, indexing="ij")

    return np.stack((x.ravel(), y.ravel()), axis=-1)


def generate_perturbed_points(
    xcoord: Array1Dfloat,
    ycoord: Array1Dfloat,
    num_points: int = 100,
    varx: float = 0.0004,
    vary: float = 0.0004,
) -> tuple[Array1Dfloat, Array1Dfloat]:
    """
    Estimate the size (num points in one boundary) based on the number of points
    assuming that the domain is square. Then, create a regualar uniform grid,
    ignore the edges to get only the "inner" points and add a noise to these
    coordinates. After that, we add the boundaries back on and voila.
    """
    np.random.seed(0)
    size = int(np.sqrt(num_points))

    xl = np.linspace(xcoord[0], xcoord[-1], size)
    yl = np.linspace(ycoord[0], ycoord[-1], size)

    x, y = np.meshgrid(xl, yl)

    x_inner = x[1:-1, 1:-1]
    y_inner = y[1:-1, 1:-1]

    # Assume dx = dy = delta
    delta = abs(xcoord[1] - xcoord[0])
    varx = varx * delta
    vary = vary * delta

    mean = (0, 0)
    cov = [[varx, 0], [0, vary]]
    uncerts = np.random.multivariate_normal(mean, cov, x_inner.shape)

    x_inner += uncerts[:, :, 0]
    y_inner += uncerts[:, :, 1]

    x[1:-1, 1:-1] = x_inner
    y[1:-1, 1:-1] = y_inner

    return np.stack([x.ravel(), y.ravel()], axis=1)


def generate_random_points(
    xcoord: Array1Dfloat,
    ycoord: Array1Dfloat,
    num_points: int = 100,
) -> tuple[Array1Dfloat, Array1Dfloat]:
    """
    Generate a 2D uniform distribution of points inside the domain.
    """
    np.random.seed(0)

    x = np.random.uniform(xcoord[0], xcoord[-1], num_points)
    y = np.random.uniform(ycoord[0], ycoord[-1], num_points)

    return np.stack((x, y), axis=-1)
