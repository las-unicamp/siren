import numpy as np


def spectral_derivative_2d(scalar, h):
    nx, ny = scalar.shape
    kx = np.fft.fftfreq(nx, h / (2 * np.pi))
    ky = np.fft.fftfreq(ny, h / (2 * np.pi))

    dpdx = np.zeros((nx, ny))
    for i in range(nx):
        dpdx[:, i] = np.fft.ifft(1j * kx * np.fft.fft(scalar[:, i])).real

    dpdy = np.zeros((nx, ny))
    for i in range(ny):
        dpdy[i, :] = np.fft.ifft(1j * ky * np.fft.fft(scalar[i, :])).real

    return dpdx, dpdy


def gradient_of_scalar(scalar, h, spectral: bool = False):
    if spectral:
        grad_x, grad_y = spectral_derivative_2d(scalar[:, :, 1], h)
        grad_z = np.zeros_like(grad_x)
    else:
        grad_x, grad_y, grad_z = np.gradient(scalar, h, edge_order=2)
    return grad_x, grad_y, grad_z


def gradient_of_vector(vector, h):
    dv1dx, dv1dy, dv1dz = gradient_of_scalar(vector[:, :, :, 0], h)
    dv2dx, dv2dy, dv2dz = gradient_of_scalar(vector[:, :, :, 1], h)
    dv3dx, dv3dy, dv3dz = gradient_of_scalar(vector[:, :, :, 2], h)

    return dv1dx, dv1dy, dv1dz, dv2dx, dv2dy, dv2dz, dv3dx, dv3dy, dv3dz


def gradient_6th_compact_3d(array, h):
    grad_x = np.zeros_like(array)
    grad_y = np.zeros_like(array)
    grad_z = np.zeros_like(array)

    nx, ny, nz = array.shape

    for j in range(ny):
        for k in range(nz):
            grad_x[:, j, k] = compact_scheme_6th(array[:, j, k], h)

    for i in range(nx):
        for k in range(nz):
            grad_y[i, :, k] = compact_scheme_6th(array[i, :, k], h)

    for i in range(nx):
        for j in range(ny):
            grad_z[i, j, :] = compact_scheme_6th(array[i, j, :], h)

    return grad_x, grad_y, grad_z


def gradient_6th_compact_2d(array, h):
    grad_x = np.zeros_like(array)
    grad_y = np.zeros_like(array)

    nrows, ncols = array.shape

    for i in range(nrows):
        grad_x[i, :] = compact_scheme_6th(array[i, :], h)

    for i in range(ncols):
        grad_y[:, i] = compact_scheme_6th(array[:, i], h)

    return grad_x, grad_y


def compact_scheme_6th(vec, h):
    """
    6th Order compact finite difference scheme (non-periodic BC).

    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42
    """
    n = len(vec)
    rhs = np.zeros(n)

    a = 14.0 / 18.0
    b = 1.0 / 36.0

    rhs[2:-2] = (vec[3:-1] - vec[1:-3]) * (a / h) + (vec[4:] - vec[0:-4]) * (b / h)

    # boundaries:
    rhs[0] = (
        (-197.0 / 60.0) * vec[0]
        + (-5.0 / 12.0) * vec[1]
        + 5.0 * vec[2]
        + (-5.0 / 3.0) * vec[3]
        + (5.0 / 12.0) * vec[4]
        + (-1.0 / 20.0) * vec[5]
    ) / h

    rhs[1] = (
        (-20.0 / 33.0) * vec[0]
        + (-35.0 / 132.0) * vec[1]
        + (34.0 / 33.0) * vec[2]
        + (-7.0 / 33.0) * vec[3]
        + (2.0 / 33.0) * vec[4]
        + (-1.0 / 132.0) * vec[5]
    ) / h

    rhs[-1] = (
        (197.0 / 60.0) * vec[-1]
        + (5.0 / 12.0) * vec[-2]
        + (-5.0) * vec[-3]
        + (5.0 / 3.0) * vec[-4]
        + (-5.0 / 12.0) * vec[-5]
        + (1.0 / 20.0) * vec[-6]
    ) / h

    rhs[-2] = (
        (20.0 / 33.0) * vec[-1]
        + (35.0 / 132.0) * vec[-2]
        + (-34.0 / 33.0) * vec[-3]
        + (7.0 / 33.0) * vec[-4]
        + (-2.0 / 33.0) * vec[-5]
        + (1.0 / 132.0) * vec[-6]
    ) / h

    alpha1 = 5.0  # j = 1 and n
    alpha2 = 2.0 / 11  # j = 2 and n-1
    alpha = 1.0 / 3.0

    db = np.ones(n)
    da = alpha * np.ones(n)
    dc = alpha * np.ones(n)

    # boundaries:
    da[1] = alpha2
    da[-1] = alpha1
    da[-2] = alpha2
    dc[0] = alpha1
    dc[1] = alpha2
    dc[-2] = alpha2

    return tdma_solver(da, db, dc, rhs)


def tdma_solver(a, b, c, d):
    """Thomas algorithm to solve tridiagonal linear systems with
    non-periodic BC.

    | b0  c0                 | | . |     | . |
    | a1  b1  c1             | | . |     | . |
    |     a2  b2  c2         | | x |  =  | d |
    |         ..........     | | . |     | . |
    |             an  bn  cn | | . |     | . |
    """
    n = len(b)

    cp = np.zeros(n)
    cp[0] = c[0] / b[0]
    for i in range(1, n - 1):
        cp[i] = c[i] / (b[i] - a[i] * cp[i - 1])

    dp = np.zeros(n)
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x
