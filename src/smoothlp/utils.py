import numpy as np
from scipy import linalg as sla
from jax import numpy as jnp, scipy as jsp, Array, lax
from jaxtyping import Float


def bspline(p, knots, x):
    """B-Splines defined through the interior knots on unit interval.

    This function returns k B-Splines, where k is the size of the knots array.
    Note, that the knots must be equally spaced.

    The return matrix has dimension i x k, where i is the size of x
    and k the number of splines.
    """
    np.testing.assert_allclose(np.diff(knots), 1.0)
    k = knots.shape[0]
    # Add boundaries
    if p > 0:
        knots = np.concat(
            (
                np.arange(knots[0] - p, knots[0]),
                knots,
                np.arange(knots[-1] + 1, knots[-1] + p + 2),
            )
        )

    # Reshape
    knots = knots.reshape(-1, 1)
    x = x.reshape(1, -1)

    # Compute the 0th order bspline
    temp = np.roll(knots, -1)
    temp[-1] = knots[-1] + 1.0
    b = ((x >= knots) & (x < temp)).astype(float)

    # Compute the higher order bsplines
    for i in range(1, p + 1):
        x1 = (x - knots) / (np.roll(knots, -i) - knots) * b
        x2 = (
            (np.roll(knots, -(i + 1)) - x)
            / (np.roll(knots, -(i + 1)) - np.roll(knots, -1))
            * np.roll(b, -1, axis=0)
        )
        b = x1 + x2

    return b[p : (p + k), :].T


def demean(
    fe: list[np.ndarray],
    data: list[np.ndarray],
    maxiter: int = 10,
    norm: float = 1e-6,
    mask: np.ndarray = None,
):
    """Iteratively demean the provided data

    If a mask was provided, ignore masked elements during the mean computation step.
    """
    # Prep the arrays to compute the mean.
    sort_arr, g_id, g_pos, g_count, inv_map = [], [], [], [], []
    for f in fe:
        t_sort_arr = np.argsort(f)
        t_g_id, t_g_pos, t_g_count = np.unique(
            f[t_sort_arr], return_index=True, return_counts=True
        )
        if mask is not None:
            t_g_count = np.add.reduceat((~mask)[t_sort_arr], t_g_pos)
        _, t_inv_map = np.unique(f, return_inverse=True)
        sort_arr.append(t_sort_arr)
        g_id.append(t_g_id)
        g_pos.append(t_g_pos)
        g_count.append(t_g_count)
        inv_map.append(t_inv_map)

    iter_norm = 1e6
    iter = 0
    while iter_norm >= norm and iter < maxiter:
        iter_norm = 0.0
        iter += 1
        for j in range(len(fe)):
            for k in range(len(data)):
                temp = data[k]
                if mask is None:
                    temp_comp_mean = temp
                else:
                    if len(temp.shape) == 1:
                        temp_comp_mean = np.where(mask, 0.0, temp)
                    else:
                        temp_comp_mean = np.where(mask.reshape((-1, 1)), 0.0, temp)
                if len(temp.shape) == 1:
                    mean = (
                        np.add.reduceat(temp_comp_mean[sort_arr[j]], g_pos[j])
                        / g_count[j]
                    )
                    temp = temp - mean[inv_map[j]]
                    iter_norm = max(iter_norm, np.abs(mean).max())
                else:
                    for d in range(temp.shape[1]):
                        mean = (
                            np.add.reduceat(temp_comp_mean[sort_arr[j], d], g_pos[j])
                            / g_count[j]
                        )
                        temp[:, d] = temp[:, d] - mean[inv_map[j]]
                        iter_norm = max(iter_norm, np.abs(mean).max())
                data[k] = temp
        if len(fe) == 1:
            break

    print(f"Exited demeaning after {iter} iterations with norm {iter_norm}.")
    return data


def _lrv_np(Xu: Float[np.ndarray, "T k"], lags: int) -> Float[np.ndarray, "k k"]:
    cross = 0.5 * Xu
    temp = Xu.copy()
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        temp = np.roll(temp, -1, axis=0)
        temp[-1, :] = 0.0
        cross += w * temp
    cross = Xu.T @ cross
    lrv = cross + cross.T
    return lrv


def _lrv_jax(Xu: Float[Array, "T k"], lags: int) -> Float[Array, "k k"]:
    def accumulate(carry, lag):
        shifted, weighted = carry

        # Advance one time step, zero-padding instead of wrapping.
        shifted = jnp.roll(shifted, -1, axis=0).at[-1].set(0.0)
        weight = 1.0 - lag / (lags + 1)
        return (shifted, weighted + weight * shifted), None

    (_, weighted), _ = lax.scan(
        accumulate,
        (Xu, 0.5 * Xu),
        jnp.arange(1, lags + 1),
    )

    cross = Xu.T @ weighted
    lrv = cross + cross.T
    return lrv


def comp_lrv(Xu, lags: int):
    if isinstance(Xu, np.ndarray):
        lrv = _lrv_np(Xu, lags)
    elif isinstance(Xu, jnp.ndarray):
        lrv = _lrv_jax(Xu, lags)
    else:
        raise ValueError("Xu must be a numpy or jax array")
    return lrv


def np_vmap(f, in_axes=0, out_axes=0):
    def mapped(x):
        temp = np.moveaxis(x, in_axes, 0)
        return np.stack(
            [f(temp[_i, ...]) for _i in range(temp.shape[0])], axis=out_axes
        )

    return mapped


def invert(X):
    U = sla.cholesky(X)
    U_inv = sla.solve_triangular(U, np.eye(X.shape[0]))
    return U_inv @ U_inv.T


def invert_jax(X):
    U = jsp.linalg.cholesky(X)
    U_inv = jsp.linalg.solve_triangular(U, jnp.eye(X.shape[0]))
    return U_inv @ U_inv.T
