import numpy as np


def bspline(p, knots, x):
    k = knots.shape[0]
    # Compute the 0th order bspline
    temp = np.roll(knots, -1)
    temp[-1] = np.inf  # If I enforce that x is within the bounds, this is unneeded.
    b = np.where(x > knots, (x <= temp).astype(float), 0.0)
    # Note, that the last element is not needed...

    # Compute the higher order bsplines
    for i in range(1, p + 1):
        x1 = (x - knots) / (np.roll(knots, -i) - knots) * b
        x2 = (
            (np.roll(knots, -(i + 1)) - x)
            / (np.roll(knots, -(i + 1)) - np.roll(knots, -1))
            * np.roll(b, -1)
        )
        b = x1 + x2

    return b[: (k - p - 1)]


def demean(
    fe: list[np.ndarray], data=list[np.ndarray], maxiter: int = 5, norm: float = 1e-6
):
    # Prep the arrays to compute the mean.
    sort_arr, g_id, g_pos, g_count, inv_map = [], [], [], [], []
    for f in fe:
        t_sort_arr = np.argsort(f)
        t_g_id, t_g_pos, t_g_count = np.unique(
            f[t_sort_arr], return_index=True, return_counts=True
        )
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
                if len(temp.shape) == 1:
                    mean = np.add.reduceat(temp[sort_arr[j]], g_pos[j]) / g_count[j]
                    temp = temp - mean[inv_map[j]]
                    iter_norm = max(iter_norm, np.abs(mean).max())
                else:
                    for l in range(temp.shape[1]):
                        mean = (
                            np.add.reduceat(temp[sort_arr[j], l], g_pos[j]) / g_count[j]
                        )
                        temp[:, l] = temp[:, l] - mean[inv_map[j]]
                        iter_norm = max(iter_norm, np.abs(mean).max())
                data[k] = temp
        if len(fe) == 1:
            break

    print(f"Exited demeaning after {iter} iterations with norm {iter_norm}.")
    return data
