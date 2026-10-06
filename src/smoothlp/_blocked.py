from jax import lax, Array
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Float


def fill(data, lower: bool = False):
    """Expand column-indexed lower block bands into a dense matrix."""
    k, n_bands, n, _ = data.shape
    res = jnp.zeros((n * k, n * k), dtype=data.dtype)
    for column in range(k):
        column_slice = slice(column * n, (column + 1) * n)
        for offset in range(min(n_bands, k - column)):
            row = column + offset
            row_slice = slice(row * n, (row + 1) * n)
            block = data[column, offset]
            res = res.at[row_slice, column_slice].set(block)
            if offset and not lower:
                res = res.at[column_slice, row_slice].set(block.T)
    return res


def blocked_cholesky(A: Float[Array, "n k k"]):
    n = A.shape[1]
    k = A.shape[2]
    print(A.shape)
    subdiag_elems = int(n * (n + 1) / 2 - n)
    carry = jnp.zeros((subdiag_elems, k, k))

    def f(carry, input):
        L11 = jnp.linalg.cholesky(input[0, ...] - carry[0, ...])
        R = []
        res = []
        for i in range(1, n):
            rhs = input[i, ...]
            if i < n - 1:
                rhs = rhs - carry[i, ...]
            Ri = jsp.linalg.solve_triangular(L11, rhs.T, lower=True)
            R.append(Ri)
            res.append(Ri.T)
        # carry = jnp.stack([R12_1.T @ R12_1 + c3, R12_2.T @ R12_1, R12_2.T @ R12_2], axis=0)
        new_carry = []
        pos = 0
        for i in range(n - 1):
            for j in range(i, n - 1):
                tmp = R[j].T @ R[i]
                prev_carry_pos = pos + (n - i - 1)
                if j < n - 2:
                    tmp += carry[prev_carry_pos, ...]
                new_carry.append(tmp)
                pos += 1
        res = jnp.stack([L11] + res, axis=0)
        return jnp.stack(new_carry, axis=0), res

    _, res = lax.scan(f, carry, A)
    return res


def block_solve(
    chol: Float[Array, "k n b b"],
    a: Float[Array, "k b l"],
    lower: bool = True,
):
    """Solve ``L x = a``, or ``L.T x = a`` when ``lower=False``."""
    k, n, b = chol.shape[:3]
    l = a.shape[-1]
    carry = jnp.zeros((n - 1, b, l), dtype=jnp.result_type(chol, a))

    def solve_lower(carry, data):
        a, L = data
        x1 = jsp.linalg.solve_triangular(L[0, ...], a - carry[0, ...], lower=True)
        nc = jnp.zeros_like(carry)
        nc = nc.at[:-1].set(carry[1:, ...])
        for i in range(1, n):
            nc = nc.at[i - 1].add(L[i, ...] @ x1)
        return nc, x1

    def solve_upper(carry, data):
        a, L = data
        rhs = a
        for i in range(1, n):
            rhs = rhs - L[i, ...].T @ carry[i - 1, ...]
        x1 = jsp.linalg.solve_triangular(L[0, ...].T, rhs, lower=False)
        nc = jnp.zeros_like(carry)
        nc = nc.at[1:].set(carry[:-1, ...])
        nc = nc.at[0].set(x1)
        return nc, x1

    if lower:
        _, x = lax.scan(solve_lower, carry, (a, chol))
    else:
        _, x = lax.scan(solve_upper, carry, (a, chol), reverse=True)
    return x
