import numpy as np

from jax import lax, Array
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Float


def fill(data: Float[Array, "n n_bands k k"], lower: bool = False):
    """Expand column-indexed lower block bands into a dense matrix."""
    k, n_bands, n, _ = data.shape

    columns = jnp.arange(k)[:, None]
    rows = columns + jnp.arange(n_bands)[None, :]

    blocks = jnp.zeros((k, k, n, n), dtype=data.dtype)
    blocks = blocks.at[rows, columns].set(data, mode="drop")
    if not lower:
        blocks = blocks.at[columns, rows[:, 1:]].set(
            data[:, 1:].swapaxes(-1, -2), mode="drop"
        )
    blocks = blocks.transpose(0, 2, 1, 3).reshape((n * k, n * k))

    return blocks


def blocked_cholesky(A: Float[Array, "n_blocks n_bands k k"]):
    """Factor lower block bands, scanning columns with batched band updates."""
    n_bands, k = A.shape[1:3]
    n_off = n_bands - 1

    # The carry stores pairs (i, j), i <= j, in row-major triangular order.
    # Build these shape-dependent indices on the host rather than tracing loops.
    pair_i, pair_j = np.triu_indices(n_off)
    has_previous = pair_j < n_off - 1
    previous = np.where(has_previous, np.arange(len(pair_i)) + n_off - pair_i, 0)
    carry = jnp.zeros((len(pair_i), k, k), dtype=A.dtype)

    def factor_column(carry, column):
        diagonal = jnp.linalg.cholesky(column[0] - carry[0])
        corrections = jnp.concatenate(
            (carry[1:n_off], jnp.zeros_like(column[:1])), axis=0
        )
        rhs = column[1:] - corrections

        # Solve all off-diagonal blocks against the same factor in one call.
        packed = rhs.transpose(2, 0, 1).reshape(k, n_off * k)
        R = jsp.linalg.solve_triangular(diagonal, packed, lower=True)
        R = R.reshape(k, n_off, k).transpose(1, 0, 2)

        products = R[pair_j].swapaxes(-1, -2) @ R[pair_i]
        next_carry = products + jnp.where(
            has_previous[:, None, None], carry[previous], 0
        )
        factor = jnp.concatenate((diagonal[None], R.swapaxes(-1, -2)), axis=0)
        return next_carry, factor

    _, res = lax.scan(factor_column, carry, A)
    return res


def block_solve(
    chol: Float[Array, "k n b b"],
    a: Float[Array, "k b l"],
    lower: bool = True,
):
    """Solve ``L x = a``, or ``L.T x = a`` when ``lower=False``."""
    k, n, b = chol.shape[:3]
    m = a.shape[-1]
    carry = jnp.zeros((n - 1, b, m), dtype=jnp.result_type(chol, a))

    def solve_lower(carry, data):
        a, L = data
        x1 = jsp.linalg.solve_triangular(L[0, ...], a - carry[0, ...], lower=True)
        shifted = jnp.concatenate((carry[1:], jnp.zeros_like(carry[:1])), axis=0)
        nc = shifted + L[1:] @ x1
        return nc, x1

    def solve_upper(carry, data):
        a, L = data
        rhs = a - jnp.einsum("aij,ail->jl", L[1:], carry)
        x1 = jsp.linalg.solve_triangular(L[0, ...].T, rhs, lower=False)
        nc = jnp.concatenate((x1[None], carry[:-1]), axis=0)
        return nc, x1

    if lower:
        _, x = lax.scan(solve_lower, carry, (a, chol))
    else:
        _, x = lax.scan(solve_upper, carry, (a, chol), reverse=True)
    return x
