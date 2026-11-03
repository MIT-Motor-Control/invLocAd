"""Windowed least-squares refit of the linear internal models."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular


def _pivoted_basic_lstsq(
    input_now_store: np.ndarray,
    output_next_store: np.ndarray,
) -> np.ndarray:
    """Return the basic least-squares solution of ``input @ X = output``.

    QR with column pivoting gives zero coefficients to linearly dependent
    columns of a rank-deficient window, as MATLAB's backslash operator does,
    rather than returning the minimum-norm solution.
    """

    q_factor, r_factor, pivot = qr(input_now_store, mode="economic", pivoting=True)
    tolerance = max(input_now_store.shape) * np.finfo(r_factor.dtype).eps * np.abs(r_factor).max()
    rank = int(np.sum(np.abs(np.diag(r_factor)) > tolerance))

    q_transpose_y = q_factor.T @ output_next_store
    coefficients = np.zeros((input_now_store.shape[1], output_next_store.shape[1]))
    if rank > 0:
        coefficients[:rank, :] = solve_triangular(
            r_factor[:rank, :rank],
            q_transpose_y[:rank, :],
            lower=False,
        )

    basic_solution = np.zeros_like(coefficients)
    basic_solution[pivot, :] = coefficients
    return basic_solution


def rlsupdate(
    aold: np.ndarray,
    beta: float,
    input_now_store: np.ndarray,
    output_next_store: np.ndarray,
    mu_measurement: float,
    num_steps_to_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit a linear model with intercept to the most recent observations.

    The model ``output = A @ [input; 1]`` is refit by least squares over the
    last ``num_steps_to_use`` columns each call.

    Args:
        aold: Previous coefficients, shape (n_outputs, n_inputs + 1); used only
            to compute its prediction error.
        beta: Unused; the fit has no exponential forgetting.
        input_now_store: Inputs, shape (n_inputs, n_samples), one column per stride.
        output_next_store: Outputs, shape (n_outputs, n_samples), or 1D for a
            single output.
        mu_measurement: Unused; the fit has no measurement weighting.
        num_steps_to_use: Number of most recent columns in the fit.

    Returns:
        Tuple ``(anew, error_old, error_new)``: refit coefficients, shape
        (n_outputs, n_inputs + 1) with the intercept in the last column, and the
        residuals of ``aold`` and ``anew`` on the most recent sample, each of
        shape (n_outputs, 1).
    """

    input_now_store = input_now_store[:, -num_steps_to_use:]

    if output_next_store.ndim == 1:
        output_next_store = output_next_store[-num_steps_to_use:]
        output_next_store = output_next_store.reshape(1, -1)
    else:
        output_next_store = output_next_store[:, -num_steps_to_use:]

    input_now_store = input_now_store.T
    output_next_store = output_next_store.T

    input_now_store = np.hstack(
        [input_now_store, np.ones((input_now_store.shape[0], 1))]
    )

    anew = _pivoted_basic_lstsq(input_now_store, output_next_store)

    aold_t = aold.T
    error_old_all_steps = output_next_store - input_now_store @ aold_t
    error_old_most_recent_step = error_old_all_steps[-1, :].reshape(-1, 1)

    error_new_all_steps = output_next_store - input_now_store @ anew
    error_new_most_recent_step = error_new_all_steps[-1, :].reshape(-1, 1)

    anew = anew.T

    return anew, error_old_most_recent_step, error_new_most_recent_step
