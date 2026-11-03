"""Internal linear model of the stride objective."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.rlsupdate import rlsupdate


def compute_energy_dynamics_model_rls_no_singular(
    adynamics_now: np.ndarray,
    state_var_now_store: np.ndarray,
    edot_store: np.ndarray,
    p_input_controller_now_store: np.ndarray,
    num_steps_to_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit the internal model of the measured stride objective from recent strides.

    The model is ``J = A @ [x; p; 1]``, with regressors as in
    ``compute_model_dynamics_model_rls_no_singular``. ``J`` is the objective of
    ``f_objective_asymmetric_nominal`` with multiplicative sensory noise.

    Args:
        adynamics_now: Current model coefficients, shape (1, 13).
        state_var_now_store: Midstance states at stride start, shape (5, n).
        edot_store: Measured objective of each stride, length n.
        p_input_controller_now_store: Controllers tried, shape (8, n).
        num_steps_to_use: Number of most recent strides in the fit.

    Returns:
        Tuple ``(adynamics_new, error_old_model, error_new_model)`` from
        ``rlsupdate``.
    """

    beta = 0.5

    aold = adynamics_now

    mu_measurement = 0.0

    min_cols = min(state_var_now_store.shape[1], p_input_controller_now_store.shape[1])

    state_var_trimmed = state_var_now_store[:, :min_cols]
    p_input_trimmed = p_input_controller_now_store[:, :min_cols]

    input_now_store = np.vstack((state_var_trimmed, p_input_trimmed))

    output_next_store = np.atleast_2d(edot_store)
    if output_next_store.shape[0] != 1:
        output_next_store = output_next_store.T

    output_next_store = output_next_store[:, :min_cols]

    input_now_store = input_now_store[1:, :]

    adynamics_new, error_old_model, error_new_model = rlsupdate(
        aold,
        beta,
        input_now_store,
        output_next_store,
        mu_measurement,
        num_steps_to_use,
    )

    return adynamics_new, error_old_model, error_new_model
