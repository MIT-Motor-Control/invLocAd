"""Internal linear model of the stride-to-stride midstance state map."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.rlsupdate import rlsupdate


def compute_model_dynamics_model_rls_no_singular(
    adynamics_now: np.ndarray,
    state_var_now_store: np.ndarray,
    state_var_next_store: np.ndarray,
    p_input_controller_now_store: np.ndarray,
    num_steps_to_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit the internal model of the midstance state map from recent strides.

    The model is ``x_next = A @ [x; p; 1]``, where ``x`` is the midstance state
    without the stance angle and ``p`` is the 8D controller tried in that
    stride. The stance angle equals ``-angleSlope`` at every midstance, so
    including it would make the regression singular.

    Args:
        adynamics_now: Current model coefficients, shape (4, 13).
        state_var_now_store: Midstance states at stride start, shape (5, n).
        state_var_next_store: Midstance states at stride end, shape (5, n).
        p_input_controller_now_store: Controllers tried, shape (8, n).
        num_steps_to_use: Number of most recent strides in the fit.

    Returns:
        Tuple ``(adynamics_new, error_old_model, error_new_model)`` from
        ``rlsupdate``.
    """

    beta = 0.5

    aold = adynamics_now

    mu_measurement = 0.0

    min_cols = min(state_var_now_store.shape[1], p_input_controller_now_store.shape[1],
                   state_var_next_store.shape[1])

    state_var_trimmed = state_var_now_store[:, :min_cols]
    p_input_trimmed = p_input_controller_now_store[:, :min_cols]
    state_next_trimmed = state_var_next_store[:, :min_cols]

    input_now_store = np.vstack((state_var_trimmed, p_input_trimmed))
    output_next_store = state_next_trimmed

    input_now_store = input_now_store[1:, :]
    output_next_store = output_next_store[1:, :]

    adynamics_new, error_old_model, error_new_model = rlsupdate(
        aold,
        beta,
        input_now_store,
        output_next_store,
        mu_measurement,
        num_steps_to_use,
    )

    return adynamics_new, error_old_model, error_new_model
