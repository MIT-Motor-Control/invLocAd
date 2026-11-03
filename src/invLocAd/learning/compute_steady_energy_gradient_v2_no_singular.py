"""Objective gradient predicted by the internal models."""

from __future__ import annotations

import numpy as np


def compute_steady_energy_gradient_v2_no_singular(
    adynamics_model: np.ndarray,
    adynamics_energy: np.ndarray,
    p_input_now: np.ndarray,
    num_learning_dimensions: int,
    num_state_dimensions: int,
    include_internal_model: bool,
) -> np.ndarray:
    """Return the gradient of the steady-state objective with respect to the controller.

    With the internal models ``x_next = A x + B p + C`` and ``J = G x + H p + K``,
    the steady state ``x = (I - A)^-1 (B p + C)`` gives
    ``dJ/dp = G (I - A)^-1 B + H``. Without the internal model, ``dJ/dp = H``.

    Args:
        adynamics_model: State-model coefficients ``[A, B, C]``, shape (4, 13).
        adynamics_energy: Objective-model coefficients ``[G, H, K]``, shape (1, 13).
        p_input_now: Current controller; unused because the gradient of the
            linear model does not depend on it.
        num_learning_dimensions: Number of learned controller parameters (8).
        num_state_dimensions: Size of the full midstance state (5), including the
            stance angle that the models exclude.
        include_internal_model: Whether to include the state-model term.

    Returns:
        Gradient ``dJ/dp``, shape (num_learning_dimensions,).
    """

    num_state_dimensions = num_state_dimensions - 1

    inow = np.eye(num_state_dimensions)

    a = adynamics_model[0:num_state_dimensions, 0:num_state_dimensions]
    b = adynamics_model[
        0:num_state_dimensions,
        num_state_dimensions : num_state_dimensions + num_learning_dimensions,
    ]

    g = adynamics_energy[0, 0:num_state_dimensions]
    h = adynamics_energy[
        0,
        num_state_dimensions : num_state_dimensions + num_learning_dimensions,
    ]

    g_energy = g @ np.linalg.solve(inow - a, b) * float(include_internal_model) + h
    return np.asarray(g_energy, dtype=float).reshape(-1)
