"""Midstance event of the stance integration."""

from __future__ import annotations

import numpy as np


def detect_midstance(
    t: float, state_var: np.ndarray, param_fixed, param_controller
) -> float:
    """Return the midstance event value for ``solve_ivp``.

    Args:
        t: Time; unused.
        state_var: Stance state ``[theta, theta_dot, y_foot]``.
        param_fixed: Model parameters with ``leglength`` and ``angleSlope``.
        param_controller: Controller state; unused.

    Returns:
        Fore-aft body position relative to the stance foot,
        ``leglength * sin(theta + angleSlope)``, which crosses zero upward at
        midstance.
    """

    angle_theta = state_var[0]

    y_now = param_fixed['leglength'] * np.sin(angle_theta + param_fixed['angleSlope'])
    value = y_now

    return float(value)
