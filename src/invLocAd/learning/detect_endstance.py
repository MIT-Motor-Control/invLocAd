"""Endstance event of the stance integration."""

from __future__ import annotations

import numpy as np


def detect_endstance(
    t: float, state_var: np.ndarray, param_fixed, param_controller
) -> float:
    """Return the endstance event value for ``solve_ivp``.

    Args:
        t: Time; unused.
        state_var: Stance state ``[theta, theta_dot, y_foot]``.
        param_fixed: Model parameters; unused.
        param_controller: Controller state with the commanded endstance angle
            ``theta_end_thisStep`` (rad).

    Returns:
        ``theta - theta_end_thisStep``, which crosses zero upward at endstance.
    """

    angle_theta = state_var[0]
    value = angle_theta - param_controller['theta_end_thisStep']
    return float(value)
