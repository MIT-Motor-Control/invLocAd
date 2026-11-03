"""Initial midstance body state."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def load_initial_body_state(p_input_controller_asymmetric_nominal: Sequence[float]) -> np.ndarray:
    """Return the initial midstance body state.

    Args:
        p_input_controller_asymmetric_nominal: 10-element controller vector; element 1,
            the odd-step midstance speed relative to the belt, sets the initial angular
            rate (leg length 1).

    Returns:
        State ``[theta, theta_dot, y_foot, sum_y, v_swing]`` (defined in
        ``simulate_many_steps_asymmetric_control``) with zero leg angle, foot position
        and summed position error, and a lab-frame swing-foot velocity of 0.35.
    """
    v_swing_initial = 0.35
    state_var0_model = np.array([
        0.0,
        p_input_controller_asymmetric_nominal[1],
        0.0,
        0.0,
        v_swing_initial,
    ])
    return state_var0_model
