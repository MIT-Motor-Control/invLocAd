"""Objective of the 8D learned controller."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)


def f_objective_asymmetric_nominal_8d_to_10d(
    p_input_controller_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    param_controller,
    param_fixed,
    t_start: float,
) -> tuple[float, np.ndarray, float, float, float]:
    """Evaluate ``f_objective_asymmetric_nominal`` for an 8D learned controller.

    The learned vector holds three odd-step targets (endstance angle, midstance
    speed, push-off impulse) and all five even-step targets. The two odd-step
    position targets, which define the spatial origin, are set to zero. The
    other arguments and the return value are as in
    ``f_objective_asymmetric_nominal``.
    """
    p_input_flat = np.asarray(p_input_controller_nominal).flatten()

    temp = np.zeros(10)
    temp[0:3] = p_input_flat[0:3]
    temp[5:10] = p_input_flat[3:8]
    p_input_controller_nominal_full = temp
    return f_objective_asymmetric_nominal(
        p_input_controller_nominal_full,
        state_var0_model,
        param_controller,
        param_fixed,
        t_start,
    )
