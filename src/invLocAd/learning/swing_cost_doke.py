"""Swing-leg metabolic cost (Doke et al., 2005)."""

from __future__ import annotations

import numpy as np


def swing_cost_doke(
    t_duration_of_change: float,
    v_initial: float,
    v_final: float,
    v_body: float,
    param_fixed,
) -> float:
    """Return the swing-leg cost of changing foot velocity (Doke et al., 2005).

    The cost is ``Coeff * sqrt(r**2 + 0.01**2) ** alpha``, a smoothed power of
    the force rate ``r = mFoot * (v_final - v_initial) / T**2``.

    Args:
        t_duration_of_change: Duration ``T`` of the velocity change.
        v_initial: Foot velocity before the change (lab frame).
        v_final: Foot velocity after the change (lab frame).
        v_body: Body velocity; it cancels in the velocity change.
        param_fixed: Parameters ``mFoot``, ``swingCost['Coeff']`` and
            ``swingCost['alpha']``.

    Returns:
        Swing cost for the interval.
    """

    v_initial_relative = v_initial - v_body
    v_final_relative = v_final - v_body

    delta_v = v_final_relative - v_initial_relative
    force = param_fixed['mFoot'] * delta_v / t_duration_of_change
    force_rate = force / t_duration_of_change

    epsilon = 0.01

    c = (
        param_fixed['swingCost']['Coeff']
        * np.sqrt(force_rate**2 + epsilon**2) ** param_fixed['swingCost']['alpha']
    )

    return float(c)
