"""Stance integration from heel strike to midstance."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from invLocAd.learning.detect_midstance import detect_midstance
from invLocAd.learning.single_pendulum_ode import single_pendulum_ode


def simulate_ip_until_midstance(
    state_var0: np.ndarray,
    tspan: np.ndarray,
    param_fixed,
    param_controller,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the stance dynamics from heel strike until midstance.

    Integration stops at the upward zero crossing of ``detect_midstance`` or at
    the end of ``tspan``; tolerances are as in ``simulate_ip_until_endstance``.

    Args:
        state_var0: Initial stance state ``[theta, theta_dot, y_foot]``.
        tspan: Times whose first and last entries bound the integration.
        param_fixed: Model parameters.
        param_controller: Controller state with the belt schedule used by
            ``single_pendulum_ode``.

    Returns:
        Tuple ``(t, states)``: solver times, shape (n,), and states, shape (n, 3),
        ending at the event if it occurs.
    """

    def ode(t, y):
        """Return the stance-state derivative from ``single_pendulum_ode``."""
        return single_pendulum_ode(t, y, param_fixed, param_controller)

    def event(t, y):
        """Return the midstance event value from ``detect_midstance``."""
        return detect_midstance(t, y, param_fixed, param_controller)

    event.terminal = True
    event.direction = 1.0

    sol = solve_ivp(
        ode,
        (float(tspan[0]), float(tspan[-1])),
        state_var0,
        events=event,
        rtol=param_fixed.get("odeRtol", 1e-10),
        atol=param_fixed.get("odeAtol", 1e-10),
        dense_output=False,
    )

    return sol.t, sol.y.T
