"""Inverted-pendulum stance dynamics on a moving belt."""

from __future__ import annotations

import numpy as np


def single_pendulum_ode(
    t: float,
    state_var: np.ndarray,
    param_fixed,
    param_controller,
) -> np.ndarray:
    """Return the time derivative of the inverted-pendulum stance state.

    ``theta`` is the stance-leg angle from vertical, positive with the body
    ahead of the foot. Belt acceleration ``a`` adds the inertial term
    ``-a cos(theta) / leglength`` to the angular acceleration unless
    ``includeAccelerationTorque`` is false. Without a belt-speed schedule in
    ``param_controller``, the foot moves at the nominal tied-belt speed -0.3276
    (1.0 m/s) with zero acceleration.

    Args:
        t: Time (nondimensional).
        state_var: Stance state ``[theta, theta_dot, y_foot]``.
        param_fixed: Model parameters ``mbody``, ``leglength`` and ``gravg``.
        param_controller: Belt-speed schedule ``tList_BeltSpeed``,
            ``PushoffFootSpeedNowList`` and ``PushoffAccelerationNowList``.
            ``PushoffFootSpeedNow`` and ``PushoffAccelerationNow`` are set to
            their values at ``t``.

    Returns:
        ``[theta_dot, theta_ddot, y_foot_dot]``.
    """

    angle_theta = state_var[0]
    dangle_theta = state_var[1]

    if ('tList_BeltSpeed' not in param_controller or
        'PushoffAccelerationNowList' not in param_controller):
        param_controller['PushoffAccelerationNow'] = 0.0
    else:
        # Acceleration is piecewise constant between schedule knots; linear
        # interpolation would add torque during constant-speed phases.
        _t_knots = param_controller['tList_BeltSpeed']
        _acc_list = param_controller['PushoffAccelerationNowList']
        _idx = int(np.searchsorted(_t_knots, t, side='right')) - 1
        _idx = min(max(_idx, 0), len(_acc_list) - 1)
        param_controller['PushoffAccelerationNow'] = float(_acc_list[_idx])

    force_due_to_acceleration = -param_controller['PushoffAccelerationNow'] * param_fixed['mbody']
    torque_due_to_acceleration = (
        force_due_to_acceleration
        * np.cos(angle_theta)
        / (param_fixed['mbody'] * param_fixed['leglength'])
    )

    if not param_fixed.get('includeAccelerationTorque', True):
        torque_due_to_acceleration = 0.0

    dd_angle_theta = (
        param_fixed['gravg'] / param_fixed['leglength'] * np.sin(angle_theta)
        + torque_due_to_acceleration
    )

    if ('tList_BeltSpeed' not in param_controller or
        'PushoffFootSpeedNowList' not in param_controller):
        param_controller['PushoffFootSpeedNow'] = -0.3276
    else:
        param_controller['PushoffFootSpeedNow'] = float(
            np.interp(
                t,
                param_controller['tList_BeltSpeed'],
                param_controller['PushoffFootSpeedNowList'],
                left=param_controller['PushoffFootSpeedNowList'][0],
                right=param_controller['PushoffFootSpeedNowList'][-1],
            )
        )

    dy_foot = param_controller['PushoffFootSpeedNow']

    return np.array([dangle_theta, dd_angle_theta, dy_foot])
