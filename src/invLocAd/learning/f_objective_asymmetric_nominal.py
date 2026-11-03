"""Objective of one learning iteration (paper Eqs. 1 and 2)."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.fall_margin_penalty import (
    compute_fall_margin_penalty,
    compute_trial_min_com_height,
)
from invLocAd.learning.simulate_many_steps_asymmetric_control import (
    simulate_many_steps_asymmetric_control,
)


def f_objective_asymmetric_nominal(
    p_input_controller_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    param_controller,
    param_fixed,
    t_start: float,
) -> tuple[float, np.ndarray, float, float, float]:
    """Simulate one learning iteration and return its objective.

    With ``lambdaEnergyVsPeriodicity = 1`` (the default, which removes the
    periodicity term), ``lambdaEnergyVsSymmetry = 1 - psi`` and
    ``symmetryMultiplier = M = 10``, the objective is
    ``J0 = (1 - psi) e + psi M s`` (paper Eq. 1). Here ``e`` is the metabolic
    power over the iteration and ``s`` is the squared difference between the
    endstance leg angles of its first two steps. A nonzero ``fallPenaltyWeight``
    ``w`` adds ``w Phi`` (paper Eq. 2), with ``Phi`` from
    ``compute_fall_margin_penalty``, multiplied by ``e`` when
    ``fallPenaltyScaleToEnergy`` is true.

    Args:
        p_input_controller_nominal: 10D controller, five targets for odd steps
            then five for even steps: endstance leg angle (rad), midstance speed
            relative to the belt, push-off impulse, midstance position, and
            summed midstance position. The odd-step position targets (entries 3
            and 4) are not used; odd steps use zero.
        state_var0_model: Midstance state at the start of the iteration (5,).
        param_controller: Controller gains and targets; updated in place.
        param_fixed: Model parameters; updated in place. ``num_steps`` is set to
            ``Learner['numStepsPerIteration']``, and ``_lastFallPenalty``,
            ``_lastMinCom`` (both NaN unless the penalty or
            ``collectFallDiagnostics`` is enabled) and ``_lastFallPenaltyMode``
            are recorded.
        t_start: Start time of the iteration.

    Returns:
        Tuple ``(f_objective, state_var0, t_end, f_energy, f_constraint)``: the
        objective, the midstance state and time at the end of the iteration, the
        metabolic power ``e``, and the periodicity error.
    """

    param_controller['Odd']['theta_end_nominal'] = float(p_input_controller_nominal[0])
    param_controller['Odd']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_nominal[1])
    param_controller['Odd']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_nominal[2])
    param_controller['Odd']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[3])
    param_controller['Odd']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[4])

    param_controller['Even']['theta_end_nominal'] = float(p_input_controller_nominal[5])
    param_controller['Even']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_nominal[6])
    param_controller['Even']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_nominal[7])
    param_controller['Even']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[8])
    param_controller['Even']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[9])

    param_fixed['num_steps'] = param_fixed['Learner']['numStepsPerIteration']
    (
        state_var0,
        t_store,
        state_store,
        emet_total_store,
        _emet_per_time_store,
        t_step_store,
        _ework_pushoff_store,
        _ework_heelstrike_store,
    ) = simulate_many_steps_asymmetric_control(
        state_var0_model, param_controller, param_fixed, t_start
    )

    f_energy = np.sum(emet_total_store) / np.sum(t_step_store)
    t_end = float(t_store[-1][-1])

    # Squared change in theta_dot and foot position over the iteration.
    f_constraint = float(
        np.linalg.norm(state_var0_model[1:3] - state_var0[1:3]) ** 2
    )

    lambda_energy_vs_periodicity = param_fixed['lambdaEnergyVsPeriodicity']
    f_objective = (
        lambda_energy_vs_periodicity * f_energy
        + (1 - lambda_energy_vs_periodicity) * f_constraint
    )

    # state_store[0] and [1] end at the endstance of steps 1 and 2.
    angle_theta_end_1 = state_store[0][-1, 0]
    angle_theta_end_2 = state_store[1][-1, 0]
    f_symmetry = float((angle_theta_end_1 - angle_theta_end_2) ** 2)

    f_objective = (
        param_fixed['lambdaEnergyVsSymmetry'] * f_objective
        + (1 - param_fixed['lambdaEnergyVsSymmetry'])
        * f_symmetry
        * param_fixed['symmetryMultiplier']
    )

    fall_penalty_weight = float(param_fixed.get("fallPenaltyWeight", 0.0))
    if fall_penalty_weight != 0.0 or param_fixed.get("collectFallDiagnostics", False):
        f_fall_margin = compute_fall_margin_penalty(state_store, param_fixed)
        if param_fixed.get("fallPenaltyScaleToEnergy", False):
            f_fall_margin = f_fall_margin * f_energy
        param_fixed["_lastFallPenalty"] = f_fall_margin
        param_fixed["_lastFallPenaltyMode"] = param_fixed.get("fallPenaltyMode", "com_margin")
        param_fixed["_lastMinCom"] = compute_trial_min_com_height(state_store, param_fixed)
    else:
        param_fixed["_lastFallPenalty"] = np.nan
        param_fixed["_lastFallPenaltyMode"] = param_fixed.get("fallPenaltyMode", "com_margin")
        param_fixed["_lastMinCom"] = np.nan

    if fall_penalty_weight != 0.0:
        f_objective = f_objective + fall_penalty_weight * f_fall_margin

    return f_objective, state_var0, t_end, f_energy, f_constraint
