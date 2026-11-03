"""Single walking step and its metabolic cost."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.simulate_ip_until_endstance import (
    simulate_ip_until_endstance,
)
from invLocAd.learning.simulate_ip_until_midstance import (
    simulate_ip_until_midstance,
)
from invLocAd.learning.swing_cost_doke import swing_cost_doke


def simulate_one_step_midstance_to_midstance_with_energy(
    p_input0: np.ndarray,
    i_step: int,
    t_start: float,
    param_controller,
    param_fixed,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
]:
    """Simulate one step from midstance to the next midstance and its energy cost.

    The feedback controller sets the endstance angle and push-off impulse from
    midstance deviations. The stance leg follows inverted-pendulum dynamics to
    that angle, the push-off impulse acts along the trailing leg, and heel strike
    removes the velocity component along the leading leg. Metabolic cost is
    ``bPos`` times push-off work plus ``bNeg`` times heel-strike work plus the
    swing cost of both half stances (``swing_cost_doke``).

    Args:
        p_input0: Midstance state ``[theta, theta_dot, y_foot, sum_y, v_swing]``
            (see ``simulate_many_steps_asymmetric_control``).
        i_step: Step number from 1; odd steps push off from belt 1 and land on
            belt 2, even steps the reverse.
        t_start: Start time.
        param_controller: Targets and gains of the current step; updated in
            place, including the commanded angle ``theta_end_thisStep``.
        param_fixed: Model parameters and ``imposedFootSpeeds`` belt schedule.

    Returns:
        Tuple ``(state_next, t_endstance, states_endstance, t_midstance,
        states_midstance, emet_total, emet_per_time, t_total, ework_pushoff,
        ework_heelstrike)``: the next midstance state (5,); time and
        ``[theta, theta_dot, y_foot]`` trajectories from midstance to endstance
        and from heel strike to midstance; the step's metabolic cost, metabolic
        power and duration; and absolute push-off and heel-strike work.
    """

    if i_step % 2 == 0:
        param_controller['tList_BeltSpeed'] = param_fixed['imposedFootSpeeds']['tList']
        param_controller['PushoffFootSpeedNowList'] = (
            param_fixed['imposedFootSpeeds']['footSpeed2List']
        )
        param_controller['HeelstrikeFootSpeedNowList'] = (
            param_fixed['imposedFootSpeeds']['footSpeed1List']
        )
        param_controller['PushoffAccelerationNowList'] = (
            param_fixed['imposedFootSpeeds']['footAcc2List']
        )
    else:
        param_controller['tList_BeltSpeed'] = param_fixed['imposedFootSpeeds']['tList']
        param_controller['PushoffFootSpeedNowList'] = (
            param_fixed['imposedFootSpeeds']['footSpeed1List']
        )
        param_controller['HeelstrikeFootSpeedNowList'] = (
            param_fixed['imposedFootSpeeds']['footSpeed2List']
        )
        param_controller['PushoffAccelerationNowList'] = (
            param_fixed['imposedFootSpeeds']['footAcc1List']
        )

    num_points_per_interval = 300
    t_stance = 30.0
    t_end = t_start + t_stance
    tspan = np.linspace(t_start, t_end, num_points_per_interval)

    angleTheta0 = float(p_input0[0])
    angleThetaDot0 = float(p_input0[1])
    yFoot0 = float(p_input0[2])
    SUMy0 = float(p_input0[3])

    vSwing_Initial_LabFrame = float(p_input0[4])
    param_controller['PushoffFootSpeedNow'] = float(
        np.interp(
            t_start,
            param_controller['tList_BeltSpeed'],
            param_controller['PushoffFootSpeedNowList'],
        )
    )
    vBody_Initial_LabFrame = angleThetaDot0 + param_controller['PushoffFootSpeedNow']

    y_atMidstance_WRTfoot = param_fixed['leglength'] * np.sin(
        angleTheta0 + param_fixed['angleSlope']
    )
    y_atMidstance_Slopeframe = y_atMidstance_WRTfoot + yFoot0

    ydot_atMidstance = angleThetaDot0 * param_fixed['leglength']

    yFootBegin_Step1 = yFoot0

    delta_ydot_atMidstance_beltframe = (
        ydot_atMidstance - param_controller['ydot_at_midstance_nominal_beltframe']
    )

    delta_y_atMidstance = (
        y_atMidstance_Slopeframe - param_controller['y_at_midstance_nominal_slopeframe']
    )
    delta_SUMy = SUMy0 - param_controller['sumy_at_midstance_nominal_slopeframe']

    ydot_atMidstance_LabFrame = ydot_atMidstance + param_controller['PushoffFootSpeedNow']
    # Reference belt speed of the belt-speed feedback terms.
    vFootNominal = -0.35
    param_controller['ydot_atMidstance_nominal_labframe'] = (
        param_controller['ydot_at_midstance_nominal_beltframe'] + vFootNominal
    )
    delta_ydot_atMidstance_labframe = (
        ydot_atMidstance_LabFrame - param_controller['ydot_atMidstance_nominal_labframe']
    )

    delta_ydot_atMidstance_beltframe = delta_ydot_atMidstance_beltframe + (
        param_fixed['velocitySensoryNoise'] * param_fixed.get('_rng', np.random).randn()
    )

    delta_vFoot_labFrame = (
        delta_ydot_atMidstance_labframe - delta_ydot_atMidstance_beltframe
    )
    param_controller['theta_end_thisStep'] = param_controller['theta_end_nominal'] + (
        param_controller['legAngle_gain_ydot'] * delta_ydot_atMidstance_beltframe
        + param_controller['legAngle_gain_y'] * delta_y_atMidstance
        + param_controller['legAngle_gain_SUMy'] * delta_SUMy
        + param_controller['legAngle_gain_BeltSpeed'] * delta_vFoot_labFrame
    )

    # Upper limit on the commanded endstance angle (rad). With ``uncapStepAngle`` the
    # limit is ``stepAngleHardLimit`` (default 0.49 pi), which permits low COM heights
    # but keeps the COM above ground.
    _cap = 0.95 * np.pi / 4
    if param_fixed.get("uncapStepAngle", False):
        _cap = param_fixed.get("stepAngleHardLimit", 0.49 * np.pi)
    if param_controller['theta_end_thisStep'] > _cap:  # noqa: PLR1730
        param_controller['theta_end_thisStep'] = _cap

    stateVar0 = np.array([angleTheta0, angleThetaDot0, yFoot0])
    tlistTillEndstance, statevarlistTillEndstance = simulate_ip_until_endstance(
        stateVar0, tspan, param_fixed, param_controller
    )

    dAngleThetaEnd = statevarlistTillEndstance[-1, 1]

    unit_vector_AlongCircle = np.array(
        [
            np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
            -np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
        ]
    )

    vEndstance_body_pushoffFootFrame = (
        unit_vector_AlongCircle * dAngleThetaEnd * param_fixed['leglength']
    )

    unit_vector_trailingleg = np.array(
        [
            np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
            np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
        ]
    )

    PushoffImpulseMagnitude_thisStep = (
        param_controller['pushoff_impulse_magnitude_nominal']
        + param_controller['pushoff_gain_ydot'] * delta_ydot_atMidstance_beltframe
        + param_controller['pushoff_gain_y'] * delta_y_atMidstance
        + param_controller['pushoff_gain_SUMy'] * delta_SUMy
        + param_controller['pushoff_gain_BeltSpeed'] * delta_vFoot_labFrame
    )

    v_afterPushoff_pushoffFootFrame = (
        vEndstance_body_pushoffFootFrame
        + PushoffImpulseMagnitude_thisStep * unit_vector_trailingleg
    )

    deltaE_pushoff = (
        0.5 * param_fixed['mbody'] * np.linalg.norm(v_afterPushoff_pushoffFootFrame) ** 2
        - 0.5 * param_fixed['mbody'] * np.linalg.norm(vEndstance_body_pushoffFootFrame) ** 2
    )

    param_controller['PushoffFootSpeedNow'] = float(
        np.interp(
            tlistTillEndstance[-1],
            param_controller['tList_BeltSpeed'],
            param_controller['PushoffFootSpeedNowList'],
        )
    )

    v_PushoffFoot_Slopeframe = np.array([param_controller['PushoffFootSpeedNow'], 0.0])
    v_afterPushoff_labframe = v_afterPushoff_pushoffFootFrame + v_PushoffFoot_Slopeframe

    param_controller['HeelstrikeFootSpeedNow'] = float(
        np.interp(
            tlistTillEndstance[-1],
            param_controller['tList_BeltSpeed'],
            param_controller['HeelstrikeFootSpeedNowList'],
        )
    )

    v_HeelstrikeFoot_labframe = np.array([param_controller['HeelstrikeFootSpeedNow'], 0.0])
    v_afterPushoff_heelstrikeFootFrame = v_afterPushoff_labframe - v_HeelstrikeFoot_labframe

    stepLength_nextStep = 2 * param_fixed['leglength'] * np.sin(
        param_controller['theta_end_thisStep'] + param_fixed['angleSlope']
    )

    vector_leadingleg = np.array(
        [
            -np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
            np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
        ]
    )
    unit_vector_leadingleg = vector_leadingleg / np.linalg.norm(vector_leadingleg)
    v_afterHeelStrike_heelstrikeFootFrame = v_afterPushoff_heelstrikeFootFrame - (
        np.dot(unit_vector_leadingleg, v_afterPushoff_heelstrikeFootFrame)
        * unit_vector_leadingleg
    )

    deltaE_heelstrike = (
        0.5 * param_fixed['mbody'] * np.linalg.norm(v_afterHeelStrike_heelstrikeFootFrame) ** 2
        - 0.5 * param_fixed['mbody'] * np.linalg.norm(v_afterPushoff_heelstrikeFootFrame) ** 2
    )

    t_start = tlistTillEndstance[-1]
    t_stance = 30.0
    t_end = t_start + t_stance
    tspan = np.linspace(t_start, t_end, num_points_per_interval)

    angleTheta0 = -param_controller['theta_end_thisStep'] - 2 * param_fixed['angleSlope']
    dAngleTheta0 = np.linalg.norm(v_afterHeelStrike_heelstrikeFootFrame) / param_fixed['leglength']
    yFoot0 = statevarlistTillEndstance[-1, 2] + stepLength_nextStep

    yFootEnd_Step1 = statevarlistTillEndstance[-1, 2]
    yFootBegin_Step2 = yFoot0

    stateVar0 = np.array([angleTheta0, dAngleTheta0, yFoot0])
    tlistTillMidstance, statevarlistMidstance = simulate_ip_until_midstance(
        stateVar0, tspan, param_fixed, param_controller
    )

    stateVar0 = statevarlistMidstance[-1, :]

    yFoot0 = stateVar0[2]
    SUMy0 = SUMy0 + (yFoot0 - param_controller['y_at_midstance_nominal_slopeframe'])

    yFootEnd_Step2 = yFoot0

    pInput0_out = np.concatenate([stateVar0, [SUMy0]])

    Ework_pushoff = abs(deltaE_pushoff)
    Ework_heelstrike = abs(deltaE_heelstrike)
    Emet_step2step = param_fixed['bPos'] * Ework_pushoff + param_fixed['bNeg'] * Ework_heelstrike

    Thalfstance1 = np.ptp(tlistTillEndstance)
    Thalfstance2 = np.ptp(tlistTillMidstance)

    vSwing_AtoC = (yFootBegin_Step2 - yFootBegin_Step1) / Thalfstance1
    vSwing_BtoD = (yFootEnd_Step2 - yFootEnd_Step1) / Thalfstance2

    C1 = 0.0
    C3 = 0.0
    C2 = 0.0

    C1 = C1 + swing_cost_doke(
        Thalfstance1,
        vSwing_Initial_LabFrame,
        vSwing_AtoC,
        vBody_Initial_LabFrame,
        param_fixed,
    )
    C3 = C3 + swing_cost_doke(
        Thalfstance2,
        param_controller['PushoffFootSpeedNow'],
        vSwing_BtoD,
        v_afterPushoff_labframe[0],
        param_fixed,
    )

    EmetSwing_total = C1 + C2 + C3

    Emet_total = Emet_step2step + EmetSwing_total
    tStance1 = np.ptp(tlistTillEndstance)
    tStance2 = np.ptp(tlistTillMidstance)
    Emet_perTime = Emet_total / (tStance1 + tStance2)
    tTotal = tStance1 + tStance2

    pInput0_out = np.concatenate([pInput0_out, [vSwing_BtoD]])

    return (
        pInput0_out,
        np.asarray(tlistTillEndstance),
        np.asarray(statevarlistTillEndstance),
        np.asarray(tlistTillMidstance),
        np.asarray(statevarlistMidstance),
        float(Emet_total),
        float(Emet_perTime),
        float(tTotal),
        float(Ework_pushoff),
        float(Ework_heelstrike),
    )
