"""Unit checks of stance dynamics, gait events, swing cost, objective values and fall penalty."""

import numpy as np
import pytest

from invLocAd.initializing.load_initial_body_state import load_initial_body_state
from invLocAd.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from invLocAd.learning.detect_endstance import detect_endstance
from invLocAd.learning.detect_midstance import detect_midstance
from invLocAd.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)
from invLocAd.learning.single_pendulum_ode import single_pendulum_ode
from invLocAd.learning.swing_cost_doke import swing_cost_doke


@pytest.mark.parametrize(
    "state",
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.3, 0.2, 0.1],
        [-0.1, -0.05, -0.02],
    ],
)
def test_pendulum_ode(model_parameters, state):
    fixed, controller = model_parameters
    derivative = single_pendulum_ode(0.0, np.array(state), fixed, controller)
    expected = [
        state[1],
        fixed["gravg"] / fixed["leglength"] * np.sin(state[0]),
        fixed["imposedFootSpeeds"]["footSpeed1List"][0],
    ]
    assert derivative.shape == (3,)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.1, 0.3, 0.5, -0.1, -0.3])
def test_detection_functions(model_parameters, angle):
    fixed, controller = model_parameters
    controller["theta_end_thisStep"] = 0.3
    state = np.array([angle, 0.1, 0.0])
    endstance = detect_endstance(0.0, state, fixed, controller)
    midstance = detect_midstance(0.0, state, fixed, controller)
    assert np.isfinite(endstance)
    assert np.isfinite(midstance)
    assert endstance == pytest.approx(angle - 0.3, abs=1e-12)
    assert midstance == pytest.approx(
        fixed["leglength"] * np.sin(angle + fixed["angleSlope"]), abs=1e-12
    )


# Costs computed with matlab/src/swingCostDoke.m and the parameters of loadBipedModelParameters.m.
@pytest.mark.parametrize(
    "duration,initial_speed,final_speed,body_speed,matlab_cost",
    [
        (0.5, 0.0, 0.1, 0.0, 0.02012461179749811),
        (0.5, 0.1, 0.0, 0.0, 0.02012461179749811),
        (1.0, 0.2, 0.3, 0.1, 0.010062305898749053),
        (0.3, -0.1, 0.1, 0.0, 0.10040418317978592),
        (0.8, 0.5, 0.5, 0.2, 0.009000000000000001),
        (0.96, -0.3276, 0.4, 0.31, 0.03664958599944431),
        (0.02, 0.0, 0.01, 0.0, 1.1250359994240184),
    ],
)
def test_swing_cost_matches_matlab(
    model_parameters, duration, initial_speed, final_speed, body_speed, matlab_cost
):
    fixed, _ = model_parameters
    cost = swing_cost_doke(duration, initial_speed, final_speed, body_speed, fixed)
    assert cost == pytest.approx(matlab_cost, rel=1e-12, abs=0)


# Outputs for offsets added to every entry of the initial controller. The objective is
# deterministic because the sensed-velocity noise is zero; the values agree to better than
# 1e-14 between OpenBLAS and Accelerate builds of NumPy and SciPy.
OBJECTIVE_REFERENCE = {
    0.0: {
        "objective": 0.035074851810527406,
        "end_time": 3.7993241410053233,
        "metabolic_power": 0.04663591093913591,
        "periodicity_error": 0.001307915917299884,
    },
    0.01: {
        "objective": 0.03840660018239915,
        "end_time": 3.8067936229458814,
        "metabolic_power": 0.05115850182164205,
        "periodicity_error": 0.0033345209769652735,
    },
    0.05: {
        "objective": 0.05490316898026243,
        "end_time": 3.835614551621519,
        "metabolic_power": 0.07307059558709247,
        "periodicity_error": 0.030820858157076397,
    },
    0.1: {
        "objective": 0.08243419768200166,
        "end_time": 3.8813220488797824,
        "metabolic_power": 0.10852627253574029,
        "periodicity_error": 0.10516266759278992,
    },
    -0.01: {
        "objective": 0.0320668549732573,
        "end_time": 3.7913233841041887,
        "metabolic_power": 0.04251205388808839,
        "periodicity_error": 0.00121376735972464,
    },
    -0.05: {
        "objective": 0.023370664548591893,
        "end_time": 3.7483480572519916,
        "metabolic_power": 0.030216710040515214,
        "periodicity_error": 0.018849462494561344,
    },
}


@pytest.mark.parametrize("controller_offset", list(OBJECTIVE_REFERENCE))
def test_objective_matches_reference(model_parameters, controller_offset):
    fixed, controller = model_parameters
    parameters = load_learnable_parameters_initial(fixed)
    initial_state = load_initial_body_state(parameters)
    objective, final_state, end_time, metabolic_power, periodicity_error = (
        f_objective_asymmetric_nominal(
            parameters + controller_offset, initial_state, controller, fixed, 0.0
        )
    )
    actual = {
        "objective": objective,
        "end_time": end_time,
        "metabolic_power": metabolic_power,
        "periodicity_error": periodicity_error,
    }
    assert np.all(np.isfinite(final_state))
    assert actual == pytest.approx(OBJECTIVE_REFERENCE[controller_offset], rel=1e-12, abs=0)


@pytest.mark.parametrize("collect_diagnostics", [False, True])
def test_omitted_fall_penalty_mode_records_com_margin(model_parameters, collect_diagnostics):
    fixed, controller = model_parameters
    fixed.pop("fallPenaltyMode", None)
    fixed["fallPenaltyWeight"] = 0.0
    fixed["collectFallDiagnostics"] = collect_diagnostics
    parameters = load_learnable_parameters_initial(fixed)
    state = load_initial_body_state(parameters)
    f_objective_asymmetric_nominal(parameters, state, controller, fixed, 0.0)
    assert fixed["_lastFallPenaltyMode"] == "com_margin"
    if collect_diagnostics:
        assert np.isfinite(fixed["_lastFallPenalty"])
    else:
        assert np.isnan(fixed["_lastFallPenalty"])
