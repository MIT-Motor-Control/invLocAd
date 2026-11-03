"""Python model defaults, objective repeatability and seeded learning behavior."""

from copy import deepcopy

import numpy as np
import pytest

from invLocAd.initializing.get_treadmill_speed import get_treadmill_speed
from invLocAd.initializing.load_initial_body_state import load_initial_body_state
from invLocAd.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from invLocAd.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)
from invLocAd.learning.simulate_learning_step_by_step import (
    simulate_learning_step_by_step,
)


@pytest.fixture(autouse=True)
def preserve_random_state():
    """Restore the global NumPy random state after each test."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("mbody", 1.0),
        ("leglength", 1.0),
        ("gravg", 1.0),
        ("efficiency_pos", 0.25),
        ("efficiency_neg", 1.2),
        ("lambdaEnergyVsPeriodicity", 1.0),
        ("lambdaEnergyVsSymmetry", 0.75),
        ("symmetryMultiplier", 10),
    ],
)
def test_model_defaults(model_parameters, name, expected):
    fixed, _ = model_parameters
    assert fixed[name] == pytest.approx(expected, rel=0, abs=1e-10)


@pytest.mark.parametrize("time", [0.0, 10.0, 100.0, 1000.0, 5000.0])
def test_protocol_speeds_are_finite(model_parameters, time):
    fixed, _ = model_parameters
    speeds = get_treadmill_speed(time, fixed["imposedFootSpeeds"])
    assert np.all(np.isfinite(speeds))
    if time == 0:
        assert speeds[0] == pytest.approx(speeds[1], rel=0, abs=1e-10)


def test_objective_is_repeatable(model_parameters):
    fixed, controller = model_parameters
    parameters = load_learnable_parameters_initial(fixed)
    state = load_initial_body_state(parameters)
    results = []
    for _ in range(3):
        np.random.seed(42)
        result = f_objective_asymmetric_nominal(
            parameters.copy(), state.copy(), deepcopy(controller), deepcopy(fixed), 0.0
        )
        results.append(result[0])
    assert np.all(np.isfinite(results))
    np.testing.assert_allclose(results, results[0], rtol=0, atol=1e-10)


def run_learning(fixed, controller, iterations):
    """Return the (8, iterations) controller history of a learning run seeded with 42."""
    parameters = load_learnable_parameters_initial(fixed)
    state = load_initial_body_state(parameters)
    context = np.array(get_treadmill_speed(0.0, fixed["imposedFootSpeeds"]))
    settings = deepcopy(fixed)
    settings["num_iterations"] = iterations
    np.random.seed(42)
    return simulate_learning_step_by_step(
        settings, parameters.copy(), state.copy(), context, deepcopy(controller)
    )


def test_seeded_learning_is_repeatable(model_parameters):
    fixed, controller = model_parameters
    first = run_learning(fixed, controller, 10)
    second = run_learning(fixed, controller, 10)
    assert first.shape == second.shape
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-10)


def test_short_run_updates_controller_parameters(model_parameters):
    fixed, controller = model_parameters
    history = run_learning(fixed, controller, 50)
    assert np.all(np.isfinite(history))
    changes = np.abs(history[:, -1] - history[:, 0])
    assert np.any(changes > 1e-6)
    assert np.all(changes < 1.0)
