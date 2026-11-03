"""Pinned reference values of short seeded simulations run through ``run_simulation``."""

import numpy as np
import pytest

from invLocAd import run_simulation

# Seed 42 with the default split and tied protocols; the 40-iteration split run is the
# ``invLocAd-simulate demo`` configuration. The internal models are first fitted after 30
# iterations, so only that run covers the estimator fits and gradient steps. The pinned values
# agree to better than 1e-13 between OpenBLAS and Accelerate builds of NumPy and SciPy. The
# final midstance angle, the memory slopes and the step asymmetries are near zero or
# differences of close values, so they vary more between builds and are not pinned.
RTOL = 1e-9
REFERENCE = {
    ("split", 30): {
        "final_controller": [
            0.32678437435631313,
            0.31032490259883033,
            0.15417865867022476,
            0.3311722037904611,
            0.31246701381720904,
            0.1532373442217609,
            -3.8071369540658404e-05,
            -0.0020050976830108987,
        ],
        "iteration_energy_rate": {
            1: 0.0462702505649017,
            10: 0.04808641311299569,
            20: 0.04704615953346461,
            30: 0.046962068697480756,
        },
        "summary": {
            "total_steps": 60,
            "total_time": 115.23899896987456,
            "average_energy_per_step": 0.09095880879824136,
            "average_energy_rate": 0.047365308608954125,
            "final_average_energy_rate": 0.046962068697480756,
        },
    },
    ("tied", 30): {
        "final_controller": [
            0.32678437435631313,
            0.31032490259883033,
            0.15417865867022498,
            0.3311722037904611,
            0.31246701381720904,
            0.15323734422176113,
            -3.8071369540377663e-05,
            -0.002005097683010618,
        ],
        "iteration_energy_rate": {
            1: 0.05613060614824807,
            10: 0.05728993574670722,
            20: 0.05611878868072924,
            30: 0.05606763208769377,
        },
        "summary": {
            "total_steps": 60,
            "total_time": 110.50853674831114,
            "average_energy_per_step": 0.10427480216754076,
            "average_energy_rate": 0.05662264961871134,
            "final_average_energy_rate": 0.05606763208769377,
        },
    },
    ("split", 40): {
        "final_controller": [
            0.327159619079943,
            0.30959663991728775,
            0.15301325105636784,
            0.3236140586600198,
            0.30772467369560275,
            0.15628938901720377,
            0.003298074075174984,
            -0.0005015547519912896,
        ],
        "iteration_energy_rate": {
            1: 0.0462702505649017,
            30: 0.046962068697480756,
            35: 0.04787366180088488,
            40: 0.047679096823881606,
        },
        "summary": {
            "total_steps": 80,
            "total_time": 153.63887119488743,
            "average_energy_per_step": 0.0909716435410132,
            "average_energy_rate": 0.04737576875325629,
            "final_average_energy_rate": 0.047679096823881606,
        },
    },
}


@pytest.fixture(scope="module", params=list(REFERENCE), ids=lambda key: f"{key[0]}-{key[1]}")
def seeded_run(request, tmp_path_factory):
    """Return the results of a seed-42 run and the reference values of its configuration."""
    condition, iterations = request.param
    results = run_simulation(
        seed=42,
        num_iterations=iterations,
        split_or_tied=condition,
        make_plots=False,
        output_dir=tmp_path_factory.mktemp(f"{condition}_{iterations}"),
    )
    return results, REFERENCE[request.param]


def test_final_controller_matches_reference(seeded_run):
    results, reference = seeded_run
    assert results.controller_history_8d.shape == (8, results.metadata["target_iterations"])
    np.testing.assert_allclose(
        results.controller_history_8d[:, -1], reference["final_controller"], rtol=RTOL, atol=0
    )


def test_iteration_energy_rates_match_reference(seeded_run):
    results, reference = seeded_run
    expected = reference["iteration_energy_rate"]
    assert results.edot_store_iteration_average.shape == (results.metadata["target_iterations"],)
    actual = {
        iteration: results.edot_store_iteration_average[iteration - 1] for iteration in expected
    }
    assert actual == pytest.approx(expected, rel=RTOL, abs=0)


def test_summary_matches_reference(seeded_run):
    results, reference = seeded_run
    expected = reference["summary"]
    actual = {name: results.summary[name] for name in expected}
    assert actual == pytest.approx(expected, rel=RTOL, abs=0)
