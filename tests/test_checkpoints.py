"""Numerical reproducibility and safe persistence for the baseline simulation."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from invLocAd import (
    load_simulation_results,
    plot_simulation_results,
    run_simulation,
)
from invLocAd.artifacts import load_archive, save_archive
from invLocAd.cli import main
from invLocAd.parameter_loading.load_protocol_parameters import (
    load_protocol_parameters,
)


@pytest.mark.parametrize("condition,physical_steps", [("split", 6400), ("tied", 2000)])
@pytest.mark.parametrize("requested_steps,effective_steps", [(2, 2), (3, 4), (4, 4)])
def test_protocol_iteration_budget_after_even_step_normalization(
    condition, physical_steps, requested_steps, effective_steps
):
    parameters = load_protocol_parameters(
        {"Learner": {"numStepsPerIteration": requested_steps}},
        split_or_tied=condition,
    )
    assert parameters["Learner"]["numStepsPerIteration"] == effective_steps
    assert parameters["numStepsToLearn"] == physical_steps
    assert parameters["num_iterations"] == parameters["numIterations"]
    assert parameters["num_iterations"] * effective_steps == physical_steps


def assert_numerically_identical(first, second):
    """Assert that two simulation results have identical arrays, diagnostics and summaries."""
    for name in (
        "controller_history_8d",
        "controller_history_10d",
        "final_state",
        "emet_total_store",
        "emet_per_time_store",
        "t_step_store",
        "ework_pushoff_store",
        "ework_heelstrike_store",
        "edot_store_iteration_average",
        "t_total_iteration_store",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
    for name in ("t_store", "state_store"):
        assert len(getattr(first, name)) == len(getattr(second, name))
        for a, b in zip(getattr(first, name), getattr(second, name)):
            np.testing.assert_array_equal(a, b)
    for name in first.learning_diagnostics:
        np.testing.assert_array_equal(
            first.learning_diagnostics[name], second.learning_diagnostics[name]
        )
    for name in first.summary:
        np.testing.assert_array_equal(first.summary[name], second.summary[name])


@pytest.mark.parametrize("condition,total,boundary", [("split", 360, 310), ("tied", 120, 65)])
def test_stochastic_resume_across_protocol_transition(tmp_path, condition, total, boundary):
    complete = run_simulation(
        seed=42, num_iterations=total, split_or_tied=condition, make_plots=False
    )
    checkpoint = tmp_path / "checkpoint.npz"
    partial = run_simulation(
        seed=42,
        num_iterations=total,
        split_or_tied=condition,
        stop_after=boundary,
        checkpoint_path=checkpoint,
        checkpoint_every=60,
        make_plots=False,
    )
    assert partial.metadata["completed_iterations"] == boundary
    assert partial.metadata["learning_complete"] is False
    snapshot = load_archive(checkpoint, "checkpoint")
    assert np.linalg.norm(snapshot["param_fixed"]["storedmemory"]["controlSlopeVsContext"]) > 0
    assert np.linalg.norm(snapshot["learning_state"]["adynamics_model_now"]) > 0
    resumed = run_simulation(resume_from=checkpoint, make_plots=False)
    assert_numerically_identical(complete, resumed)
    assert (
        complete.learning_diagnostics["t_end_store"][-1]
        > complete.param_fixed["imposedFootSpeeds"]["tList"][3 if condition == "split" else 1]
    )
    assert np.all(np.isfinite(complete.controller_history_8d))


def test_repeatability_and_no_global_random_state_change():
    np.random.seed(812)
    before = np.random.get_state()
    first = run_simulation(seed=17, num_iterations=40, make_plots=False)
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    second = run_simulation(seed=17, num_iterations=40, make_plots=False)
    assert_numerically_identical(first, second)
    other = run_simulation(seed=18, num_iterations=40, make_plots=False)
    assert not np.array_equal(first.controller_history_8d, other.controller_history_8d)


def test_resume_extension_and_completed_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    short = run_simulation(seed=3, num_iterations=8, checkpoint_path=checkpoint, make_plots=False)
    same = run_simulation(resume_from=checkpoint, checkpoint_path=checkpoint, make_plots=False)
    assert_numerically_identical(short, same)
    extended = run_simulation(resume_from=checkpoint, num_iterations=40, make_plots=False)
    full = run_simulation(seed=3, num_iterations=40, make_plots=False)
    assert_numerically_identical(extended, full)


@pytest.mark.parametrize("condition", ["split", "tied"])
def test_safe_results_roundtrip_and_replot(tmp_path, monkeypatch, condition):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    result = run_simulation(
        seed=1, num_iterations=1, split_or_tied=condition, make_plots=False, output_dir=tmp_path
    )
    loaded = load_simulation_results(tmp_path / "run.npz")
    assert_numerically_identical(result, loaded)
    with np.load(tmp_path / "run.npz", allow_pickle=False) as saved:
        assert all(not saved[key].dtype.hasobject for key in saved.files)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["seed"] == 1
    assert metadata["artifact_role"] == "baseline_model_example_not_a_published_fit"
    assert (tmp_path / f"{condition}_belt_summary.png").is_file()
    before = deepcopy(loaded.summary)
    plot_simulation_results(loaded, tmp_path / "replotted")
    for key in before:
        np.testing.assert_array_equal(before[key], loaded.summary[key])


@pytest.fixture
def checkpoint(tmp_path):
    """Return the path of a seed-42 checkpoint paused after 8 of 10 iterations."""
    path = tmp_path / "checkpoint.npz"
    run_simulation(seed=42, num_iterations=10, stop_after=8, checkpoint_path=path, make_plots=False)
    return path


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 41},
        {"split_or_tied": "tied"},
        {"speed_protocol": "single speed"},
        {"transition_time": 10},
        {"split_speed_diff_ms": 0.7},
        {"num_iterations": 7},
    ],
)
def test_incompatible_resume_rejected(checkpoint, override):
    with pytest.raises(ValueError):
        run_simulation(resume_from=checkpoint, make_plots=False, **override)


@pytest.mark.parametrize(
    "corruption",
    [
        "length",
        "shape",
        "nan",
        "time",
        "memory",
        "config",
        "runtime",
        "rng",
        "rng_position",
        "rng_keys",
        "rng_cache",
    ],
)
def test_corrupted_checkpoint_rejected(checkpoint, corruption):
    payload = load_archive(checkpoint, "checkpoint")
    state = payload["learning_state"]
    if corruption == "length":
        state["edot_store"].pop()
    elif corruption == "shape":
        state["adynamics_energy_now"] = np.zeros((1, 4))
    elif corruption == "nan":
        state["g_energy_gradient_now"][0] = np.nan
    elif corruption == "time":
        state["t_start"] += 1
    elif corruption == "memory":
        payload["param_fixed"]["storedmemory"]["controlSlopeVsContext"] = np.zeros((1, 2))
    elif corruption == "config":
        payload["param_fixed"]["Learner"]["LearningRate"] *= 2
    elif corruption == "runtime":
        payload["metadata"]["numpy"] = "unsupported"
    elif corruption.startswith("rng_"):
        rng_state = list(payload["rng_state"])
        if corruption == "rng_position":
            rng_state[2] = 625
        elif corruption == "rng_keys":
            rng_state[1] = np.zeros(624, dtype=float)
        else:
            rng_state[4] = float("nan")
        payload["rng_state"] = tuple(rng_state)
    else:
        payload["rng_state"] = ("unsupported", np.array([1]))
    save_archive(checkpoint, "checkpoint", payload)
    with pytest.raises(ValueError):
        run_simulation(resume_from=checkpoint, make_plots=False)


def test_schema_and_pickle_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata_json=np.array(json.dumps({"schema_version": 999})))
    with pytest.raises(ValueError, match="schema"):
        load_archive(path, "checkpoint")
    np.savez(path, metadata_json=np.array({"pickled": "object"}, dtype=object))
    with pytest.raises(ValueError):
        load_archive(path, "checkpoint")


def test_atomic_checkpoint_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    save_archive(path, "checkpoint", {"value": np.arange(4)})
    before = path.read_bytes()

    def fail(*args, **kwargs):
        """Raise OSError as a write to a full disk would."""
        raise OSError("simulated full disk")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="full disk"):
        save_archive(path, "checkpoint", {"value": np.arange(8)})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "arguments",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"num_iterations": 0},
        {"num_iterations": 1.5},
        {"transition_time": 0},
        {"transition_time": float("nan")},
        {"transition_time": 500},
        {"split_speed_diff_ms": float("inf")},
        {"split_speed_diff_ms": -0.4},
        {"split_or_tied": "unknown"},
        {"split_or_tied": "tied", "split_speed_diff_ms": 0.4},
        {"checkpoint_every": 0},
        {"checkpoint_every": 5},
        {"stop_after": 4},
    ],
)
def test_invalid_inputs(arguments):
    with pytest.raises(ValueError):
        run_simulation(make_plots=False, **arguments)


def test_cli_resume_and_validation(tmp_path, capsys):
    path = tmp_path / "checkpoint.npz"
    assert (
        main(
            [
                "--seed",
                "42",
                "--iterations",
                "4",
                "--stop-after",
                "2",
                "--checkpoint",
                str(path),
                "--no-plots",
            ]
        )
        == 0
    )
    assert "paused" in capsys.readouterr().out
    assert main(["--resume-from", str(path), "--no-plots"]) == 0
    assert "Iterations: 4/4" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--iterations", "-1", "--no-plots"])
    assert error.value.code == 2
