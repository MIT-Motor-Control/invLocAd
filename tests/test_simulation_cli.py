"""Command-line simulations and checkpoint persistence."""

from types import SimpleNamespace

import numpy as np
import pytest

from invLocAd import cli, load_simulation_results, run_simulation
from invLocAd.artifacts import load_archive


def test_demo_saves_reusable_results_without_interactive_windows(tmp_path, monkeypatch, capsys):
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("The demo must not open windows"))
    assert cli.main(["demo"]) == 0
    output = tmp_path / "outputs/demo"
    result = load_simulation_results(output / "run.npz")
    checkpoint = load_archive(output / "checkpoint.npz", "checkpoint")
    assert result.metadata["seed"] == 42
    assert result.metadata["completed_iterations"] == 40
    assert result.metadata["target_iterations"] == 40
    assert checkpoint["metadata"]["completed_iterations"] == 40
    assert (output / "metadata.json").is_file()
    assert (output / "checkpoint.metadata.json").is_file()
    assert list(output.glob("*.png"))
    assert "may finish before split-belt adaptation begins" in capsys.readouterr().out


def test_run_pause_and_resume_inherit_state_without_changing_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--seed", "17", "--iterations", "8", "--stop-after", "3"]) == 0
    source = tmp_path / "outputs/run/checkpoint.npz"
    before = source.read_bytes()
    assert load_archive(source, "checkpoint")["metadata"]["completed_iterations"] == 3
    assert cli.main(["resume", str(source)]) == 0
    result = load_simulation_results(tmp_path / "outputs/resumed/run.npz")
    assert result.metadata["seed"] == 17
    assert result.metadata["target_iterations"] == 8
    assert result.metadata["completed_iterations"] == 8
    assert source.read_bytes() == before
    assert (tmp_path / "outputs/resumed/checkpoint.npz").is_file()
    fresh = run_simulation(seed=17, num_iterations=8, make_plots=False)
    np.testing.assert_array_equal(result.controller_history_8d, fresh.controller_history_8d)


@pytest.fixture
def simulation_calls(monkeypatch):
    """Replace ``run_simulation`` with a stub; return the list of its keyword arguments."""
    calls = []

    def simulate(**kwargs):
        """Record the keyword arguments and return a minimal completed result."""
        calls.append(kwargs)
        return SimpleNamespace(
            summary={"average_energy_rate": 0.1, "final_average_energy_rate": 0.1},
            metadata={
                "learning_complete": True,
                "seed": 42,
                "completed_iterations": 40,
                "target_iterations": 40,
            },
            param_fixed={
                "SplitOrTied": "split",
                "speedProtocol": "classic split belt",
                "splitSpeedDiffMS": 1.0,
            },
        )

    monkeypatch.setattr(cli, "run_simulation", simulate)
    return calls


def test_run_preserves_model_defaults_and_supports_custom_output(simulation_calls, tmp_path):
    output = tmp_path / "my-run"
    checkpoint = tmp_path / "my-checkpoint.npz"
    assert (
        cli.main(
            [
                "run",
                "--output-dir",
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-every",
                "10",
                "--show-plots",
            ]
        )
        == 0
    )
    request = simulation_calls[0]
    for name in (
        "seed",
        "num_iterations",
        "split_or_tied",
        "speed_protocol",
        "transition_time",
        "split_speed_diff_ms",
    ):
        assert request[name] is None
    assert request["output_dir"] == output
    assert request["checkpoint_path"] == checkpoint
    assert request["checkpoint_every"] == 10
    assert request["make_plots"] is True


def test_options_without_command_save_nothing_by_default(simulation_calls):
    assert cli.main(["--seed", "7", "--iterations", "12"]) == 0
    request = simulation_calls[0]
    assert request["seed"] == 7
    assert request["num_iterations"] == 12
    assert request["output_dir"] is None
    assert request["checkpoint_path"] is None
    assert request["checkpoint_every"] is None
    assert request["make_plots"] is True


def test_bare_command_shows_help_without_starting_a_simulation(simulation_calls, capsys):
    assert cli.main([]) == 0
    help_text = capsys.readouterr().out
    assert "invLocAd-simulate demo" in help_text
    assert "invLocAd-simulate resume" in help_text
    assert not simulation_calls


def test_resume_never_overwrites_source_by_default(tmp_path, monkeypatch, simulation_calls, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "outputs/resumed/checkpoint.npz"
    with pytest.raises(SystemExit) as error:
        cli.main(["resume", str(source)])
    assert error.value.code == 2
    assert "Choose another --output-dir" in capsys.readouterr().err
    assert not simulation_calls
    assert cli.main(["resume", str(source), "--checkpoint", str(source)]) == 0
    assert simulation_calls[0]["resume_from"] == source
    assert simulation_calls[0]["checkpoint_path"] == source


def test_resume_results_cannot_overwrite_input_checkpoint(tmp_path, simulation_calls):
    source = tmp_path / "run.npz"
    with pytest.raises(SystemExit) as error:
        cli.main(["resume", str(source), "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert not simulation_calls
