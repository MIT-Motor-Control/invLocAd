"""Scientific convention checks and prepared-data numerical regressions."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from invLocAd.publication.fall_penalty_analysis import fit_energy_exponential
from invLocAd.publication.regenerated import (
    FIGURE2C_ROLLOUTS,
    TRIAL_AVERAGE_GRID,
    figure2c_model_tables,
    grid_finite_counts,
    normalize_raw_trajectory,
)
from invLocAd.publication.supplements import (
    ENERGY,
    compare_exponential_timescales,
    fall_penalty_energy_percent,
    matlab_movmean,
    recompute_s1_matlab,
    render_supplements,
    stored_fit_metrics,
)


def read_rows(path):
    """Return the rows of a CSV file as dictionaries."""
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_matlab_even_window_and_shrink_endpoints():
    signal = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # MATLAB movmean with window 4 averages samples i-2 through i+1.
    np.testing.assert_allclose(matlab_movmean(signal, 4), [1.5, 7 / 3, 15 / 4, 30 / 4, 28 / 3])
    np.testing.assert_allclose(matlab_movmean(signal, 1), signal)


def test_stored_fit_recovers_scale_without_refitting_shape():
    normalized = np.array([0.0, -10.0, -20.0, -30.0])
    signal = 2.5 * (1 + normalized / 100)
    r2, scale = stored_fit_metrics(signal, normalized)
    assert r2 == pytest.approx(1)
    assert scale == pytest.approx(2.5)
    r2_rescaled, scale_rescaled = stored_fit_metrics(signal * 3, normalized)
    assert r2_rescaled == pytest.approx(r2)
    assert scale_rescaled == pytest.approx(7.5)
    with pytest.raises(ValueError, match="equal-length"):
        stored_fit_metrics(signal[:-1], normalized)
    with pytest.raises(ValueError, match="zero variance"):
        stored_fit_metrics(np.ones(4), normalized)


def test_figure2b_normalization_preserves_matlab_trim_and_padding():
    signal = np.linspace(3, 1, 150)
    padded = np.r_[np.nan, 0, signal, np.inf, 0]
    smooth = matlab_movmean(signal, 100)
    # Zero and nonfinite padding is dropped. For the 150 remaining samples the smoothed
    # trajectory starts at sample max(30, round(0.03 * 150)) = 30 (Python index 29).
    trimmed = smooth[29:]
    expected = (trimmed - trimmed[:120].mean()) / trimmed[:120].mean() * 100
    np.testing.assert_array_equal(normalize_raw_trajectory(padded), expected)
    with pytest.raises(ValueError, match="startup-trim"):
        normalize_raw_trajectory(np.ones(20))


def test_grid_finite_counts_exclude_padded_participant(tmp_path):
    path = tmp_path / TRIAL_AVERAGE_GRID
    path.parent.mkdir(parents=True)
    grid = np.full((3, 9, 20, 20), np.nan)
    for condition, count in enumerate((8, 9, 9)):
        grid[condition, :count, 0, :10] = 1
        if condition < 2:
            grid[condition, :count, 1, :] = 2
        if condition == 0:
            grid[condition, :count, 2, :] = 3
    grid[0, 8] = 1  # Padding slot of the 8-participant condition, not a 27th participant.
    savemat(path, {"rmse_matrix": grid, "exp_subj_num": [8, 9, 9]})
    assert [r["admissible_cells"] for r in grid_finite_counts(tmp_path)] == [10, 30, 50]


def test_figure2c_trials_use_tied_window_nonfall_medians_and_shared_cells(tmp_path):
    params = np.empty((3, 9), object)
    for condition in range(3):
        for subject in range(9):
            params[condition, subject] = {"learning_rate": 0.001, "sym_cost": 0.1}
    original = tmp_path / "original/2d_sweep_trial50_param_result.mat"
    original.parent.mkdir(parents=True)
    savemat(original, {"static_params": params, "exp_subj_num": [8, 9, 9],
                       "learning_pool": [0.001, 0.002], "sym_cost_pool": [0.1, 0.2]})
    sweep = tmp_path / FIGURE2C_ROLLOUTS
    sweep.mkdir(parents=True)
    last = None
    for condition, (slow, fast) in enumerate(((0.8, 1.2), (0.65, 1.35), (0.5, 1.5)), 1):
        for trial in range(1, 51):
            energy = np.full(300, 2.0)
            energy[-100:] = 2 * (1 + trial / 100 + condition / 10)
            last = sweep / (f"model_split45_trans20_slow{slow}_fast{fast}_noise0.01_"
                            f"learning0.001_energysym0.1_swingcost1_trial{trial}.mat")
            savemat(last, {"energy": energy, "fall_flag": [int(trial == 50), 0]})
    trials, participants, summary = figure2c_model_tables(tmp_path)
    assert len(trials) == 150 and len(participants) == 26
    np.testing.assert_allclose([r["mean_energy_change_pct"] for r in summary], [35, 45, 55])
    assert [r["participant_weighted_falls"] for r in summary] == [8, 9, 9]
    assert [r["unique_cells"] for r in summary] == [1, 1, 1]
    assert sum(r["fall_flag"] for r in trials) == 3
    assert all(r["energy_change_pct"] is None for r in trials if r["fall_flag"])
    for flag in ([0.5, 0], [np.nan, 0], [0]):
        savemat(last, {"energy": energy, "fall_flag": flag})
        with pytest.raises(ValueError, match="fall flag"):
            figure2c_model_tables(tmp_path)
    savemat(last, {"energy": np.r_[np.nan, energy[1:]], "fall_flag": [1, 0]})
    with pytest.raises(ValueError, match="energy signal"):
        figure2c_model_tables(tmp_path)
    savemat(last, {"energy": energy, "fall_flag": [1, 0]})
    off_grid = last.with_name(last.name.replace("learning0.001_", "learning0.0014_"))
    last.rename(off_grid)
    with pytest.raises(ValueError, match="does not identify"):
        figure2c_model_tables(tmp_path)
    off_grid.rename(last)
    last.unlink()
    with pytest.raises(FileNotFoundError, match="1 Figure 2C rollouts"):
        figure2c_model_tables(tmp_path)


def test_fall_penalty_energy_symmetric_window_and_onset():
    raw = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # Smoothing with window=2 is a centered three-point mean with shrinking endpoints.
    smooth = np.array([1.5, 7 / 3, 14 / 3, 28 / 3, 12.0])
    expected = (smooth - smooth[:2].mean()) / smooth[:2].mean() * 100
    np.testing.assert_allclose(
        fall_penalty_energy_percent(raw, window=2, onset_window=2), expected
    )
    with pytest.raises(ValueError, match="baseline"):
        fall_penalty_energy_percent(np.zeros(5))


def test_fall_penalty_refit_recovers_known_exponential_and_rejects_incomplete_input():
    time = np.arange(2700)
    values = -22 + 27 * np.exp(-time / 350)
    np.testing.assert_allclose(fit_energy_exponential(values), values, atol=1e-8)
    with pytest.raises(ValueError, match="2700"):
        fit_energy_exponential(values[:-1])
    with pytest.raises(ValueError, match="ten finite"):
        fit_energy_exponential(np.full(2700, np.nan))


def test_timescale_diagnostic_detects_known_second_component():
    time = np.arange(1, 501)
    values = 3 + 2 * np.exp(-time / 15) + 4 * np.exp(-time / 200)
    result = compare_exponential_timescales(values)
    assert result["r2_double_refit_raw"] == pytest.approx(1, abs=1e-10)
    assert result["delta_r2_reported"] > 0.01
    with pytest.raises(ValueError, match="nonconstant"):
        compare_exponential_timescales(np.ones(500))


def test_missing_sources_fail_instead_of_reporting_success(tmp_path):
    for name in ("figure2", "s1", "s2", "s4", "r2"):
        with pytest.raises(FileNotFoundError, match="Required publication input"):
            render_supplements(tmp_path / "absent", tmp_path / "out", [name])
    with pytest.raises(ValueError, match="Unknown"):
        render_supplements(tmp_path, tmp_path / "out", ["unknown"])
    assert render_supplements(tmp_path, tmp_path / "out", []) == []


def test_figure2c_recomputes_experimental_values_but_never_claims_missing_model(tmp_path):
    root = tmp_path / "data"
    energy = root / ENERGY
    energy.parent.mkdir(parents=True)
    tied, split = {}, {}
    for i, count in enumerate((8, 9, 9), 1):
        tied[f"condition_{i}"] = np.full((count, 200), 2.0)
        split[f"condition_{i}"] = np.full((count, 200), 2.0 * (1 + i / 10))
    savemat(energy, {"split_energy": split, "tied1_energy": tied})
    result = render_supplements(root, tmp_path / "out", ["figure2"])[0]
    # Figure 2B lacks its regenerated inputs, so the merged record is blocked.
    assert result["id"] == "figure2"
    assert result["status"] == "blocked"
    assert result["details"]["panels"] == {"2B": "blocked", "2C": "partial"}
    assert result["caveats"][0].startswith("Figure 2B is blocked: Required publication input")
    figure2c = result["details"]["figure2c"]
    assert figure2c["missing_series"] == ["inverse_adaptation_model"]
    np.testing.assert_allclose(figure2c["experimental_means"], [10.0, 20.0, 30.0])
    assert result["outputs"] and all(Path(path).is_file() for path in result["outputs"])
    assert "figure2c_partial.png" in {Path(path).name for path in result["outputs"]}
    assert len(read_rows(tmp_path / "out/tables/figure2c_experimental_participants.csv")) == 26
    assert not any("model" in Path(path).stem for path in result["outputs"])


def test_figure2_merges_parts_with_least_complete_status(tmp_path, monkeypatch):
    from invLocAd.publication import supplements

    def part(name, status):
        """Return a stand-in renderer whose record is labeled ``name``."""
        return lambda root, output: supplements._record(
            "figure2", [f"{name}.png"], [f"{name}.mat"], name, [f"{name} note"], status, label=name
        )

    for statuses, expected in (
        (("passed", "passed"), "passed"),
        (("passed", "partial"), "partial"),
        (("blocked", "partial"), "blocked"),
        (("failed", "blocked"), "failed"),
    ):
        monkeypatch.setattr(supplements, "_figure2b", part("2b", statuses[0]))
        monkeypatch.setattr(supplements, "_figure2c", part("2c", statuses[1]))
        record = render_supplements(tmp_path, tmp_path / "out", ["figure2"])[0]
        assert record["status"] == expected
        assert record["outputs"] == ["2b.png", "2c.png"]
        assert record["caveats"] == ["2b note", "2c note"]
        assert record["details"]["sources"] == ["2b.mat", "2c.mat"]
        assert record["details"]["panels"] == dict(zip(("2B", "2C"), statuses))
        assert record["details"]["figure2b"]["label"] == "2b"
        assert record["details"]["figure2c"]["label"] == "2c"


def test_figure2_part_error_does_not_suppress_other_part(tmp_path, monkeypatch):
    from invLocAd.publication import supplements

    def broken(root, output):
        """Raise an analysis error in place of Figure 2B."""
        raise ValueError("bad trajectory")

    monkeypatch.setattr(supplements, "_figure2b", broken)
    monkeypatch.setattr(
        supplements,
        "_figure2c",
        lambda root, output: supplements._record("figure2", ["2c.png"], ["2c.mat"], "2c"),
    )
    output = tmp_path / "out"
    record = render_supplements(tmp_path, output, ["figure2"])[0]
    assert record["status"] == "failed"
    assert record["details"]["panels"] == {"2B": "failed", "2C": "passed"}
    assert record["outputs"] == ["2c.png"]
    assert "ValueError: bad trajectory" in record["error"]
    assert "bad trajectory" in (output / "logs/figure2b.txt").read_text()

def test_optional_matlab_requires_inputs_before_starting_process(tmp_path):
    with pytest.raises(FileNotFoundError, match="trialavg.mat"):
        recompute_s1_matlab(tmp_path / "data", tmp_path / "out", "no-such-matlab")


@pytest.fixture
def publication_data():
    """Return the prepared publication data folder, or skip without the full dataset."""
    root = Path(__file__).resolve().parents[1] / "data/publication"
    if not (root / "manifest.json").is_file():
        pytest.skip("Run invLocAd-publication prepare to enable prepared-data regressions")
    if json.loads((root / "manifest.json").read_text()).get("dataset_kind") == "zenodo_experimental":
        pytest.skip("Model reference regressions require the companion publication-data.zip")
    return root


@pytest.mark.integration
def test_all_supplement_values_against_publication_sources(publication_data, tmp_path):
    if not (publication_data / "extensions/s5/analysis_summary.json").is_file():
        pytest.skip("Fall-penalty analysis sources are not in the prepared data")
    results = {r["id"]: r for r in render_supplements(publication_data, tmp_path)}
    has_rollouts = (publication_data / FIGURE2C_ROLLOUTS).is_dir()
    assert {key: r["status"] for key, r in results.items()} == {
        "figure2": "passed" if has_rollouts else "partial",
        "s1": "passed",
        "s2": "passed",
        "s4": "passed",
        "r2": "passed",
    }
    assert results["figure2"]["details"]["panels"] == {
        "2B": "passed",
        "2C": "passed" if has_rollouts else "partial",
    }
    figure2 = results["figure2"]["details"]["figure2b"]
    assert figure2["n_data"] == figure2["n_model"] == 9
    assert figure2["samples_per_trajectory"] == 270
    assert figure2["pearson_r"] == pytest.approx(0.9445316549299377)
    assert figure2["rmse_pct"] == pytest.approx(3.021338300307935)
    assert figure2["data_plateau_pct"] == pytest.approx(-26.373025592007593)
    assert figure2["model_plateau_pct"] == pytest.approx(-25.333063008683794)
    group_fits = read_rows(tmp_path / "tables/figure2b_group_fits.csv")
    np.testing.assert_allclose([float(r["tau_exposure_pct"]) for r in group_fits],
                               [10.483039601467793, 11.685268815874876], rtol=1e-7)
    assert len(read_rows(tmp_path / "tables/figure2b_participants.csv")) == 18 * 270
    s1 = results["s1"]["details"]
    assert s1["mode"] == "saved_reference_replot"
    assert s1["median_condition_specific"] == pytest.approx(8.87651802846845)
    assert s1["median_common_size_matched"] == pytest.approx(16.9351104934933)
    assert s1["wilcoxon_two_sided_exact_scipy"] == pytest.approx(2.9802322387695312e-8)
    assert s1["wilcoxon_two_sided_matlab_signrank_default"] == pytest.approx(8.29809930635733e-6)
    if s1["grid_sensitivity_available"]:
        rows = s1["grid_sensitivity_summary"]
        assert [int(r["admissible_cells"]) for r in rows] == [138, 307, 384]
        np.testing.assert_allclose([float(r["median_common_size_matched"]) for r in rows],
                                   [16.93511049349334, 21.07943040972788, 25.09174114501218])
    figure2c = results["figure2"]["details"]["figure2c"]
    if has_rollouts:
        np.testing.assert_allclose(figure2c["model_means"],
                                   [30.681741357807397, 106.79465294047486, 318.6162033053468])
        assert figure2c["unique_model_trials"] == 1150
        summary = read_rows(tmp_path / "tables/figure2c_model_summary.csv")
        assert [int(r["participant_weighted_falls"]) for r in summary] == [193, 253, 306]
        assert [int(r["unique_cells"]) for r in summary] == [8, 8, 7]
    assert results["s2"]["details"]["max_reference_error"] < 1e-12
    assert results["s2"]["details"]["participants"] == 20
    assert results["r2"]["details"]["reference_verified"]
    assert results["r2"]["details"]["max_coefficient_scale_disagreement_pct"] < 1e-9
    np.testing.assert_allclose(
        figure2c["experimental_means"],
        [-8.571140401974557, -0.5279256055629971, 20.61772447061896],
    )
    selected = results["s4"]["details"]["selections"]
    assert [row["fall_weight"] for row in selected if row["model"] == "C"] == [0.0, 10.0, 100.0]
    assert all(row["successful_seeds"] == 50 for row in selected)
    assert results["s4"]["details"]["analysis"]["cached_cells_refitted"] == 750
    summary = read_rows(tmp_path / "tables/s4_loso_summary.csv")
    pooled = {r["model"]: float(r["mean_rmse"]) for r in summary if r["delta_mps"] == "pooled"}
    assert pooled == pytest.approx(
        {"C": 8.676033071530686, "E": 10.204003450301913, "noise_floor": 8.948329842079502},
        abs=1e-8,
    )
    high = next(
        r for r in read_rows(tmp_path / "tables/s4_loso_tests.csv") if r["delta_mps"] == "1.0"
    )
    assert float(high["wilcoxon_two_sided_p"]) == pytest.approx(0.0078125)
    assert float(high["holm_p_within_three_conditions"]) == pytest.approx(0.0234375)
    folds = read_rows(tmp_path / "tables/s4_loso_folds.csv")
    assert len(folds) == 26
    assert all(
        int(r["training_participants"]) == (7 if r["delta_mps"] == "0.4" else 8) for r in folds
    )
    rmse = read_rows(tmp_path / "tables/inverse_fit_rmse_summary.csv")
    assert float(rmse[0]["mean_rmse"]) == pytest.approx(0.3714788929774202)
    timescale = read_rows(tmp_path / "tables/r2_timescale_summary.csv")
    np.testing.assert_allclose(
        [float(r["mean_delta_r2"]) for r in timescale],
        [0.0147342521, 0.0408614680, 0.0268716229, 0.0551922854, 0.0165944674, 0.0112064244],
        atol=2e-5,
        rtol=0,
    )
    assert all(Path(path).is_file() for result in results.values() for path in result["outputs"])


@pytest.mark.integration
def test_r2_rejects_changed_reference(publication_data, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "original").symlink_to(publication_data / "original", target_is_directory=True)
    reference = root / "reference_tables/expfit_r2.csv"
    reference.parent.mkdir()
    rows = read_rows(publication_data / "reference_tables/expfit_r2.csv")
    rows[0]["r2_experimental"] = "0.1234"
    with reference.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="R² differs"):
        render_supplements(root, tmp_path / "out", ["r2"])
