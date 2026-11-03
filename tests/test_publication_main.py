"""Numerical regression and data-loss guards for cached publication reproduction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from invLocAd.publication.data import prepare_data, sha256
from invLocAd.publication.main_figures import (
    _rank_sum_probability,
    _sanchez,
    fall_filename,
    fall_probability_tests,
    fall_tables,
    figure1_tables,
    fit_exp1,
    landscape_data,
    landscape_tables,
    matlab_box_summary,
    parameter_statistics,
    render_main,
)
from invLocAd.publication.trajectory import (
    SOURCE_FILES,
    read_storage,
    source_energy_cost,
    trajectory_source_tables,
)


def test_prepare_exact_energy_recovery_and_idempotence(tmp_path):
    bundle = tmp_path / "bundle"
    original = bundle / "code/original_data"
    original.mkdir(parents=True)
    expected = {
        name: {"condition_1": np.array([[1.23456789, 4.321]])}
        for name in ("split_energy", "tied1_energy", "tied2_energy")
    }
    source = original / "2d_sweep_trial50_param_result.mat"
    savemat(source, expected)
    before = sha256(source)
    destination = tmp_path / "prepared"
    first = prepare_data(bundle, destination)
    energy = destination / "original/data/cosmed/processed/processedEnergyData.mat"
    recovered = loadmat(energy, simplify_cells=True)
    for name in expected:
        np.testing.assert_array_equal(
            recovered[name]["condition_1"], expected[name]["condition_1"].ravel()
        )
    second = prepare_data(bundle, destination)
    assert first["files"] == second["files"]
    assert sha256(source) == before
    entry = next(
        item for item in first["files"] if item["path"].endswith("processedEnergyData.mat")
    )
    assert entry["sha256"] == hashlib.sha256(energy.read_bytes()).hexdigest()
    assert entry["source_sha256"] == before
    assert "Extracted exact" in entry["transformation"]
    assert str(tmp_path) not in json.dumps(first)


def test_conflicting_fall_archive_fails_before_copy(tmp_path):
    bundle = tmp_path / "bundle"
    falls = bundle / "code/original_data/data/fall probability"
    falls.mkdir(parents=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    (falls / "trial.mat").write_bytes(b"author")
    (archive / "trial.mat").write_bytes(b"different")
    target = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Conflicting scientific input"):
        prepare_data(bundle, target, archive)
    assert not target.exists()


def test_prepare_rejects_source_and_destination_escapes(tmp_path):
    bundle = tmp_path / "bundle"
    original = bundle / "code/original_data"
    original.mkdir(parents=True)
    unrelated = tmp_path / "outside.mat"
    unrelated.write_bytes(b"outside")
    (original / "escape.mat").symlink_to(unrelated)
    with pytest.raises(ValueError, match="escapes its source"):
        prepare_data(bundle, tmp_path / "prepared")
    (original / "escape.mat").unlink()
    (original / "cached.mat").write_bytes(b"scientific input")
    target = tmp_path / "prepared"
    target.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (target / "original").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes its destination"):
        prepare_data(bundle, target)
    assert not list(elsewhere.iterdir())
    with pytest.raises(ValueError, match="not contain"):
        prepare_data(bundle, tmp_path)


def test_prepare_refuses_omitted_previously_imported_archive(tmp_path):
    bundle, private, target = tmp_path / "bundle", tmp_path / "private", tmp_path / "prepared"
    (bundle / "code/original_data").mkdir(parents=True)
    (private / "Data").mkdir(parents=True)
    (private / "Data/split_energy.csv").write_text("participant,energy\n1,3.1\n")
    prepare_data(bundle, target, private_archive=private)
    before = (target / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="repeat all original source"):
        prepare_data(bundle, target)
    assert (target / "manifest.json").read_bytes() == before
    assert (target / "extensions/experimental_csv/split_energy.csv").is_file()


def test_matlab_box_percentiles_and_whiskers():
    box = matlab_box_summary(np.arange(8))
    assert [box["q1"], box["median"], box["q3"]] == [1.5, 3.5, 5.5]
    box = matlab_box_summary([0, 1, 2, 3, 4, 5, 100])
    assert [box["whisker_low"], box["whisker_high"], box["hidden_outliers"]] == [0, 5, 1]


def test_sanchez_preserves_shared_energy_indices_and_signal_specific_long_windows(tmp_path):
    folder = tmp_path / "original/data/sanchez data"
    folder.mkdir(parents=True)
    energy = {
        f"P{i}": {"Base": np.full(200, 2.0), "Split": 2 + 0.001 * np.arange(1500)}
        for i in range(1, 16)
    }
    sla = {
        f"P{i}": {"Base": np.full(300, 0.1), "Split": 0.1 + 0.0001 * np.arange(2400)}
        for i in range(1, 16)
    }
    savemat(folder / "sanchez_energy.mat", {"MetabolicPower": energy})
    savemat(folder / "sanchez_sla.mat", {"StepLengthAsymmetry": sla})
    short, long = _sanchez(tmp_path)[:2]
    assert short["energy_window_first_sample"] == short["sla_window_first_sample"] == 101
    assert short["energy_window_last_sample"] == short["sla_window_last_sample"] == 200
    np.testing.assert_allclose(
        [short["energy_change_pct"], short["sla_change_percentage_points"]],
        [7.475, 1.495],
        atol=1e-12,
    )
    assert long["energy_window_first_sample"] == 1401
    assert long["sla_window_first_sample"] == 2301
    np.testing.assert_allclose(
        [long["energy_change_pct"], long["sla_change_percentage_points"]],
        [72.475, 23.495],
        atol=1e-12,
    )
    energy["P1"]["Base"][0] = np.nan
    savemat(folder / "sanchez_energy.mat", {"MetabolicPower": energy})
    with pytest.raises(ValueError, match="finite values"):
        _sanchez(tmp_path)


def test_exact_rank_sum_includes_equal_allocations_with_ties():
    # Pooled midranks are (2, 2, 2, 4). Three of the six equiprobable allocations of the
    # first sample give the minimum rank sum of 4, so P(W <= 4) = 0.5.
    statistic, p, method = _rank_sum_probability([1, 1], [1, 2], "less")
    assert statistic == 4
    assert p == 0.5
    assert "exact" in method
    assert _rank_sum_probability([1, 2], [1, 1], "greater")[1] == 0.5


@pytest.mark.parametrize(
    "content",
    [
        "time x\n0 1\n1 2\n",  # No storage header terminator.
        "endheader\ntime x x\n0 1 2\n1 2 3\n",  # Ambiguous columns.
        "endheader\ntime x\n0 1\n0 2\n",  # Duplicate times.
        "endheader\ntime x\n0 1\n1 nan\n",  # Missing scientific value.
    ],
)
def test_storage_rejects_incomplete_or_ambiguous_sources(tmp_path, content):
    path = tmp_path / "source.sto"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_storage(path)


def test_storage_retains_column_labels_and_values(tmp_path):
    path = tmp_path / "source.sto"
    path.write_text("version=3\nendheader\ntime\t/forceset/test|activation\n0\t0.1\n1\t0.2\n")
    labels, values = read_storage(path)
    assert labels == ["time", "/forceset/test|activation"]
    np.testing.assert_array_equal(values, [[0, 0.1], [1, 0.2]])
    assert trajectory_source_tables(tmp_path) == []


def test_exp1_recovers_known_nonlinear_model():
    x = np.linspace(0.4, 1, 7)
    fit = fit_exp1(x, 1200 * np.exp(-1.8 * x))
    np.testing.assert_allclose([fit["a"], fit["b"]], [1200, -1.8], rtol=1e-10)
    assert fit["sse"] < 1e-18
    with pytest.raises(ValueError, match="distinct speeds"):
        fit_exp1([1, 1, 1], [2, 3, 4])


def test_missing_fall_trial_blocks_without_changing_denominator(tmp_path):
    with pytest.raises(FileNotFoundError, match="1,750 trials: 1750 missing"):
        fall_tables(tmp_path)
    out = tmp_path / "out"
    record = render_main(tmp_path, out, ["figure3"])[0]
    assert record["status"] == "blocked"
    assert record["outputs"] == []
    assert not out.exists()


def test_inconsistent_data_is_not_reported_as_missing(tmp_path, monkeypatch):
    from invLocAd.publication import main_figures

    def inconsistent(_):
        """Raise the error of an inconsistent input."""
        raise ValueError("Scientific shape mismatch")

    monkeypatch.setattr(main_figures, "figure1_tables", inconsistent)
    with pytest.raises(ValueError, match="Scientific shape mismatch"):
        render_main(tmp_path, tmp_path / "out", ["figure1"])


@pytest.fixture
def publication_data():
    """Return the prepared publication data folder, or skip without the full dataset."""
    path = Path(__file__).resolve().parents[1] / "data/publication"
    if not (path / "manifest.json").is_file():
        pytest.skip("Prepared publication data is required for audited numerical regression")
    if (
        json.loads((path / "manifest.json").read_text()).get("dataset_kind")
        == "zenodo_experimental"
    ):
        pytest.skip(
            "Full-precision reference regressions require the companion publication-data.zip"
        )
    return path


@pytest.mark.integration
def test_figure1_audited_participant_values(publication_data):
    tables = figure1_tables(publication_data)
    expected = {
        ("energy_participants", "short"): [4.340751418297072, 21.94825582391544, 45.10066097685599],
        ("energy_participants", "long"): [
            -8.571140401974557,
            -0.5279256055629971,
            20.61772447061896,
        ],
        ("sla_participants", "short"): [
            -2.9403338296665384,
            -4.19500993594461,
            -11.775913304592443,
        ],
        ("sla_participants", "long"): [-1.9695521054905414, 2.193997111107009, 0.7645353372066099],
    }
    for (name, exposure), values in expected.items():
        column = (
            "energy_change_pct" if name.startswith("energy") else "sla_change_percentage_points"
        )
        actual = [
            np.mean(
                [
                    r[column]
                    for r in tables[name]
                    if r["condition"] == condition and r["exposure"].startswith(exposure)
                ]
            )
            for condition in (1, 2, 3)
        ]
        np.testing.assert_allclose(actual, values, atol=1e-11, rtol=1e-12)
    assert [
        sum(r["significant_p_lt_005"] for r in tables["paired_tests"] if r["condition"] == c)
        for c in (1, 2, 3)
    ] == [1, 2, 3]
    assert len(tables["energy_participants"]) == 52
    assert len(tables["sla_participants"]) == 46
    assert len(tables["energy_timeseries"]) == 156


@pytest.mark.integration
def test_figure1e_significant_intervals_match_paper(publication_data):
    """Figure 1E: significant decreases through 9, 18 and 27 min at 0.4, 0.7 and 1.0 m/s."""
    tables = figure1_tables(publication_data)
    expected_p = [
        [
            0.001801405675968146,
            0.13241463971933762,
            0.9331854723863924,
            0.22325456930407078,
            0.34813046140735926,
        ],
        [
            0.0010360511764730398,
            0.019063204755437917,
            0.15483546906755108,
            0.9487549844426032,
            0.1712213902754225,
        ],
        [
            0.0009260530020777565,
            0.005592057834613511,
            0.013640968202585028,
            0.5898225546808779,
            0.9699553228646448,
        ],
    ]
    for condition, minutes, p_values in zip((1, 2, 3), ([9], [9, 18], [9, 18, 27]), expected_p):
        rows = [r for r in tables["paired_tests"] if r["condition"] == condition]
        np.testing.assert_allclose(
            [r["p_two_sided"] for r in rows], p_values, rtol=1e-12, atol=1e-14
        )
        assert [r["current_min"] for r in rows if r["significant_p_lt_005"]] == minutes
        assert all(r["t_statistic"] < 0 for r in rows if r["significant_p_lt_005"])


@pytest.mark.integration
def test_fall_plot_conventions_match_paper(publication_data, tmp_path, monkeypatch):
    from invLocAd.publication import main_figures

    seen = {}

    def inspect_plot(_, name, figure):
        """Check the axes of one Figure 3 panel instead of saving it."""
        from matplotlib import pyplot as plt

        if name == "figure3_c":
            axis = figure.axes[0]
            assert axis.get_ylim() == (-0.05, 1.05)
            assert max(max(line.get_ydata()) for line in axis.lines) == 1
        elif name == "figure3_b":
            assert [axis.get_xlim() for axis in figure.axes] == [(0, 2500), (0, 2000)]
        elif name == "figure3_d":
            assert all(axis.get_ylim() == (0, 2000) for axis in figure.axes)
            assert all(len(axis.collections) == 1 for axis in figure.axes)
        seen[name] = True
        plt.close(figure)
        return []

    monkeypatch.setattr(main_figures, "_save", inspect_plot)
    record = render_main(publication_data, tmp_path, ["figure3"])[0]
    assert record["status"] == "passed"
    assert set(seen) == {"figure3_b", "figure3_c", "figure3_d"}


@pytest.mark.integration
def test_figure3_complete_binomial_counts(publication_data):
    trials, cells = fall_tables(publication_data)
    assert len(trials) == 1750 and len(cells) == 35
    assert all(row["trials"] == 50 for row in cells)
    cell = next(r for r in cells if r["delta_mps"] == 0.4 and r["learning_rate"] == 0.04)
    assert cell["falls"] == 21
    assert cell["fall_probability_pct"] == 42
    assert sum(row["stride_from_time_length"] for row in trials) == 50
    assert fall_filename(0.9, 0.05, 6) == (
        "model_split45_trans20_slow0.55_fast1.45_noise0.01_learning0.05_energysym1_swingcost1_trial6.mat"
    )
    # Reference coefficients were fitted independently with MATLAB fminsearch.
    for rate, a, b, sse in (
        (0.02, 1069.84240207159, -1.41806478936392, 121664.239816847),
        (0.05, 2332.71803938419, -1.87067908491541, 127171.491037044),
    ):
        selected = [r for r in cells if r["learning_rate"] == rate]
        fit = fit_exp1(
            [r["delta_mps"] for r in selected], [r["mean_strides_until_fall"] for r in selected]
        )
        np.testing.assert_allclose([fit["a"], fit["b"]], [a, b], rtol=5e-8, atol=1e-8)
        assert abs(fit["sse"] - sse) < 1e-7


@pytest.mark.integration
def test_original_landscape_audited_ellipse_values(publication_data):
    tables = landscape_tables(publication_data)
    assert len(tables["landscape"]) == 1200
    assert len(tables["participant_parameters"]) == 26
    current = [
        r for r in tables["ellipse_risk"] if r["landscape_delta_mps"] == r["ellipse_delta_mps"]
    ]
    assert [r["displayed_integer_pct"] for r in current] == [52, 50, 55]
    assert current[-1]["included_grid_nodes"] == 2
    alpha, psi, risk = landscape_data(publication_data)
    assert alpha[0] == 0.005
    np.testing.assert_allclose([psi[0], psi[9], psi[-1]], [0, 0.4736842105263158, 1])
    np.testing.assert_allclose(risk[0, -1, :] - risk[0, 0, :], [62, 42, -16], atol=1e-12, rtol=0)


@pytest.mark.integration
def test_parameter_comparisons_match_independent_matlab_with_ties(publication_data):
    # Reference values from MATLAB R2025a kruskalwallis, anova1, fitlm, ttest2 and ranksum,
    # evaluated on the original and regenerated parameter fits.
    expected = {
        ("original", "learning_rate"): [
            9.1357212475633514,
            0.010380143011048181,
            0.10031126982056092,
            -0.02214464733261725,
            0.034174881826052737,
            0.0094837700053482053,
            0.00090497737556561111,
        ],
        ("original", "symmetry_weight"): [
            9.1855517307324561,
            0.01012471448533666,
            0.0017957593771143585,
            1.0005171659307002,
            0.000392805731869932,
            2.3168041127059721e-5,
            0.00032908268202385847,
        ],
        ("regenerated", "learning_rate"): [
            3.038920574566963,
            0.21882996039886082,
            0.23659918924390666,
            -0.015306122448979612,
            0.14461561661052991,
            0.078370018059416846,
            0.086137392019744963,
        ],
        ("regenerated", "symmetry_weight"): [
            5.33281937272901,
            0.069501309395593056,
            0.0702727230437279,
            0.57803238254366129,
            0.023124724768154532,
            0.011084507365048738,
            0.006663924310983135,
        ],
    }
    tables = parameter_statistics(publication_data)
    assert len(tables["parameter_tests"]) == 20
    for (source, parameter), numbers in expected.items():
        rows = {
            r["test"]: r
            for r in tables["parameter_tests"]
            if r["fit_source"] == source and r["parameter"] == parameter
        }
        actual = [
            rows["kruskal_wallis"]["statistic"],
            rows["kruskal_wallis"]["p_value"],
            rows["one_way_anova"]["p_value"],
            rows["linear_trend"]["statistic"],
            rows["linear_trend"]["p_value"],
            rows["two_sample_t"]["p_value"],
            rows["rank_sum"]["p_value"],
        ]
        np.testing.assert_allclose(actual, numbers, rtol=2e-11, atol=2e-14)
        assert all(
            r["sampling_unit"] == "participant, independent condition cohorts"
            for r in rows.values()
        )


@pytest.mark.integration
def test_sanchez_matches_original_matlab_windows(publication_data):
    if not (publication_data / "original/data/sanchez data/sanchez_energy.mat").is_file():
        pytest.skip("Sánchez energy and SLA inputs are required")
    rows = _sanchez(publication_data)
    # P1 and P15 reference values from the authors' MATLAB R2025a analysis for Figure 1C-D.
    for participant, expected in (
        ("P1", [39.841766146199127, 34.101988532304752, -5.7686971140829719, 2.3811102200814673]),
        ("P15", [32.2157545896442, 11.812356790051719, -6.4204800056208731, 2.3510865075084189]),
    ):
        short, long = [r for r in rows if r["participant"] == participant]
        actual = [
            short["energy_change_pct"],
            long["energy_change_pct"],
            short["sla_change_percentage_points"],
            long["sla_change_percentage_points"],
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(rows) == 30


@pytest.mark.integration
def test_optimization_source_subset_matches_original_energy_equations(publication_data):
    folder = publication_data / "original/data/trajectory optimization"
    if not (folder / "model_13d24m_sbt.osim").is_file():
        pytest.skip("OpenSim trajectory-optimization inputs are required")
    # Reference values from MATLAB computeOsimEnergy.m, with the muscle parameters read
    # from the OSIM XML rather than through the OpenSim API.
    expected = [3.5183137519303593, 3.251264129677224, 3.2334139295303213, 2.8268725674956321]
    actual = [
        source_energy_cost(folder / name, folder / "model_13d24m_sbt.osim")["net_cost_j_kg_m"]
        for name in SOURCE_FILES.values()
    ]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=1e-13)
    rows = trajectory_source_tables(publication_data)
    assert [r["delta_mps"] for r in rows] == [0, 0.4, 0.7, 1.0]
    assert rows[0]["energy_change_pct"] == 0


def test_fall_probability_tests_pair_participants_across_landscapes():
    alpha, psi = np.array([0.01, 0.02]), np.array([0.0, 1.0])
    rates = np.zeros((2, 2, 3))
    rates[:, :, 0] = [[10, 20], [30, 40]]
    rates[:, :, 1] = [[15, 25], [35, 45]]
    rates[:, :, 2] = [[50, 60], [70, 80]]
    fitted = [
        (np.array([0.01, 0.01, 0.02, 0.02, 0.01, 0.02]), np.array([0, 1, 0, 1, 1, 0.0])),
        (np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.01]), np.array([1, 0, 0, 1, 1, 1.0])),
        (np.array([0.01]), np.array([0.0])),
    ]
    rows = fall_probability_tests(alpha, psi, rates, fitted)
    assert [(r["fitted_delta_mps"], r["landscape_delta_mps"]) for r in rows] == [
        (0.4, 0.7), (0.4, 1.0), (0.7, 1.0)
    ]
    assert [r["n_increased"] for r in rows] == [6, 6, 6]
    assert rows[0]["mean_larger_landscape_pct"] - rows[0]["mean_own_landscape_pct"] == 5
    assert all(r["p_one_sided_greater"] == pytest.approx(1 / 64) for r in rows)
    with pytest.raises(ValueError, match="not on the fall-risk landscape grid"):
        fall_probability_tests(alpha, psi, rates, [(np.array([0.015]), np.array([0.0]))])

