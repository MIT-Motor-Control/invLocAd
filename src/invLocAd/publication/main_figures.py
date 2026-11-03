"""Reproduce Figures 1, 3 and 4 and the fall-risk landscape slices from cached data.

The renderer ``s3`` produces the landscape slices of Supplementary Figure 3.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.stats import (
    f_oneway,
    kruskal,
    linregress,
    mannwhitneyu,
    rankdata,
    t,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

from .trajectory import trajectory_source_tables

SPEEDS = (0.4, 0.7, 1.0)
# Rounded trajectory-optimization predictions plotted in Figure 1C/D: energy change (%)
# from the tied-belt cost of 3.147 W/kg and SLA change (percentage points).
OPT_SPEEDS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
OPT_ENERGY = (np.array([3.161, 3.183, 3.138, 3.046, 2.968]) - 3.147) / 3.147 * 100
OPT_SLA = np.array([11.268, 16.814, 23.186, 29.740, 42.391])


def _mat(path: Path, **kwargs) -> dict:
    """Load a MAT file with MATLAB structs as attribute objects."""
    with path.open("rb") as stream:
        return loadmat(stream, struct_as_record=False, squeeze_me=True, **kwargs)


def _csv(root: Path, name: str, rows: list[dict]) -> str:
    """Write nonempty ``rows`` to ``root/tables/<name>.csv`` and return the resolved path."""
    if not rows:
        raise ValueError(f"Refusing to export an empty source table: {name}")
    path = root / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path.resolve())


def _plotting():
    """Return ``pyplot`` set up for headless rendering with editable vector text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(root: Path, name: str, figure) -> list[str]:
    """Save ``figure`` as PNG, PDF and SVG in ``root/figures``, close it, return the paths."""
    from matplotlib import pyplot as plt

    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = folder / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def _record(identifier, status="passed", outputs=None, caveats=None, details=None):
    """Return a figure result record for the reproduction report."""
    return {
        "id": identifier,
        "status": status,
        "outputs": outputs or [],
        "caveats": caveats or [],
        "details": details or {},
    }


def matlab_box_summary(values) -> dict:
    """Return MATLAB ``boxplot`` statistics for one group.

    Quartiles use midpoint (Hazen) interpolation, as MATLAB ``prctile`` does. Whiskers
    extend to the most extreme values within 1.5 IQR of the box.

    Args:
        values: Finite, nonempty one-dimensional participant values.

    Returns:
        Dict with ``n``, ``q1``, ``median``, ``q3``, ``whisker_low``, ``whisker_high`` and
        the number of ``hidden_outliers`` beyond the whiskers.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Boxplots need finite, nonempty one-dimensional participant data")
    q1, median, q3 = np.percentile(values, [25, 50, 75], method="hazen")
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {
        "n": len(values),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": float(min(q1, values[values >= low].min())),
        "whisker_high": float(max(q3, values[values <= high].max())),
        "hidden_outliers": int(np.sum((values < low) | (values > high))),
    }


def _boxes(axis, groups, positions, widths=0.5) -> list[dict]:
    """Draw MATLAB-style box plots without outliers on ``axis``; return their summaries."""
    summaries = [matlab_box_summary(group) for group in groups]
    axis.bxp(
        [
            {
                "q1": s["q1"],
                "med": s["median"],
                "q3": s["q3"],
                "whislo": s["whisker_low"],
                "whishi": s["whisker_high"],
            }
            for s in summaries
        ],
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#990000", "alpha": 0.5},
        medianprops={"color": "black"},
    )
    return summaries


def figure1_tables(data_root: Path) -> dict[str, list[dict]]:
    """Tabulate Figure 1C-E experimental energy and SLA changes and the Figure 1E tests.

    Energy (1 Hz) is compared with the mean of the last 3 tied-belt minutes; the short and
    long exposures are split-belt minutes 3-6 and 42-45. SLA changes compare 100-stride
    means with the last 100 tied strides; the short SLA window ends at stride
    ceil(6/45 x split strides) and the long window is the last 100 strides. Figure 1E compares
    each 3-minute mean with the preceding one by a two-sided paired t-test.

    Args:
        data_root: Prepared publication data directory.

    Returns:
        Dict of row lists keyed by ``energy_participants``, ``sla_participants``,
        ``energy_timeseries``, ``paired_tests`` and ``optimization``.
    """
    original = Path(data_root) / "original"
    energy = _mat(original / "data/cosmed/processed/processedEnergyData.mat")
    sla = _mat(original / "processedSLAData_23Subj.mat")
    energy_rows, sla_rows, time_rows, test_rows = [], [], [], []
    for condition, (speed, ne, ns) in enumerate(zip(SPEEDS, (8, 9, 9), (7, 8, 8)), 1):
        split = np.atleast_2d(getattr(energy["split_energy"], f"condition_{condition}"))
        tied = np.atleast_2d(getattr(energy["tied1_energy"], f"condition_{condition}"))
        if split.shape != (ne, 2700) or tied.shape != (ne, 360):
            raise ValueError(f"Unexpected metabolic cohort/shape for condition {condition}")
        if not np.isfinite(split).all() or not np.isfinite(tied).all():
            raise ValueError("Energy arrays contain non-finite values")
        baseline = tied[:, -180:].mean(axis=1)
        if np.any(baseline == 0):
            raise ValueError("A tied-belt metabolic baseline is zero")
        for index in range(ne):
            for label, window in (
                ("short_6_min", slice(180, 360)),
                ("long_45_min", slice(-180, None)),
            ):
                value = float(split[index, window].mean())
                energy_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "exposure": label,
                        "tied_baseline_w_kg": float(baseline[index]),
                        "split_w_kg": value,
                        "energy_change_pct": float(
                            (value - baseline[index]) / baseline[index] * 100
                        ),
                    }
                )
        times = (3, 9, 18, 27, 36, 45)
        time_values = np.stack(
            [split[:, t * 60 - 180 : t * 60].mean(axis=1) for t in times], axis=1
        )
        for index in range(ne):
            for column, minute in enumerate(times):
                time_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "time_min": minute,
                        "metabolic_cost_w_kg": float(time_values[index, column]),
                    }
                )
        for column in range(1, len(times)):
            test = ttest_rel(time_values[:, column], time_values[:, column - 1])
            test_rows.append(
                {
                    "condition": condition,
                    "delta_mps": speed,
                    "n": ne,
                    "previous_min": times[column - 1],
                    "current_min": times[column],
                    "t_statistic": float(test.statistic),
                    "df": ne - 1,
                    "p_two_sided": float(test.pvalue),
                    "significant_p_lt_005": bool(test.pvalue < 0.05),
                    "correction": "none",
                }
            )
        split_sla = np.atleast_1d(sla[f"split_sla_cond{condition}"])
        tied_sla = np.atleast_1d(sla[f"tied1_sla_cond{condition}"])
        if sum(np.asarray(v).size > 0 for v in split_sla) != ns:
            raise ValueError(f"Unexpected SLA cohort for condition {condition}")
        indices = np.atleast_1d(sla.get(f"short_idx_cond{condition}", np.full(ns, np.nan)))
        for index in range(ns):
            s, t = np.asarray(split_sla[index]).ravel(), np.asarray(tied_sla[index]).ravel()
            end = int(np.ceil(6 / 45 * len(s)))
            if end < 100 or len(t) < 100 or not np.isfinite(s).all() or not np.isfinite(t).all():
                raise ValueError("SLA record is too short or nonfinite for the 100-stride windows")
            baseline_sla = float(t[-100:].mean() * 100)
            for label, values in (
                ("short_6_min_fractional_stride", s[end - 100 : end]),
                ("long_45_min", s[-100:]),
            ):
                value = float(values.mean() * 100)
                sla_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "exposure": label,
                        "tied_sla_pct": baseline_sla,
                        "split_sla_pct": value,
                        "sla_change_percentage_points": value - baseline_sla,
                        "short_end_stride": end,
                        "stored_short_index_not_used": float(indices[index]),
                    }
                )
    optimization = [
        {
            "delta_mps": float(speed),
            "tied_energy_w_kg": 3.147,
            "energy_change_pct": float(e),
            "sla_change_percentage_points": float(s),
            "source": (
                "Rounded trajectory-optimization values plotted in Figure 1C/D; "
                "independent of the OpenSim source subset"
            ),
        }
        for speed, e, s in zip(OPT_SPEEDS, OPT_ENERGY, OPT_SLA)
    ]
    return {
        "energy_participants": energy_rows,
        "sla_participants": sla_rows,
        "energy_timeseries": time_rows,
        "paired_tests": test_rows,
        "optimization": optimization,
    }


def _sanchez(data_root: Path) -> list[dict]:
    """Return short and long energy and SLA changes for the 15 Sánchez et al. (2021) participants.

    Baselines are the last 100 tied samples. The short window is the 100 samples ending at
    round(6/45 x energy samples), or the first 100 samples if fewer precede it, and is
    applied to both signals. The long window is the last 100 samples of each signal.
    """
    folder = data_root / "original/data/sanchez data"
    energy = _mat(folder / "sanchez_energy.mat")["MetabolicPower"]
    sla = _mat(folder / "sanchez_sla.mat")["StepLengthAsymmetry"]
    rows = []
    for participant in range(1, 16):
        e, s = getattr(energy, f"P{participant}"), getattr(sla, f"P{participant}")
        es, ss = np.asarray(e.Split).ravel(), np.asarray(s.Split).ravel()
        base_energy, base_sla = np.asarray(e.Base).ravel(), np.asarray(s.Base).ravel()
        if any(len(v) < 100 or not np.isfinite(v).all() for v in (es, ss, base_energy, base_sla)):
            raise ValueError("Sanchez inputs require at least 100 finite values in every record")
        eb, sb = float(base_energy[-100:].mean()), float(base_sla[-100:].mean() * 100)
        end = int(np.floor(6 / 45 * len(es) + 0.5))
        window = slice(end - 100, end) if end > 100 else slice(0, 100)
        if max(window.stop, 100) > min(len(es), len(ss)) or eb == 0:
            raise ValueError("Sanchez input is too short for its windows or has a zero baseline")
        for label, w in (("short_6_min", window), ("long_45_min", slice(-100, None))):
            rows.append(
                {
                    "participant": f"P{participant}",
                    "delta_mps": 1.0,
                    "exposure": label,
                    "energy_change_pct": float((es[w].mean() - eb) / eb * 100),
                    "sla_change_percentage_points": float(ss[w].mean() * 100 - sb),
                    "energy_baseline_w_kg": eb,
                    "sla_baseline_pct": sb,
                    "energy_split_samples": len(es),
                    "sla_split_samples": len(ss),
                    "energy_window_first_sample": w.start + 1
                    if w.start >= 0
                    else len(es) + w.start + 1,
                    "energy_window_last_sample": w.stop if w.stop is not None else len(es),
                    "sla_window_first_sample": w.start + 1
                    if w.start >= 0
                    else len(ss) + w.start + 1,
                    "sla_window_last_sample": w.stop if w.stop is not None else len(ss),
                    "baseline_samples": 100,
                    "window_rule": "Short: round(6/45 * energy length), shared indices for both signals; long: last 100 of each signal",
                }
            )
    return rows


def _figure1(data_root: Path, output_root: Path) -> dict:
    """Export Figure 1C-E tables and plots; 1C/D are partial without the Sánchez inputs."""
    tables = figure1_tables(data_root)
    outputs = [_csv(output_root, f"figure1_{name}", rows) for name, rows in tables.items()]
    caveats = [
        "Participant indices identify ordering within each condition, not global participant IDs.",
        (
            "Short SLA uses the 100 strides ending at stride ceil(6/45 × total strides), "
            "a stride-count approximation of the 6-minute point; stored short_idx variables "
            "are not used."
        ),
        "The five trajectory-optimization points are fixed rounded values, not recomputed here.",
        (
            "Figure 1E tests are two-sided paired t-tests between adjacent time points, without "
            "multiplicity correction."
        ),
    ]
    missing = [
        name
        for name in ("sanchez_energy.mat", "sanchez_sla.mat")
        if not (data_root / "original/data/sanchez data" / name).is_file()
    ]
    sanchez = [] if missing else _sanchez(data_root)
    if sanchez:
        outputs.append(_csv(output_root, "figure1_sanchez_participants", sanchez))
        caveats.append(
            "Sánchez short exposure applies the energy-derived sample indices to both energy and "
            "SLA, as in the original analysis, although the two signals have different sample "
            "counts."
        )
    else:
        caveats.append("Sánchez comparison inputs are missing; Figures 1C and 1D are partial.")
    trajectory = trajectory_source_tables(data_root)
    if trajectory:
        outputs.append(_csv(output_root, "figure1_optimization_source_subset", trajectory))
        caveats.append(
            "The saved OpenSim outputs cover tied, 0.4, 0.7 and 1.0 m/s and do not reproduce "
            "the five plotted constants; their energy values are exported separately and are "
            "not plotted."
        )
    plt = _plotting()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    summaries = []
    for row, (name, column, ylabel) in enumerate(
        (
            ("energy_participants", "energy_change_pct", "Energy change vs tied (%)"),
            (
                "sla_participants",
                "sla_change_percentage_points",
                "SLA change vs tied (percentage points)",
            ),
        )
    ):
        opt = OPT_ENERGY if row == 0 else OPT_SLA
        x = np.linspace(0.2, 1, 500)
        axes[row, 0].scatter(
            OPT_SPEEDS, opt, facecolors="#e4f0e4", edgecolors="black", linewidths=0.5
        )
        axes[row, 0].plot(
            x, np.polyval(np.polyfit(OPT_SPEEDS, opt, 2 if row == 0 else 1), x), color="#009900"
        )
        axes[row, 0].set_title("Trajectory optimization")
        axes[row, 0].set(xlim=(0.1, 1.1), ylim=(-10, 5) if row == 0 else (0, 50))
        for col, exposure in enumerate(("short", "long"), 1):
            groups = [
                np.array(
                    [
                        r[column]
                        for r in tables[name]
                        if r["delta_mps"] == speed and r["exposure"].startswith(exposure)
                    ]
                )
                for speed in SPEEDS
            ]
            boxes = _boxes(axes[row, col], groups, SPEEDS, widths=0.075)
            means = [float(v.mean()) for v in groups]
            xx = np.linspace(0.4, 1, 500)
            axes[row, col].plot(xx, np.polyval(np.polyfit(SPEEDS, means, 2), xx), color="#990000")
            for speed, values, box in zip(SPEEDS, groups, boxes):
                summaries.append(
                    {
                        "panel": "1C" if row == 0 else "1D",
                        "exposure": exposure,
                        "delta_mps": speed,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "units": ylabel,
                        **box,
                    }
                )
            if sanchez:
                value = np.mean([r[column] for r in sanchez if r["exposure"].startswith(exposure)])
                axes[row, col].scatter(
                    [1], [value], color="#000099", label="Sánchez 2021", zorder=4
                )
                axes[row, col].legend(frameon=False)
            axes[row, col].set_title(f"Experiment: {'6' if col == 1 else '45'} min")
            axes[row, col].set(xlim=(0.3, 1.1), ylim=(-45, 80) if row == 0 else (-30, 10))
        for axis in axes[row]:
            axis.set_xlabel("Belt-speed difference (m/s)")
            axis.set_ylabel(ylabel)
    fig.suptitle("Figure 1C/D" + (" (partial: Sánchez comparison unavailable)" if missing else ""))
    fig.tight_layout()
    outputs += _save(output_root, "figure1_cd", fig)
    outputs.append(_csv(output_root, "figure1_cd_summary", summaries))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    times = (3, 9, 18, 27, 36, 45)
    box_rows = []
    for axis, speed in zip(axes, SPEEDS):
        groups = [
            [
                r["metabolic_cost_w_kg"]
                for r in tables["energy_timeseries"]
                if r["delta_mps"] == speed and r["time_min"] == t
            ]
            for t in times
        ]
        boxes = _boxes(axis, groups, range(1, 7))
        box_rows.extend(
            {"delta_mps": speed, "time_min": minute, **box} for minute, box in zip(times, boxes)
        )
        axis.set_xticks(range(1, 7), times)
        for index, test in enumerate(
            [r for r in tables["paired_tests"] if r["delta_mps"] == speed]
        ):
            if test["significant_p_lt_005"]:
                axis.text(
                    index + 1.5, max(max(v) for v in groups) + 0.2, "*", ha="center", fontsize=15
                )
        axis.set_title(f"{speed:g} m/s")
        axis.set_xlabel("Time (min)")
    axes[0].set_ylabel("Metabolic cost (W/kg)")
    axes[0].set_ylim(top=max(r["metabolic_cost_w_kg"] for r in tables["energy_timeseries"]) + 0.7)
    fig.suptitle("Figure 1E: three-minute means and adjacent paired tests")
    fig.tight_layout()
    outputs += _save(output_root, "figure1_e", fig)
    outputs.append(_csv(output_root, "figure1_e_box_summary", box_rows))
    return _record(
        "figure1",
        "partial" if missing else "passed",
        outputs,
        caveats,
        {
            "energy_n": [8, 9, 9],
            "sla_n": [7, 8, 8],
            "missing_sanchez_files": missing,
            "sanchez_n": 15 if sanchez else 0,
            "optimization_source_subset": {
                "available": bool(trajectory),
                "matches_published_five_point_series": False,
            },
            "panels": {
                "1C": "partial" if missing else "passed",
                "1D": "partial" if missing else "passed",
                "1E": "passed",
            },
        },
    )


def fall_filename(speed: float, rate: float, trial: int) -> str:
    """Return the cached Figure 3 trial filename.

    Args:
        speed: Belt-speed difference (m/s); the name stores belt speeds 1 - speed/2 and
            1 + speed/2.
        rate: Learning rate α.
        trial: Trial number, 1-50.
    """
    return (
        f"model_split45_trans20_slow{1 - speed / 2:g}_fast{1 + speed / 2:g}_noise0.01_"
        f"learning{rate:g}_energysym1_swingcost1_trial{trial}.mat"
    )


def fall_tables(data_root: Path) -> tuple[list[dict], list[dict]]:
    """Load all 1,750 Figure 3 fall-probability trials at ψ = 0.

    The grid is 7 belt-speed differences x 5 learning rates x 50 trials. ``fall_flag``
    holds (fell, fall stride); a fall recorded at stride 0 takes its stride from the
    length of ``time``.

    Args:
        data_root: Prepared publication data directory.

    Returns:
        Tuple of per-trial rows and per-cell rows with fall counts, probabilities and mean
        strides until fall.

    Raises:
        FileNotFoundError: If any trial file is missing.
    """
    folder = Path(data_root) / "fall_probability"
    speeds, rates = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0.01, 0.02, 0.03, 0.04, 0.05)
    expected = [
        (speed, rate, trial, folder / fall_filename(speed, rate, trial))
        for speed in speeds
        for rate in rates
        for trial in range(1, 51)
    ]
    missing = [p.name for _, _, _, p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Figure 3 needs all 1,750 trials: {len(missing)} missing; first: {missing[0]}"
        )
    trials, summary = [], []
    for speed, rate, trial, path in expected:
        value = _mat(path, variable_names=["fall_flag", "time"])
        flag = np.asarray(value["fall_flag"]).ravel()
        if len(flag) != 2 or flag[0] not in (0, 1):
            raise ValueError(f"Invalid fall_flag in {path.name}")
        fell = bool(flag[0])
        fallback = bool(fell and flag[1] == 0)
        stride = int(np.asarray(value["time"]).size if fallback else flag[1]) if fell else None
        trials.append(
            {
                "delta_mps": speed,
                "learning_rate": rate,
                "symmetry_weight": 0,
                "trial": trial,
                "fell": fell,
                "fall_stride": stride,
                "stride_from_time_length": fallback,
                "source_file": f"fall_probability/{path.name}",
            }
        )
    for speed in speeds:
        for rate in rates:
            values = [r for r in trials if r["delta_mps"] == speed and r["learning_rate"] == rate]
            strides = [r["fall_stride"] for r in values if r["fell"]]
            summary.append(
                {
                    "delta_mps": speed,
                    "learning_rate": rate,
                    "symmetry_weight": 0,
                    "falls": len(strides),
                    "trials": len(values),
                    "fall_probability": len(strides) / len(values),
                    "fall_probability_pct": 100 * len(strides) / len(values),
                    "mean_strides_until_fall": float(np.mean(strides)) if strides else "",
                }
            )
    return trials, summary


def fit_exp1(x, y) -> dict:
    """Fit ``y = a * exp(b * x)`` by unweighted nonlinear least squares.

    The fit is parameterized by log(a). It starts from a log-linear fit and from five
    fixed slopes and keeps the converged solution with the lowest cost.

    Args:
        x: Belt-speed differences (m/s); at least three values, not all equal.
        y: Positive mean strides until fall at each ``x``.

    Returns:
        Dict with ``a``, ``b``, ``sse``, ``rmse``, ``rmse_df_adjusted`` (sqrt(SSE / (n - 2))),
        ``n``, ``residuals`` (fitted minus observed) and ``objective`` and ``model``
        descriptions.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.shape != y.shape
        or x.ndim != 1
        or len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("exp1 needs at least three finite paired observations")
    if np.any(y <= 0) or np.ptp(x) == 0:
        raise ValueError("exp1 requires positive mean fall strides and distinct speeds")

    def residual(parameters):
        """Return fitted minus observed strides for parameters (log a, b)."""
        return np.exp(parameters[0] + parameters[1] * x) - y

    def jacobian(parameters):
        """Return the residual Jacobian with respect to (log a, b)."""
        prediction = np.exp(parameters[0] + parameters[1] * x)
        return np.column_stack((prediction, x * prediction))

    initial_b, initial_log_a = np.polyfit(x, np.log(y), 1)
    candidates = [
        least_squares(
            residual,
            [initial_log_a, initial_b],
            jac=jacobian,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=10000,
        )
    ]
    for b in (-10.0, -2.0, 0.0, 2.0, 10.0):
        candidates.append(
            least_squares(
                residual,
                [np.log(y.mean()) - b * x.mean(), b],
                jac=jacobian,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=10000,
            )
        )
    successful = [r for r in candidates if r.success and np.isfinite(r.cost)]
    if not successful:
        raise ValueError("The exponential least-squares fit did not converge")
    result = min(successful, key=lambda r: r.cost)
    sse = float(np.sum(result.fun**2))
    return {
        "a": float(np.exp(result.x[0])),
        "b": float(result.x[1]),
        "sse": sse,
        "rmse": float(np.sqrt(sse / len(x))),
        "rmse_df_adjusted": float(np.sqrt(sse / (len(x) - 2))),
        "n": len(x),
        "residuals": result.fun.tolist(),
        "objective": "Unweighted sum of squared residuals of the per-speed mean fall strides",
        "model": "a * exp(b * delta_mps)",
    }


def _figure3(data_root: Path, output_root: Path) -> dict:
    """Export Figure 3B-D tables and plots from the cached fall trials."""
    trials, summary = fall_tables(data_root)
    outputs = [
        _csv(output_root, "figure3_trials", trials),
        _csv(output_root, "figure3_c_counts", summary),
    ]
    plt = _plotting()
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    for rate, color in zip((0.01, 0.02, 0.03, 0.04, 0.05), plt.cm.Reds(np.linspace(0.35, 0.95, 5))):
        values = [r for r in summary if r["learning_rate"] == rate]
        axis.plot(
            [r["delta_mps"] for r in values],
            [r["fall_probability"] for r in values],
            "o-",
            color=color,
            label=f"α = {rate:g}",
        )
    axis.set(
        xlabel="Belt-speed difference (m/s)",
        ylabel="Fall probability",
        xlim=(0.35, 1.05),
        ylim=(-0.05, 1.05),
        title="Figure 3C: 50 trials per cell; ψ = 0",
    )
    axis.legend(frameon=False)
    outputs += _save(output_root, "figure3_c", fig)

    # Figure 3B examples at α = 0.05, searching 0.9 then 1.0 m/s in trial order: the first
    # fall after stride 1000 and the first trial without a fall.
    examples = []
    for speed in (0.9, 1.0):
        for trial in range(1, 51):
            row = next(
                r
                for r in trials
                if r["delta_mps"] == speed and r["learning_rate"] == 0.05 and r["trial"] == trial
            )
            value = _mat(
                data_root / row["source_file"], variable_names=["fall_flag", "min_vert_pos"]
            )
            flag = np.asarray(value["fall_flag"]).ravel()
            kind = (
                "fall" if flag[0] == 1 and flag[1] > 1000 else "no_fall" if flag[0] == 0 else None
            )
            if kind and kind not in [r["kind"] for r in examples]:
                height = np.asarray(value["min_vert_pos"])[:, 1]
                if kind == "fall":
                    height = np.append(height, 0)
                examples.append(
                    {"kind": kind, "source_file": row["source_file"], "heights": height}
                )
    caveats = [
        "Cached trial reconstruction; not a fresh stochastic rerun.",
        (
            "Cached belt-speed profiles use a 25-second transition and physical belt speeds divided "
            "by 2. Their trans20 filenames are retained as source identifiers."
        ),
        (
            "The representative fall trace appends a terminal zero, following the original plotting script; "
            "the cached height field contains the per-stance minimum, rather than a separately stored end-stance sample."
        ),
        (
            "Figure 3D preserves the original 0-2000-stride axis; all falling trials, including points above "
            "that display range, remain in the source table and exponential-fit inputs."
        ),
        (
            "Figure 3D fits the exp1 model a*exp(b*x) to cell means by unweighted nonlinear "
            "least squares. SciPy replaces the MATLAB Curve Fitting Toolbox solver, so "
            "bit-identical coefficients are not claimed."
        ),
    ]
    panels = {"3C": "passed", "3D": "passed", "3B": "passed" if len(examples) == 2 else "blocked"}
    if len(examples) == 2:
        example_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for axis, kind in zip(axes, ("fall", "no_fall")):
            example = next(v for v in examples if v["kind"] == kind)
            height = example["heights"]
            for index, value in enumerate(height, 1):
                example_rows.append(
                    {
                        "example": kind,
                        "stride": index,
                        "min_vertical_height": float(value),
                        "source_file": example["source_file"],
                        "appended_terminal_zero": bool(kind == "fall" and index == len(height)),
                    }
                )
            axis.plot(np.arange(1, len(height) + 1), height, color="#990000")
            axis.set(
                xlabel="Stride",
                ylabel="Minimum vertical height",
                title=kind.replace("_", " "),
                ylim=(0, 1.1),
                xlim=(0, 2500) if kind == "fall" else (0, 2000),
            )
        fig.suptitle("Figure 3B: original representative-trial selection")
        fig.tight_layout()
        outputs += [_csv(output_root, "figure3_b_traces", example_rows)] + _save(
            output_root, "figure3_b", fig
        )
    else:
        caveats.append("Original representative-trial selection did not find both examples.")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    rows_d, fit_rows, curve_rows = [], [], []
    for axis, rate, color in zip(axes, (0.02, 0.05), ("#d98e8e", "#660000")):
        selected = [r for r in trials if r["learning_rate"] == rate and r["fell"]]
        axis.scatter(
            [r["delta_mps"] for r in selected],
            [r["fall_stride"] for r in selected],
            s=8,
            alpha=0.2,
            color=color,
        )
        means = [r for r in summary if r["learning_rate"] == rate]
        observed = [r for r in means if r["falls"]]
        fit = fit_exp1(
            [r["delta_mps"] for r in observed], [r["mean_strides_until_fall"] for r in observed]
        )
        fit_rows.append(
            dict(
                learning_rate=rate,
                **{key: value for key, value in fit.items() if key != "residuals"},
            )
        )
        for row in means:
            fitted = float(fit["a"] * np.exp(fit["b"] * row["delta_mps"]))
            rows_d.append(
                {
                    "delta_mps": row["delta_mps"],
                    "learning_rate": rate,
                    "falls": row["falls"],
                    "mean_strides_until_fall": row["mean_strides_until_fall"],
                    "fitted_mean_strides": fitted,
                    "residual_predicted_minus_observed": fitted - row["mean_strides_until_fall"]
                    if row["falls"]
                    else "",
                }
            )
        x_fit = np.linspace(0.4, 1.0, 100)
        y_fit = fit["a"] * np.exp(fit["b"] * x_fit)
        curve_rows.extend(
            {"learning_rate": rate, "delta_mps": float(x), "fitted_mean_strides": float(y)}
            for x, y in zip(x_fit, y_fit)
        )
        axis.plot(x_fit, y_fit, color=color)
        axis.set(
            xlabel="Belt-speed difference (m/s)",
            ylabel="Strides until fall",
            title=f"α = {rate:g}",
            xlim=(0.35, 1.05),
            ylim=(0, 2000),
        )
    fig.suptitle("Figure 3D: original exponential model fitted to the seven cell means")
    fig.tight_layout()
    outputs += [
        _csv(output_root, "figure3_d_fit_inputs", rows_d),
        _csv(output_root, "figure3_d_fit_coefficients", fit_rows),
        _csv(output_root, "figure3_d_fit_curves", curve_rows),
    ] + _save(output_root, "figure3_d", fig)
    return _record(
        "figure3",
        "passed" if len(examples) == 2 else "partial",
        outputs,
        caveats,
        {
            "required_trials": 1750,
            "available_trials": len(trials),
            "cells": len(summary),
            "panels": panels,
            "zero_stride_fallbacks": sum(r["stride_from_time_length"] for r in trials),
        },
    )


def landscape_data(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the original Figure 4 fall-risk landscape.

    Args:
        data_root: Prepared publication data directory.

    Returns:
        Tuple of learning rates α, ascending symmetry weights ψ = 1 - sym_cost, and fall
        probabilities (%) with shape (len(α), len(ψ), 3) for 0.4, 0.7 and 1.0 m/s.
    """
    value = _mat(Path(data_root) / "original/iso_fall_risk_heat_map.mat")
    alpha = np.asarray(value["learning_pool"], dtype=float).ravel()
    cost = np.asarray(value["sym_cost_pool"], dtype=float).ravel()
    order = np.argsort(1 - cost)
    psi = (1 - cost)[order]
    rates = np.asarray(value["fall_rate_matrix_result"], dtype=float)[:, order, :]
    if (
        rates.shape != (len(alpha), len(psi), 3)
        or not np.isfinite(rates).all()
        or np.any((rates < 0) | (rates > 100))
    ):
        raise ValueError("Landscape is incomplete or contains invalid fall probabilities")
    return alpha, psi, rates


def _rank_sum_probability(x, y, alternative: str) -> tuple[float, float, str]:
    """Return the one-sided Wilcoxon rank-sum test of ``x`` against ``y`` as MATLAB ``ranksum``.

    The exact distribution of average ranks, including ties, is used when
    min(n_x, n_y) < 10 and n_x + n_y < 20; otherwise the normal approximation with tie
    and continuity corrections.

    Args:
        x: First sample; its rank sum is the statistic.
        y: Second sample.
        alternative: ``"greater"`` or ``"less"``.

    Returns:
        Tuple of rank sum of ``x``, p-value and method description.
    """
    x, y = np.asarray(x), np.asarray(y)
    ranks = rankdata(np.concatenate([x, y]))
    observed = float(ranks[: len(x)].sum())
    if min(len(x), len(y)) < 10 and len(x) + len(y) < 20:
        sums = np.fromiter((sum(group) for group in combinations(ranks, len(x))), dtype=float)
        p = np.mean(sums >= observed if alternative == "greater" else sums <= observed)
        return observed, float(p), "exact allocation of average ranks, including ties"
    test = mannwhitneyu(x, y, alternative=alternative, method="asymptotic", use_continuity=True)
    return observed, float(test.pvalue), "normal approximation with ties and continuity correction"


def parameter_statistics(data_root: Path) -> dict[str, list[dict]]:
    """Compare fitted α and ψ across the independent 8/9/9 participant cohorts.

    For the original fits and, when present, the regenerated fits, each parameter is tested
    by Kruskal-Wallis, one-way ANOVA and a linear trend on belt-speed difference, and by
    one-sided 0.4 vs 1.0 m/s t and rank-sum tests (α greater, ψ smaller at 0.4 m/s).
    No multiplicity correction is applied.

    Args:
        data_root: Prepared publication data directory.

    Returns:
        Dict with ``parameter_summary`` and ``parameter_tests`` row lists.
    """
    summaries, tests = [], []
    for source in ("original", "regenerated"):
        path = Path(data_root) / source / "2d_sweep_trial50_param_result.mat"
        if source == "regenerated" and not path.is_file():
            continue
        value = _mat(path)
        counts = np.asarray(value["exp_subj_num"]).ravel()
        parameters = np.asarray(value["static_params"])
        if not np.array_equal(counts, [8, 9, 9]) or parameters.shape != (3, 9):
            raise ValueError("Parameter comparisons require the independent 8/9/9 cohorts")
        for name in ("learning_rate", "symmetry_weight"):
            groups = [
                np.array(
                    [
                        float(parameters[c, s].learning_rate)
                        if name == "learning_rate"
                        else 1 - float(parameters[c, s].sym_cost)
                        for s in range(int(n))
                    ]
                )
                for c, n in enumerate(counts)
            ]
            values = np.concatenate(groups)
            if not np.isfinite(values).all() or (
                np.any(values <= 0)
                if name == "learning_rate"
                else np.any((values < 0) | (values > 1))
            ):
                raise ValueError("Invalid fitted values in parameter comparisons")
            for speed, group in zip(SPEEDS, groups):
                summaries.append(
                    {
                        "fit_source": source,
                        "parameter": name,
                        "delta_mps": speed,
                        "n": len(group),
                        "median": float(np.median(group)),
                        "mean": float(group.mean()),
                        "sd": float(group.std(ddof=1)),
                        "units": "model learning-rate parameter"
                        if name == "learning_rate"
                        else "dimensionless (1 - sym_cost)",
                    }
                )
            common = {
                "fit_source": source,
                "parameter": name,
                "n_04": 8,
                "n_07": 9,
                "n_10": 9,
                "sampling_unit": "participant, independent condition cohorts",
                "multiplicity_correction": "none",
            }
            kw = kruskal(*groups)
            anova = f_oneway(*groups)
            regression = linregress(np.repeat(SPEEDS, counts.astype(int)), values)
            margin = t.ppf(0.975, len(values) - 2) * regression.stderr
            alternative = "greater" if name == "learning_rate" else "less"
            extreme = ttest_ind(groups[0], groups[2], equal_var=True, alternative=alternative)
            rank_sum, rank_p, rank_method = _rank_sum_probability(groups[0], groups[2], alternative)
            for test, statistic, p, comparison, tail, method, lower, upper in (
                (
                    "kruskal_wallis",
                    kw.statistic,
                    kw.pvalue,
                    "all three conditions",
                    "any condition distribution differs",
                    "tie-corrected H, chi-square reference, df=2",
                    "",
                    "",
                ),
                (
                    "one_way_anova",
                    anova.statistic,
                    anova.pvalue,
                    "all three conditions",
                    "any condition mean differs",
                    "independent groups, df=2,23",
                    "",
                    "",
                ),
                (
                    "linear_trend",
                    regression.slope,
                    regression.pvalue,
                    "parameter vs belt-speed difference (m/s)",
                    "two-sided",
                    "OLS slope per m/s; t reference df=24",
                    regression.slope - margin,
                    regression.slope + margin,
                ),
                (
                    "two_sample_t",
                    extreme.statistic,
                    extreme.pvalue,
                    "0.4 vs 1.0 m/s",
                    alternative,
                    "equal-variance independent t-test, df=15",
                    "",
                    "",
                ),
                ("rank_sum", rank_sum, rank_p, "0.4 vs 1.0 m/s", alternative, rank_method, "", ""),
            ):
                if not np.isfinite([statistic, p]).all():
                    raise ValueError("Parameter comparison returned nonfinite statistics")
                tests.append(
                    {
                        **common,
                        "test": test,
                        "statistic": float(statistic),
                        "p_value": float(p),
                        "comparison": comparison,
                        "alternative": tail,
                        "method": method,
                        "slope_ci_lower_95": float(lower) if lower != "" else "",
                        "slope_ci_upper_95": float(upper) if upper != "" else "",
                    }
                )
    return {"parameter_summary": summaries, "parameter_tests": tests}


def fall_probability_tests(alpha, psi, rates, fitted) -> list[dict]:
    """Test whether each participant's fitted parameters are riskier at larger speed differences.

    For the participants of each condition, the fall probability at their fitted (α, ψ) grid
    cell on their own condition's landscape is paired with the probability at the same cell
    on each landscape of a larger belt-speed difference, and compared with a Wilcoxon
    signed-rank test.

    Args:
        alpha: Landscape learning rates.
        psi: Landscape symmetry weights.
        rates: Fall probabilities (%), shape (len(alpha), len(psi), 3).
        fitted: Per condition, a pair of arrays (learning rates, symmetry weights).

    Returns:
        One row per (fitted condition, larger landscape) pair with means, counts and p-values.

    Raises:
        ValueError: If a fitted parameter does not lie on the landscape grid.
    """
    rows = []
    for c, (lr, sym) in enumerate(fitted):
        cells = []
        for a, p in zip(lr, sym):
            i, j = np.flatnonzero(np.isclose(alpha, a)), np.flatnonzero(np.isclose(psi, p))
            if len(i) != 1 or len(j) != 1:
                raise ValueError("A fitted parameter pair is not on the fall-risk landscape grid")
            cells.append((i[0], j[0]))
        own = np.array([rates[i, j, c] for i, j in cells])
        for larger in range(c + 1, len(SPEEDS)):
            other = np.array([rates[i, j, larger] for i, j in cells])
            one_sided = wilcoxon(other, own, alternative="greater")
            two_sided = wilcoxon(other, own)
            rows.append(
                {
                    "fitted_delta_mps": SPEEDS[c],
                    "landscape_delta_mps": SPEEDS[larger],
                    "n": len(own),
                    "mean_own_landscape_pct": float(own.mean()),
                    "mean_larger_landscape_pct": float(other.mean()),
                    "n_increased": int(np.sum(other > own)),
                    "n_equal": int(np.sum(other == own)),
                    "n_decreased": int(np.sum(other < own)),
                    "wilcoxon_statistic": float(one_sided.statistic),
                    "p_one_sided_greater": float(one_sided.pvalue),
                    "p_two_sided": float(two_sided.pvalue),
                    "method": "paired Wilcoxon signed-rank, scipy.stats.wilcoxon",
                }
            )
    return rows


def landscape_tables(data_root: Path) -> dict[str, list[dict]]:
    """Tabulate the Figure 4 landscape, participant fits and ellipse fall risk.

    Each condition's ellipse is centered at the participant medians of ψ and α, with
    semi-axes equal to their standard errors of the mean. Its fall risk is the mean
    probability of the grid nodes inside it, evaluated on its own landscape and on those of
    larger belt-speed differences, and rounded half up for display.

    Args:
        data_root: Prepared publication data directory.

    Returns:
        Dict of row lists: ``landscape``, ``participant_parameters``, ``ellipses``,
        ``ellipse_risk``, ``fall_probability_tests`` and the tables of
        ``parameter_statistics``.
    """
    alpha, psi, rates = landscape_data(data_root)
    grid = [
        {
            "delta_mps": speed,
            "learning_rate": float(a),
            "symmetry_weight": float(p),
            "fall_probability_pct": float(rates[i, j, c]),
        }
        for c, speed in enumerate(SPEEDS)
        for i, a in enumerate(alpha)
        for j, p in enumerate(psi)
    ]
    value = _mat(Path(data_root) / "original/2d_sweep_trial50_param_result.mat")
    n = np.asarray(value["exp_subj_num"]).ravel().astype(int)
    parameters = np.asarray(value["static_params"])
    participants, ellipses, risks, fitted = [], [], [], []
    x, y = np.meshgrid(psi, alpha)
    for c, speed in enumerate(SPEEDS):
        lr = np.array([parameters[c, s].learning_rate for s in range(n[c])], dtype=float)
        sym = 1 - np.array([parameters[c, s].sym_cost for s in range(n[c])], dtype=float)
        if not np.isfinite(lr).all() or not np.isfinite(sym).all():
            raise ValueError("Non-finite fitted parameters")
        fitted.append((lr, sym))
        cx, cy = float(np.median(sym)), float(np.median(lr))
        rx, ry = float(sym.std(ddof=1) / np.sqrt(n[c])), float(lr.std(ddof=1) / np.sqrt(n[c]))
        if rx == 0 or ry == 0:
            raise ValueError(
                "A zero-width ellipse cannot reproduce the original grid inclusion rule"
            )
        ellipses.append(
            {
                "delta_mps": speed,
                "n": int(n[c]),
                "center_symmetry_weight": cx,
                "center_learning_rate": cy,
                "sem_symmetry_weight": rx,
                "sem_learning_rate": ry,
            }
        )
        for s, (a, p) in enumerate(zip(lr, sym), 1):
            participants.append(
                {
                    "delta_mps": speed,
                    "participant_index": s,
                    "learning_rate": float(a),
                    "symmetry_weight": float(p),
                }
            )
        for prior, ellipse in enumerate(ellipses):
            mask = (
                (x - ellipse["center_symmetry_weight"]) / ellipse["sem_symmetry_weight"]
            ) ** 2 + (
                (y - ellipse["center_learning_rate"]) / ellipse["sem_learning_rate"]
            ) ** 2 <= 1
            mean = float(rates[:, :, c][mask].mean()) if mask.any() else float("nan")
            risks.append(
                {
                    "landscape_delta_mps": speed,
                    "ellipse_delta_mps": SPEEDS[prior],
                    "included_grid_nodes": int(mask.sum()),
                    "mean_fall_probability_pct": mean,
                    "displayed_integer_pct": int(np.floor(mean + 0.5)) if np.isfinite(mean) else "",
                }
            )
    return {
        "landscape": grid,
        "participant_parameters": participants,
        "ellipses": ellipses,
        "ellipse_risk": risks,
        "fall_probability_tests": fall_probability_tests(alpha, psi, rates, fitted),
        **parameter_statistics(data_root),
    }


def _figure4(data_root: Path, output_root: Path) -> dict:
    """Export Figure 4 tables and the landscape plot with parameter ellipses."""
    tables = landscape_tables(data_root)
    outputs = [_csv(output_root, f"figure4_{key}", rows) for key, rows in tables.items()]
    alpha, psi, rates = landscape_data(data_root)
    plt = _plotting()
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True, layout="constrained")
    for c, (axis, speed) in enumerate(zip(axes, SPEEDS)):
        display = axis.pcolormesh(
            psi, alpha, rates[:, :, c] / 100, shading="nearest", vmin=0, vmax=1, cmap="viridis"
        )
        for prior in range(c + 1):
            row = tables["ellipses"][prior]
            risk = next(
                r
                for r in tables["ellipse_risk"]
                if r["landscape_delta_mps"] == speed and r["ellipse_delta_mps"] == SPEEDS[prior]
            )
            cx, cy = row["center_symmetry_weight"], row["center_learning_rate"]
            axis.add_patch(
                Ellipse(
                    (cx, cy),
                    2 * row["sem_symmetry_weight"],
                    2 * row["sem_learning_rate"],
                    facecolor="#b00000" if prior == c else "gray",
                    alpha=0.6,
                    edgecolor="black",
                )
            )
            axis.plot(cx, cy, "o", color="black", markersize=2)
            axis.annotate(
                f"{risk['displayed_integer_pct'] / 100:.2f}",
                (cx, cy),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set(
            xlabel="Symmetry weight ψ", title=f"{speed:g} m/s", xlim=(0, 1), ylim=(0.005, 0.05)
        )
    axes[0].set_ylabel("Learning rate α")
    fig.colorbar(display, ax=axes, label="Fall probability", fraction=0.025, pad=0.03)
    fig.suptitle("Figure 4: original cached landscape and inferred parameter regions")
    outputs += _save(output_root, "figure4", fig)
    return _record(
        "figure4",
        "passed",
        outputs,
        [
            "Original cached landscape; raw trial counts, seeds and simulation revision are absent from its MAT metadata.",
            "Ellipse values are arithmetic means of included grid nodes, not significance tests.",
            (
                "The increase in fall probability for unchanged parameters is tested per "
                "participant: fall probability at the fitted grid cell on the own-condition "
                "landscape versus each larger belt-speed landscape, paired Wilcoxon signed-rank "
                "test (figure4_fall_probability_tests.csv)."
            ),
            "Across-condition parameter tests compare fitted learning rates and symmetry weights.",
            (
                "The manually drawn arrows and MATLAB colormap of the published figure are not "
                "reproduced."
            ),
        ],
        {
            "grid_shape": list(rates.shape),
            "participant_n": [8, 9, 9],
            "parameter_kruskal_wallis": [
                {
                    "fit_source": r["fit_source"],
                    "parameter": r["parameter"],
                    "H": r["statistic"],
                    "p_value": r["p_value"],
                }
                for r in tables["parameter_tests"]
                if r["test"] == "kruskal_wallis"
            ],
            "fall_probability_tests": [
                {
                    "fitted_delta_mps": r["fitted_delta_mps"],
                    "landscape_delta_mps": r["landscape_delta_mps"],
                    "p_one_sided_greater": r["p_one_sided_greater"],
                }
                for r in tables["fall_probability_tests"]
            ],
            "current_ellipse_displayed_pct": [
                r["displayed_integer_pct"]
                for r in tables["ellipse_risk"]
                if r["landscape_delta_mps"] == r["ellipse_delta_mps"]
            ],
        },
    )


def _s3(data_root: Path, output_root: Path) -> dict:
    """Export Supplementary Figure 3: fall risk versus ψ at fixed α and versus α at fixed ψ."""
    alpha, psi, rates = landscape_data(data_root)
    if rates.shape != (20, 20, 3):
        raise ValueError("Supplementary Figure 3 requires the original 20 × 20 × 3 landscape")
    # Grid indices of the fixed α and ψ values in the original plotting script.
    ai, pi = [0, 2, 6, 15], [0, 9, 19]
    rows, summary = [], []
    plt = _plotting()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    for c, speed in enumerate(SPEEDS):
        for row, (indices, axis_values, fixed_name, swept_name) in enumerate(
            (
                (ai, psi, "learning_rate", "symmetry_weight"),
                (pi, alpha, "symmetry_weight", "learning_rate"),
            )
        ):
            for index in indices:
                fixed = float(alpha[index] if row == 0 else psi[index])
                values = rates[index, :, c] if row == 0 else rates[:, index, c]
                axes[row, c].plot(
                    axis_values,
                    values,
                    "o-",
                    markersize=2,
                    label=f"{'α' if row == 0 else 'ψ'}={fixed:.4g}",
                )
                for x, value in zip(axis_values, values):
                    rows.append(
                        {
                            "delta_mps": speed,
                            "sweep": "psi_sweep" if row == 0 else "alpha_sweep",
                            "fixed_parameter": fixed_name,
                            "fixed_value": fixed,
                            "swept_parameter": swept_name,
                            "swept_value": float(x),
                            "fall_probability_pct": float(value),
                        }
                    )
                summary.append(
                    {
                        "sweep": "psi_sweep" if row == 0 else "alpha_sweep",
                        "delta_mps": speed,
                        "held_fixed": "alpha" if row == 0 else "psi",
                        "fixed_value": fixed,
                        "fall_pct_at_low_end": float(values[0]),
                        "fall_pct_at_high_end": float(values[-1]),
                        "end_to_end_change_pp": float(values[-1] - values[0]),
                    }
                )
            axes[row, c].set(
                xlabel="Symmetry weight ψ" if row == 0 else "Learning rate α",
                ylim=(-4, 104),
                title=f"{speed:g} m/s",
            )
            axes[row, c].legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("Fall probability (%)")
    axes[1, 0].set_ylabel("Fall probability (%)")
    fig.suptitle("Supplementary Figure 3: slices of the original cached landscape")
    fig.tight_layout()
    outputs = [
        _csv(output_root, "s3_plotted_points", rows),
        _csv(output_root, "s3_endpoint_summary", summary),
    ]
    outputs += _save(output_root, "s3_fall_risk_slices", fig)
    return _record(
        "s3",
        "passed",
        outputs,
        [
            (
                "Slices use the grid indices of the original plotting script; the corresponding "
                "parameter values are exported."
            ),
            "Cached probabilities are reproduced without assuming unrecorded trial counts or seeds.",
        ],
        {"plotted_points": len(rows), "summary_rows": len(summary)},
    )


def render_main(data_root: Path, output_root: Path, figures: list[str] | None = None) -> list[dict]:
    """Render Figures 1, 3 and 4 and the Supplementary Figure 3 landscape slices (``s3``).

    Args:
        data_root: Prepared publication data directory.
        output_root: Directory that receives ``figures/`` and ``tables/``.
        figures: Subset of ``figure1``, ``figure3``, ``figure4`` and ``s3``; all if None.

    Returns:
        One result record per figure. A figure with missing inputs is recorded as
        ``blocked``; other errors propagate.

    Raises:
        ValueError: If ``figures`` names an unknown figure.
    """
    data_root, output_root = Path(data_root), Path(output_root)
    functions = {"figure1": _figure1, "figure3": _figure3, "figure4": _figure4, "s3": _s3}
    selected = list(functions) if figures is None else list(figures)
    unknown = set(selected) - functions.keys()
    if unknown:
        raise ValueError(f"Unknown main figure selection: {', '.join(sorted(unknown))}")
    records = []
    for name in selected:
        try:
            records.append(functions[name](data_root, output_root))
        except FileNotFoundError as error:
            records.append(_record(name, "blocked", caveats=[f"{type(error).__name__}: {error}"]))
    return records
