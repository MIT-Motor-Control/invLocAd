"""Refit the fall-penalty model grid and evaluate leave-one-subject-out predictions.

Used by the ``s4`` renderer for Supplementary Figure 4 (models with and without a fall penalty).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def fit_energy_exponential(values: np.ndarray) -> np.ndarray:
    """Return the bounded least-squares fit of A + B exp(-t / tau) on a 2700-s grid.

    Only finite samples are fitted. The fit starts from (last value, first minus last
    value, 300 s) with bounds |A| <= 200, |B| <= 500 and 5 <= tau <= 8100 s.

    Args:
        values: Energy change (%) at 1 Hz, shape (2700,).

    Returns:
        Fitted curve at t = 0, ..., 2699 s.
    """
    y = np.asarray(values, float)
    if y.shape != (2700,):
        raise ValueError("Fall-penalty exponential fitting requires 2700 one-Hz samples")
    finite = np.isfinite(y)
    if finite.sum() < 10:
        raise ValueError("Fall-penalty exponential fitting requires at least ten finite samples")
    t = np.arange(2700, dtype=float)

    def model(time, a, b, tau):
        """Return A + B exp(-t / tau) at ``time`` (s)."""
        return a + b * np.exp(-time / tau)

    v = y[finite]
    parameters, _ = curve_fit(
        model,
        t[finite],
        v,
        p0=[v[-1], v[0] - v[-1], 300.0],
        bounds=([-200, -500, 5.0], [200, 500, 8100.0]),
        maxfev=10000,
    )
    fitted = model(t, *parameters)
    if not np.all(np.isfinite(fitted)):
        raise ValueError("Fall-penalty exponential fitting produced nonfinite values")
    return fitted


def _rmse(first, second):
    """Return the RMSE over samples where both inputs are finite."""
    difference = np.asarray(first) - np.asarray(second)
    finite = difference[np.isfinite(difference)]
    if not finite.size:
        raise ValueError("Fall-penalty RMSE has no finite paired samples")
    return float(np.sqrt(np.mean(finite**2)))


def _paired(rows, speed):
    """Return the paired Wilcoxon test of held-out RMSE C - E at one speed or pooled."""
    selected = [r for r in rows if speed == "pooled" or r["delta_mps"] == speed]
    difference = np.array([r["rmse_C"] - r["rmse_E"] for r in selected])
    nonzero = int(np.count_nonzero(difference))
    return {
        "delta_mps": speed,
        "n": len(selected),
        "n_effective": nonzero,
        "n_zero_differences": len(selected) - nonzero,
        "mean_difference_C_minus_E": float(difference.mean()),
        "median_difference_C_minus_E": float(np.median(difference)),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(difference).pvalue) if nonzero else 1.0,
        "method": "scipy.wilcoxon, two-sided, zero_method=wilcox, method=auto",
    }


def analyze_fall_penalty_grid(
    data_root: Path, experimental: dict, participant_ids: dict
) -> dict:
    """Refit the fall-penalty cells, select models C and E, and run leave-one-subject-out tests.

    Model C selects learning rate, symmetry weight and fall weight per speed; model E fixes
    the fall weight at zero. Cell mean trajectories and participant-group mean trajectories
    are summarized by ``fit_energy_exponential``, and models are selected by the minimum
    RMSE between fits. Each fold selects on the remaining participants and scores the
    held-out participant; the noise floor is the RMSE between the training-group and
    held-out fits. Cell metadata and seeds are validated against ``grid_config.json``;
    saved selections and summaries are not used.

    Args:
        data_root: Prepared publication data directory containing ``extensions/s5``.
        experimental: Speed to (participants, 2700) array of smoothed, onset-normalized
            energy change (%) at 1 Hz.
        participant_ids: Speed to participant identifiers in the row order of
            ``experimental``.

    Returns:
        Dict with ``tables`` (folds, summaries, Wilcoxon tests with Holm adjustment across
        the three speeds, group selections, per-cell mechanical summaries and their
        fall-weight slices at the model C selections), ``sources`` (relative input paths)
        and ``details``.
    """
    base = Path(data_root) / "extensions/s5"
    config = json.loads((base / "grid_config.json").read_text())
    speeds = config["speeds"]
    axes = [config[k] for k in ("learning_rates", "symmetry_weights", "fall_weights")]
    if len(axes[2]) != len(set(axes[2])) or 0.0 not in axes[2]:
        raise ValueError("Fall-weight grid must have distinct values including zero")
    cells, mechanical, sources = {}, [], ["extensions/s5/grid_config.json"]
    for speed, lr, sym, weight in product(speeds, *axes):
        relative = f"extensions/s5/cells/sp{speed:g}_lr{lr:.3e}_sym{sym:g}_w{weight:g}.npz"
        with np.load(Path(data_root) / relative, allow_pickle=False) as cell:
            for name, expected in (
                ("speed", speed),
                ("learning_rate", lr),
                ("symmetry_weight", sym),
                ("fall_weight", weight),
            ):
                if not np.isclose(float(cell[name]), expected, rtol=1e-12, atol=0):
                    raise ValueError(f"Fall-penalty cell metadata differs: {relative}: {name}")
            seeds, failed = np.asarray(cell["ok_seeds"]), np.asarray(cell["failed_seeds"])
            if sorted(np.r_[seeds, failed].tolist()) != sorted(config["seeds"]):
                raise ValueError(f"Fall-penalty cell has missing or duplicate seeds: {relative}")
            energy = np.asarray(cell["energy_pct"], float)
            penalty, height = (
                np.asarray(cell["mean_penalty"], float),
                np.asarray(cell["min_com"], float),
            )
            if energy.shape != (seeds.size, 2700) or not seeds.size:
                raise ValueError(f"Fall-penalty cell has invalid energy trajectories: {relative}")
            if penalty.shape != seeds.shape or height.shape != seeds.shape:
                raise ValueError(f"Fall-penalty cell has invalid mechanical records: {relative}")
            if not np.all(np.isfinite(energy)) or not np.all(np.isfinite([penalty, height])):
                raise ValueError(f"Fall-penalty cell contains nonfinite values: {relative}")
            curve = fit_energy_exponential(np.mean(energy, axis=0))
            key = (speed, lr, sym, weight)
            cells[key] = curve
            mechanical.append(
                {
                    "delta_mps": speed,
                    "learning_rate": lr,
                    "symmetry_weight": sym,
                    "fall_weight": weight,
                    "successful_seeds": int(seeds.size),
                    "failed_seeds": int(failed.size),
                    "mean_penalty": float(penalty.mean()),
                    "min_com_height_over_seeds": float(height.min()),
                    "max_seed_min_com_height": float(height.max()),
                    "min_seed_mean_penalty": float(penalty.min()),
                    "max_seed_mean_penalty": float(penalty.max()),
                }
            )
        sources.append(relative)
    selections, folds = [], []
    for speed in speeds:
        matrix = np.asarray(experimental[speed], float)
        ids = participant_ids[speed]
        if matrix.ndim != 2 or matrix.shape != (len(ids), 2700) or len(ids) < 2:
            raise ValueError("Fall-penalty participant matrix and identifiers differ")
        keys = [key for key in cells if key[0] == speed]
        zero_keys = [key for key in keys if key[-1] == 0]
        group = fit_energy_exponential(np.nanmean(matrix, axis=0))
        for model, options in (("C", keys), ("E", zero_keys)):
            chosen = min(options, key=lambda key: _rmse(cells[key], group))
            selections.append(
                {
                    "delta_mps": speed,
                    "model": model,
                    "learning_rate": chosen[1],
                    "symmetry_weight": chosen[2],
                    "fall_weight": chosen[3],
                    "group_fit_rmse": _rmse(cells[chosen], group),
                }
            )
        for i, participant in enumerate(ids):
            held = fit_energy_exponential(matrix[i])
            train = fit_energy_exponential(np.nanmean(np.delete(matrix, i, axis=0), axis=0))
            row = {
                "delta_mps": speed,
                "subject_id": participant,
                "training_participants": len(ids) - 1,
                "noise_floor_rmse": _rmse(train, held),
            }
            for model, options in (("C", keys), ("E", zero_keys)):
                chosen = min(options, key=lambda key: _rmse(cells[key], train))
                row.update(
                    {
                        f"rmse_{model}": _rmse(cells[chosen], held),
                        f"learning_rate_{model}": chosen[1],
                        f"symmetry_weight_{model}": chosen[2],
                        f"fall_weight_{model}": chosen[3],
                    }
                )
            folds.append(row)
    summaries = []
    for speed in (*speeds, "pooled"):
        rows = [r for r in folds if speed == "pooled" or r["delta_mps"] == speed]
        for label, column in (
            ("C", "rmse_C"),
            ("E", "rmse_E"),
            ("noise_floor", "noise_floor_rmse"),
        ):
            values = np.array([r[column] for r in rows])
            summaries.append(
                {
                    "delta_mps": speed,
                    "model": label,
                    "n": len(rows),
                    "mean_rmse": float(values.mean()),
                    "median_rmse": float(np.median(values)),
                }
            )
    tests = [_paired(folds, speed) for speed in (*speeds, "pooled")]
    ordered = sorted(tests[:-1], key=lambda row: row["wilcoxon_two_sided_p"])
    running = 0.0
    for i, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * row["wilcoxon_two_sided_p"]))
        row["holm_p_within_three_conditions"] = running
    tests[-1]["holm_p_within_three_conditions"] = "not_in_family"
    # Hold the fitted learning rate and symmetry weight fixed while varying fall weight.
    mechanical_slices = [
        row
        for selected in selections
        if selected["model"] == "C"
        for row in mechanical
        if all(row[k] == selected[k] for k in ("delta_mps", "learning_rate", "symmetry_weight"))
    ]
    return {
        "tables": {
            "s4_loso_folds": folds,
            "s4_loso_summary": summaries,
            "s4_loso_tests": tests,
            "s4_recomputed_selection": selections,
            "s4_mechanical_cells": mechanical,
            "s4_mechanical_weight_slices": mechanical_slices,
        },
        "sources": sources,
        "details": {
            "cached_cells_refitted": len(cells),
            "loso_folds": len(folds),
            "simulation_rerun": False,
            "fit_fallbacks": 0,
            "primary_metric": "RMSE between bounded exponential fits over 2700 seconds",
            "noise_floor": "RMSE between the training-group and held-participant exponential fits",
            "mechanical_resolution": "Per-seed mean penalty and minimum COM height; individual step penalties are unavailable",
        },
    }
