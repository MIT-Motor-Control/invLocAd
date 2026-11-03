"""Analyze the regenerated-realization inputs for Figures 2B and 2C and the grid check.

Figure 2B and the cross-validation grid-sensitivity check of S1 Text Section 1 use the
regenerated fits. The inverse adaptation model energy changes of Figure 2C apply the original
published participant parameters to a separate regenerated rollout realization.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REGENERATED_PARAMETERS = "regenerated/2d_sweep_trial50_param_result.mat"
REGENERATED_ENERGY = "regenerated/data/cosmed/processed/processedEnergyData.mat"
TRIAL_AVERAGE_GRID = "regenerated/2d_sweep_trial50_param_result_trialavg.mat"
FIGURE2C_ROLLOUTS = "regenerated/data/2d sweep fine"
S1_GRID_PREFIX = "regenerated/figures/comment1_grid_sensitivity"


def _load(root: Path, relative: str) -> dict:
    """Load ``root/relative`` as a MAT file; raise FileNotFoundError if it is absent."""
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required publication input is missing: {relative}")
    return loadmat(path, struct_as_record=False, squeeze_me=True)


def normalize_raw_trajectory(values: np.ndarray) -> np.ndarray:
    """Return a raw energy trajectory as percent change from its split-onset mean.

    Zero and nonfinite samples are dropped and the rest is smoothed with a 100-sample
    centered moving mean, as MATLAB ``movmean(x, 100)``. The smoothed signal starts at
    sample max(30, round(0.03 n)) (1-based), where n is the number of retained samples,
    and is normalized to the mean of its first 120 samples.

    Args:
        values: Raw experimental (1 Hz) or model (per-stride) energy samples.
    """
    values = np.asarray(values, float).ravel()
    values = values[np.isfinite(values) & (values != 0)]
    if values.size < 30:
        raise ValueError("Figure 2B raw trajectory cannot support its startup-trim window")
    index = np.arange(values.size)
    lower, upper = np.maximum(0, index - 50), np.minimum(values.size, index + 50)
    cumulative = np.r_[0.0, np.cumsum(values)]
    smooth = (cumulative[upper] - cumulative[lower]) / (upper - lower)
    onset = max(30, round(0.03 * values.size))
    smooth = smooth[onset - 1 :]
    baseline = float(smooth[:120].mean())
    if baseline == 0 or not np.isfinite(baseline):
        raise ValueError("Figure 2B split-onset baseline must be finite and nonzero")
    return (smooth - baseline) / baseline * 100


def figure2_group_trajectories(root: Path) -> tuple[dict, list[dict], dict]:
    """Return the Figure 2B experiment and model energy trajectories at 0.7 m/s.

    Each of the nine condition-2 participants and their nine regenerated model fits is
    normalized by ``normalize_raw_trajectory`` and interpolated onto 270 points spanning
    0-100 % of the exposure.

    Args:
        root: Prepared publication data directory.

    Returns:
        Tuple of a dict mapping ``experiment`` and ``model`` to (9, 270) arrays, long-format
        sample rows, and group-mean agreement metrics. Plateau metrics average the last 27
        samples (the final 10 % of exposure).
    """
    energy = _load(root, REGENERATED_ENERGY)
    parameters = _load(root, REGENERATED_PARAMETERS)["static_params"]
    raw_experiment = np.asarray(energy["split_energy"].condition_2, float)
    if raw_experiment.shape != (9, 2700) or parameters.shape != (3, 9):
        raise ValueError(
            "Figure 2B requires the regenerated 3×9 fit and nine condition-2 subjects"
        )
    grid = np.linspace(0, 100, 270)
    groups, rows = {}, []
    raw_model = [np.asarray(p.model_energy_raw, float).ravel() for p in parameters[1]]
    for label, signals in (("experiment", raw_experiment), ("model", raw_model)):
        trajectories = []
        for subject, signal in enumerate(signals, 1):
            normalized = normalize_raw_trajectory(signal)
            trajectory = np.interp(grid, np.linspace(0, 100, normalized.size), normalized)
            trajectories.append(trajectory)
            rows.extend(
                {
                    "source": label,
                    "condition": 2,
                    "delta_mps": 0.7,
                    "subject": subject,
                    "sample_index": index,
                    "exposure_pct": float(exposure),
                    "energy_change_pct": float(value),
                }
                for index, (exposure, value) in enumerate(zip(grid, trajectory))
            )
        groups[label] = np.asarray(trajectories)
    experiment, model = (groups[name].mean(axis=0) for name in ("experiment", "model"))
    metrics = {
        "condition": 2,
        "delta_mps": 0.7,
        "n_data": groups["experiment"].shape[0],
        "n_model": groups["model"].shape[0],
        "pearson_r": float(np.corrcoef(experiment, model)[0, 1]),
        "rmse_pct": float(np.sqrt(np.mean((experiment - model) ** 2))),
        "data_plateau_pct": float(experiment[243:].mean()),
        "model_plateau_pct": float(model[243:].mean()),
    }
    return groups, rows, metrics


def grid_finite_counts(root: Path) -> list[dict]:
    """Count trial-average grid cells with finite RMSE for at least 26, 17 or 8 participants.

    Args:
        root: Prepared publication data directory.

    Returns:
        One row per threshold with the number of admissible cells out of 400.
    """
    source = _load(root, TRIAL_AVERAGE_GRID)
    rmse, nsub = np.asarray(source["rmse_matrix"]), np.asarray(source["exp_subj_num"])
    if rmse.shape != (3, 9, 20, 20) or not np.array_equal(nsub, [8, 9, 9]):
        raise ValueError("Grid sensitivity requires the regenerated 26-subject 20×20 RMSE grid")
    counts = sum(np.isfinite(rmse[c, s]) for c in range(3) for s in range(int(nsub[c])))
    return [
        {"minimum_finite_subjects": threshold, "admissible_cells": int((counts >= threshold).sum()),
         "total_cells": 400, "participants": 26}
        for threshold in (26, 17, 8)
    ]


def figure2c_model_tables(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Summarize the Figure 2C model rollouts at each participant's fitted grid cell.

    Each original participant fit is mapped to the nearest learning-rate and sym_cost grid
    values. A nonfalling trial's energy change compares the mean of the last 100 strides
    with the tied mean over strides 31-70; a participant's value is the median over the
    nonfalling trials of the cell. Participants that share a cell share its trials, so
    condition totals weight each cell by its number of participants.

    Args:
        root: Prepared publication data directory.

    Returns:
        Tuple of per-trial rows, per-participant rows and per-condition summaries.
    """
    source = _load(root, "original/2d_sweep_trial50_param_result.mat")
    parameters = source["static_params"]
    nsub = np.asarray(source["exp_subj_num"], int)
    learning, symmetry = np.asarray(source["learning_pool"]), np.asarray(source["sym_cost_pool"])
    if parameters.shape != (3, 9) or not np.array_equal(nsub, [8, 9, 9]):
        raise ValueError("Figure 2C requires the original published 8/9/9 participant parameters")
    for pool in (learning, symmetry):
        if pool.ndim != 1 or not pool.size or not np.isfinite(pool).all() or np.any(np.diff(pool) <= 0):
            raise ValueError(
                "Figure 2C parameter pools must be finite, increasing one-dimensional grids"
            )
    needed = {}
    for condition, speed in enumerate((0.4, 0.7, 1.0), 1):
        for subject in range(1, nsub[condition - 1] + 1):
            p = parameters[condition - 1, subject - 1]
            if not np.isfinite([p.learning_rate, p.sym_cost]).all():
                raise ValueError("Figure 2C fitted participant parameters must be finite")
            key = (round(1 - speed / 2, 3), int(np.argmin(abs(learning - p.learning_rate))),
                   int(np.argmin(abs(symmetry - p.sym_cost))))
            needed.setdefault(key, []).append((condition, speed, subject))
    pattern = re.compile(
        r"model_split45_trans20_slow([0-9.]+)_fast([0-9.]+)_noise0\.01_"
        r"learning([0-9.]+)_energysym([0-9.]+)_swingcost1_trial(\d+)\.mat"
    )
    indexed = {key: {} for key in needed}
    for path in sorted((root / FIGURE2C_ROLLOUTS).glob("*.mat")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        slow, fast, alpha, psi = map(float, match.groups()[:4])
        if not np.isclose(slow + fast, 2, rtol=0, atol=1e-12):
            raise ValueError(f"Unexpected Figure 2C treadmill speeds in {path.name}")
        key = (round(slow, 3), int(np.argmin(abs(learning - alpha))),
               int(np.argmin(abs(symmetry - psi))))
        if key not in needed:
            continue
        # Filenames give the learning rate and sym_cost to five significant digits.
        if (
            not np.isclose(slow, key[0], rtol=0, atol=1e-12)
            or abs(alpha - learning[key[1]]) > 5.1e-7
            or abs(psi - symmetry[key[2]]) > 5.1e-6
        ):
            raise ValueError(
                f"Figure 2C rollout filename does not identify its fitted grid cell: {path.name}"
            )
        trial = int(match.group(5))
        if trial in indexed[key]:
            raise ValueError(f"Duplicate Figure 2C trial {trial} at fitted grid cell {key}")
        indexed[key][trial] = path
    missing = sum(len(set(range(1, 51)) - set(paths)) for paths in indexed.values())
    if missing:
        raise FileNotFoundError(
            "Required publication input is missing: "
            f"{missing} Figure 2C rollouts in {FIGURE2C_ROLLOUTS}"
        )
    trials, participants = [], []
    for (slow, ia, ip), paths in indexed.items():
        if set(paths) != set(range(1, 51)):
            raise ValueError("Figure 2C fitted cells require exactly trials 1–50")
        values, fallen = [], 0
        condition, speed, _ = needed[(slow, ia, ip)][0]
        for trial, path in sorted(paths.items()):
            data = loadmat(path, struct_as_record=False, squeeze_me=True,
                           variable_names=["energy", "fall_flag"])
            energy = np.asarray(data["energy"], float).ravel()
            flag = np.asarray(data["fall_flag"], float).ravel()
            if flag.size != 2 or not np.isfinite(flag).all() or flag[0] not in (0, 1):
                raise ValueError(f"Invalid Figure 2C fall flag: {path.name}")
            if not energy.size or not np.isfinite(energy).all():
                raise ValueError(f"Invalid Figure 2C energy signal: {path.name}")
            fall = int(flag[0])
            baseline = adapted = change = None
            if fall:
                fallen += 1
            else:
                if energy.size < 100:
                    raise ValueError(f"Incomplete nonfalling Figure 2C energy signal: {path.name}")
                baseline, adapted = float(energy[30:70].mean()), float(energy[-100:].mean())
                if baseline == 0:
                    raise ValueError(f"Zero Figure 2C tied baseline: {path.name}")
                change = (adapted - baseline) / baseline * 100
                values.append(change)
            trials.append({
                "condition": condition, "delta_mps": speed, "learning_grid_index": ia + 1,
                "symmetry_grid_index": ip + 1, "trial": trial, "fall_flag": fall,
                "tied_stride31_70_mean": baseline, "adapted_last100_mean": adapted,
                "energy_change_pct": change, "input": path.relative_to(root).as_posix(),
            })
        if not values:
            raise ValueError(f"No nonfalling Figure 2C trials at fitted grid cell {(slow, ia, ip)}")
        for condition, speed, subject in needed[(slow, ia, ip)]:
            participants.append({
                "condition": condition, "delta_mps": speed, "subject": subject,
                "learning_rate": float(learning[ia]), "sym_cost": float(symmetry[ip]),
                "learning_grid_index": ia + 1, "symmetry_grid_index": ip + 1,
                "trials": 50, "falling_trials": fallen, "successful_trials": len(values),
                "median_energy_change_pct": float(np.median(values)),
            })
    participants.sort(key=lambda r: (r["condition"], r["subject"]))
    summary = []
    for condition, speed in enumerate((0.4, 0.7, 1.0), 1):
        rows = [r for r in participants if r["condition"] == condition]
        values = np.array([r["median_energy_change_pct"] for r in rows])
        summary.append({
            "condition": condition, "delta_mps": speed, "participants": len(rows),
            "mean_energy_change_pct": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "participant_weighted_trials": sum(r["trials"] for r in rows),
            "participant_weighted_falls": sum(r["falling_trials"] for r in rows),
            "unique_cells": len({(r["learning_grid_index"], r["symmetry_grid_index"]) for r in rows}),
        })
    return trials, participants, summary
