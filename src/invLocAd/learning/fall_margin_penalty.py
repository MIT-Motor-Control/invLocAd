"""Instability penalty ``Phi`` of the fall-penalized objective (paper Eq. 2).

Penalties are averaged over the stance phases of one learning iteration
(``state_store`` from ``simulate_many_steps_asymmetric_control``). The default
``com_margin`` mode is the COM-height penalty described with Eq. 2;
``capture_point`` is an alternative based on the extrapolated center of mass
(XCoM).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def compute_step_min_com_heights(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> np.ndarray:
    """Return the minimum COM height ``leglength * cos(theta)`` of each stance phase.

    Args:
        state_store: Stance trajectories, rows ``[theta, theta_dot, y_foot]``;
            empty entries are skipped.
        param_fixed: Model parameters with ``leglength``.

    Returns:
        Minimum COM height of each non-empty stance phase.
    """
    leg_length = float(param_fixed["leglength"])
    min_heights: list[float] = []
    for step_state in state_store:
        if step_state is None or len(step_state) == 0:
            continue
        theta = np.asarray(step_state)[:, 0]
        min_heights.append(float(np.min(leg_length * np.cos(theta))))
    return np.asarray(min_heights, dtype=float)


def compute_trial_min_com_height(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Return the lowest COM height across all stance phases, or NaN if there are none.

    Arguments are as in ``compute_step_min_com_heights``.
    """
    min_heights = compute_step_min_com_heights(state_store, param_fixed)
    if min_heights.size == 0:
        return float("nan")
    return float(np.min(min_heights))


def compute_step_max_abs_capture_points(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> np.ndarray:
    """Return the maximum absolute XCoM of each stance phase.

    The XCoM is ``x + v / omega`` with ``omega = sqrt(gravg / leglength)``,
    where ``x`` and ``v`` are the fore-aft COM position and velocity relative to
    the stance foot. Arguments are as in ``compute_step_min_com_heights``;
    ``param_fixed`` also needs ``gravg``.
    """
    leg_length = float(param_fixed["leglength"])
    omega = np.sqrt(float(param_fixed["gravg"]) / leg_length)
    angle_slope = float(param_fixed.get("angleSlope", 0.0))
    max_capture_points: list[float] = []
    for step_state in state_store:
        if step_state is None or len(step_state) == 0:
            continue
        step_state = np.asarray(step_state)
        theta = step_state[:, 0]
        theta_dot = step_state[:, 1]
        com_position = leg_length * np.sin(theta + angle_slope)
        com_velocity = leg_length * np.cos(theta + angle_slope) * theta_dot
        capture_point = com_position + com_velocity / omega
        max_capture_points.append(float(np.max(np.abs(capture_point))))
    return np.asarray(max_capture_points, dtype=float)


def compute_com_margin_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Return the COM-height penalty (``com_margin`` mode).

    ``Phi = mean(clip((h_safe - h_min) / (h_safe - h_ground), 0, 1) ** power)``
    over stance phases, where ``h_min`` is the minimum COM height of a phase.
    ``Phi`` is zero while the COM stays above ``fallSafeComHeight`` (``h_safe``,
    default 0.95 leg lengths) and reaches 1 at ``fallGroundComHeight``
    (``h_ground``, default 0.90); the default ``fallPenaltyPower`` of 2 makes it
    quadratic in between.

    Args:
        state_store: Stance trajectories, as in ``compute_step_min_com_heights``.
        param_fixed: Model parameters with ``leglength`` and the optional
            settings above.

    Returns:
        ``Phi`` in [0, 1]; zero when there are no stance phases.
    """
    power = float(param_fixed.get("fallPenaltyPower", 2.0))
    eps = float(param_fixed.get("fallPenaltyEpsilon", 1e-6))
    safe_height = float(param_fixed.get("fallSafeComHeight", 0.95))
    ground_height = float(param_fixed.get("fallGroundComHeight", 0.90))
    span = max(safe_height - ground_height, eps)

    min_heights = compute_step_min_com_heights(state_store, param_fixed)
    if min_heights.size == 0:
        return 0.0
    margins = np.clip((safe_height - min_heights) / span, 0.0, 1.0)
    return float(np.mean(margins**power))


def compute_capture_point_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Return the mean bounded XCoM excess over stance phases.

    Each phase contributes ``clip((max|XCoM| - limit) / scale, 0, 1) ** power``,
    with ``fallCapturePointLimit`` and ``fallCapturePointScale``.

    Args:
        state_store: Stance trajectories, as in ``compute_step_min_com_heights``.
        param_fixed: Model parameters with ``leglength``, ``gravg`` and the
            optional settings above.

    Returns:
        Penalty in [0, 1]; zero when there are no stance phases.
    """
    power = float(param_fixed.get("fallPenaltyPower", 2.0))
    eps = float(param_fixed.get("fallPenaltyEpsilon", 1e-6))
    capture_points = compute_step_max_abs_capture_points(state_store, param_fixed)
    if capture_points.size == 0:
        return 0.0
    limit = float(param_fixed.get("fallCapturePointLimit", 1.0))
    scale = float(param_fixed.get("fallCapturePointScale", 1.0))
    deficits = np.maximum(0.0, capture_points - limit)
    bounded = np.clip(deficits / max(scale, eps), 0.0, 1.0)
    return float(np.mean(bounded**power))


def compute_fall_margin_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Return the penalty selected by ``fallPenaltyMode`` (default ``com_margin``).

    Arguments are as in ``compute_capture_point_penalty``.

    Raises:
        ValueError: If ``fallPenaltyMode`` is not ``com_margin`` or ``capture_point``.
    """
    mode = str(param_fixed.get("fallPenaltyMode", "com_margin"))
    if mode == "com_margin":
        return compute_com_margin_penalty(state_store, param_fixed)
    if mode == "capture_point":
        return compute_capture_point_penalty(state_store, param_fixed)
    raise ValueError(f"Unknown fallPenaltyMode: {mode!r} (use 'com_margin' or 'capture_point')")
