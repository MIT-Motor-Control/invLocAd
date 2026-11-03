"""Tied-belt treadmill speed protocols."""

from __future__ import annotations

import numpy as np


def _phase_arrays(durations: list[float], speeds: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return phase boundary times and speeds, repeating the first speed at time zero."""
    t_store = np.cumsum(np.array([0.0, *durations], dtype=float))
    speed_store = np.array([speeds[0], *speeds], dtype=float)
    return t_store, speed_store


def _with_transitions(
    t_store: np.ndarray,
    foot_speed1_store: np.ndarray,
    foot_speed2_store: np.ndarray,
    transition_duration: float,
) -> dict[str, np.ndarray]:
    """Return the schedule with a linear ramp after each interior phase boundary."""
    t_store_new = [float(t_store[0])]
    foot_speed1_new = [float(foot_speed1_store[0])]
    foot_speed2_new = [float(foot_speed2_store[0])]

    for index in range(1, len(t_store)):
        if index < len(t_store) - 1:
            t_store_new.extend([float(t_store[index]), float(t_store[index] + transition_duration)])
            foot_speed1_new.extend(
                [float(foot_speed1_store[index]), float(foot_speed1_store[index + 1])]
            )
            foot_speed2_new.extend(
                [float(foot_speed2_store[index]), float(foot_speed2_store[index + 1])]
            )
        else:
            t_store_new.append(float(t_store[index]))
            foot_speed1_new.append(float(foot_speed1_store[index]))
            foot_speed2_new.append(float(foot_speed2_store[index]))

    t_list = np.array(t_store_new)
    foot_speed1_list = np.array(foot_speed1_new)
    foot_speed2_list = np.array(foot_speed2_new)
    foot_acc1_list = np.append(np.diff(foot_speed1_list) / np.diff(t_list), 0.0)
    foot_acc2_list = np.append(np.diff(foot_speed2_list) / np.diff(t_list), 0.0)
    return {
        "tList": t_list,
        "footSpeed1List": foot_speed1_list,
        "footSpeed2List": foot_speed2_list,
        "footAcc1List": foot_acc1_list,
        "footAcc2List": foot_acc2_list,
    }


def make_treadmill_speed_tied(param_fixed: dict) -> dict[str, np.ndarray]:
    """Return the tied-belt speed schedule with linear ramps between phases.

    Both belts follow 0.5, 0.75, 1, 1.25 or 1.5 times the normal speed -0.35 (units of
    sqrt(g * L), L = 0.95 m). Phase durations in minutes are listed in ``protocol_specs``.

    Args:
        param_fixed: Parameters with ``speedProtocol`` and optionally ``transitionTime``
            in seconds (default 3, written back).

    Returns:
        Schedule with the same keys as ``make_treadmill_speed_split``.

    Raises:
        ValueError: For an unknown protocol.
    """
    leg_length = 0.95
    grav = 9.81
    time_scaling = np.sqrt(leg_length / grav)
    time_per_phase_in_min = 2

    transition_time = float(param_fixed.get("transitionTime", 3))
    param_fixed["transitionTime"] = transition_time

    v_normal = -0.35
    v_slow = -0.35 * 0.75
    v_very_slow = -0.35 * 0.5
    v_fast = -0.35 * 1.25
    v_very_fast = -0.35 * 1.5

    transition_duration = param_fixed["transitionTime"] / time_scaling
    protocol = param_fixed["speedProtocol"]

    protocol_specs = {
        "single speed": (
            [9, 1, 1, 1],
            [v_normal, v_normal, v_normal, v_normal],
        ),
        "single speed change pulse": (
            [4, 1, time_per_phase_in_min, time_per_phase_in_min],
            [v_normal, v_normal, v_fast, v_normal],
        ),
        "single speed change": (
            [4, 1, time_per_phase_in_min],
            [v_normal, v_normal, v_fast],
        ),
        "two speed changes": (
            [time_per_phase_in_min] * 5,
            [v_normal, v_fast, v_normal, v_slow, v_normal],
        ),
        "four speed changes": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_slow, v_normal, v_fast, v_normal, v_very_fast, v_normal, v_very_slow, v_normal],
        ),
        "four speed changes 1": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_fast, v_normal, v_slow, v_normal, v_very_slow, v_normal, v_very_fast, v_normal],
        ),
        "four speed changes 2": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_very_slow, v_normal, v_fast, v_normal, v_slow, v_normal, v_very_fast, v_normal],
        ),
        "four speed changes 3": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_very_fast, v_normal, v_fast, v_normal, v_very_slow, v_normal, v_slow, v_normal],
        ),
        "four speed changes 4": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_fast, v_normal, v_very_slow, v_normal, v_very_fast, v_normal, v_slow, v_normal],
        ),
        "four speed changes 5": (
            [time_per_phase_in_min] * 9,
            [v_normal, v_very_slow, v_normal, v_very_fast, v_normal, v_fast, v_normal, v_slow, v_normal],
        ),
    }

    if protocol not in protocol_specs:
        raise ValueError(f"Unknown tied-belt speedProtocol: {protocol}")

    duration_minutes, speed_profile = protocol_specs[protocol]
    durations = [minutes * 60 / time_scaling for minutes in duration_minutes]
    t_store, foot_speed1_store = _phase_arrays(durations, speed_profile)
    _, foot_speed2_store = _phase_arrays(durations, speed_profile)
    return _with_transitions(
        t_store,
        foot_speed1_store,
        foot_speed2_store,
        transition_duration,
    )
