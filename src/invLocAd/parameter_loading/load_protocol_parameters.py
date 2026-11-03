"""Treadmill protocol and learning-iteration budget."""
from __future__ import annotations

import numpy as np

from invLocAd.parameter_loading.make_treadmill_speed_split import (
    make_treadmill_speed_split,
)
from invLocAd.parameter_loading.make_treadmill_speed_tied import (
    make_treadmill_speed_tied,
)


def load_protocol_parameters(
    param_fixed: dict,
    split_or_tied: str | None = None,
    speed_protocol: str | None = None,
    transition_time: float | None = None,
    split_speed_diff_ms: float | None = None,
) -> dict:
    """Add the treadmill protocol and the number of learning iterations.

    Arguments left as None fall back to values already in ``param_fixed``, then to
    the defaults: split belt, ``"classic split belt"`` or ``"four speed changes"``,
    a 15 s (split) or 3 s (tied) transition and a 1.0 m/s split speed difference.

    Args:
        param_fixed: Parameter dictionary, updated in place; its learner steps per
            iteration set the iteration count.
        split_or_tied: ``"split"`` or ``"tied"``.
        speed_protocol: Protocol name understood by the split or tied speed builder.
        transition_time: Duration of each belt-speed ramp in seconds.
        split_speed_diff_ms: Split-belt speed difference in m/s.

    Returns:
        The updated ``param_fixed``, including ``imposedFootSpeeds`` and ``num_iterations``.

    Raises:
        ValueError: For an unknown configuration or protocol, or an invalid transition
            time or speed difference.
    """
    split_or_tied = split_or_tied or param_fixed.get("SplitOrTied", "split")
    effective_transition = transition_time if transition_time is not None else param_fixed.get("transitionTime", 15 if split_or_tied == "split" else 3)
    if not np.isscalar(effective_transition) or not np.isfinite(effective_transition) or effective_transition <= 0:
        raise ValueError("transition_time must be finite and positive")
    if split_speed_diff_ms is not None:
        if split_or_tied != "split":
            raise ValueError("split_speed_diff_ms is only valid for split-belt runs")
        if not np.isscalar(split_speed_diff_ms) or not np.isfinite(split_speed_diff_ms) or split_speed_diff_ms < 0:
            raise ValueError("split_speed_diff_ms must be finite and non-negative")

    if split_or_tied == "split":
        param_fixed["SplitOrTied"] = "split"
        param_fixed["speedProtocol"] = (
            speed_protocol
            or param_fixed.get("speedProtocol")
            or "classic split belt"
        )
        param_fixed["transitionTime"] = (
            transition_time
            if transition_time is not None
            else param_fixed.get("transitionTime", 15)
        )
        if split_speed_diff_ms is not None:
            param_fixed["splitSpeedDiffMS"] = float(split_speed_diff_ms)
        param_fixed["imposedFootSpeeds"] = make_treadmill_speed_split(param_fixed)
    elif split_or_tied == "tied":
        param_fixed["SplitOrTied"] = "tied"
        param_fixed["speedProtocol"] = (
            speed_protocol
            or param_fixed.get("speedProtocol")
            or "four speed changes"
        )
        param_fixed["transitionTime"] = (
            transition_time
            if transition_time is not None
            else param_fixed.get("transitionTime", 3)
        )
        param_fixed["imposedFootSpeeds"] = make_treadmill_speed_tied(param_fixed)
    else:
        raise ValueError(f"Unknown treadmill configuration: {split_or_tied}")

    # Level treadmill.
    param_fixed["angleSlope"] = 0

    # Step budget: protocol duration over a nominal dimensionless step time, to the nearest 100.
    nominal_step_time = 1.7
    t_list = param_fixed["imposedFootSpeeds"]["tList"]
    if not np.all(np.isfinite(t_list)) or not np.all(np.diff(t_list) > 0):
        raise ValueError("transition_time must be shorter than every protocol phase")
    param_fixed["numStepsToLearn"] = t_list[-1] / nominal_step_time
    param_fixed["numStepsToLearn"] = round(param_fixed["numStepsToLearn"] / 100) * 100

    learner = param_fixed.get("Learner", {})
    # Round the steps per iteration up to an even number so both legs step equally often.
    if learner.get("numStepsPerIteration", 1) % 2 != 0:
        learner["numStepsPerIteration"] = learner.get("numStepsPerIteration", 1) + 1
        param_fixed["Learner"] = learner

    num_iterations = int(
        np.floor(param_fixed["numStepsToLearn"] / learner.get("numStepsPerIteration", 1))
    )
    param_fixed["num_iterations"] = num_iterations
    param_fixed["numIterations"] = num_iterations

    return param_fixed
