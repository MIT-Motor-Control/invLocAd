"""Prediction error of the context-dependent controller memory."""

from __future__ import annotations

import numpy as np


def error_from_memory_compute(
    slope_j_controller_vs_context: np.ndarray,
    p_input_now_considered_good: np.ndarray,
    param_fixed,
    context_gait_now: np.ndarray,
) -> float:
    """Return the squared error of the memory's controller prediction.

    The memory predicts the controller as
    ``nominalControl + slope @ (context - nominalContext)``.

    Args:
        slope_j_controller_vs_context: Controller-versus-context slope, shape (8, 2).
        p_input_now_considered_good: Current accepted controller, shape (8,).
        param_fixed: Parameters with ``storedmemory['nominalControl']`` and
            ``storedmemory['nominalContext']``.
        context_gait_now: Current context, the two belt speeds, shape (2,).

    Returns:
        Squared Euclidean distance between the accepted and predicted controllers.
    """

    p_input_memory_now = (
        param_fixed['storedmemory']['nominalControl']
        + slope_j_controller_vs_context
        @ (context_gait_now - param_fixed['storedmemory']['nominalContext'])
    )

    diff = p_input_now_considered_good - p_input_memory_now

    diff_flat = np.asarray(diff).flatten()

    f = float(np.sum(diff_flat * diff_flat))

    return f
