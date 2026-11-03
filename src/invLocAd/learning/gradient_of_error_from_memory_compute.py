"""Gradient of the memory prediction error with respect to the memory slope."""

from __future__ import annotations

import numpy as np

from invLocAd.learning.error_from_memory_compute import error_from_memory_compute


def gradient_of_error_from_memory_compute(
    slope_j_controller_vs_context: np.ndarray,
    p_input_now_considered_good: np.ndarray,
    param_fixed,
    context_gait_now: np.ndarray,
) -> np.ndarray:
    """Return the gradient of ``error_from_memory_compute`` with respect to the slope.

    Forward differences with step ``1e-4`` are taken for each slope entry; the
    arguments are as in ``error_from_memory_compute``.

    Returns:
        Gradient with the shape of ``slope_j_controller_vs_context``.
    """

    f0 = error_from_memory_compute(
        slope_j_controller_vs_context,
        p_input_now_considered_good,
        param_fixed,
        context_gait_now,
    )

    h = 1e-4

    g = np.zeros_like(slope_j_controller_vs_context)

    for i_count in range(slope_j_controller_vs_context.shape[0]):
        for j_count in range(slope_j_controller_vs_context.shape[1]):
            slope_now = slope_j_controller_vs_context.copy()
            slope_now[i_count, j_count] += h

            f = error_from_memory_compute(
                slope_now,
                p_input_now_considered_good,
                param_fixed,
                context_gait_now,
            )

            g[i_count, j_count] = (f - f0) / h

    return g
