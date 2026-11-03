"""Interpolation of imposed treadmill belt speeds."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _interp_extrap(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return ``y`` interpolated linearly at ``t``, extrapolating from the end segments."""
    result = np.empty_like(t, dtype=float)
    idxs = np.searchsorted(x, t, side="left")
    for i, (ti, idx) in enumerate(zip(t, idxs)):
        if idx == 0:
            x1, x2 = x[0], x[1]
            y1, y2 = y[0], y[1]
        elif idx >= len(x):
            x1, x2 = x[-2], x[-1]
            y1, y2 = y[-2], y[-1]
        else:
            x1, x2 = x[idx - 1], x[idx]
            y1, y2 = y[idx - 1], y[idx]
        result[i] = y1 + (y2 - y1) * (ti - x1) / (x2 - x1)
    return result


def get_treadmill_speed(
    t: Sequence[float] | float,
    belt_speeds_imposed: Mapping[str, np.ndarray],
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Return the two imposed belt speeds at time ``t``.

    Times outside the protocol are extrapolated linearly from the nearest two samples.

    Args:
        t: Dimensionless time, scalar or 1D sequence.
        belt_speeds_imposed: Protocol with ``tList``, ``footSpeed1List`` and ``footSpeed2List``.

    Returns:
        Belt 1 and belt 2 speeds; floats for scalar ``t``, otherwise arrays.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    foot_speed1 = _interp_extrap(
        t_arr,
        np.asarray(belt_speeds_imposed["tList"], dtype=float),
        np.asarray(belt_speeds_imposed["footSpeed1List"], dtype=float),
    )
    foot_speed2 = _interp_extrap(
        t_arr,
        np.asarray(belt_speeds_imposed["tList"], dtype=float),
        np.asarray(belt_speeds_imposed["footSpeed2List"], dtype=float),
    )
    if np.ndim(t) == 0:
        return foot_speed1.item(), foot_speed2.item()
    return foot_speed1, foot_speed2
