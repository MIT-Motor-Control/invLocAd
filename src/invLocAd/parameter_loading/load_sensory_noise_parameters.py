"""Sensory noise parameters."""
from __future__ import annotations


def load_sensory_noise_parameters(param_fixed: dict | None = None) -> dict:
    """Add the sensory noise standard deviations.

    Args:
        param_fixed: Parameter dictionary to update in place; created when None.

    Returns:
        The updated ``param_fixed``.
    """
    if param_fixed is None:
        param_fixed = {}

    # Relative (multiplicative) noise on the sensed energy rate. The learner also adds one
    # draw with this standard deviation to both sensed belt speeds.
    param_fixed['noiseEnergySensory'] = 0.0001

    # Additive noise on the sensed midstance velocity.
    param_fixed['velocitySensoryNoise'] = 0

    return param_fixed
