"""Initial nominal controller parameters."""
from __future__ import annotations

import numpy as np


def load_learnable_parameters_initial(param_fixed: dict) -> np.ndarray:
    """Return the initial 10-element nominal controller.

    Elements 0-4 apply to odd steps and 5-9 to even steps. Each block holds the nominal
    endstance leg angle, midstance speed relative to the belt, push-off impulse,
    midstance position and summed midstance position error.

    Args:
        param_fixed: Fixed parameters; only ``swingCost["Coeff"] == 0.9`` is supported.

    Raises:
        NotImplementedError: For any other swing cost coefficient.
    """
    if param_fixed.get('swingCost', {}).get('Coeff') == 0.9:
        p_input_controller_asymmetric_nominal = np.array([
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0.0,
            0.0,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735,
        ])
    else:
        raise NotImplementedError(
            "Only swingCost.Coeff=0.9 configuration supported"
        )
    return p_input_controller_asymmetric_nominal
