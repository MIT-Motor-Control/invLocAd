"""Fixed biped, energy-cost and objective parameters."""
from __future__ import annotations


def load_biped_model_parameters(param_fixed: dict | None = None) -> dict:
    """Add the dimensionless biped, energy-cost and objective parameters.

    Args:
        param_fixed: Parameter dictionary to update in place; created when None.

    Returns:
        The updated ``param_fixed``.
    """
    if param_fixed is None:
        param_fixed = {}

    # Include the inertial torque from belt acceleration in the stance dynamics.
    param_fixed['includeAccelerationTorque'] = 1

    # Body mass, leg length and gravity set the units of the dimensionless model.
    param_fixed['mbody'] = 1
    param_fixed['leglength'] = 1
    param_fixed['gravg'] = 1

    # Muscle efficiencies; step-to-step cost is bPos * push-off work + bNeg * heel-strike work.
    param_fixed['efficiency_neg'] = 1.2
    param_fixed['efficiency_pos'] = 0.25
    param_fixed['bPos'] = 1 / param_fixed['efficiency_pos']
    param_fixed['bNeg'] = 1 / param_fixed['efficiency_neg']

    # Foot mass, coefficient and exponent of the swing-leg force-rate cost.
    param_fixed['mFoot'] = 0.05
    param_fixed['swingCost'] = {
        'Coeff': 0.9,
        'alpha': 1.0,
    }

    # Weight on energy versus periodicity (1: energy only).
    param_fixed['lambdaEnergyVsPeriodicity'] = 1

    # Energy weight 1 - psi and symmetry multiplier M of the paper's objective (Eq. 1).
    param_fixed['lambdaEnergyVsSymmetry'] = 0.75
    param_fixed['symmetryMultiplier'] = 10

    # Fall penalty weight w of paper Eq. 2 and its settings (see fall_margin_penalty);
    # w = 0 disables the penalty.
    param_fixed['fallPenaltyWeight'] = 0.0
    param_fixed['fallPenaltyMode'] = 'com_margin'
    param_fixed['fallPenaltyPower'] = 2.0
    param_fixed['fallPenaltyEpsilon'] = 1e-6
    # Dimensionless COM heights at which the com_margin penalty starts and saturates.
    param_fixed['fallSafeComHeight'] = 0.95
    param_fixed['fallGroundComHeight'] = 0.90
    # Settings of the alternative capture_point mode.
    param_fixed['fallCapturePointLimit'] = 1.0
    param_fixed['fallCapturePointScale'] = 1.0
    # When True, the penalty is multiplied by the iteration's energy rate.
    param_fixed['fallPenaltyScaleToEnergy'] = False

    return param_fixed
