"""Compare the Python implementation with a deterministic MATLAB reference run.

The reference is written by matlab/validation/export_reference_run.m. Both runs use the
default parameters with exploratory and sensory energy noise set to zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from invLocAd.initializing.get_treadmill_speed import (
    get_treadmill_speed,
)
from invLocAd.initializing.load_initial_body_state import (
    load_initial_body_state,
)
from invLocAd.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from invLocAd.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)
from invLocAd.learning.simulate_learning_step_by_step import (
    simulate_learning_step_by_step,
)
from invLocAd.parameter_loading.load_biped_model_parameters import (
    load_biped_model_parameters,
)
from invLocAd.parameter_loading.load_controller_gain_parameters import (
    load_controller_gain_parameters,
)
from invLocAd.parameter_loading.load_learner_parameters import (
    load_learner_parameters,
)
from invLocAd.parameter_loading.load_protocol_parameters import (
    load_protocol_parameters,
)
from invLocAd.parameter_loading.load_sensory_noise_parameters import (
    load_sensory_noise_parameters,
)
from invLocAd.parameter_loading.load_stored_memory_parameters_control_vs_speed import (
    load_stored_memory_parameters_control_vs_speed,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the reference path and the absolute tolerance."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Python implementation against a deterministic MATLAB reference "
            "exported with matlab/validation/export_reference_run.m."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Path to the MATLAB .mat reference file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Maximum allowed absolute difference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Rerun the reference configuration in Python and compare it with the MATLAB export.

    Compares, by maximum absolute difference, the initial objective value, 10D
    controller, body state and belt speeds, and the controller tried at each learning
    iteration (``controllerStore8D``).

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        0 if every difference is finite and within the tolerance, otherwise 1.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and non-negative")
    matlab_reference = loadmat(args.reference)

    param_fixed = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(param_fixed)
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
    param_fixed["Learner"]["noiseSTDExploratory"] = 0
    param_fixed["noiseEnergySensory"] = 0
    param_fixed["num_iterations"] = int(matlab_reference["controllerStore8D"].shape[1])
    param_fixed["numIterations"] = param_fixed["num_iterations"]

    p_input = load_learnable_parameters_initial(param_fixed)
    state_var0 = load_initial_body_state(p_input)
    v_a, v_b = get_treadmill_speed(0, param_fixed["imposedFootSpeeds"])
    context_now = np.array([v_a, v_b])
    objective_value = f_objective_asymmetric_nominal(
        p_input,
        state_var0,
        param_controller_gains,
        param_fixed,
        0.0,
    )[0]
    controller_store_8d = simulate_learning_step_by_step(
        param_fixed,
        p_input,
        state_var0,
        context_now,
        param_controller_gains,
    )

    comparisons = {
        "objectiveValue": abs(
            float(matlab_reference["objectiveValue"].squeeze()) - float(objective_value)
        ),
        "pInput": float(
            np.max(np.abs(matlab_reference["pInput"].squeeze() - p_input))
        ),
        "stateVar0": float(
            np.max(np.abs(matlab_reference["stateVar0"].squeeze() - state_var0))
        ),
        "contextNow": float(
            np.max(np.abs(matlab_reference["contextNow"].squeeze() - context_now))
        ),
        "controllerStore8D": float(
            np.max(np.abs(matlab_reference["controllerStore8D"] - controller_store_8d))
        ),
    }

    print("MATLAB/Python deterministic parity check")
    for name, value in comparisons.items():
        print(f"{name}: {value:.3e}")

    if any(not np.isfinite(value) or value > args.tolerance for value in comparisons.values()):
        print("Parity check failed.")
        return 1

    print("Parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
