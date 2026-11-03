"""Iteration-by-iteration learning loop and validation of its checkpoint state."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from invLocAd.initializing.get_treadmill_speed import get_treadmill_speed
from invLocAd.learning.compute_energy_dynamics_model_rls_no_singular import (
    compute_energy_dynamics_model_rls_no_singular,
)
from invLocAd.learning.compute_model_dynamics_model_rls_no_singular import (
    compute_model_dynamics_model_rls_no_singular,
)
from invLocAd.learning.compute_steady_energy_gradient_v2_no_singular import (
    compute_steady_energy_gradient_v2_no_singular,
)
from invLocAd.learning.error_from_memory_compute import error_from_memory_compute
from invLocAd.learning.f_objective_asymmetric_nominal_8d_to_10d import (
    f_objective_asymmetric_nominal_8d_to_10d,
)
from invLocAd.learning.gradient_of_error_from_memory_compute import (
    gradient_of_error_from_memory_compute,
)


def safe_float(value):
    """Return a scalar, or the sum of an array, as a float (zero for an empty array)."""
    if np.isscalar(value):
        return float(value)

    arr = np.asarray(value).flatten()
    if arr.size == 1:
        return float(arr[0])
    elif arr.size == 0:
        return 0.0
    else:
        return float(np.sum(arr))


def simulate_learning_step_by_step(
    param_fixed,
    p_input_controller_asymmetric_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    context_now: np.ndarray,
    param_controller_gains,
    return_diagnostics: bool = False,
    verbose: bool = True,
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_every: int | None = None,
    stop_after: int | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Simulate learning of the 8D controller over successive iterations.

    Each iteration (one stride) moves the accepted controller down the estimated
    objective gradient, within a trust region if enabled, and toward the
    controller that memory predicts for the current context. It then evaluates
    the accepted controller plus exploratory noise. After the first
    ``numStepsToUseForEstimator`` iterations, the internal state and objective
    models are refit to recent iterations and give the next gradient estimate;
    with ``shouldWeThresholdPredictionError``, the estimate is set to zero when
    an objective-model residual exceeds ``predictionErrorThreshold``. The memory
    slope is then updated toward the accepted controller.

    Args:
        param_fixed: Model, protocol, noise, learner and memory parameters;
            ``_rng`` supplies the random draws.
            ``storedmemory['controlSlopeVsContext']`` is updated in place.
        p_input_controller_asymmetric_nominal: Initial 10D controller; entries
            3 and 4 are not learned.
        state_var0_model: Initial midstance state (5,).
        context_now: Initial context, the two belt speeds (2,).
        param_controller_gains: Controller gains and targets; updated in place.
        return_diagnostics: Also return per-iteration diagnostics.
        verbose: Print progress every 50 iterations.
        resume_state: Learner state from ``checkpoint_callback`` to continue from.
        checkpoint_callback: Called with a copy of the learner state after
            every ``checkpoint_every`` iterations and after the last iteration.
        checkpoint_every: Checkpoint interval in iterations.
        stop_after: Last iteration to run; defaults to ``num_iterations``.

    Returns:
        Controllers tried in all completed iterations, shape (8, n). With
        ``return_diagnostics``, a tuple of that array and a dict of
        per-iteration arrays: ``objective_store`` (noisy objective),
        ``energy_store`` (metabolic power), ``t_end_store``,
        ``fall_penalty_store``, ``min_com_store`` and
        ``nominal_step_length_asymmetry_store``.

    Raises:
        ValueError: If the iteration arguments or ``resume_state`` are invalid.
    """

    if verbose:
        print(
            "Simulating learning while walking "
            f"({param_fixed['num_iterations']} learning iterations)"
        )

    rng = param_fixed.get("_rng", np.random)
    noise_std = param_fixed["Learner"]["noiseSTDExploratory"]
    num_iterations = param_fixed["num_iterations"]
    include_internal_model = param_fixed["Learner"]["includeInternalModel"]

    p_input_controller_asymmetric_nominal = p_input_controller_asymmetric_nominal[
        [0, 1, 2, 5, 6, 7, 8, 9]
    ]
    num_learning_dimensions = len(p_input_controller_asymmetric_nominal)
    num_state_dimensions = len(state_var0_model)

    alpha_energy_learning_rate = param_fixed["Learner"]["LearningRate"]

    p_input_now = p_input_controller_asymmetric_nominal.copy()
    t_start = 0.0

    adynamics_model_now = np.zeros(
        (num_state_dimensions - 1, num_learning_dimensions + num_state_dimensions)
    )
    adynamics_energy_now = np.zeros((1, num_learning_dimensions + num_state_dimensions))

    g_energy_gradient_now = np.zeros_like(p_input_controller_asymmetric_nominal)

    p_input_controller_store_ones_tried: list[np.ndarray] = []
    state_var_now_store: list[np.ndarray] = []
    state_var_next_store: list[np.ndarray] = []
    edot_store: list[float] = []
    energy_store: list[float] = []
    t_end_store: list[float] = []
    fall_penalty_store: list[float] = []
    min_com_store: list[float] = []
    nominal_step_length_asymmetry_store: list[float] = []
    p_input_controller_store_ones_considered_good: list[np.ndarray] = []
    gradient_energy_estimate_store: list[np.ndarray] = []
    adynamics_model_store: list[np.ndarray] = []
    adynamics_energy_store: list[np.ndarray] = []

    error_old_dynamics_model_store: list[np.ndarray] = []
    error_new_dynamics_model_store: list[np.ndarray] = []
    error_old_energy_model_store: list[np.ndarray] = []
    error_new_energy_model_store: list[np.ndarray] = []
    learning_on_or_not_store: list[int] = []
    memory_to_gradient_direction_cosine_store: list[float] = []

    g_gradient_for_memory_update_store: list[np.ndarray] = []
    error_from_memory_store: list[float] = []
    p_input_memory_now_store: list[np.ndarray] = []

    num_strides_to_use = param_fixed["Learner"]["numStepsToUseForEstimator"]
    state_var0_model_now = state_var0_model.copy()
    p_input_now_considered_good = p_input_now.copy()

    completed = 0
    if not isinstance(num_iterations, (int, np.integer)) or num_iterations < 1:
        raise ValueError("num_iterations must be a positive integer")
    limit = num_iterations if stop_after is None else stop_after
    if not isinstance(limit, (int, np.integer)) or not 1 <= limit <= num_iterations:
        raise ValueError("stop_after must be between 1 and num_iterations")
    if checkpoint_every is not None and (
        not isinstance(checkpoint_every, int) or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    if resume_state is not None:
        validate_learning_state(resume_state, param_fixed)
        completed = resume_state["completed_iterations"]
        if completed > limit:
            raise ValueError("Cannot resume to an iteration before the checkpoint")
        t_start = deepcopy(resume_state["t_start"])
        context_now = deepcopy(resume_state["context_now"])
        state_var0_model_now = deepcopy(resume_state["state_var0_model_now"])
        p_input_now_considered_good = deepcopy(resume_state["p_input_now_considered_good"])
        adynamics_model_now = deepcopy(resume_state["adynamics_model_now"])
        adynamics_energy_now = deepcopy(resume_state["adynamics_energy_now"])
        g_energy_gradient_now = deepcopy(resume_state["g_energy_gradient_now"])
        p_input_controller_store_ones_tried = deepcopy(
            resume_state["p_input_controller_store_ones_tried"]
        )
        state_var_now_store = deepcopy(resume_state["state_var_now_store"])
        state_var_next_store = deepcopy(resume_state["state_var_next_store"])
        edot_store = deepcopy(resume_state["edot_store"])
        energy_store = deepcopy(resume_state["energy_store"])
        t_end_store = deepcopy(resume_state["t_end_store"])
        fall_penalty_store = deepcopy(resume_state["fall_penalty_store"])
        min_com_store = deepcopy(resume_state["min_com_store"])
        nominal_step_length_asymmetry_store = deepcopy(
            resume_state["nominal_step_length_asymmetry_store"]
        )
        p_input_controller_store_ones_considered_good = deepcopy(
            resume_state["p_input_controller_store_ones_considered_good"]
        )
        gradient_energy_estimate_store = deepcopy(resume_state["gradient_energy_estimate_store"])
        adynamics_model_store = deepcopy(resume_state["adynamics_model_store"])
        adynamics_energy_store = deepcopy(resume_state["adynamics_energy_store"])
        error_old_dynamics_model_store = deepcopy(resume_state["error_old_dynamics_model_store"])
        error_new_dynamics_model_store = deepcopy(resume_state["error_new_dynamics_model_store"])
        error_old_energy_model_store = deepcopy(resume_state["error_old_energy_model_store"])
        error_new_energy_model_store = deepcopy(resume_state["error_new_energy_model_store"])
        learning_on_or_not_store = deepcopy(resume_state["learning_on_or_not_store"])
        memory_to_gradient_direction_cosine_store = deepcopy(
            resume_state["memory_to_gradient_direction_cosine_store"]
        )
        g_gradient_for_memory_update_store = deepcopy(
            resume_state["g_gradient_for_memory_update_store"]
        )
        error_from_memory_store = deepcopy(resume_state["error_from_memory_store"])
        p_input_memory_now_store = deepcopy(resume_state["p_input_memory_now_store"])

    for i_stride in range(completed + 1, limit + 1):
        if verbose and i_stride % 50 == 0:
            print(f"iStride = {i_stride}")

        delta_pinput_noise = noise_std * rng.randn(num_learning_dimensions)
        delta_pinput_gradient = alpha_energy_learning_rate * (-g_energy_gradient_now)

        if param_fixed["Learner"]["shouldWeUseTrustRegion"]:
            norm_grad = np.linalg.norm(delta_pinput_gradient)
            if norm_grad > param_fixed["Learner"]["trustRegionSize"] * np.sqrt(
                num_learning_dimensions
            ):
                delta_pinput_gradient = (
                    delta_pinput_gradient
                    / norm_grad
                    * param_fixed["Learner"]["trustRegionSize"]
                    * np.sqrt(num_learning_dimensions)
                )

        p_input_memory_now = param_fixed["storedmemory"]["nominalControl"] + param_fixed[
            "storedmemory"
        ]["controlSlopeVsContext"] @ (context_now - param_fixed["storedmemory"]["nominalContext"])
        p_input_memory_now_store.append(p_input_memory_now)

        # The step toward memory has rate LearningRateTowardMemory * ((1 + c) / 2) ** power,
        # where c is the cosine between the memory direction and the descent direction.
        dir_toward_memory = p_input_memory_now - p_input_now_considered_good
        temp = safe_float(
            np.dot(dir_toward_memory, -g_energy_gradient_now)
            / (np.linalg.norm(dir_toward_memory) * np.linalg.norm(g_energy_gradient_now) + 1e-8)
        )
        memory_to_gradient_direction_cosine_store.append(temp)

        power_move_to_memory = param_fixed["Learner"]["powerToTheMoveToMemory"]
        if np.isnan(temp):
            temp = 1 * param_fixed["Learner"]["LearningRateTowardMemory"]
        else:
            temp = (1 + temp) / 2
            temp = temp**power_move_to_memory
            temp = temp * param_fixed["Learner"]["LearningRateTowardMemory"]

        delta_pinput_memory = temp * dir_toward_memory

        p_input_now_considered_good = p_input_now_considered_good + delta_pinput_memory
        p_input_now_considered_good = p_input_now_considered_good + delta_pinput_gradient
        p_input_now_to_try = p_input_now_considered_good + delta_pinput_noise

        (
            edot_now,
            state_var0_model_next,
            t_end,
            f_energy_now,
            _,
        ) = f_objective_asymmetric_nominal_8d_to_10d(
            p_input_now_to_try,
            state_var0_model_now,
            param_controller_gains,
            param_fixed,
            t_start,
        )
        edot_now = edot_now * (1 + rng.randn() * param_fixed["noiseEnergySensory"])

        state_var_now_store.append(state_var0_model_now)
        state_var_next_store.append(state_var0_model_next)
        p_input_controller_store_ones_tried.append(p_input_now_to_try)
        p_input_controller_store_ones_considered_good.append(p_input_now_considered_good)
        edot_store.append(edot_now)
        energy_store.append(f_energy_now)
        t_end_store.append(t_end)
        fall_penalty_store.append(float(param_fixed.get("_lastFallPenalty", np.nan)))
        min_com_store.append(float(param_fixed.get("_lastMinCom", np.nan)))

        # Odd steps end with heel strike on belt 2, the slow belt in split walking.
        step_length_slow = abs(2 * param_fixed["leglength"] * np.sin(p_input_now_to_try[0]))
        step_length_fast = abs(2 * param_fixed["leglength"] * np.sin(p_input_now_to_try[3]))
        nominal_step_length_asymmetry_store.append(
            safe_float(
                (step_length_fast - step_length_slow)
                / (step_length_fast + step_length_slow + 1e-12)
            )
        )

        if i_stride > num_strides_to_use:
            max_elements = min(
                len(state_var_now_store),
                len(state_var_next_store),
                len(p_input_controller_store_ones_tried),
            )
            max_elements = min(max_elements, num_strides_to_use + 10)

            state_var_recent = state_var_now_store[-max_elements:]
            state_var_next_recent = state_var_next_store[-max_elements:]
            p_input_recent = p_input_controller_store_ones_tried[-max_elements:]

            adynamics_model_now, error_old_dynamics_model, error_new_dynamics_model = (
                compute_model_dynamics_model_rls_no_singular(
                    adynamics_model_now,
                    np.column_stack(state_var_recent),
                    np.column_stack(state_var_next_recent),
                    np.column_stack(p_input_recent),
                    num_strides_to_use,
                )
            )
            error_old_dynamics_model_store.append(error_old_dynamics_model)
            error_new_dynamics_model_store.append(error_new_dynamics_model)

        if i_stride > num_strides_to_use:
            max_elements = min(
                len(state_var_now_store),
                len(edot_store),
                len(p_input_controller_store_ones_tried),
            )
            max_elements = min(max_elements, num_strides_to_use + 10)

            state_var_recent = state_var_now_store[-max_elements:]
            edot_recent = edot_store[-max_elements:]
            p_input_recent = p_input_controller_store_ones_tried[-max_elements:]

            adynamics_energy_now, error_old_energy_model, error_new_energy_model = (
                compute_energy_dynamics_model_rls_no_singular(
                    adynamics_energy_now,
                    np.column_stack(state_var_recent),
                    np.array(edot_recent),
                    np.column_stack(p_input_recent),
                    num_strides_to_use,
                )
            )
            error_old_energy_model_store.append(error_old_energy_model)
            error_new_energy_model_store.append(error_new_energy_model)

        if i_stride > num_strides_to_use:
            g_energy_gradient_now = compute_steady_energy_gradient_v2_no_singular(
                adynamics_model_now,
                adynamics_energy_now,
                p_input_now_to_try,
                num_learning_dimensions,
                num_state_dimensions,
                include_internal_model,
            )
        gradient_energy_estimate_store.append(g_energy_gradient_now)
        adynamics_model_store.append(adynamics_model_now)
        adynamics_energy_store.append(adynamics_energy_now)

        if (
            param_fixed["Learner"]["shouldWeThresholdPredictionError"]
            and i_stride > num_strides_to_use
        ):
            prediction_error_list_now = np.array([error_old_energy_model, error_new_energy_model])
            if np.any(
                np.abs(prediction_error_list_now)
                > param_fixed["Learner"]["predictionErrorThreshold"]
            ):
                g_energy_gradient_now = 0 * g_energy_gradient_now
                learning_on_or_not_store.append(0)
            else:
                learning_on_or_not_store.append(1)

        g_gradient_for_memory_update = gradient_of_error_from_memory_compute(
            param_fixed["storedmemory"]["controlSlopeVsContext"],
            p_input_now_considered_good,
            param_fixed,
            context_now,
        )

        # The memory-slope update has the same form, with c the cosine between the
        # controller's motion and the direction from memory to the controller.
        n_memory_to_current_controller = -dir_toward_memory
        n_memory_to_current_controller = n_memory_to_current_controller / (
            np.linalg.norm(n_memory_to_current_controller) + 1e-8
        )

        n_v_controller = delta_pinput_gradient + delta_pinput_memory
        n_v_controller = n_v_controller / (np.linalg.norm(n_v_controller) + 1e-8)

        dot_product = np.dot(n_memory_to_current_controller, n_v_controller)
        dot_memory_to_current_v_current = safe_float(dot_product)

        temp = dot_memory_to_current_v_current
        power_for_memory_formation = param_fixed["Learner"]["powerToTheMemoryFormation"]
        if np.isnan(temp):
            temp = 0.0
        else:
            temp = (1 + temp) / 2
            temp = temp**power_for_memory_formation
            temp = temp * param_fixed["Learner"]["LearningRateForMemoryFormationUpdates"]

        param_fixed["storedmemory"]["controlSlopeVsContext"] = param_fixed["storedmemory"][
            "controlSlopeVsContext"
        ] + temp * (-g_gradient_for_memory_update)

        g_gradient_for_memory_update_store.append(g_gradient_for_memory_update)
        error_from_memory_store.append(
            error_from_memory_compute(
                param_fixed["storedmemory"]["controlSlopeVsContext"],
                p_input_now_considered_good,
                param_fixed,
                context_now,
            )
        )

        t_start = t_end
        state_var0_model_now = state_var0_model_next

        v_a, v_b = get_treadmill_speed(t_start, param_fixed["imposedFootSpeeds"])
        context_now = np.array([v_a, v_b])
        context_now = context_now + param_fixed["noiseEnergySensory"] * rng.randn()

        # Checkpoint after all updates and random draws of this iteration.
        if checkpoint_callback is not None and (
            i_stride == limit or (checkpoint_every is not None and i_stride % checkpoint_every == 0)
        ):
            namespace = locals()
            snapshot = {name: deepcopy(namespace[name]) for name in _STATE_FIELDS}
            snapshot["completed_iterations"] = i_stride
            checkpoint_callback(snapshot)

    if completed == limit and checkpoint_callback is not None:
        checkpoint_callback(deepcopy(resume_state))

    controller_history = np.column_stack(p_input_controller_store_ones_tried)
    if not return_diagnostics:
        return controller_history

    diagnostics = {
        "objective_store": np.asarray(edot_store, dtype=float),
        "energy_store": np.asarray(energy_store, dtype=float),
        "t_end_store": np.asarray(t_end_store, dtype=float),
        "fall_penalty_store": np.asarray(fall_penalty_store, dtype=float),
        "min_com_store": np.asarray(min_com_store, dtype=float),
        "nominal_step_length_asymmetry_store": np.asarray(
            nominal_step_length_asymmetry_store, dtype=float
        ),
    }
    return controller_history, diagnostics


# Learner state saved in checkpoints; entries from index 7 are per-iteration histories.
_STATE_FIELDS = (
    "t_start",
    "context_now",
    "state_var0_model_now",
    "p_input_now_considered_good",
    "adynamics_model_now",
    "adynamics_energy_now",
    "g_energy_gradient_now",
    "p_input_controller_store_ones_tried",
    "state_var_now_store",
    "state_var_next_store",
    "edot_store",
    "energy_store",
    "t_end_store",
    "fall_penalty_store",
    "min_com_store",
    "nominal_step_length_asymmetry_store",
    "p_input_controller_store_ones_considered_good",
    "gradient_energy_estimate_store",
    "adynamics_model_store",
    "adynamics_energy_store",
    "error_old_dynamics_model_store",
    "error_new_dynamics_model_store",
    "error_old_energy_model_store",
    "error_new_energy_model_store",
    "learning_on_or_not_store",
    "memory_to_gradient_direction_cosine_store",
    "g_gradient_for_memory_update_store",
    "error_from_memory_store",
    "p_input_memory_now_store",
)


def validate_learning_state(state: dict[str, Any], param_fixed: dict[str, Any]) -> None:
    """Check that a learner checkpoint state is complete and consistent.

    Args:
        state: Learner state passed to ``checkpoint_callback``.
        param_fixed: Parameters of the run to be continued.

    Raises:
        ValueError: If fields, shapes, history lengths or times are invalid.
    """
    if not isinstance(state, dict) or set(state) != set(_STATE_FIELDS) | {"completed_iterations"}:
        raise ValueError("Checkpoint learner state fields are incomplete or unsupported")
    n = state["completed_iterations"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Checkpoint iteration count must be a positive integer")
    shapes = {
        "context_now": (2,),
        "state_var0_model_now": (5,),
        "p_input_now_considered_good": (8,),
        "adynamics_model_now": (4, 13),
        "adynamics_energy_now": (1, 13),
        "g_energy_gradient_now": (8,),
    }
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype.kind not in "fiu"
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"Invalid checkpoint learner array: {key}")
    if (
        not isinstance(state["t_start"], (int, float, np.integer, np.floating))
        or not np.isfinite(state["t_start"])
        or state["t_start"] <= 0
    ):
        raise ValueError("Invalid checkpoint simulation time")
    num_estimator = param_fixed["Learner"]["numStepsToUseForEstimator"]
    errors = {
        "error_old_dynamics_model_store",
        "error_new_dynamics_model_store",
        "error_old_energy_model_store",
        "error_new_energy_model_store",
    }
    shapes.update(
        {
            "p_input_controller_store_ones_tried": (8,),
            "state_var_now_store": (5,),
            "state_var_next_store": (5,),
            "p_input_controller_store_ones_considered_good": (8,),
            "gradient_energy_estimate_store": (8,),
            "adynamics_model_store": (4, 13),
            "adynamics_energy_store": (1, 13),
            "p_input_memory_now_store": (8,),
            "g_gradient_for_memory_update_store": (8, 2),
            "error_old_dynamics_model_store": (4, 1),
            "error_new_dynamics_model_store": (4, 1),
            "error_old_energy_model_store": (1, 1),
            "error_new_energy_model_store": (1, 1),
        }
    )
    for key in _STATE_FIELDS[7:]:
        values = state[key]
        expected = max(0, n - num_estimator) if key in errors else n
        if (
            key == "learning_on_or_not_store"
            and not param_fixed["Learner"]["shouldWeThresholdPredictionError"]
        ):
            expected = 0
        elif key == "learning_on_or_not_store":
            expected = max(0, n - num_estimator)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"Invalid checkpoint history length: {key}")
        for value in values:
            array = np.asarray(value)
            if array.shape != shapes.get(key, ()) or array.dtype.kind not in "fiu":
                raise ValueError(f"Invalid checkpoint history shape: {key}")
            if key not in ("fall_penalty_store", "min_com_store") and not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"Non-finite checkpoint history: {key}")
    if float(state["t_end_store"][-1]) != float(state["t_start"]):
        raise ValueError("Checkpoint time does not agree with history")
