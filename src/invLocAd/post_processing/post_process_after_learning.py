"""Replay of a learned controller history."""
from __future__ import annotations

import numpy as np

from invLocAd.learning.simulate_many_steps_asymmetric_control import (
    simulate_many_steps_asymmetric_control,
)


def post_process_after_learning(
    p_input_store: np.ndarray,
    state_var0_model: np.ndarray,
    param_controller,
    param_fixed,
    do_animate: bool = False,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Replay a controller history to recover step-by-step trajectories and energy costs.

    Args:
        p_input_store: Controller vectors, shape (10, n), one column per learning iteration.
        state_var0_model: Initial 5-element midstance body state.
        param_controller: Feedback gains; its ``Odd`` and ``Even`` targets are overwritten.
        param_fixed: Fixed parameters; ``num_steps`` is set to the steps per iteration.
        do_animate: Unused.

    Returns:
        Final body state, stance-phase times, stance-phase states, metabolic energy per
        step, metabolic rate per step, step durations, push-off work per step,
        heel-strike work per step, metabolic rate per iteration and iteration durations.
    """

    num_iterations = p_input_store.shape[1]

    t_store = []
    state_store = []
    emet_total_store = []
    emet_per_time_store = []
    t_step_store = []
    ework_pushoff_store = []
    ework_heelstrike_store = []

    edot_store_iteration_average = np.zeros(num_iterations)
    t_total_iteration_store = np.zeros(num_iterations)

    t_start = 0.0

    print(f'Post-processing all the walking data ... ({num_iterations} iterations)')

    for i_stride in range(num_iterations):

        if (i_stride + 1) % 50 == 0:
            print(f'  iStride = {i_stride + 1}')

        p_input_controller_asymmetric_nominal = p_input_store[:, i_stride]

        param_controller['Odd']['theta_end_nominal'] = float(p_input_controller_asymmetric_nominal[0])
        param_controller['Odd']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_asymmetric_nominal[1])
        param_controller['Odd']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_asymmetric_nominal[2])
        param_controller['Odd']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_asymmetric_nominal[3])
        param_controller['Odd']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_asymmetric_nominal[4])

        param_controller['Even']['theta_end_nominal'] = float(p_input_controller_asymmetric_nominal[5])
        param_controller['Even']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_asymmetric_nominal[6])
        param_controller['Even']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_asymmetric_nominal[7])
        param_controller['Even']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_asymmetric_nominal[8])
        param_controller['Even']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_asymmetric_nominal[9])

        param_fixed['num_steps'] = param_fixed['Learner']['numStepsPerIteration']
        (
            state_var0_model,
            t_store_now,
            state_store_now,
            emet_total_store_now,
            emet_per_time_store_now,
            t_step_store_now,
            ework_pushoff_store_now,
            ework_heelstrike_store_now,
        ) = simulate_many_steps_asymmetric_control(
            state_var0_model, param_controller, param_fixed, t_start
        )

        t_start = float(np.asarray(t_store_now[-1]).flatten()[-1])

        edot_store_iteration_average[i_stride] = np.sum(emet_total_store_now) / np.sum(t_step_store_now)
        t_total_iteration_store[i_stride] = np.sum(t_step_store_now)

        if i_stride == 0:
            t_store = t_store_now.copy()
            state_store = state_store_now.copy()
            emet_total_store = emet_total_store_now.copy()
            emet_per_time_store = emet_per_time_store_now.copy()
            t_step_store = t_step_store_now.copy()
            ework_pushoff_store = ework_pushoff_store_now.copy()
            ework_heelstrike_store = ework_heelstrike_store_now.copy()
        else:
            # Join the final half stance of the previous iteration to the first of this one.
            t_store[-1] = np.concatenate([t_store[-1], t_store_now[0]])
            state_store[-1] = np.vstack([state_store[-1], state_store_now[0]])

            t_store.extend(t_store_now[1:])
            state_store.extend(state_store_now[1:])

            emet_total_store = np.concatenate([emet_total_store, emet_total_store_now])
            emet_per_time_store = np.concatenate([emet_per_time_store, emet_per_time_store_now])
            t_step_store = np.concatenate([t_step_store, t_step_store_now])
            ework_pushoff_store = np.concatenate([ework_pushoff_store, ework_pushoff_store_now])
            ework_heelstrike_store = np.concatenate([ework_heelstrike_store, ework_heelstrike_store_now])

    print('Post-processing complete.')

    print(f'Total steps simulated: {len(emet_total_store)}')
    print(f'Average energy consumption: {np.mean(edot_store_iteration_average):.6f}')
    print(f'Final energy consumption: {edot_store_iteration_average[-1]:.6f}')

    return (
        state_var0_model,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        ework_pushoff_store,
        ework_heelstrike_store,
        edot_store_iteration_average,
        t_total_iteration_store,
    )
