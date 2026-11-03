"""Gait asymmetry and metabolic-rate summaries and figures for simulation results."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from invLocAd.initializing.get_treadmill_speed import get_treadmill_speed
from invLocAd.post_processing.convert_met_to_vo2 import (
    convert_met_to_vo2,
)


def post_process_helper_plots(
    state_var0: np.ndarray,
    t_store: list,
    state_store: list,
    emet_total_store: np.ndarray,
    emet_per_time_store: np.ndarray,
    t_step_store: np.ndarray,
    param_controller,
    param_fixed,
    do_animate: bool = False,
    ework_pushoff_store: np.ndarray = None,
    ework_heelstrike_store: np.ndarray = None,
    edot_store_iteration_average: np.ndarray = None,
    t_total_iteration_store: np.ndarray = None,
    create_plots: bool = True,
    show_plots: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """Compute gait asymmetry and metabolic-rate summaries and optionally plot them.

    Args:
        state_var0: Final body state (unused).
        t_store: Time samples of each stance phase.
        state_store: States of each stance phase; column 0 is the stance-leg angle.
        emet_total_store: Metabolic energy per step.
        emet_per_time_store: Metabolic rate per step.
        t_step_store: Step durations.
        param_controller: Controller parameters (unused).
        param_fixed: Fixed parameters, including the protocol and ``Learner`` settings.
        do_animate: Unused.
        ework_pushoff_store: Push-off work per step (unused).
        ework_heelstrike_store: Heel-strike work per step (unused).
        edot_store_iteration_average: Metabolic rate per learning iteration.
        t_total_iteration_store: Duration of each learning iteration.
        create_plots: Create the split- or tied-belt summary figure.
        show_plots: Open interactive windows; otherwise figures are closed.
        output_dir: Directory for PNG figures; None saves nothing.

    Returns:
        Dictionary of step count, total time, mean step duration, mean energy per step,
        mean energy rate, first and last iteration energy rates, step time and step
        length asymmetry per stride, and the VO2-filtered metabolic rate with its times.
    """
    t_stance_list = np.array([np.ptp(step_times) for step_times in t_store], dtype=float)
    step_length_list = np.array(
        [
            abs(2 * param_fixed["leglength"] * np.sin(step_state[-1, 0]))
            for step_state in state_store
        ],
        dtype=float,
    )

    # Odd-numbered stance phases (even indices) are on belt 1, the fast belt when split.
    t_stance_fast = t_stance_list[0::2]
    t_stance_slow = t_stance_list[1::2]
    pair_count = min(len(t_stance_fast), len(t_stance_slow))
    t_stance_fast = t_stance_fast[:pair_count]
    t_stance_slow = t_stance_slow[:pair_count]
    step_time_asymmetry = (t_stance_slow - t_stance_fast) / (t_stance_slow + t_stance_fast)

    # Step length 2 L |sin(theta)| at the end of each stance phase is assigned to the belt
    # on which the leading foot lands. SLA = (fast - slow) / (fast + slow) (paper Eq. 3).
    step_length_slow = step_length_list[0::2]
    step_length_fast = step_length_list[1::2]
    pair_count = min(len(step_length_slow), len(step_length_fast))
    step_length_slow = step_length_slow[:pair_count]
    step_length_fast = step_length_fast[:pair_count]
    step_length_asymmetry = (step_length_fast - step_length_slow) / (
        step_length_fast + step_length_slow
    )

    params = {
        "tList": np.cumsum(np.asarray(t_total_iteration_store, dtype=float)),
        "EmetRateList": np.asarray(edot_store_iteration_average, dtype=float),
    }
    t_span_smoothed, emet_smoothed = convert_met_to_vo2(params)

    summary = {
        "total_steps": len(emet_total_store),
        "total_time": float(np.sum(t_step_store)),
        "average_step_duration": float(np.mean(t_step_store)),
        "average_energy_per_step": float(np.mean(emet_total_store)),
        "average_energy_rate": float(np.mean(emet_per_time_store)),
        "initial_average_energy_rate": float(edot_store_iteration_average[0]),
        "final_average_energy_rate": float(edot_store_iteration_average[-1]),
        "step_time_asymmetry": step_time_asymmetry.copy(),
        "step_length_asymmetry": step_length_asymmetry,
        "smoothed_metabolic_rate_time": t_span_smoothed,
        "smoothed_metabolic_rate": emet_smoothed,
    }

    print("\nSimulation summary")
    print(f"Total steps: {summary['total_steps']}")
    print(f"Total simulation time: {summary['total_time']:.3f}")
    print(f"Average step duration: {summary['average_step_duration']:.4f}")
    print(f"Average energy per step: {summary['average_energy_per_step']:.6f}")
    print(f"Average energy rate: {summary['average_energy_rate']:.6f}")
    print(f"Initial average energy rate: {summary['initial_average_energy_rate']:.6f}")
    print(f"Final average energy rate: {summary['final_average_energy_rate']:.6f}")

    if not create_plots:
        return summary

    import matplotlib.pyplot as plt

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures = []
    skip_plot = 10

    if param_fixed["SplitOrTied"] == "split":
        stride_count_list = np.arange(1, len(step_time_asymmetry) + 1)
        # The first stance phase starts at midstance.
        if len(step_time_asymmetry) > 0:
            step_time_asymmetry[0] = np.nan

        figure = plt.figure(200, figsize=(14, 4))
        figure.clf()

        ax_belts = figure.add_subplot(1, 3, 1)
        foot1_speed_list, foot2_speed_list = get_treadmill_speed(
            params["tList"],
            param_fixed["imposedFootSpeeds"],
        )
        ax_belts.plot(
            params["tList"][::skip_plot],
            np.abs(foot1_speed_list)[::skip_plot],
            linewidth=2,
            label="fast belt",
        )
        ax_belts.plot(
            params["tList"][::skip_plot],
            np.abs(foot2_speed_list)[::skip_plot],
            linewidth=2,
            label="slow belt",
        )
        ax_belts.set_xlabel("time")
        ax_belts.set_ylabel("treadmill belt speeds")
        ax_belts.set_xlim([0, np.max(params["tList"])])
        ax_belts.set_ylim([0, 0.6])
        ax_belts.set_xticklabels([])
        ax_belts.legend()
        ax_belts.set_box_aspect(1)

        ax_asymmetry = figure.add_subplot(1, 3, 2)
        ax_asymmetry.plot(
            stride_count_list[1::skip_plot],
            step_length_asymmetry[1::skip_plot],
            "-",
            linewidth=2,
        )
        ax_asymmetry.set_xlabel("stride index")
        ax_asymmetry.set_ylabel("step length symmetry")
        ax_asymmetry.set_ylim([-0.5, 0.5])
        ax_asymmetry.set_xlim([0, np.max(stride_count_list)])
        ax_asymmetry.set_box_aspect(1)

        ax_energy = figure.add_subplot(1, 3, 3)
        stride_list = (
            np.arange(1, len(params["EmetRateList"]) + 1)
            * param_fixed["Learner"]["numStepsPerIteration"]
            / 2
        )
        ax_energy.plot(
            stride_list[::skip_plot],
            params["EmetRateList"][::skip_plot],
            "-",
            label=f"{param_fixed['Learner']['numStepsPerIteration']} step average",
        )
        ax_energy.plot(
            stride_list[::skip_plot],
            emet_smoothed[::skip_plot],
            linewidth=2,
            label="Edot smoothed by VO2",
        )
        ax_energy.set_xlabel("stride index")
        ax_energy.set_ylabel("Edot, met rate")
        ax_energy.set_ylim([0, np.max(params["EmetRateList"])])
        ax_energy.legend()
        ax_energy.set_box_aspect(1)

        figure.tight_layout()
        figures.append((figure, "split_belt_summary.png"))

    if param_fixed["SplitOrTied"] == "tied":
        figure = plt.figure(201, figsize=(10, 4))
        figure.clf()

        ax_belts = figure.add_subplot(1, 2, 1)
        foot1_speed_list, foot2_speed_list = get_treadmill_speed(
            params["tList"],
            param_fixed["imposedFootSpeeds"],
        )
        ax_belts.plot(
            params["tList"][::skip_plot],
            np.abs(foot1_speed_list)[::skip_plot],
            linewidth=2,
            label="fast belt",
        )
        ax_belts.plot(
            params["tList"][::skip_plot],
            np.abs(foot2_speed_list)[::skip_plot],
            linewidth=2,
            label="slow belt",
        )
        ax_belts.set_xlabel("time")
        ax_belts.set_ylabel("treadmill belt speeds")
        ax_belts.set_xlim([0, np.max(params["tList"])])
        ax_belts.set_ylim([0, 0.6])
        ax_belts.set_xticklabels([])
        ax_belts.legend()
        ax_belts.set_box_aspect(1)

        ax_frequency = figure.add_subplot(1, 2, 2)
        t_stance_per_stride = (t_stance_fast + t_stance_slow) / 2
        t_list_step_begin = np.cumsum(t_stance_list)
        ax_frequency.plot(t_list_step_begin[0::2][:len(t_stance_per_stride)], 1 / t_stance_per_stride)
        ax_frequency.set_xlabel("time (non dim)")
        ax_frequency.set_ylabel("step freq, averaged over 2 steps")
        ax_frequency.set_ylim([0, 0.8])
        ax_frequency.set_box_aspect(1)

        figure.suptitle(
            "Tied Treadmill: Step frequency changes in response to speed changes"
        )
        figure.tight_layout()
        figures.append((figure, "tied_belt_summary.png"))

    if output_path is not None:
        for figure, filename in figures:
            figure.savefig(output_path / filename, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        for figure, _ in figures:
            plt.close(figure)

    return summary
