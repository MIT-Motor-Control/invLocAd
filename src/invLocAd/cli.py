"""Command-line interface for running the Python simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .workflow import run_simulation


def _add_options(parser: argparse.ArgumentParser, command: str | None = None) -> None:
    """Add simulation options and command-specific output defaults.

    Args:
        parser: Parser or subcommand parser to extend.
        command: ``"demo"``, ``"run"`` or ``"resume"``; None for the option-only parser.
    """
    output = (
        Path("outputs") / {"demo": "demo", "run": "run", "resume": "resumed"}[command]
        if command
        else None
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42 if command == "demo" else None,
        help=(
            "Random seed (demo default: 42)."
            if command == "demo"
            else "Random seed. Omit to generate and record one, or inherit it on resume."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40 if command == "demo" else None,
        help=(
            "Total learning iterations, including completed iterations when resuming. "
            + ("Default: 40." if command == "demo" else "Omit to use the model or saved target.")
        ),
    )
    parser.add_argument(
        "--split-or-tied",
        choices=("split", "tied"),
        default=None,
        help="Choose the treadmill context used for the simulation.",
    )
    parser.add_argument(
        "--speed-protocol",
        default=None,
        help=(
            "Optional protocol name. New runs default to classic split-belt or four speed "
            "changes for tied walking. Resume inherits the saved protocol."
        ),
    )
    parser.add_argument(
        "--transition-time",
        type=float,
        default=None,
        help="Optional treadmill transition time override in seconds.",
    )
    parser.add_argument(
        "--split-speed-diff",
        type=float,
        default=None,
        help=(
            "Optional split-belt speed difference in m/s. New split runs default to "
            "1.0 m/s; resume inherits the saved value."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output,
        help=(
            "Save run.npz, metadata.json, and summary PNG figures here. "
            + (f"Default: {output}." if output else "Omit to leave results unsaved.")
        ),
    )
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument(
        "--no-plots",
        action="store_true",
        default=command is not None,
        help="Skip interactive figure windows (default for demo, run, and resume).",
    )
    plots.add_argument(
        "--show-plots",
        dest="no_plots",
        action="store_false",
        help="Open interactive figure windows as well as saving figures.",
    )
    if command is None:
        parser.add_argument(
            "--resume-from", type=Path, help="Resume a versioned learning checkpoint (.npz)."
        )
    else:
        parser.set_defaults(resume_from=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Save resumable state here (.npz). "
            + ("Default: checkpoint.npz in the output directory." if command else "")
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25 if command else None,
        help=(
            "Save every N completed iterations and at the end "
            "(default for demo, run, and resume: 25)."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help=(
            "Stop at this total iteration count; resume later from the saved checkpoint."
            if command
            else "Stop at this total iteration count; requires --checkpoint."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the demo, run and resume commands."""
    parser = argparse.ArgumentParser(
        prog="invLocAd-simulate",
        description="Run and resume baseline model examples. These are not published participant fits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Start here:\n"
            "  invLocAd-simulate demo\n"
            "  invLocAd-simulate run --seed 42 --iterations 100 --stop-after 40\n"
            "  invLocAd-simulate resume outputs/run/checkpoint.npz\n\n"
            "Use COMMAND --help for options. Options without a command run a single simulation."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "demo": (
            "Run a short example with seed 42 and 40 learning iterations; save to outputs/demo. "
            "This example may finish before split-belt adaptation begins."
        ),
        "run": (
            "Run a new simulation with the configured model settings. "
            "Save results and checkpoints to outputs/run; no interactive windows by default."
        ),
        "resume": (
            "Continue a checkpoint using its saved seed, settings and total target. "
            "Save results and a new checkpoint to outputs/resumed; leave the source checkpoint "
            "unchanged unless explicitly chosen with --checkpoint."
        ),
    }
    for name, help_text in (
        ("demo", "Try a short, reproducible example."),
        ("run", "Run a new simulation and save its results."),
        ("resume", "Continue from a saved checkpoint."),
    ):
        child = commands.add_parser(name, help=help_text, description=descriptions[name])
        if name == "resume":
            child.add_argument(
                "checkpoint_input",
                type=Path,
                metavar="CHECKPOINT",
                help="Path to a learning checkpoint (.npz).",
            )
        _add_options(child, name)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``invLocAd-simulate`` command line.

    Arguments that start with an option rather than a command use the option-only
    parser, which saves nothing unless ``--output-dir`` or ``--checkpoint`` is given.

    Args:
        argv: Command-line arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Exit status 0. Invalid arguments, and ``ValueError`` or ``OSError`` raised by the
        simulation, are reported through ``parser.error`` (exit status 2).
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("-") and arguments[0] not in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description="Run one simulation from options; results are saved only on request."
        )
        parser.set_defaults(command=None)
        _add_options(parser)
    else:
        parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.command:
        if args.command == "resume":
            args.resume_from = args.checkpoint_input
            if args.resume_from.resolve() == (args.output_dir / "run.npz").resolve():
                parser.error(
                    "Choose another --output-dir so run.npz cannot replace the source checkpoint."
                )
        if args.checkpoint is None:
            args.checkpoint = args.output_dir / "checkpoint.npz"
            if (
                args.resume_from is not None
                and args.checkpoint.resolve() == args.resume_from.resolve()
            ):
                parser.error(
                    "The default checkpoint output is the source checkpoint. Choose another "
                    "--output-dir, or explicitly use --checkpoint to update that file."
                )
        if args.command == "demo":
            print("Short baseline example; not a published participant fit.")
            if args.split_or_tied in (None, "split"):
                print("This short example may finish before split-belt adaptation begins.")
    try:
        results = run_simulation(
            seed=args.seed,
            num_iterations=args.iterations,
            split_or_tied=args.split_or_tied,
            speed_protocol=args.speed_protocol,
            transition_time=args.transition_time,
            split_speed_diff_ms=args.split_speed_diff,
            make_plots=not args.no_plots,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            checkpoint_path=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
            stop_after=args.stop_after,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))

    summary = results.summary
    print(
        "Simulation complete."
        if results.metadata["learning_complete"]
        else "Simulation paused at checkpoint."
    )
    print(f"Split/tied condition: {results.param_fixed['SplitOrTied']}")
    print(f"Speed protocol: {results.param_fixed['speedProtocol']}")
    if results.param_fixed["SplitOrTied"] == "split":
        print(f"Split speed difference: {results.param_fixed['splitSpeedDiffMS']:.3f} m/s")
    print(f"Seed: {results.metadata['seed']}")
    print(
        f"Iterations: {results.metadata['completed_iterations']}/{results.metadata['target_iterations']}"
    )
    print(f"Average energy rate: {summary['average_energy_rate']:.6f}")
    print(f"Final average energy rate: {summary['final_average_energy_rate']:.6f}")
    if args.output_dir is not None:
        print(f"Saved numerical results and figures to {args.output_dir.resolve()}")
    if args.checkpoint is not None:
        print(f"Saved checkpoint to {args.checkpoint.resolve()}")
    return 0
