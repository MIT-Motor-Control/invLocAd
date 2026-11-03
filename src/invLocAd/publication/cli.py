"""The ``invLocAd-publication`` commands for preparing inputs and reproducing figures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

from .reporting import sha256, verify_data, write_report

# Analysis identifiers: figure1-figure4 are Figures 1C-E, 2B-C, 3B-D and 4; s1-s4 are
# Supplementary Figures 1-4; r2 is the in-sample fit RMSE and exponential-fit R².
FIGURES = ("figure1", "figure2", "figure3", "figure4", "s1", "s2", "s3", "s4", "r2")
MAIN_FIGURES = {"figure1", "figure3", "figure4", "s3"}


def build_parser() -> argparse.ArgumentParser:
    """Return the ``invLocAd-publication`` argument parser."""
    parser = argparse.ArgumentParser(description="Reproduce the figures and numerical tables of the paper.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare the downloaded Zenodo data for analysis.")
    source = prepare.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="Experimental data folder (default: data/invLocAd-DATA).")
    source.add_argument("--bundle", type=Path, help="Maintainers: build the dataset from the original author files instead.")
    prepare.add_argument("--model-inputs", type=Path, help="publication-data.zip to install (default: found in the source folder or data/).")
    prepare.add_argument("--fall-archive", type=Path, help="Maintainers: folder of additional fall-probability MAT files.")
    prepare.add_argument("--private-archive", type=Path, help="Maintainers: research folder with the Supplementary Figure 2 and 4 inputs.")
    prepare.add_argument("--data", default=Path("data/publication"), type=Path, help="Folder for the prepared data (default: data/publication).")
    verify = commands.add_parser("verify-data", help="Check every prepared input against its SHA-256 manifest.")
    verify.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    addendum = commands.add_parser("import-addendum", help="Maintainers: merge supplementary inputs into a prepared dataset.")
    addendum.add_argument("--source", required=True, type=Path, help="Folder with the supplementary author inputs.")
    addendum.add_argument("--previous-bundle", type=Path, help="Earlier author folder whose original inputs are also merged.")
    addendum.add_argument("--generated-s1", type=Path, help="Folder with MATLAB grid-sensitivity CSVs to add.")
    addendum.add_argument("--generated-s1-script", type=Path, help="MATLAB script that produced those CSVs; required with --generated-s1.")
    addendum.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    reproduce = commands.add_parser("reproduce", help="Write figures, source tables and a completion report.")
    reproduce.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    reproduce.add_argument("--output", default=Path("outputs/publication"), type=Path, help="Output folder (default: outputs/publication).")
    reproduce.add_argument(
        "--figures", nargs="+", choices=FIGURES, default=list(FIGURES),
        help="Analyses to run: figure1-figure4 (Figures 1-4), s1-s4 (Supplementary Figures 1-4) "
        "and r2 (in-sample fit RMSE and R²). Default: all.",
    )
    reproduce.add_argument("--strict", action="store_true", help="Exit 2 if any selected analysis is partial or blocked.")
    reproduce.add_argument("--overwrite", action="store_true", help="Replace a prior managed reproduction output.")
    package = commands.add_parser("package-data", help="Build a portable ZIP of the verified input dataset.")
    package.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    package.add_argument("--output", default=Path("dist/publication-data.zip"), type=Path, help="ZIP file to write (default: dist/publication-data.zip).")
    s1 = commands.add_parser(
        "recompute-s1",
        help="Recompute the Supplementary Figure 1 cross-validation and grid-sensitivity analyses in MATLAB.",
    )
    s1.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    s1.add_argument("--output", default=Path("outputs/s1-recomputed"), type=Path, help="Output folder (default: outputs/s1-recomputed).")
    s1.add_argument("--matlab", default="matlab", help="MATLAB executable (default: matlab).")
    audit = commands.add_parser("validate-manuscript", help="Compare reproduced values with those reported in the paper.")
    audit.add_argument(
        "--manuscript", type=Path, default=None,
        help="Optional PDF of the paper; its SHA-256 is recorded in the report.",
    )
    audit.add_argument("--data", default=Path("data/publication"), type=Path, help="Prepared data folder (default: data/publication).")
    audit.add_argument("--results", default=Path("outputs/publication"), type=Path, help="Folder written by reproduce (default: outputs/publication).")
    audit.add_argument("--output", default=Path("outputs/manuscript-validation"), type=Path, help="Output folder (default: outputs/manuscript-validation).")
    return parser


def require_integrity(data_root: Path) -> dict:
    """Return the ``verify_data`` summary for ``data_root``.

    Raises:
        ValueError: If the dataset is not prepared or any input fails verification.
    """
    if not (data_root / "manifest.json").is_file():
        raise ValueError(
            f"Prepared data not found in {data_root}. Place the downloaded "
            "invLocAd-DATA folder inside data/, then run invLocAd-publication prepare. "
            "See data/README.md for the companion model inputs."
        )
    integrity = verify_data(data_root)
    if integrity["errors"]:
        details = "; ".join(f"{item['path']}: {item['reason']}" for item in integrity["errors"][:5])
        raise ValueError(f"Input integrity check failed: {details}")
    return integrity


def _companion_archive(source: Path) -> Path | None:
    """Return the first publication-data.zip in the DATA folder, ``source`` or data/, or None."""
    from .zenodo import _dataset

    source = source.expanduser().resolve()
    dataset = _dataset(source)
    for archive in (
        dataset / "publication-data.zip",
        source / "publication-data.zip",
        Path("data/publication-data.zip"),
    ):
        if archive.is_file():
            return archive
    return None


def reproduce(args) -> int:
    """Render the selected figures and tables into ``args.output`` with a report.

    Each figure runs independently: a missing input marks it blocked, and an exception
    marks it failed and writes the traceback to ``logs/``.

    Args:
        args: Parsed ``reproduce`` arguments.

    Returns:
        Exit code: 1 if any figure failed, 2 with ``--strict`` if any did not pass, else 0.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .main_figures import render_main
    from .supplements import render_supplements

    data_root, destination = args.data.resolve(), args.output.resolve()
    if destination.is_relative_to(data_root) or data_root.is_relative_to(destination):
        raise ValueError("Output and input directories must be separate, non-overlapping folders.")
    if destination.exists():
        report_path = destination / "report.json"
        if not args.overwrite or not report_path.is_file():
            raise ValueError("Output already exists. Choose a new folder or use --overwrite for a prior reproduction run.")
        if json.loads(report_path.read_text()).get("schema_version") != 1:
            raise ValueError("Refusing to replace an unrecognized output folder.")
    integrity = require_integrity(data_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        results = []
        for figure in dict.fromkeys(args.figures):
            renderer = render_main if figure in MAIN_FIGURES else render_supplements
            try:
                result = renderer(data_root, staging, [figure])
                if len(result) != 1 or result[0].get("id") != figure or result[0].get("status") not in {"passed", "partial", "blocked", "failed"}:
                    raise ValueError(f"Invalid result record for {figure}")
                results.extend(result)
            except FileNotFoundError as error:
                results.append({
                    "id": figure,
                    "status": "blocked",
                    "outputs": [],
                    "caveats": [
                        f"Required input is unavailable: {error}",
                        ("Experimental-only downloads do not contain model fits or simulation sweeps. "
                         "See data/README.md for the companion publication-data.zip."),
                    ],
                })
            except Exception as error:  # noqa: BLE001
                logs = staging / "logs"
                logs.mkdir(exist_ok=True)
                (logs / f"{figure}.txt").write_text(traceback.format_exc())
                results.append({"id": figure, "status": "failed", "outputs": [], "caveats": [str(error)], "error": f"{type(error).__name__}: {error}"})
            print(f"{figure}: {results[-1]['status']}", flush=True)
        write_report(staging, results, integrity)
        if destination.exists():
            backup = Path(temporary) / "previous"
            destination.rename(backup)
            try:
                staging.rename(destination)
            except OSError:
                backup.rename(destination)
                raise
        else:
            staging.rename(destination)
    print(f"Report: {destination / 'report.md'}")
    if any(item["status"] == "failed" for item in results):
        return 1
    return 2 if args.strict and any(item["status"] != "passed" for item in results) else 0


def package_data(data_root: Path, destination: Path) -> None:
    """Write the verified dataset to a ZIP under ``publication/`` with a ``.sha256`` file.

    Args:
        data_root: Prepared dataset to package.
        destination: Output ZIP path outside ``data_root``.
    """
    require_integrity(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    paths = sorted({record["path"] for record in manifest["files"]} | {"manifest.json"})
    if destination.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("Place the release archive outside its input dataset.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive_path = Path(temporary) / "data.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in paths:
                info = zipfile.ZipInfo("publication/" + relative, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with (data_root / relative).open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target)
        os.replace(archive_path, destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{sha256(destination)}  {destination.name}\n")
    print(f"Data archive: {destination.resolve()}")


def main(argv: list[str] | None = None) -> int:
    """Run one ``invLocAd-publication`` command.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Exit code: 0 on success; 1 on an error, a failed figure or a Supplementary Figure 1
        recomputation that did not pass; 2 for a ``--strict`` reproduction with an unpassed
        figure or a comparison with the paper that did not match.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if args.bundle is not None:
                if args.model_inputs is not None:
                    raise ValueError("Use --model-inputs with the public --source workflow, not --bundle.")
                from .data import prepare_data

                manifest = prepare_data(args.bundle, args.data, args.fall_archive, private_archive=args.private_archive)
            else:
                if args.fall_archive is not None or args.private_archive is not None:
                    raise ValueError("--fall-archive and --private-archive require the maintainer --bundle option.")
                from .zenodo import prepare_zenodo_data

                source = args.source or Path("data/invLocAd-DATA")
                archive = args.model_inputs if args.model_inputs is not None else _companion_archive(source)
                if archive is not None:
                    from .archive import install_publication_archive

                    manifest = install_publication_archive(archive, args.data, source)
                else:
                    manifest = prepare_zenodo_data(source, args.data)
            print(f"Prepared {len(manifest['files'])} inputs in {args.data.resolve()}")
            if manifest.get("dataset_kind") == "zenodo_experimental":
                print("Experimental data only: Figures 1 and 2C are partial; model analyses need publication-data.zip. See data/README.md.")
        elif args.command == "import-addendum":
            from .data import integrate_addendum

            manifest = integrate_addendum(
                args.source, args.data, previous_bundle=args.previous_bundle,
                generated_s1=args.generated_s1, generated_s1_script=args.generated_s1_script,
            )
            print(f"Prepared {len(manifest['files'])} inputs in {args.data.resolve()}")
        elif args.command == "verify-data":
            result = require_integrity(args.data)
            print(f"Verified {result['checked_files']} input files.")
        elif args.command == "reproduce":
            return reproduce(args)
        elif args.command == "package-data":
            package_data(args.data, args.output)
        elif args.command == "validate-manuscript":
            from .manuscript import validate_manuscript

            report = validate_manuscript(args.manuscript, args.data, args.results, args.output)
            print(f"Agreement with the paper: {report['status']}; report: {args.output / 'report.md'}")
            return 2 if report["status"] != "matched" else 0
        elif args.command == "recompute-s1":
            from .supplements import recompute_s1_matlab

            integrity = require_integrity(args.data)
            if args.output.resolve().is_relative_to(args.data.resolve()) or args.data.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("Output and input directories must be separate, non-overlapping folders.")
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Choose an empty output directory for the MATLAB Supplementary Figure 1 recomputation.")
            args.output.mkdir(parents=True, exist_ok=True)
            result = recompute_s1_matlab(args.data.resolve(), args.output.resolve(), args.matlab)
            write_report(args.output.resolve(), [result], integrity)
            print(f"Supplementary Figure 1 recomputation: {result['status']}; report: {args.output / 'report.md'}")
            return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
