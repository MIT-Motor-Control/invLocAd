"""Build and extend the prepared publication dataset from the original author files."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .reporting import sha256, verify_data


def prepare_data(
    bundle: Path,
    destination: Path,
    fall_archive: Path | None = None,
    *,
    private_archive: Path | None = None,
) -> dict:
    """Copy the original inputs into ``destination`` and write its SHA-256 manifest.

    Args:
        bundle: Author folder with ``code/original_data``, ``code/verify`` and ``results``.
        destination: Prepared-data folder to create or extend.
        fall_archive: Optional folder of additional fall-probability MAT files.
        private_archive: Optional research folder with the Supplementary Figure 2 and 4
            results and the experimental CSVs.

    Returns:
        The written manifest.

    Raises:
        ValueError: If sources conflict or overlap ``destination``, or ``destination`` holds
            different or unlisted inputs.
    """
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if not (bundle / "code/original_data").is_dir():
        raise FileNotFoundError(f"Original author files not found: {bundle / 'code/original_data'}")
    roots = {"bundle": bundle}
    for name, source in (
        ("bundle", bundle),
        ("fall_archive", fall_archive),
        ("private_archive", private_archive),
    ):
        if source is not None:
            source = Path(source).resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Source archive does not exist: {name}")
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError(
                    "The prepared-data destination must be outside and not contain any source archive"
                )
            roots[name] = source

    plan: dict[str, dict] = {}

    def add(path: Path, relative: str, source: str, transformation: str = "Copied byte-for-byte"):
        """Plan a copy of ``path`` to ``relative``; an identical duplicate corroborates it."""
        if not path.is_file():
            return
        if not path.resolve().is_relative_to(roots[source.split(":", 1)[0]]):
            raise ValueError(f"Scientific input escapes its source archive: {source}")
        digest = sha256(path)
        if relative in plan:
            if plan[relative]["sha256"] != digest:
                raise ValueError(f"Conflicting scientific input: {relative} ({source})")
            plan[relative]["corroborating_sources"].append(source)
            return
        plan[relative] = {
            "path": relative,
            "sha256": digest,
            "size": path.stat().st_size,
            "source": source,
            "transformation": transformation,
            "corroborating_sources": [],
            "_source_path": path,
        }

    original = bundle / "code/original_data"
    for path in sorted(original.glob("*.mat")):
        add(path, f"original/{path.name}", f"bundle:/code/original_data/{path.name}")
    for relative in (
        "data/cosmed/processed/processedEnergyData.mat",
        "data/sanchez data/sanchez_energy.mat",
        "data/sanchez data/sanchez_sla.mat",
    ):
        add(original / relative, f"original/{relative}", f"bundle:/code/original_data/{relative}")
    for path in sorted((original / "Submitted Data").glob("*.csv")):
        add(
            path,
            f"original/Submitted Data/{path.name}",
            f"bundle:/code/original_data/Submitted Data/{path.name}",
        )
    for path in sorted((bundle / "code/verify").glob("*.mat")):
        add(path, f"regenerated/{path.name}", f"bundle:/code/verify/{path.name}")
    for path in sorted((bundle / "code/verify/data/2d sweep fine").glob("*.mat")):
        add(
            path,
            f"regenerated/data/2d sweep fine/{path.name}",
            f"bundle:/code/verify/data/2d sweep fine/{path.name}",
        )
    for path in sorted((bundle / "code/verify/figures").glob("comment1_loso_cv_*.csv")):
        add(path, f"regenerated/figures/{path.name}", f"bundle:/code/verify/figures/{path.name}")
    for path in sorted((bundle / "results").glob("*.csv")):
        add(
            path,
            f"reference_tables/{path.name}",
            f"bundle:/results/{path.name}",
            "Author reference table; not assumed to represent the final plotted analysis",
        )

    for folder, prefix in (
        (original / "data/fall probability", "bundle:/code/original_data/data/fall probability"),
        (Path(fall_archive) if fall_archive else None, "fall_archive:"),
    ):
        if folder is not None:
            for path in sorted(folder.glob("*.mat")):
                add(path, f"fall_probability/{path.name}", f"{prefix}/{path.name}")

    if private_archive is not None:
        private_archive = Path(private_archive)
        for comment, extension in (("comment_2", "s2"), ("comment_4", "s5")):
            result = private_archive / "reviewer_comments" / comment / "results"
            for path in sorted(result.iterdir()) if result.is_dir() else []:
                if path.is_file() and path.suffix in {".npz", ".csv", ".json"}:
                    add(
                        path,
                        f"extensions/{extension}/{path.name}",
                        f"private_archive:/reviewer_comments/{comment}/results/{path.name}",
                    )
            if extension == "s5":
                for path in sorted((result / "cells").glob("*.npz")):
                    add(
                        path,
                        f"extensions/s5/cells/{path.name}",
                        f"private_archive:/reviewer_comments/comment_4/results/cells/{path.name}",
                    )
        for path in sorted((private_archive / "Data").glob("*.csv")):
            add(
                path,
                f"extensions/experimental_csv/{path.name}",
                f"private_archive:/Data/{path.name}",
            )

    energy_relative = "original/data/cosmed/processed/processedEnergyData.mat"
    # Without the standalone energy file, extract its structs from the original fit file.
    if energy_relative not in plan:
        embedded = original / "2d_sweep_trial50_param_result.mat"
        if embedded.is_file():
            from scipy.io import loadmat, savemat

            fields = ("split_energy", "tied1_energy", "tied2_energy")
            values = loadmat(embedded, variable_names=list(fields))
            if all(field in values for field in fields):
                stream = io.BytesIO()
                savemat(stream, {field: values[field] for field in fields}, do_compression=True)
                payload = stream.getvalue()
                header = (
                    b"MATLAB 5.0 MAT-file, recovered publication energy fields; see manifest.json"
                )
                # Timestamp-free header for a stable hash, as in zenodo._mat_payload.
                payload = header.ljust(116, b" ") + payload[116:]
                plan[energy_relative] = {
                    "path": energy_relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "source": "bundle:/code/original_data/2d_sweep_trial50_param_result.mat",
                    "transformation": "Extracted exact split_energy, tied1_energy, tied2_energy MATLAB structs; "
                    "no numerical processing; deterministic MAT header",
                    "source_sha256": sha256(embedded),
                    "corroborating_sources": [],
                    "_payload": payload,
                }

    if destination.exists():
        unlisted = [
            p.relative_to(destination).as_posix()
            for p in destination.rglob("*")
            if p.is_file()
            and p.relative_to(destination).as_posix() != "manifest.json"
            and p.relative_to(destination).as_posix() not in plan
        ]
        if unlisted:
            raise ValueError(
                "Existing prepared inputs are absent from this import plan; repeat all original source "
                f"arguments or use a fresh destination. First unlisted input: {min(unlisted)}"
            )
    for relative, item in plan.items():
        existing = destination / relative
        if not existing.resolve().is_relative_to(destination):
            raise ValueError(f"Prepared-data path escapes its destination: {relative}")
        if existing.exists() and (not existing.is_file() or sha256(existing) != item["sha256"]):
            raise ValueError(f"Refusing to replace a different existing input: {relative}")
    for relative, item in plan.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if "_payload" in item:
                target.write_bytes(item["_payload"])
            else:
                shutil.copy2(item["_source_path"], target)

    expected = [
        energy_relative,
        "original/processedSLAData_23Subj.mat",
        "original/data/sanchez data/sanchez_energy.mat",
        "original/data/sanchez data/sanchez_sla.mat",
        "original/iso_fall_risk_heat_map.mat",
    ]
    rollout_metadata = {
        "path": "regenerated/data/2d sweep fine",
        "required_trials_per_cell": 50,
        "available_files": sum(k.startswith("regenerated/data/2d sweep fine/") for k in plan),
        "required_unique_cells": None,
        "required_files": None,
        "note": "Figure 2C requires the complete 50-trial library at each unique selected published-fit cell; "
        "best-trial fitted trajectories do not substitute for those rollouts. "
        "File presence alone does not verify the required cell/trial identities.",
    }
    parameter_file = original / "2d_sweep_trial50_param_result.mat"
    if parameter_file.is_file():
        import numpy as np
        from scipy.io import loadmat

        fields = ["static_params", "exp_subj_num", "learning_pool", "sym_cost_pool"]
        values = loadmat(
            parameter_file, variable_names=fields, struct_as_record=False, squeeze_me=True
        )
        if all(field in values for field in fields):
            params, n = values["static_params"], np.asarray(values["exp_subj_num"]).astype(int)
            lp, sp = np.asarray(values["learning_pool"]), np.asarray(values["sym_cost_pool"])
            cells = {
                (
                    c,
                    int(np.argmin(abs(lp - params[c, s].learning_rate))),
                    int(np.argmin(abs(sp - params[c, s].sym_cost))),
                )
                for c in range(len(n))
                for s in range(n[c])
            }
            rollout_metadata.update(
                required_unique_cells=len(cells),
                required_files=50 * len(cells),
                condition_unique_cells=[sum(c == i for c, _, _ in cells) for i in range(len(n))],
                expectation_source="Original published static_params nearest learning_pool and sym_cost_pool grid cells; 50 trials per cell",
            )
    manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "purpose": "Cached numerical reproduction of the published analyses; not a raw simulation rerun",
        "files": [
            {key: value for key, value in plan[relative].items() if not key.startswith("_")}
            for relative in sorted(plan)
        ],
        "missing_inputs": [relative for relative in expected if relative not in plan],
        "missing_analyses": {"s3_rollouts": rollout_metadata},
        "counts": {
            "files": len(plan),
            "fall_probability_files": sum(k.startswith("fall_probability/") for k in plan),
        },
        "caveats": [
            "Original and regenerated realizations are retained separately.",
            "Raw COSMED and OpenSim optimization inputs are not supplied by this import.",
            "Reference tables may describe earlier analyses; use per-figure exported tables for plotted data.",
            "No regenerated Figure 2C rollout subset is synthesized or inferred from fitted parameters.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    return manifest


def _same_energy(left: Path, right: Path) -> bool:
    """Return True if both MAT files hold equal, finite split, tied1 and tied2 energy structs."""
    import numpy as np
    from scipy.io import loadmat

    a, b = [loadmat(path, struct_as_record=False, squeeze_me=True) for path in (left, right)]
    fields = {"split_energy", "tied1_energy", "tied2_energy"}
    if any({key for key in values if not key.startswith("__")} != fields for values in (a, b)):
        return False
    for field in fields:
        if set(getattr(a[field], "_fieldnames", [])) != set(getattr(b[field], "_fieldnames", [])):
            return False
        for condition in a[field]._fieldnames:
            x, y = (
                np.asarray(getattr(a[field], condition)),
                np.asarray(getattr(b[field], condition)),
            )
            if not np.isfinite(x).all() or not np.array_equal(x, y):
                return False
    return True


def _validate_figure2c_rollouts(parameter_file: Path, rollouts: dict[str, Path]) -> dict:
    """Verify the Figure 2C model rollout library and return its manifest summary.

    Each fitted participant's nearest (learning rate, symmetry weight) grid cell must have
    trials 1-50, each with a 0/1 fall flag and a finite energy trace that has at least 100
    samples unless the trial fell.

    Args:
        parameter_file: Original fit file with ``static_params`` and the parameter grids.
        rollouts: Prepared relative path mapped to the rollout MAT file.

    Returns:
        The ``missing_analyses["s3_rollouts"]`` manifest entry.

    Raises:
        ValueError: If a rollout is unrecognized, duplicated, invalid or missing.
    """
    import numpy as np
    from scipy.io import loadmat

    values = loadmat(parameter_file, struct_as_record=False, squeeze_me=True)
    n = np.atleast_1d(values["exp_subj_num"]).astype(int)
    params = np.asarray(values["static_params"], dtype=object).reshape(3, -1)
    learning, symmetry = [
        np.atleast_1d(values[key]).astype(float) for key in ("learning_pool", "sym_cost_pool")
    ]
    if len(n) != 3 or np.any(n < 1) or np.any(n > params.shape[1]):
        raise ValueError("Invalid original-fit participant layout for Figure 2C")
    if any(
        len(pool) < 2 or not np.isfinite(pool).all() or np.any(np.diff(pool) <= 0)
        for pool in (learning, symmetry)
    ):
        raise ValueError("Invalid Figure 2C parameter grid")
    expected = {
        (
            condition,
            int(np.argmin(abs(learning - params[condition, subject].learning_rate))),
            int(np.argmin(abs(symmetry - params[condition, subject].sym_cost))),
        )
        for condition in range(3)
        for subject in range(n[condition])
    }
    pattern = re.compile(
        r"model_split45_trans20_slow([0-9.]+)_fast([0-9.]+)_noise0\.01_"
        r"learning([0-9.]+)_energysym([0-9.]+)_swingcost1_trial(\d+)\.mat"
    )
    trials, falls = set(), 0
    for relative, path in rollouts.items():
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognized Figure 2C rollout filename: {path.name}")
        slow, fast, rate, weight = map(float, match.groups()[:4])
        trial = int(match.group(5))
        condition = int(np.argmin(abs(np.array([0.8, 0.65, 0.5]) - slow)))
        ia, ip = int(np.argmin(abs(learning - rate))), int(np.argmin(abs(symmetry - weight)))
        if (
            abs(slow - (0.8, 0.65, 0.5)[condition]) > 1e-9
            or abs(slow + fast - 2) > 1e-9
            or abs(learning[ia] - rate) > 5.1e-7
            or abs(symmetry[ip] - weight) > 5.1e-6
        ):
            raise ValueError(f"Figure 2C rollout does not match the published grid: {relative}")
        key = (condition, ia, ip, trial)
        if (condition, ia, ip) not in expected or not 1 <= trial <= 50 or key in trials:
            raise ValueError(f"Unexpected or duplicate Figure 2C cell/trial: {relative}")
        trials.add(key)
        result = loadmat(path, variable_names=["energy", "fall_flag"], squeeze_me=True)
        energy, flag = np.atleast_1d(result["energy"]), np.atleast_1d(result["fall_flag"])
        if flag.shape != (2,) or flag[0] not in (0, 1) or not np.isfinite(energy).all():
            raise ValueError(f"Invalid Figure 2C energy or fall flag: {relative}")
        if not flag[0] and energy.size < 100:
            raise ValueError(f"Nonfalling Figure 2C rollout lacks the analysis windows: {relative}")
        falls += int(flag[0])
    required = {(*cell, trial) for cell in expected for trial in range(1, 51)}
    if trials != required:
        raise ValueError(
            f"Incomplete Figure 2C rollout library: {len(required - trials)} required trials "
            "missing"
        )
    return {
        "path": "regenerated/data/2d sweep fine",
        "required_trials_per_cell": 50,
        "available_files": len(trials),
        "required_files": len(required),
        "required_unique_cells": len(expected),
        "condition_unique_cells": [sum(cell[0] == c for cell in expected) for c in range(3)],
        "verified_cell_trial_identities": True,
        "falling_trials": falls,
        "expectation_source": "Original published static_params nearest learning_pool and sym_cost_pool grid cells; 50 trials per cell",
        "note": "Every required cell/trial filename and finite energy array was verified; stochastic simulations were not rerun.",
    }


def integrate_addendum(
    source: Path,
    destination: Path,
    *,
    previous_bundle: Path | None = None,
    generated_s1: Path | None = None,
    generated_s1_script: Path | None = None,
) -> dict:
    """Merge supplementary inputs into a verified prepared dataset.

    Inputs already in the manifest must be byte-identical, except the aggregate energy MAT
    file, which may differ in encoding when its three energy structs are equal; the existing
    file is kept. The Figure 2C model rollout library must be complete.

    Args:
        source: Supplementary author folder containing ``code/verify``.
        destination: Existing prepared dataset.
        previous_bundle: Optional earlier author folder whose original inputs are also merged.
        generated_s1: Optional folder with MATLAB cross-validation grid-sensitivity CSVs.
        generated_s1_script: MATLAB script that produced ``generated_s1``; required with it.

    Returns:
        The updated manifest.

    Raises:
        ValueError: If an input conflicts with the dataset or fails validation.
    """
    if Path(destination).is_symlink():
        raise ValueError("The prepared dataset must not be a symlink")
    destination = Path(destination).resolve()
    integrity = verify_data(destination)
    if integrity["errors"]:
        raise ValueError(
            f"Existing prepared dataset failed integrity verification: {integrity['errors'][0]}"
        )
    manifest_path = destination / "manifest.json"
    initial_manifest = manifest_path.read_bytes()
    manifest = json.loads(initial_manifest)
    records = {entry["path"]: entry for entry in manifest["files"]}
    roots = {"addendum": Path(source).resolve()}
    if previous_bundle is not None:
        roots["previous_bundle"] = Path(previous_bundle).resolve()
    if (generated_s1 is None) != (generated_s1_script is None):
        raise ValueError(
            "Generated grid-sensitivity tables require both their folder and MATLAB source script"
        )
    if generated_s1 is not None:
        roots["generated_s1"] = Path(generated_s1).resolve()
    for label, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Scientific source folder not found: {label}")
        if root.is_relative_to(destination) or destination.is_relative_to(root):
            raise ValueError("Scientific sources and destination must not overlap")
    if not (roots["addendum"] / "code/verify").is_dir():
        raise FileNotFoundError("Expected the addendum's code/verify folder")
    additions: dict[str, Path] = {}
    observations = []

    def add(path: Path, relative: str, label: str, generated: dict | None = None):
        """Stage ``path`` as ``relative`` or record it as corroborating an existing input."""
        if not path.is_file():
            return
        if path.is_symlink() or not path.resolve().is_relative_to(roots[label]):
            raise ValueError(f"Scientific input escapes its source: {path.name}")
        provenance = f"{label}:/{path.relative_to(roots[label]).as_posix()}"
        digest = sha256(path)
        observations.append({"source": provenance, "sha256": digest})
        comparison = "SHA-256 identical"
        if relative in records:
            existing = additions.get(relative, destination / relative)
            if records[relative]["sha256"] != digest:
                if (
                    relative != "original/data/cosmed/processed/processedEnergyData.mat"
                    or not _same_energy(existing, path)
                ):
                    raise ValueError(f"Conflicting scientific input: {relative}")
                comparison = "All split_energy, tied1_energy and tied2_energy fields exactly equal; existing MAT bytes preserved"
            record = records[relative]
            if provenance != record["source"] and provenance not in record.setdefault(
                "corroborating_sources", []
            ):
                record["corroborating_sources"].append(provenance)
            if provenance != record["source"] or comparison != "SHA-256 identical":
                detail = {"source": provenance, "sha256": digest, "comparison": comparison}
                if detail not in record.setdefault("corroborating_files", []):
                    record["corroborating_files"].append(detail)
        else:
            records[relative] = {
                "path": relative,
                "sha256": digest,
                "size": path.stat().st_size,
                "source": provenance,
                "transformation": "Copied byte-for-byte",
                "corroborating_sources": [],
            }
            additions[relative] = path
        if generated:
            records[relative]["generation"] = generated

    for label, root in roots.items():
        if label == "generated_s1":
            continue
        original = root / "code/original_data"
        paths = list(original.glob("*.mat"))
        paths += list((original / "data/cosmed/processed").glob("*.mat"))
        paths += list((original / "data/sanchez data").glob("sanchez_*.mat"))
        paths += [
            p
            for p in (original / "data/trajectory optimization").rglob("*")
            if p.suffix in {".sto", ".osim"}
        ]
        for path in sorted(paths):
            add(path, "original/" + path.relative_to(original).as_posix(), label)
        if label != "addendum":
            continue
        verify = root / "code/verify"
        paths = list(verify.glob("*.mat"))
        paths += [p for p in (verify / "data/cosmed").rglob("*") if p.suffix in {".mat", ".csv"}]
        paths += list((verify / "data/2d sweep fine").glob("*.mat"))
        paths += list((verify / "figures").glob("comment1_loso_cv_*.csv"))
        for path in sorted(paths):
            add(path, "regenerated/" + path.relative_to(verify).as_posix(), label)
    if generated_s1 is not None:
        import csv

        import numpy as np

        script = Path(generated_s1_script)
        generated = {
            "method": "MATLAB grid-sensitivity analysis",
            "script": script.name,
            "script_sha256": sha256(script),
            "inputs": [
                {
                    "path": "regenerated/2d_sweep_trial50_param_result_trialavg.mat",
                    "sha256": records["regenerated/2d_sweep_trial50_param_result_trialavg.mat"][
                        "sha256"
                    ],
                }
            ],
        }
        observations.append(
            {"source": f"analysis:/{script.name}", "sha256": generated["script_sha256"]}
        )
        for name in (
            "comment1_grid_sensitivity_summary.csv",
            "comment1_grid_sensitivity_persubject.csv",
        ):
            path = roots["generated_s1"] / name
            if not path.is_file():
                raise FileNotFoundError(f"Generated grid-sensitivity table missing: {name}")
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows or any(None in row or None in row.values() for row in rows):
                raise ValueError(f"Malformed generated grid-sensitivity table: {name}")
            for row in rows:
                for value in row.values():
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if not np.isfinite(number):
                        raise ValueError(f"Nonfinite generated grid-sensitivity value: {name}")
            add(path, f"regenerated/figures/{name}", "generated_s1", generated)
    original_fit = "original/2d_sweep_trial50_param_result.mat"
    rollout_paths = {
        relative: additions.get(relative, destination / relative)
        for relative in records
        if relative.startswith("regenerated/data/2d sweep fine/")
    }
    rollout_info = _validate_figure2c_rollouts(
        additions.get(original_fit, destination / original_fit), rollout_paths
    )
    manifest["files"] = [records[key] for key in sorted(records)]
    manifest["missing_inputs"] = [
        key for key in manifest.get("missing_inputs", []) if key not in records
    ]
    manifest.setdefault("missing_analyses", {})["s3_rollouts"] = rollout_info
    manifest.setdefault("counts", {}).update(
        files=len(records), s3_rollout_files=len(rollout_paths)
    )
    manifest["caveats"] = [
        note
        for note in manifest.get("caveats", [])
        if note != "Raw COSMED and OpenSim optimization inputs are not supplied by this import."
    ]
    note = "Supplied COSMED recordings and OpenSim source outputs are retained; available optimization conditions do not by themselves establish a complete optimization rerun."
    if note not in manifest["caveats"]:
        manifest["caveats"].append(note)
    identity = hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()
    history = manifest.setdefault("scientific_addenda", [])
    if not any(item["source_inventory_sha256"] == identity for item in history):
        history.append(
            {
                "source_inventory_sha256": identity,
                "integrated_at": datetime.now(UTC).isoformat(),
                "added_files": sorted(additions),
                "source_files": observations,
            }
        )
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    if json.loads(initial_manifest) == manifest:
        return manifest
    with tempfile.TemporaryDirectory(
        prefix=".publication-addendum-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "publication"
        shutil.copytree(destination, staging)
        for relative, path in additions.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (staging / "manifest.json").write_bytes(encoded)
        checked = verify_data(staging)
        if checked["errors"]:
            raise ValueError(
                f"Staged addendum failed integrity verification: {checked['errors'][0]}"
            )
        if manifest_path.read_bytes() != initial_manifest or verify_data(destination)["errors"]:
            raise ValueError("Prepared dataset changed during addendum integration")
        backup = destination.with_name(destination.name + ".before-addendum")
        if backup.exists():
            raise ValueError(f"A prior dataset backup needs attention: {backup.name}")
        destination.rename(backup)
        try:
            staging.rename(destination)
        except OSError:
            backup.rename(destination)
            raise
        shutil.rmtree(backup)
    return manifest
