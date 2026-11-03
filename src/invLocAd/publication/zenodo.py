"""Prepare the experimental figure inputs from the six public DATA measurement CSVs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .reporting import utc_now, verify_data

CSV_NAMES = (
    "split_ds_asym.csv",
    "split_energy.csv",
    "split_sla.csv",
    "split_step_time_asym.csv",
    "tied_energy.csv",
    "tied_sla.csv",
)
ENERGY_PATH = "original/data/cosmed/processed/processedEnergyData.mat"
SLA_PATH = "original/processedSLAData_23Subj.mat"
EXPERIMENTAL_PATHS = frozenset(
    [f"extensions/experimental_csv/{name}" for name in CSV_NAMES] + [ENERGY_PATH, SLA_PATH]
)
# Belt-speed difference (m/s) and coded participant IDs of each condition.
CONDITIONS = {"condition_1": 0.4, "condition_2": 0.7, "condition_3": 1.0}
CONDITION_IDS = {
    "condition_1": {3, 8, 9, 10, 13, 17, 19, 23, 25},
    "condition_2": {2, 4, 7, 11, 14, 18, 21, 24, 26},
    "condition_3": {1, 5, 6, 12, 15, 16, 20, 22, 27},
}
# Participant IDs present in the energy, SLA and step-time/DSA CSVs.
COHORTS = {
    "energy": set(range(1, 28)) - {19},
    "sla": set(range(1, 28)) - {11, 23, 25, 27},
    "temporal": {1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24},
}


def _dataset(source: Path) -> Path:
    """Return the DATA folder with ``processed_csv/``: ``source`` or its single subfolder."""
    if not source.is_dir():
        raise FileNotFoundError(f"Public DATA folder not found: {source}")
    if (source / "processed_csv").is_dir():
        return source
    candidates = [p for p in source.iterdir() if p.is_dir() and (p / "processed_csv").is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected processed_csv/ in the public DATA folder or exactly one immediate wrapper folder"
        )
    if not candidates[0].resolve().is_relative_to(source):
        raise ValueError("Public DATA wrapper escapes its source folder")
    return candidates[0].resolve()


def _read_csv(name: str, payload: bytes) -> list[dict]:
    """Parse and validate one public measurement CSV.

    Args:
        name: CSV file name, which identifies the modality (energy, SLA or temporal).
        payload: Raw CSV bytes.

    Returns:
        One record per participant with ``subject_id``, ``condition``, ``values``,
        ``source_row``, ``declared_samples`` and ``missing_terminal_samples``, sorted by
        condition and participant ID.
    """
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)
    fields = reader.fieldnames or []
    fixed = ["subject_id", "condition", "speed_diff_ms"]
    modality = "energy" if "energy" in name else "sla" if "sla" in name else "temporal"
    count_name = (
        None if modality == "energy" else "num_strides" if modality == "sla" else "num_steps"
    )
    prefix = "t" if modality == "energy" else "stride" if modality == "sla" else "step"
    if count_name:
        fixed += [count_name]
    length = len(fields) - len(fixed)
    if length < 1 or fields != fixed + [f"{prefix}{i}" for i in range(length)]:
        raise ValueError(f"{name}: invalid or duplicate CSV columns")
    if modality == "energy" and length != (2700 if name.startswith("split") else 360):
        raise ValueError(f"{name}: truncated or unexpected energy time axis")
    rows, seen = [], set()
    expected = {f"AB{i:02d}" for i in COHORTS[modality]}
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{name}:{line}: truncated or overlong CSV row")
        identifier, condition = row["subject_id"], row["condition"]
        if identifier not in expected or identifier in seen:
            raise ValueError(f"{name}:{line}: unexpected or duplicate participant {identifier}")
        seen.add(identifier)
        if condition not in CONDITIONS or int(identifier[2:]) not in CONDITION_IDS[condition]:
            raise ValueError(f"{name}:{line}: participant condition mismatch")
        speed = float(row["speed_diff_ms"])
        if not np.isfinite(speed) or speed != CONDITIONS[condition]:
            raise ValueError(f"{name}:{line}: condition speed mismatch")
        count = int(row[count_name]) if count_name else length
        if not 0 < count <= length:
            raise ValueError(f"{name}:{line}: invalid sample count")
        # Eight public DSA records lack only the final toe-off-derived value.
        terminal_missing = int(
            name == "split_ds_asym.csv" and not row[f"{prefix}{count - 1}"].strip()
        )
        values = np.array([float(row[f"{prefix}{i}"]) for i in range(count - terminal_missing)])
        if not np.isfinite(values).all():
            raise ValueError(f"{name}:{line}: nonfinite scientific values")
        if any(row[f"{prefix}{i}"].strip() for i in range(count, length)):
            raise ValueError(f"{name}:{line}: nonempty values in sample padding")
        # Records must cover the 100-stride windows of the Figure 1 SLA analysis.
        if modality == "sla" and (
            count < 100 or (name.startswith("split") and np.ceil(count * 6 / 45) < 100)
        ):
            raise ValueError(
                f"{name}:{line}: insufficient strides for the 100-stride analysis windows"
            )
        rows.append(
            {
                "subject_id": identifier,
                "condition": condition,
                "values": values,
                "source_row": line,
                "declared_samples": count,
                "missing_terminal_samples": terminal_missing,
            }
        )
    if seen != expected:
        raise ValueError(
            f"{name}: incomplete participant cohort; missing {sorted(expected - seen)}"
        )
    return sorted(rows, key=lambda row: (row["condition"], row["subject_id"]))


def _mat_payload(fields: dict) -> bytes:
    """Return compressed MAT-file bytes whose header text has no timestamp, for stable hashes."""
    stream = io.BytesIO()
    savemat(stream, fields, do_compression=True)
    payload = stream.getvalue()
    header = b"MATLAB 5.0 MAT-file, public CSV experimental inputs; see manifest.json"
    return header.ljust(116, b" ") + payload[116:]


def _arrays(tables: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Arrange parsed CSV records in the layout of the original energy and SLA MAT files.

    Returns:
        ``energy``: ``split_energy`` and ``tied1_energy``, each mapping condition to a
        participants x samples array; ``sla``: ``{split,tied1}_sla_cond{1,2,3}`` 1 x 9
        object arrays of column vectors, padded with empty cells.

    Raises:
        ValueError: If a tied-belt normalization baseline (mean of the final 180 samples)
            is zero.
    """
    energy, sla = {}, {}
    for stem, field in (("split", "split_energy"), ("tied", "tied1_energy")):
        rows = tables[f"{stem}_energy.csv"]
        energy[field] = {
            condition: np.stack([row["values"] for row in rows if row["condition"] == condition])
            for condition in CONDITIONS
        }
        if stem == "tied" and any(
            np.any(values[:, -180:].mean(axis=1) == 0) for values in energy[field].values()
        ):
            raise ValueError("Public energy data has a zero tied-belt normalization baseline")
        for number, condition in enumerate(CONDITIONS, 1):
            rows = [row for row in tables[f"{stem}_sla.csv"] if row["condition"] == condition]
            cells = np.empty((1, 9), dtype=object)
            for index in range(9):
                cells[0, index] = (
                    rows[index]["values"].reshape(-1, 1) if index < len(rows) else np.empty((0, 0))
                )
            sla[f"{'split' if stem == 'split' else 'tied1'}_sla_cond{number}"] = cells
    return energy, sla


def _compare_existing(
    destination: Path, payloads: dict[str, bytes], energy: dict, sla: dict
) -> dict:
    """Check an existing prepared dataset against the public CSVs and return its manifest.

    Arrays must be equal in an experimental-only dataset. Full-precision author arrays may
    differ from the six-decimal CSV values by at most 5.00001e-7 (half a unit in the sixth
    decimal plus floating-point slack).
    """
    integrity = verify_data(destination)
    if integrity["errors"]:
        raise ValueError(
            f"Existing prepared dataset failed integrity verification: {integrity['errors'][0]}"
        )
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, payload in payloads.items():
        path = destination / "extensions/experimental_csv" / name
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Existing prepared dataset conflicts with public CSV: {name}")
    tolerance = 0.0 if manifest.get("dataset_kind") == "zenodo_experimental" else 5.00001e-7
    e = loadmat(destination / ENERGY_PATH, squeeze_me=True, struct_as_record=False)
    s = loadmat(destination / SLA_PATH, squeeze_me=True, struct_as_record=False)

    def compare(actual, expected, label):
        """Raise if the arrays differ in shape or by more than ``tolerance``."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0, atol=tolerance
        ):
            raise ValueError(
                f"Existing prepared experimental arrays conflict with public CSV: {label}"
            )

    for field, conditions in energy.items():
        if field not in e:
            raise ValueError(f"Existing prepared energy field is missing: {field}")
        for condition, values in conditions.items():
            compare(getattr(e[field], condition), values, f"{field}/{condition}")
    for field, cells in sla.items():
        if field not in s:
            raise ValueError(f"Existing prepared SLA field is missing: {field}")
        existing = np.atleast_1d(s[field])
        if len(existing) != cells.size:
            raise ValueError(
                f"Existing prepared SLA cell layout conflicts with public CSV: {field}"
            )
        for index, values in enumerate(cells.ravel()):
            compare(np.asarray(existing[index]).ravel(), values.ravel(), f"{field}/{index + 1}")
    return manifest


def prepare_zenodo_data(source: Path, destination: Path) -> dict:
    """Prepare the public measurement CSVs, or check an existing dataset against them.

    A new ``destination`` receives byte copies of the six CSVs, energy and SLA MAT files
    built from them and a manifest; an existing ``destination`` is only checked.

    Args:
        source: Public DATA folder, or its parent when that holds a single DATA folder.
        destination: Prepared-data folder.

    Returns:
        The manifest of the new or existing dataset.

    Raises:
        ValueError: If a CSV fails validation or conflicts with an existing ``destination``.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser()
    if destination.is_symlink():
        raise ValueError("Prepared-data destination must not be a symlink")
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must not overlap or contain one another")
    dataset = _dataset(source)
    payloads = {}
    for name in CSV_NAMES:
        path = dataset / "processed_csv" / name
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Public input escapes its source folder: {name}")
        payloads[name] = path.read_bytes()
    tables = {name: _read_csv(name, payload) for name, payload in payloads.items()}
    energy, sla = _arrays(tables)
    if destination.exists():
        return _compare_existing(destination, payloads, energy, sla)

    source_records = [
        {
            "path": f"processed_csv/{name}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        for name, payload in payloads.items()
    ]
    prepared = {
        f"extensions/experimental_csv/{name}": payload for name, payload in payloads.items()
    }
    prepared[ENERGY_PATH], prepared[SLA_PATH] = _mat_payload(energy), _mat_payload(sla)
    entries = []
    for relative, payload in sorted(prepared.items()):
        copied = relative.startswith("extensions/")
        names = (
            [Path(relative).name]
            if copied
            else [
                f"{stem}_{'energy' if relative == ENERGY_PATH else 'sla'}.csv"
                for stem in ("split", "tied")
            ]
        )
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "source": "zenodo:/processed_csv/" + names[0]
                if copied
                else "zenodo:/processed_csv",
                "source_files": [
                    record for record in source_records if Path(record["path"]).name in names
                ],
                "transformation": "Copied byte-for-byte"
                if copied
                else (
                    "Parsed CSV values without additional rounding or interpolation; sorted by condition and coded participant ID; "
                    "stored as MATLAB structs/cells with deterministic header. "
                    + (
                        "Only split_energy and tied1_energy are available; tied2 was not invented."
                        if relative == ENERGY_PATH
                        else "Empty MATLAB cells pad each condition to nine; no short_idx values were invented."
                    )
                ),
                "corroborating_sources": [],
            }
        )
    missing = [
        "original/data/sanchez data/sanchez_energy.mat",
        "original/data/sanchez data/sanchez_sla.mat",
        "original/2d_sweep_trial50_param_result.mat",
        "original/iso_fall_risk_heat_map.mat",
        "regenerated/2d_sweep_trial50_param_result_trialavg.mat",
        "regenerated/figures/comment1_loso_cv_persubject.csv",
        "regenerated/figures/comment1_loso_cv_stats.csv",
        "regenerated/figures/comment1_loso_cv_test.csv",
        "fall_probability/",
        "regenerated/data/2d sweep fine/",
        "extensions/s2/",
        "extensions/s5/",
    ]
    manifest = {
        "schema_version": 1,
        "dataset_kind": "zenodo_experimental",
        "prepared_at": utc_now(),
        "purpose": "Public experimental data for partial Figure 1 and Figure 2C reproduction; model caches are absent",
        "source_identity": {"kind": "public_DATA_processed_csv", "files": source_records},
        "files": entries,
        "missing_inputs": missing,
        "counts": {
            "files": len(entries),
            "fall_probability_files": 0,
            "participants": {key: len(value) for key, value in COHORTS.items()},
        },
        "participant_order": {
            modality: {
                condition: [
                    row["subject_id"] for row in tables[name] if row["condition"] == condition
                ]
                for condition in CONDITIONS
            }
            for modality, name in (
                ("energy", "split_energy.csv"),
                ("sla", "split_sla.csv"),
                ("temporal", "split_step_time_asym.csv"),
            )
        },
        "participant_mapping": {
            name: [
                {
                    "subject_id": row["subject_id"],
                    "condition": row["condition"],
                    "source_csv_row": row["source_row"],
                    "condition_row_1based": sum(
                        previous["condition"] == row["condition"] for previous in rows[: index + 1]
                    ),
                    "declared_samples": row["declared_samples"],
                    "missing_terminal_samples": row["missing_terminal_samples"],
                }
                for index, row in enumerate(rows)
            ]
            for name, rows in tables.items()
        },
        "missing_analyses": {
            "model_results": {
                "available": False,
                "note": "Supply a separate prepared publication cache to reproduce saved model analyses.",
            },
            "s3_rollouts": {
                "available_files": 0,
                "note": "No model rollouts are included in the experimental DATA folder.",
            },
        },
        "caveats": [
            "Input integrity does not imply complete manuscript reproduction: model fits, fall trials, supplementary model caches, Sanchez data and optimization solutions are absent.",
            "Public CSVs are six-decimal exports; reconstruction preserves CSV values and is not byte-exact recovery of full-precision author MAT files. Source rounding is at most 0.5e-6 in measured units for the published dataset.",
            "Per-subject processed MAT files, raw C3D recordings and participant metadata are not copied or required for this CSV import.",
            "Missing tied2 energy and SLA short_idx values are not synthesized; the manuscript analysis uses available tied/split measurements and fractional-stride windows.",
            "The public DSA CSV includes absent final samples in some records; their positions are retained and listed in participant_mapping, not filled or interpolated.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative, payload in prepared.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if destination.exists():
            raise FileExistsError(
                f"Prepared-data destination appeared during import: {destination}"
            )
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
