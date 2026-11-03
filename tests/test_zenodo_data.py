"""Import of the public experimental CSVs into a publication dataset."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from invLocAd.publication.main_figures import figure1_tables
from invLocAd.publication.reporting import verify_data
from invLocAd.publication.zenodo import (
    COHORTS,
    CONDITION_IDS,
    CONDITIONS,
    CSV_NAMES,
    ENERGY_PATH,
    EXPERIMENTAL_PATHS,
    SLA_PATH,
    prepare_zenodo_data,
)


@pytest.fixture
def public_data(tmp_path):
    """Return a synthetic DATA folder with every public CSV for the published cohorts."""
    source = tmp_path / "wrapper/DATA"
    folder = source / "processed_csv"
    folder.mkdir(parents=True)
    for name in CSV_NAMES:
        modality = "energy" if "energy" in name else "sla" if "sla" in name else "temporal"
        fixed = ["subject_id", "condition", "speed_diff_ms"]
        prefix = "t" if modality == "energy" else "stride" if modality == "sla" else "step"
        length = (
            (2700 if name.startswith("split") else 360)
            if modality == "energy"
            else (900 if name.startswith("split") else 120)
        )
        if modality != "energy":
            fixed.append("num_strides" if modality == "sla" else "num_steps")
        with (folder / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fixed + [f"{prefix}{i}" for i in range(length)])
            for subject in sorted(COHORTS[modality], reverse=True):
                condition = next(key for key, values in CONDITION_IDS.items() if subject in values)
                count = length if modality == "energy" else length - subject % 3
                row = [f"AB{subject:02d}", condition, CONDITIONS[condition]]
                if modality != "energy":
                    row += [count]
                values = [str(2 + subject / 100 + index / 10000) for index in range(count)]
                writer.writerow(row + values + [""] * (length - count))
    return source


def change_csv(path, change):
    """Apply ``change`` to the rows of a CSV file in place."""
    with path.open() as stream:
        rows = list(csv.reader(stream))
    change(rows)
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_public_import_is_deterministic_and_idempotent(public_data, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    manifest = prepare_zenodo_data(public_data, first)
    prepared = prepare_zenodo_data(public_data.parent, second)
    assert manifest["dataset_kind"] == "zenodo_experimental"
    assert manifest["counts"]["participants"] == {"energy": 26, "sla": 23, "temporal": 20}
    assert {entry["path"] for entry in manifest["files"]} == EXPERIMENTAL_PATHS
    assert manifest["files"] == prepared["files"]
    assert verify_data(first)["errors"] == []
    before = (first / "manifest.json").read_bytes()
    assert prepare_zenodo_data(public_data, first) == manifest
    assert (first / "manifest.json").read_bytes() == before
    energy = loadmat(first / ENERGY_PATH, simplify_cells=True)
    sla = loadmat(first / SLA_PATH, simplify_cells=True)
    assert "tied2_energy" not in energy
    assert not any(key.startswith("short_idx") for key in sla)
    assert energy["split_energy"]["condition_1"][0, 0] == 2.03
    assert manifest["participant_order"]["energy"]["condition_1"][0] == "AB03"
    assert str(tmp_path) not in json.dumps(manifest)
    assert "fall_probability/" in manifest["missing_inputs"]


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("split_energy.csv", lambda rows: rows[1].__setitem__(0, "AB99"), "participant"),
        (
            "split_energy.csv",
            lambda rows: rows[1].__setitem__(1, "condition_1"),
            "condition mismatch",
        ),
        ("split_energy.csv", lambda rows: rows.pop(), "incomplete participant"),
        ("split_energy.csv", lambda rows: rows[1].pop(), "truncated"),
        ("split_energy.csv", lambda rows: rows[1].__setitem__(3, "nan"), "nonfinite"),
        ("split_energy.csv", lambda rows: rows[0].__setitem__(4, "t0"), "columns"),
        ("split_sla.csv", lambda rows: rows[1].__setitem__(3, "2"), "padding"),
    ],
)
def test_rejects_invalid_public_data_before_creating_destination(
    public_data, tmp_path, name, change, message
):
    change_csv(public_data / "processed_csv" / name, change)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match=message):
        prepare_zenodo_data(public_data, destination)
    assert not destination.exists()


def test_rejects_overlap_and_escaping_sources(public_data, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        prepare_zenodo_data(public_data, public_data / "prepared")
    with pytest.raises(ValueError, match="overlap"):
        prepare_zenodo_data(public_data, tmp_path)
    path = public_data / "processed_csv/split_energy.csv"
    outside = tmp_path / "outside.csv"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        prepare_zenodo_data(public_data, tmp_path / "prepared")


def test_records_missing_terminal_dsa_without_filling_it(public_data, tmp_path):
    path = public_data / "processed_csv/split_ds_asym.csv"
    change_csv(path, lambda rows: rows[1].__setitem__(3 + int(rows[1][3]), ""))
    before = path.read_bytes()
    destination = tmp_path / "prepared"
    manifest = prepare_zenodo_data(public_data, destination)
    assert (destination / "extensions/experimental_csv/split_ds_asym.csv").read_bytes() == before
    assert (
        sum(
            row["missing_terminal_samples"]
            for row in manifest["participant_mapping"]["split_ds_asym.csv"]
        )
        == 1
    )


def test_preserves_verified_full_dataset_and_checks_numeric_equivalence(public_data, tmp_path):
    destination = tmp_path / "prepared"
    manifest = prepare_zenodo_data(public_data, destination)
    manifest.pop("dataset_kind")
    model = destination / "model-cache.bin"
    model.write_bytes(b"existing model scientific data")
    manifest["files"].append(
        {"path": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    )
    path = destination / ENERGY_PATH
    fields = {key: value for key, value in loadmat(path).items() if not key.startswith("_")}
    fields["split_energy"]["condition_1"][0, 0][0, 0] += 2e-7
    savemat(path, fields)
    entry = next(item for item in manifest["files"] if item["path"] == ENERGY_PATH)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["size"] = path.stat().st_size
    (destination / "manifest.json").write_text(json.dumps(manifest))
    before = {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    }
    assert prepare_zenodo_data(public_data, destination) == manifest
    assert all((destination / path).read_bytes() == content for path, content in before.items())
    fields["split_energy"]["condition_1"][0, 0][0, 0] += 1e-3
    savemat(path, fields)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="arrays conflict"):
        prepare_zenodo_data(public_data, destination)
    assert model.read_bytes() == b"existing model scientific data"


def test_existing_destination_conflicts_are_not_replaced(public_data, tmp_path):
    destination = tmp_path / "prepared"
    prepare_zenodo_data(public_data, destination)
    before = (destination / "manifest.json").read_bytes()
    change_csv(
        public_data / "processed_csv/split_energy.csv", lambda rows: rows[1].__setitem__(3, "12.0")
    )
    with pytest.raises(ValueError, match="conflicts with public CSV"):
        prepare_zenodo_data(public_data, destination)
    assert (destination / "manifest.json").read_bytes() == before
    (destination / "unlisted.bin").write_bytes(b"must not overwrite")
    with pytest.raises(ValueError, match="integrity"):
        prepare_zenodo_data(public_data, destination)


@pytest.mark.integration
def test_audited_public_csv_reproduces_experimental_values_with_rounding_tolerance(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/invLocAd-DATA"
    if not (source / "processed_csv").is_dir():
        pytest.skip("The public experimental DATA folder is unavailable")
    destination = tmp_path / "prepared"
    prepare_zenodo_data(source, destination)
    tables = figure1_tables(destination)
    expected = {
        "energy_participants": (
            "energy_change_pct",
            [-8.571140401974557, -0.5279256055629971, 20.61772447061896],
        ),
        "sla_participants": (
            "sla_change_percentage_points",
            [-1.9695521054905414, 2.193997111107009, 0.7645353372066099],
        ),
    }
    for name, (column, values) in expected.items():
        actual = [
            np.mean(
                [
                    row[column]
                    for row in tables[name]
                    if row["condition"] == condition and row["exposure"].startswith("long")
                ]
            )
            for condition in (1, 2, 3)
        ]
        np.testing.assert_allclose(actual, values, rtol=0, atol=1e-4)
    assert [
        sum(
            row["significant_p_lt_005"]
            for row in tables["paired_tests"]
            if row["condition"] == condition
        )
        for condition in (1, 2, 3)
    ] == [1, 2, 3]
