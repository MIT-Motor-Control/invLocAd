"""Integration of supplementary source data into an existing publication dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from invLocAd.publication.data import integrate_addendum
from invLocAd.publication.reporting import sha256, verify_data

ENERGY = "original/data/cosmed/processed/processedEnergyData.mat"


def save(path, values):
    """Write a compressed MAT file, creating its parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, values, do_compression=True)


@pytest.fixture
def dataset(tmp_path):
    """Return (source, target): a supplementary tree with 150 Figure 2C rollouts and a dataset."""
    target, source = tmp_path / "publication", tmp_path / "supplementary"
    params = np.empty((3, 1), dtype=[("learning_rate", object), ("sym_cost", object)])
    for c in range(3):
        params[c, 0] = (0.01, 0.0)
    fit = {
        "static_params": params,
        "exp_subj_num": np.array([1, 1, 1]),
        "learning_pool": np.array([0.01, 0.02]),
        "sym_cost_pool": np.array([0.0, 1.0]),
    }
    save(target / "original/2d_sweep_trial50_param_result.mat", fit)
    energy = {
        field: {f"condition_{c}": np.arange(6).reshape(2, 3) + c for c in (1, 2, 3)}
        for field in ("split_energy", "tied1_energy", "tied2_energy")
    }
    save(target / ENERGY, energy)
    files = [
        {
            "path": path.relative_to(target).as_posix(),
            "sha256": sha256(path),
            "source": "initial:/" + path.name,
            "size": path.stat().st_size,
        }
        for path in sorted(target.rglob("*.mat"))
    ]
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": files,
                "missing_inputs": ["original/data/sanchez data/sanchez_energy.mat"],
                "counts": {"files": len(files)},
            }
        )
    )
    incoming = source / "code/original_data/data/cosmed/processed/processedEnergyData.mat"
    save(incoming, energy)
    incoming.write_bytes(b"alternate MAT header".ljust(116) + incoming.read_bytes()[116:])
    save(source / "code/original_data/data/sanchez data/sanchez_energy.mat", {"data": np.arange(8)})
    save(source / "code/verify/2d_sweep_trial50_param_result.mat", fit)
    save(source / "code/verify/2d_sweep_trial50_param_result_trialavg.mat", fit)
    folder = source / "code/verify/data/2d sweep fine"
    for slow, fast in ((0.8, 1.2), (0.65, 1.35), (0.5, 1.5)):
        for trial in range(1, 51):
            name = f"model_split45_trans20_slow{slow:g}_fast{fast:g}_noise0.01_learning0.01_energysym0_swingcost1_trial{trial}.mat"
            save(
                folder / name,
                {"energy": np.linspace(1, 2, 200), "fall_flag": [int(trial == 1), 100]},
            )
    return source, target


def test_import_preserves_original_bytes_and_is_idempotent(dataset):
    source, target = dataset
    energy_before = (target / ENERGY).read_bytes()
    before = sha256(source / "code/original_data/data/cosmed/processed/processedEnergyData.mat")
    manifest = integrate_addendum(source, target)
    assert verify_data(target)["errors"] == []
    assert (target / ENERGY).read_bytes() == energy_before
    assert (
        sha256(source / "code/original_data/data/cosmed/processed/processedEnergyData.mat")
        == before
    )
    assert manifest["missing_inputs"] == []
    info = manifest["missing_analyses"]["s3_rollouts"]
    assert info["available_files"] == 150
    assert info["required_unique_cells"] == 3
    assert info["verified_cell_trial_identities"]
    assert info["falling_trials"] == 3
    energy_record = next(item for item in manifest["files"] if item["path"] == ENERGY)
    assert "existing MAT bytes preserved" in energy_record["corroborating_files"][0]["comparison"]
    preserved = (target / "manifest.json").read_bytes()
    assert integrate_addendum(source, target) == manifest
    assert (target / "manifest.json").read_bytes() == preserved
    assert not list(target.parent.glob(".publication-addendum-*"))
    assert not target.with_name("publication.before-addendum").exists()


@pytest.mark.parametrize(
    "problem", ["missing_trial", "conflicting_energy", "invalid_grid", "nonfinite"]
)
def test_bad_import_never_changes_existing_dataset(dataset, problem):
    source, target = dataset
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    rollout = next((source / "code/verify/data/2d sweep fine").glob("*.mat"))
    if problem == "missing_trial":
        rollout.unlink()
    elif problem == "invalid_grid":
        rollout.rename(rollout.with_name(rollout.name.replace("learning0.01", "learning0.011")))
    elif problem == "nonfinite":
        save(rollout, {"energy": [1.0, np.nan], "fall_flag": [1, 2]})
    else:
        path = source / "code/original_data/data/cosmed/processed/processedEnergyData.mat"
        values = {k: v for k, v in loadmat(path).items() if not k.startswith("_")}
        values["split_energy"]["condition_1"][0, 0][0, 0] += 1
        save(path, values)
    with pytest.raises(ValueError):
        integrate_addendum(source, target)
    assert all((target / path).read_bytes() == content for path, content in before.items())
    assert len([p for p in target.rglob("*") if p.is_file()]) == len(before)


def test_import_rejects_symlink_and_overlapping_sources(dataset, tmp_path):
    source, target = dataset
    outside = tmp_path / "outside.mat"
    save(outside, {"data": [1, 2]})
    (source / "code/original_data/escape.mat").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        integrate_addendum(source, target)
    with pytest.raises(ValueError, match="overlap"):
        integrate_addendum(tmp_path, target)


def test_previous_bundle_fills_missing_sources_and_records_hashes(dataset, tmp_path):
    source, target = dataset
    old = tmp_path / "previous"
    path = old / "code/original_data/data/trajectory optimization/model.osim"
    path.parent.mkdir(parents=True)
    path.write_text("<OpenSimDocument/>")
    manifest = integrate_addendum(source, target, previous_bundle=old)
    entry = next(item for item in manifest["files"] if item["path"].endswith("model.osim"))
    assert (
        entry["source"]
        == "previous_bundle:/code/original_data/data/trajectory optimization/model.osim"
    )
    assert entry["sha256"] == sha256(path)


def test_generated_grid_tables_require_and_retain_code_provenance(dataset, tmp_path):
    source, target = dataset
    generated = tmp_path / "grid"
    generated.mkdir()
    for name in ("summary", "persubject"):
        (generated / f"comment1_grid_sensitivity_{name}.csv").write_text("grid,value\navg,1.2\n")
    script = tmp_path / "grid_sensitivity.m"
    script.write_text("rng(0);\n")
    with pytest.raises(ValueError, match="both"):
        integrate_addendum(source, target, generated_s1=generated)
    manifest = integrate_addendum(
        source, target, generated_s1=generated, generated_s1_script=script
    )
    entries = [item for item in manifest["files"] if "grid_sensitivity" in item["path"]]
    assert len(entries) == 2
    assert all(item["generation"]["script_sha256"] == sha256(script) for item in entries)
    assert verify_data(target)["errors"] == []


def test_failed_install_restores_existing_dataset(dataset, monkeypatch):
    source, target = dataset
    before = (target / "manifest.json").read_bytes()
    rename = Path.rename

    def fail_staging(path, destination):
        """Fail the rename of the staged dataset; rename other paths."""
        if path.parent.name.startswith(".publication-addendum-"):
            raise OSError("Simulated install failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_staging)
    with pytest.raises(OSError, match="Simulated"):
        integrate_addendum(source, target)
    assert (target / "manifest.json").read_bytes() == before
    assert verify_data(target)["errors"] == []
