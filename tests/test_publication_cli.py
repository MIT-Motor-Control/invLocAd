"""Publication command line: data verification, packaging, reproduction and preparation."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from invLocAd.publication.cli import main, package_data
from invLocAd.publication.reporting import sha256, verify_data


def prepared(tmp_path):
    """Return a dataset folder with one CSV and a matching manifest."""
    root = tmp_path / "input"
    root.mkdir()
    source = root / "value.csv"
    source.write_text("condition,value\n0.4,42\n")
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": [{"path": "value.csv", "sha256": sha256(source)}]}))
    return root


def test_manifest_rejects_changed_and_missing_data(tmp_path):
    root = prepared(tmp_path)
    assert verify_data(root)["errors"] == []
    (root / "value.csv").write_text("altered")
    assert verify_data(root)["errors"][0]["reason"] == "checksum mismatch"
    (root / "value.csv").unlink()
    assert verify_data(root)["errors"][0]["reason"] == "missing"
    assert main(["verify-data", "--data", str(root)]) == 1


def test_manifest_rejects_escape(tmp_path):
    root = prepared(tmp_path)
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": [{"path": "../outside", "sha256": "unused"}]}))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_data(root)


def test_manifest_rejects_untracked_scientific_input(tmp_path):
    root = prepared(tmp_path)
    (root / "stale.csv").write_text("stale,data\n1,2\n")
    assert verify_data(root)["errors"] == [{"path": "stale.csv", "reason": "not listed in manifest"}]


def test_release_archive_roundtrip(tmp_path):
    root = prepared(tmp_path)
    archive = tmp_path / "release.zip"
    package_data(root, archive)
    with ZipFile(archive) as zipped:
        zipped.extractall(tmp_path / "extracted")
    assert verify_data(tmp_path / "extracted" / "publication")["errors"] == []
    assert archive.with_suffix(".zip.sha256").read_text().startswith(sha256(archive))
    first_hash = sha256(archive)
    package_data(root, archive)
    assert sha256(archive) == first_hash


def test_package_rejects_nested_destination(tmp_path):
    root = prepared(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        package_data(root, root / "release.zip")


def test_reproduce_reports_partial_and_strict_exit(tmp_path, monkeypatch):
    from invLocAd.publication import main_figures

    root = prepared(tmp_path)
    monkeypatch.setattr(main_figures, "render_main", lambda *_: [{"id": "figure1", "status": "partial", "caveats": ["Missing comparison data"], "outputs": []}])
    output = tmp_path / "output"
    command = ["reproduce", "--data", str(root), "--output", str(output), "--figures", "figure1"]
    assert main(command) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["results"][0]["status"] == "partial"
    assert main(command + ["--overwrite", "--strict"]) == 2


def test_failed_figure_has_log_and_nonzero_exit(tmp_path, monkeypatch):
    from invLocAd.publication import main_figures

    root = prepared(tmp_path)

    def fail(*_):
        """Raise an input error from the renderer."""
        raise ValueError("test input invalid")

    monkeypatch.setattr(main_figures, "render_main", fail)
    output = tmp_path / "output"
    assert main(["reproduce", "--data", str(root), "--output", str(output), "--figures", "figure1"]) == 1
    assert "test input invalid" in (output / "logs/figure1.txt").read_text()
    assert json.loads((output / "report.json").read_text())["results"][0]["status"] == "failed"


def test_reproduce_protects_unmanaged_output(tmp_path):
    root = prepared(tmp_path)
    output = tmp_path / "unmanaged"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    assert main(["reproduce", "--data", str(root), "--output", str(output), "--overwrite"]) == 1
    assert marker.read_text() == "keep"


def test_missing_model_inputs_are_blocked_not_execution_failures(tmp_path):
    root = prepared(tmp_path)
    output = tmp_path / "output"
    figures = ["figure2", "s1", "figure4", "s3", "s4"]
    command = ["reproduce", "--data", str(root), "--output", str(output), "--figures", *figures]
    assert main(command) == 0
    report = json.loads((output / "report.json").read_text())
    assert {item["id"]: item["status"] for item in report["results"]} == dict.fromkeys(
        figures, "blocked"
    )
    assert not (output / "logs").exists()
    assert main(command + ["--overwrite", "--strict"]) == 2


def test_fresh_clone_data_error_has_setup_instructions(tmp_path, capsys):
    assert main(["verify-data", "--data", str(tmp_path / "missing")]) == 1
    message = capsys.readouterr().err
    assert "invLocAd-publication prepare" in message
    assert "invLocAd-DATA" in message


@pytest.mark.parametrize("source_option", [None, "download/DATA", "download"])
def test_prepare_discovers_embedded_archive_from_relative_default_and_wrapper_sources(
    tmp_path, monkeypatch, source_option
):
    from invLocAd.publication import archive as installer

    monkeypatch.chdir(tmp_path)
    dataset = Path("data/invLocAd-DATA" if source_option is None else "download/DATA")
    (dataset / "processed_csv").mkdir(parents=True)
    companion = dataset / "publication-data.zip"
    companion.write_bytes(b"embedded archive")
    external = Path("data/publication-data.zip")
    external.parent.mkdir(exist_ok=True)
    external.write_bytes(b"external archive in data/")
    calls = []

    def install(archive, destination, source):
        """Record the installation arguments and return a one-file manifest."""
        calls.append((archive.resolve(), destination, source))
        return {"files": [{"path": "model input"}]}

    monkeypatch.setattr(installer, "install_publication_archive", install)
    command = ["prepare"] + (["--source", source_option] if source_option else [])
    assert main(command) == 0
    assert calls == [(companion.resolve(), Path("data/publication"), Path(source_option) if source_option else dataset)]


@pytest.mark.parametrize("mode", ["wrapper", "data_folder", "explicit"])
def test_prepare_preserves_external_archive_choices(tmp_path, monkeypatch, mode):
    from invLocAd.publication import archive as installer

    monkeypatch.chdir(tmp_path)
    (Path("download/DATA") / "processed_csv").mkdir(parents=True)
    expected = {"wrapper": "download/publication-data.zip", "data_folder": "data/publication-data.zip", "explicit": "chosen.zip"}[mode]
    selected = Path(expected)
    selected.parent.mkdir(exist_ok=True)
    selected.write_bytes(b"selected archive")
    if mode == "explicit":
        Path("download/DATA/publication-data.zip").write_bytes(b"unused automatic choice")
    calls = []

    def install(archive, *_):
        """Record the selected archive and return an empty manifest."""
        calls.append(archive.resolve())
        return {"files": []}

    monkeypatch.setattr(installer, "install_publication_archive", install)
    command = ["prepare", "--source", "download"]
    if mode == "explicit":
        command += ["--model-inputs", str(selected)]
    assert main(command) == 0
    assert calls == [selected.resolve()]


def test_prepare_without_companion_keeps_experimental_import(tmp_path, monkeypatch):
    from invLocAd.publication import zenodo

    monkeypatch.chdir(tmp_path)
    source = Path("data/invLocAd-DATA")
    (source / "processed_csv").mkdir(parents=True)
    calls = []

    def prepare(actual_source, destination):
        """Record the import arguments and return an experimental-only manifest."""
        calls.append((actual_source, destination))
        return {"files": [], "dataset_kind": "zenodo_experimental"}

    monkeypatch.setattr(zenodo, "prepare_zenodo_data", prepare)
    assert main(["prepare"]) == 0
    assert calls == [(source, Path("data/publication"))]
