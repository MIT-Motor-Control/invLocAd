"""ZIP installation, idempotency and dataset preservation."""

import hashlib
import json
import stat
import warnings
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from invLocAd.publication import archive as installer


def write_dataset(root, files, kind=None):
    """Write ``files`` and their manifest under ``root``; return the manifest."""
    root.mkdir(parents=True)
    manifest = {"schema_version": 1, "files": []}
    if kind:
        manifest["dataset_kind"] = kind
    for relative, payload in sorted(files.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest["files"].append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest()})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_archive(path, files=None, extra=()):
    """Write a publication ZIP with its manifest and extra members; return the manifest."""
    files = files or {"original/model.dat": b"model result"}
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(files.items())
        ],
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with ZipFile(path, "w") as archive:
            archive.writestr("publication/manifest.json", json.dumps(manifest))
            for name, payload in files.items():
                archive.writestr(f"publication/{name}", payload)
            for name, payload in extra:
                archive.writestr(name, payload)
    return manifest


@pytest.fixture
def public_source(tmp_path, monkeypatch):
    """Return an experimental source folder and stub out experimental-data validation."""
    source = tmp_path / "experimental"
    source.mkdir()

    def validate(actual_source, destination):
        """Check the source folder and return the verified destination manifest."""
        assert actual_source == source
        return installer._verified_manifest(destination)

    monkeypatch.setattr(installer, "prepare_zenodo_data", validate)
    return source


def test_install_and_repeat_leave_matching_cache_unchanged(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    expected = write_archive(archive)
    assert installer.install_publication_archive(archive, destination, public_source) == expected
    assert (destination / "original/model.dat").read_bytes() == b"model result"
    before = {p.relative_to(destination): p.stat().st_mtime_ns for p in destination.rglob("*")}
    assert installer.install_publication_archive(archive, destination, public_source) == expected
    assert before == {
        p.relative_to(destination): p.stat().st_mtime_ns for p in destination.rglob("*")
    }
    assert not list(tmp_path.glob("publication-install-*"))
    assert not list(tmp_path.glob("publication.backup-*"))


def test_only_recognized_experimental_dataset_can_be_upgraded(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    old_files = {name: b"experimental" for name in installer.EXPERIMENTAL_PATHS}
    write_dataset(destination, old_files, "zenodo_experimental")
    new_files = dict(old_files, **{"original/model.dat": b"model result"})
    expected = write_archive(archive, new_files)
    assert installer.install_publication_archive(archive, destination, public_source) == expected
    assert (destination / "original/model.dat").is_file()
    assert installer._verified_manifest(destination) == expected
    assert not list(tmp_path.glob("publication.backup-*"))


def test_refuse_replacing_different_full_cache(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    old = write_dataset(destination, {"original/model.dat": b"different model result"})
    with pytest.raises(ValueError, match="different existing publication cache"):
        installer.install_publication_archive(archive, destination, public_source)
    assert installer._verified_manifest(destination) == old
    assert (destination / "original/model.dat").read_bytes() == b"different model result"


def test_experimental_marker_does_not_authorize_deleting_extra_files(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    files = {name: b"experimental" for name in installer.EXPERIMENTAL_PATHS}
    files["my_analysis.csv"] = b"user analysis"
    old = write_dataset(destination, files, "zenodo_experimental")
    with pytest.raises(ValueError, match="unrecognized experimental-only"):
        installer.install_publication_archive(archive, destination, public_source)
    assert installer._verified_manifest(destination) == old
    assert (destination / "my_analysis.csv").read_bytes() == b"user analysis"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/publication/escape.txt",
        "different/escape.txt",
        "publication/../escape.txt",
        "publication/./escape.txt",
        "publication//escape.txt",
        "publication\\escape.txt",
        "publication/.hidden",
        "publication/.hidden/value.csv",
        "publication/C:/escape.txt",
        "publication/CON",
        "publication/trailing. ",
        "publication/manifest.json",
        "publication/MANIFEST.JSON",
    ],
)
def test_reject_unsafe_or_duplicate_members_before_install(tmp_path, public_source, name):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive, extra=[(name, b"untrusted")])
    with pytest.raises(ValueError, match="Unsafe|Duplicate"):
        installer.install_publication_archive(archive, destination, public_source)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not list(tmp_path.glob("publication-install-*"))


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_reject_link_and_special_file_metadata(tmp_path, public_source, file_type):
    entry = ZipInfo("publication/linked")
    entry.create_system = 3
    entry.external_attr = (file_type | 0o777) << 16
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive, extra=[(entry, b"../../elsewhere")])
    with pytest.raises(ValueError, match="Links, special files"):
        installer.install_publication_archive(archive, destination, public_source)
    assert not destination.exists()


def test_reject_file_used_as_directory(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive, extra=[("publication/original", b"conflicting file")])
    with pytest.raises(ValueError, match="Conflicting publication archive paths"):
        installer.install_publication_archive(archive, destination, public_source)
    assert not destination.exists()


def test_reject_unlisted_archive_file(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive, extra=[("publication/extra.csv", b"not in manifest")])
    with pytest.raises(ValueError, match="not listed in manifest"):
        installer.install_publication_archive(archive, destination, public_source)
    assert not destination.exists()


def test_reject_corrupt_member_without_changing_existing_data(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    archive.write_bytes(archive.read_bytes().replace(b"model result", b"MODEL result"))
    old = write_dataset(
        destination, {name: b"old" for name in installer.EXPERIMENTAL_PATHS}, "zenodo_experimental"
    )
    with pytest.raises(ValueError, match="Cannot read publication archive"):
        installer.install_publication_archive(archive, destination, public_source)
    assert installer._verified_manifest(destination) == old


def test_cross_dataset_validation_failure_leaves_no_install(tmp_path, public_source, monkeypatch):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)

    def mismatch(*_):
        """Raise the error of a failed experimental-data comparison."""
        raise ValueError("Public experimental CSVs differ")

    monkeypatch.setattr(installer, "prepare_zenodo_data", mismatch)
    with pytest.raises(ValueError, match="experimental CSVs differ"):
        installer.install_publication_archive(archive, destination, public_source)
    assert not destination.exists()


def test_destination_symlink_is_never_replaced(tmp_path, public_source):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    real = tmp_path / "elsewhere"
    real.mkdir()
    destination.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        installer.install_publication_archive(archive, destination, public_source)
    assert destination.is_symlink()
    assert list(real.iterdir()) == []


def test_failed_upgrade_restores_original_dataset(tmp_path, public_source, monkeypatch):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    old = write_dataset(
        destination, {name: b"old" for name in installer.EXPERIMENTAL_PATHS}, "zenodo_experimental"
    )
    rename = Path.rename

    def fail_install(path, target):
        """Fail the rename of the staged installation; rename other paths."""
        if path.parent.name.startswith("publication-install-"):
            raise OSError("Simulated install failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(OSError, match="original dataset was restored"):
        installer.install_publication_archive(archive, destination, public_source)
    assert installer._verified_manifest(destination) == old
    assert not list(tmp_path.glob("publication.backup-*"))


def test_failed_rollback_keeps_original_backup_outside_temporary_cleanup(
    tmp_path, public_source, monkeypatch
):
    archive, destination = tmp_path / "cache.zip", tmp_path / "publication"
    write_archive(archive)
    old = write_dataset(
        destination, {name: b"old" for name in installer.EXPERIMENTAL_PATHS}, "zenodo_experimental"
    )
    rename = Path.rename

    def fail_install_and_restore(path, target):
        """Fail the renames of the staged installation and of the backup."""
        if path.parent.name.startswith("publication-install-") or path.name.startswith(
            "publication.backup-"
        ):
            raise OSError("Simulated rename failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_and_restore)
    with pytest.raises(OSError, match="original dataset is preserved at"):
        installer.install_publication_archive(archive, destination, public_source)
    backups = list(tmp_path.glob("publication.backup-*"))
    assert len(backups) == 1
    assert installer._verified_manifest(backups[0]) == old
    assert not destination.exists()
