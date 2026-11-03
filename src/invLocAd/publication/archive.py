"""Validate and install publication input archives."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import unicodedata
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from .reporting import sha256, verify_data
from .zenodo import EXPERIMENTAL_PATHS, prepare_zenodo_data


def _verified_manifest(root: Path) -> dict:
    """Return the manifest of ``root`` after verifying every listed checksum."""
    integrity = verify_data(root)
    if integrity["errors"]:
        first = integrity["errors"][0]
        raise ValueError(
            f"Publication data failed verification: {first['path']}: {first['reason']}"
        )
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _fingerprints(root: Path, manifest: dict) -> dict[str, str]:
    """Map each listed path to its manifest SHA-256 and add the SHA-256 of ``manifest.json``."""
    return {
        **{item["path"]: item["sha256"] for item in manifest["files"]},
        "manifest.json": sha256(root / "manifest.json"),
    }


def _entries(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Return (entry, path below ``publication/``) pairs after validating every member.

    Paths must be relative, portable and under ``publication/``; links, special files,
    encrypted members and case- or Unicode-equivalent duplicates are rejected.
    """
    result, seen = [], {}
    for entry in archive.infolist():
        name = entry.orig_filename
        parts = (name[:-1] if entry.is_dir() else name).split("/")
        if (
            not name
            or name != entry.filename
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
            or any(part.rstrip(". ") != part for part in parts)
            or any(":" in part or PureWindowsPath(part).is_reserved() for part in parts)
            or parts[0] != "publication"
            or (len(parts) == 1 and not entry.is_dir())
        ):
            raise ValueError(f"Unsafe publication archive path: {name!r}")
        file_type = stat.S_IFMT(entry.external_attr >> 16)
        allowed = {0, stat.S_IFDIR} if entry.is_dir() else {0, stat.S_IFREG}
        if file_type not in allowed or entry.flag_bits & 1:
            raise ValueError(
                f"Links, special files and encrypted entries are not supported: {name!r}"
            )
        # macOS/Windows can alias paths differing only in case or Unicode form.
        key = unicodedata.normalize("NFC", "/".join(parts)).casefold()
        if key in seen:
            raise ValueError(f"Duplicate publication archive path: {name!r}")
        seen[key] = entry.is_dir()
        result.append((entry, Path(*parts[1:])))
    for key in seen:
        for parent in PurePosixPath(key).parents:
            if str(parent) in seen and not seen[str(parent)]:
                raise ValueError(f"Conflicting publication archive paths: {key!r}")
    return result


def _extract(archive_path: Path, destination: Path) -> None:
    """Extract a validated archive into the new folder ``destination``."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = _entries(archive)
            destination.mkdir()
            for entry, relative in entries:
                target = destination / relative
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"Archive path escapes its destination: {entry.filename!r}")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ValueError(f"Cannot read publication archive: {error}") from error


def install_publication_archive(archive: Path, destination: Path, source: Path) -> dict:
    """Install a ``publication/`` input ZIP after checking it against the public CSVs.

    An existing experimental-only dataset is upgraded, an identical dataset is kept and any
    other existing dataset is rejected.

    Args:
        archive: ZIP with all members under ``publication/``, including ``manifest.json``.
        destination: Prepared-data folder to create or upgrade.
        source: Public DATA folder whose measurement CSVs must agree with the archive.

    Returns:
        The manifest of the installed dataset.

    Raises:
        ValueError: If the archive fails verification or conflicts with ``destination``.
    """
    archive, destination, source = Path(archive), Path(destination), Path(source)
    if destination.is_symlink():
        raise ValueError("The publication destination must not be a symbolic link.")
    archive, destination, source = archive.resolve(), destination.resolve(), source.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Publication archive not found: {archive}")
    if archive.is_relative_to(destination):
        raise ValueError("Keep the publication archive outside the destination directory.")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("The experimental source and publication destination must not overlap.")
    previous = None
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("The publication destination is not a directory.")
        previous = _verified_manifest(destination)
        if previous.get("dataset_kind") == "zenodo_experimental":
            if {item["path"] for item in previous["files"]} != EXPERIMENTAL_PATHS:
                raise ValueError("Refusing to replace an unrecognized experimental-only dataset.")
            prepare_zenodo_data(source, destination)
        previous_fingerprints = _fingerprints(destination, previous)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="publication-install-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "publication"
        _extract(archive, staging)
        manifest = _verified_manifest(staging)
        prepare_zenodo_data(source, staging)
        if previous is not None:
            # Recheck the destination before committing the staged installation.
            current = _verified_manifest(destination)
            if _fingerprints(destination, current) != previous_fingerprints:
                raise ValueError(
                    "Publication destination changed during installation; nothing was replaced."
                )
            if previous_fingerprints == _fingerprints(staging, manifest):
                return previous
            if previous.get("dataset_kind") != "zenodo_experimental":
                raise ValueError("Refusing to replace a different existing publication cache.")
            backup = destination.with_name(f"{destination.name}.backup-{uuid.uuid4().hex}")
            destination.rename(backup)
            try:
                staging.rename(destination)
            except OSError as error:
                try:
                    backup.rename(destination)
                except OSError as rollback_error:
                    raise OSError(
                        f"Installation and rollback failed; the original dataset is preserved at {backup}"
                    ) from rollback_error
                raise OSError("Installation failed; the original dataset was restored.") from error
            shutil.rmtree(backup)
        else:
            if destination.exists():
                raise ValueError(
                    "Publication destination appeared during installation; nothing was replaced."
                )
            staging.rename(destination)
    return manifest
