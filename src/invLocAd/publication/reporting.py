"""Verify prepared inputs against their manifest and write reproduction reports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment() -> dict:
    """Return the Python version, platform and installed versions of key packages."""
    versions = {}
    for name in ("invLocAd", "numpy", "scipy", "matplotlib", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def verify_data(data_root: Path) -> dict:
    """Check every file in a prepared dataset against ``manifest.json``.

    Args:
        data_root: Prepared dataset containing ``manifest.json``.

    Returns:
        Summary with ``checked_files``, ``manifest_sha256``, ``dataset_kind``, ``purpose``,
        ``caveats`` and ``errors`` (missing, mismatched or unlisted files).

    Raises:
        ValueError: If the manifest is unsupported or lists an unsafe or duplicate path.
    """
    data_root = data_root.resolve()
    manifest_path = data_root / "manifest.json"
    if not manifest_path.resolve().is_relative_to(data_root):
        raise ValueError("The input manifest must be inside the prepared dataset.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported data manifest; prepare the inputs again.")
    errors, seen = [], set()
    for record in manifest["files"]:
        relative = record["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or relative in seen:
            raise ValueError(f"Unsafe or duplicate manifest path: {relative}")
        seen.add(relative)
        if not path.is_file():
            errors.append({"path": relative, "reason": "missing"})
        elif sha256(path) != record["sha256"]:
            errors.append({"path": relative, "reason": "checksum mismatch"})
    for path in data_root.rglob("*"):
        if path.is_file() and path.name not in {".DS_Store", "Thumbs.db"}:
            relative = path.relative_to(data_root).as_posix()
            if relative not in seen and relative != "manifest.json":
                errors.append({"path": relative, "reason": "not listed in manifest"})
    return {
        "checked_files": len(seen),
        "manifest_sha256": sha256(manifest_path),
        "dataset_kind": manifest.get("dataset_kind", "publication_cache"),
        "purpose": manifest.get("purpose", ""),
        "caveats": manifest.get("caveats", []),
        "errors": errors,
    }


def source_hashes() -> dict[str, str]:
    """Return the SHA-256 of every Python and MATLAB source file in the package."""
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.suffix in {".py", ".m"}
    }


def write_report(output_root: Path, results: list[dict], integrity: dict) -> dict:
    """Write ``report.json`` and ``report.md`` to ``output_root``.

    Args:
        output_root: Folder holding the figure and table outputs.
        results: One record per analysis with ``id``, ``status``, ``outputs`` and ``caveats``.
        integrity: Summary returned by ``verify_data``.

    Returns:
        The report saved as ``report.json``, with output paths relative to ``output_root``.
    """

    def relative(value):
        """Return ``value`` with paths under ``output_root`` made relative."""
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and value.startswith(str(output_root) + "/"):
            return str(Path(value).relative_to(output_root))
        if isinstance(value, list):
            return [relative(item) for item in value]
        if isinstance(value, dict):
            return {key: relative(item) for key, item in value.items()}
        return value

    results = relative(results)
    artifacts = [
        {"path": path.relative_to(output_root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name not in {"report.json", "report.md"}
    ]
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "environment": environment(),
        "inputs": integrity,
        "source_sha256": source_hashes(),
        "results": results,
        "artifacts": artifacts,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Reproduction report", "",
        f"Generated {report['created_at']}. Input checksums verified for {integrity['checked_files']} files.", "",
        "These analyses recompute results from saved measurements and simulations; they do not rerun the simulation sweeps.", "",
    ]
    if integrity.get("purpose"):
        lines += [f"Input dataset: {integrity['purpose']}.", ""]
    if integrity.get("dataset_kind") == "zenodo_experimental":
        lines += [f"- {note}" for note in integrity.get("caveats", [])] + [""]
    lines += ["| Analysis | Status | Output files |", "|---|---|---:|"]
    for result in results:
        lines.append(f"| {result['id']} | {result['status']} | {len(result.get('outputs', []))} |")
    lines += ["", "## Outputs", "", "Numerical tables are in `tables/`; figures are in `figures/`. `report.json` records input, code and output checksums, dependencies, and per-analysis details.", ""]
    for result in results:
        caveats = result.get("caveats", [])
        if caveats:
            lines += [f"### {result['id']}", ""]
            lines += [f"- {note}" for note in caveats]
            lines += [""]
        if result.get("error"):
            lines += [f"**{result['id']}:** {result['error']}", ""]
    (output_root / "report.md").write_text("\n".join(lines))
    return report
