"""Download, verify and install the DATA release using Python's standard library.

Run from the repository root: python scripts/download_data.py
Downloads the complete, checksum-pinned ZIP and resumes interrupted transfers.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/23904047/files/"
DATA_ROOT = "invLocAd-DATA"
ARCHIVE_NAME = f"{DATA_ROOT}.zip"
ARCHIVE_SHA256 = "16ce92fd0fac4fc5a3686e4bcd0180f80f5bd84a0ec928e95e8de4826cdb7ffd"
ARCHIVE_BYTES = 13_269_858_121
CHUNK_SIZE = 4 * 1024 * 1024


def regular(path: Path) -> bool:
    """Return whether ``path`` exists as a regular file.

    Args:
        path: File to check; it may be missing.

    Returns:
        True for an existing regular file, False if ``path`` does not exist.

    Raises:
        ValueError: If ``path`` is a symlink (including a dangling one) or a non-file.
    """
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Expected a regular file, not a link or directory: {path}")
    return path.exists()


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, item: dict) -> bool:
    """Return whether ``path`` is a regular file with the size and SHA-256 in ``item``.

    Args:
        path: File to check; a missing file is not verified.
        item: Mapping with the expected ``bytes`` and ``sha256``.

    Raises:
        ValueError: If ``path`` is a symlink or a non-file.
    """
    return regular(path) and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]


def request(url: str, *, headers: dict | None = None):
    """Open ``url`` with the package User-Agent, extra ``headers`` and a 60 s timeout.

    Args:
        url: URL to open.
        headers: Additional request headers, such as ``Range``.

    Returns:
        The open HTTP response.
    """
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "invLocAd/0.4.0", **(headers or {})}
        ),
        timeout=60,
    )


def free_space(directory: Path, needed: int):
    """Raise OSError if ``directory`` has fewer than ``needed`` free bytes.

    Args:
        directory: Existing directory on the target file system.
        needed: Required free space in bytes.
    """
    if shutil.disk_usage(directory).free < needed:
        raise OSError(f"Need at least {needed / 1e9:.2f} GB free for the next step in {directory}")


def download_file(
    item: dict, cache: Path, *, resume: bool = False
) -> Path:
    """Download one release file into ``cache`` and verify its size and SHA-256.

    A verified cached copy is reused. Network and file errors other than HTTP 403 and 404
    are retried, for at most three attempts.

    Args:
        item: Mapping with the file ``name``, ``bytes`` and ``sha256``.
        cache: Download directory.
        resume: Continue a ``.partial`` file with an HTTP Range request; otherwise restart.

    Returns:
        Path of the verified file.

    Raises:
        FileNotFoundError: If Zenodo responds 403 or 404.
        ValueError: If the HTTP status, range, encoding, size or checksum is unexpected.
    """
    target = cache / item["name"]
    if verified(target, item):
        print(f"Verified cached {item['name']}", flush=True)
        return target
    partial = target.with_name(target.name + ".partial")
    regular(partial)
    url = RECORD_URL + urllib.parse.quote(item["name"])
    for attempt in range(1, 4):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError(
                f"Cached partial exceeds expected size; remove {partial} before retrying"
            )
        if offset == item["bytes"]:
            if verified(partial, item):
                os.replace(partial, target)
                return target
            partial.unlink()
            raise ValueError(f"Cached partial failed its checksum and was removed: {item['name']}")
        free_space(cache, item["bytes"] - offset)
        action = f"Resuming at {offset / 1e6:.1f} MB:" if offset else "Downloading"
        print(f"{action} {item['name']} ({item['bytes'] / 1e6:.1f} MB)", flush=True)
        digest, count = hashlib.sha256(), 0
        try:
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            response = request(url, headers=headers) if resume else request(url)
            with response:
                status = getattr(response, "status", 200)
                if status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if not match or tuple(map(int, match.groups())) != (
                        offset,
                        item["bytes"] - 1,
                        item["bytes"],
                    ):
                        raise ValueError(f"Unexpected partial-download range for {item['name']}")
                elif status == 200:
                    # A server may ignore Range: restart rather than append the full response.
                    offset = 0
                else:
                    raise ValueError(f"Unexpected HTTP status {status} for {item['name']}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError(f"Unexpected content encoding for {item['name']}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != item["bytes"] - offset:
                    raise ValueError(f"Unexpected download size for {item['name']}")
                if offset:
                    with partial.open("rb") as existing:
                        for block in iter(lambda: existing.read(CHUNK_SIZE), b""):
                            digest.update(block)
                    count = offset
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(min(CHUNK_SIZE, item["bytes"] - count + 1)):
                        count += len(block)
                        if count > item["bytes"]:
                            raise ValueError(f"Download exceeds expected size: {item['name']}")
                        digest.update(block)
                        output.write(block)
            if count < item["bytes"]:
                raise http.client.IncompleteRead(b"", item["bytes"] - count)
            if digest.hexdigest() != item["sha256"]:
                if resume:
                    partial.unlink()
                raise ValueError(f"Downloaded file failed its size/checksum check: {item['name']}")
            os.replace(partial, target)
            return target
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in {403, 404}:
                raise FileNotFoundError(
                    f"Zenodo DATA is unavailable for public download (HTTP {error.code}). "
                    "The matching release may be unpublished or access-restricted. "
                    "Check https://zenodo.org/records/23904047, or install an existing ZIP "
                    "with python scripts/download_data.py --archive /path/to/invLocAd-DATA.zip."
                ) from None
            if attempt == 3:
                raise
            print(
                f"Connection interrupted; retrying this file (attempt {attempt + 1}/3)", flush=True
            )
            time.sleep(attempt)
    raise AssertionError("Unreachable download state")


def zip_members(
    archive: zipfile.ZipFile,
) -> tuple[list[tuple[zipfile.ZipInfo, tuple[str, ...]]], int]:
    """Validate every ZIP entry before any file is extracted.

    Entries must lie under ``DATA_ROOT`` or ``__MACOSX``, be unencrypted regular files or
    directories, and have unique, case-consistent paths. Each ``__MACOSX`` entry must
    correspond to a DATA entry; ``__MACOSX`` and ``.DS_Store`` entries are not extracted.

    Args:
        archive: Open DATA ZIP.

    Returns:
        Tuple ``(members, total)``: the files to extract with their path components, and
        their total uncompressed size in bytes.

    Raises:
        ValueError: If any entry is unsafe, duplicated or inconsistent.
    """
    result, nodes, spelling, explicit = [], {}, {}, set()
    metadata_targets = []
    total = 0
    for info in archive.infolist():
        name = info.orig_filename
        if "\\" in name or any(ord(c) < 32 for c in name) or any(c in name for c in ":<>|?*"):
            raise ValueError(f"Unsafe ZIP path: {name!r}")
        pieces = tuple(name.rstrip("/").split("/"))
        if (
            not pieces
            or pieces[0] not in (DATA_ROOT, "__MACOSX")
            or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in pieces)
        ):
            raise ValueError(f"ZIP entry is outside the expected DATA folder: {name!r}")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 1:
            raise ValueError(
                f"ZIP links, special files and encrypted files are unsupported: {name!r}"
            )
        is_dir = info.is_dir()
        if (mode == stat.S_IFDIR and not is_dir) or (len(pieces) == 1 and not is_dir):
            raise ValueError(f"ZIP entry has conflicting file/directory metadata: {name!r}")
        key = "/".join(pieces).casefold()
        if key in explicit:
            raise ValueError(f"Duplicate ZIP entry: {name!r}")
        explicit.add(key)
        for length in range(1, len(pieces) + 1):
            prefix = "/".join(pieces[:length])
            folded = prefix.casefold()
            if folded in spelling and spelling[folded] != prefix:
                raise ValueError(f"Case-colliding ZIP paths: {name!r}")
            spelling[folded] = prefix
            kind = "directory" if length < len(pieces) or is_dir else "file"
            if folded in nodes and nodes[folded] != kind:
                raise ValueError(f"ZIP file/directory collision: {name!r}")
            nodes[folded] = kind
        if pieces[0] == "__MACOSX":
            if is_dir:
                target = pieces[1:]
            elif pieces[-1].startswith("._"):
                target = (*pieces[1:-1], pieces[-1][2:])
            else:
                raise ValueError(f"Unrecognized Finder metadata path: {name!r}")
            if target:
                if target[0] != DATA_ROOT or any(part in ("", ".", "..") for part in target):
                    raise ValueError(f"Finder metadata is outside the DATA folder: {name!r}")
                metadata_targets.append((name, target, is_dir))
            elif not is_dir:
                raise ValueError(f"Unrecognized Finder metadata path: {name!r}")
        elif pieces[-1] == ".DS_Store":
            if is_dir:
                raise ValueError(f"Finder metadata must be a regular file: {name!r}")
        elif not is_dir:
            total += info.file_size
            result.append((info, pieces))
    for name, target, is_dir in metadata_targets:
        key = "/".join(target)
        if spelling.get(key.casefold()) != key or (
            is_dir and nodes.get(key.casefold()) != "directory"
        ):
            raise ValueError(f"Finder metadata has no matching DATA entry: {name!r}")
    if not result:
        raise ValueError("DATA archive contains no regular files")
    return result, total


def extract(archive_path: Path, destination: Path) -> int:
    """Extract the DATA folder into a new ``destination`` and return the file count.

    Files are staged in a temporary directory beside ``destination`` and renamed into
    place in one step.

    Args:
        archive_path: DATA ZIP; all entries are checked with ``zip_members`` first.
        destination: DATA folder to create.

    Returns:
        Number of extracted files.

    Raises:
        FileExistsError: If ``destination`` exists before or appears during extraction.
        ValueError: If a ZIP entry is unsafe or an extracted file has the wrong size.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists and will not be changed: {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        members, total = zip_members(archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        free_space(destination.parent, total)
        print(f"Extracting {len(members)} files ({total / 1e9:.2f} GB)", flush=True)
        with tempfile.TemporaryDirectory(prefix="data-extract-", dir=destination.parent) as scratch:
            staged = Path(scratch)
            for info, pieces in members:
                path = staged.joinpath(*pieces)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output, CHUNK_SIZE)
                if path.stat().st_size != info.file_size:
                    raise ValueError(f"Extracted file has the wrong size: {info.filename}")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Destination appeared during extraction: {destination}")
            os.rename(staged / DATA_ROOT, destination)
    return len(members)


def installation_paths(destination: Path, cache: Path):
    """Check and return absolute ``(destination, cache)`` paths, creating the cache.

    Args:
        destination: DATA folder to create.
        cache: Download cache directory.

    Raises:
        FileExistsError: If ``destination`` already exists.
        ValueError: If the cache is a link or non-directory, or lies inside ``destination``.
    """
    destination, cache = Path(destination).absolute(), Path(cache).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Destination already exists and will not be changed: {destination}. "
            "Use the existing dataset with invLocAd-publication prepare, or choose another --destination."
        )
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError(f"Download cache must be a directory, not a link: {cache}")
    if cache.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Download cache must be outside the installation destination")
    cache.mkdir(parents=True, exist_ok=True)
    return destination, cache


def finish_install(
    archive: Path,
    item: dict,
    destination: Path,
    cache: Path,
    downloaded: list[Path],
    keep_downloads: bool,
) -> dict:
    """Extract ``archive``; unless ``keep_downloads``, delete ``downloaded`` and the cache if empty.

    Args:
        archive: Verified DATA ZIP.
        item: Archive entry whose ``sha256`` is reported.
        destination: DATA folder to create.
        cache: Download cache directory.
        downloaded: Cached files to delete after extraction.
        keep_downloads: Keep ``downloaded`` and the cache.

    Returns:
        Installation summary with ``destination``, ``files`` and ``archive_sha256``.
    """
    count = extract(archive, destination)
    if not keep_downloads:
        for path in downloaded:
            try:
                path.unlink()
            except OSError as error:
                print(f"DATA installed; could not remove cached {path.name}: {error}", flush=True)
        # rmdir removes the cache only when empty, so other files in it are kept.
        try:
            cache.rmdir()
        except OSError:
            pass
    return {"destination": str(destination), "files": count, "archive_sha256": item["sha256"]}


def install_release(destination: Path, cache: Path, *, keep_downloads: bool = False) -> dict:
    """Download, verify and extract the single pinned DATA ZIP.

    An interrupted download resumes from its ``.partial`` file in ``cache``.

    Args:
        destination: DATA folder to create.
        cache: Download cache directory.
        keep_downloads: Keep the downloaded ZIP after extraction.

    Returns:
        Installation summary from ``finish_install``.
    """
    destination, cache = installation_paths(destination, cache)
    item = {"name": ARCHIVE_NAME, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256}
    archive = download_file(item, cache, resume=True)
    return finish_install(archive, item, destination, cache, [archive], keep_downloads)


def install_local_archive(archive: Path, destination: Path) -> dict:
    """Verify a local DATA ZIP against the pinned size and SHA-256, then extract it.

    The archive is left in place after extraction.

    Args:
        archive: Local DATA ZIP.
        destination: DATA folder to create.

    Returns:
        Installation summary with ``destination``, ``files`` and ``archive_sha256``.

    Raises:
        FileExistsError: If ``destination`` already exists.
        FileNotFoundError: If ``archive`` does not exist.
        ValueError: If ``archive`` is a link or directory, fails the size or checksum check,
            or changes during verification.
    """
    archive, destination = Path(archive).expanduser().absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Destination already exists and is unchanged: {destination}. "
            "Run invLocAd-publication prepare, or choose another --destination."
        )
    if not regular(archive):
        raise FileNotFoundError(f"Local DATA ZIP not found: {archive}")
    before = archive.stat()
    item = {"name": ARCHIVE_NAME, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256}
    print(f"Verifying local DATA archive: {archive}", flush=True)
    if not verified(archive, item):
        raise ValueError("Local DATA archive failed its pinned size/checksum check; nothing was extracted")
    after = archive.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("Local DATA archive changed during verification; nothing was extracted")
    count = extract(archive, destination)
    return {"destination": str(destination), "files": count, "archive_sha256": item["sha256"]}


def main(argv=None) -> int:
    """Install DATA by the method chosen on the command line.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        0 on success, or 130 if interrupted. Installation errors exit with status 1 and
        invalid arguments with status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Verify and install an existing DATA ZIP without downloading or changing it"
    )
    parser.add_argument("--destination", type=Path, default=Path("data") / DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/data-download/23904047"))
    parser.add_argument(
        "--keep-downloads", action="store_true", help="Retain the verified cached ZIP"
    )
    args = parser.parse_args(argv)
    try:
        # Check the destination before any network request.
        if args.destination.exists() or args.destination.is_symlink():
            raise FileExistsError(
                f"Destination already exists and is unchanged: {args.destination}. "
                "Run invLocAd-publication prepare, or choose another --destination."
            )
        if args.archive is not None:
            result = install_local_archive(args.archive, args.destination)
        else:
            result = install_release(
                args.destination, args.cache_dir, keep_downloads=args.keep_downloads
            )
    except KeyboardInterrupt:
        print(
            "\nInterrupted. The local archive is unchanged; run the same command again."
            if args.archive is not None
            else "\nInterrupted. Cached downloads are retained; run the same command to continue."
        )
        return 130
    except (OSError, ValueError, zipfile.BadZipFile, http.client.HTTPException) as error:
        retained = "The local archive was not modified." if args.archive is not None else "Cached downloads are retained."
        parser.exit(
            1, f"DATA installation stopped: {error}\n{retained}\n"
        )
    print(f"DATA installed in {result['destination']} ({result['files']} files).")
    print(f"Verified archive SHA-256: {result['archive_sha256']}")
    print("Next: invLocAd-publication prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
