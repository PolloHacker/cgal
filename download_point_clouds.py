#!/usr/bin/env python3
"""Download point-cloud files listed by the S3 Sync Manager."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://150.164.212.12:8010"
DEFAULT_FOLDER = "point-clouds"
S3_BUCKET_URL = "https://ufmg-vale-esp-artefatos.s3.us-east-2.amazonaws.com"
CHUNK_SIZE = 1024 * 1024


def safe_relative_path(rel_path: str, folder: str) -> PurePosixPath:
    """Return a validated path below the requested remote folder."""
    path = PurePosixPath(rel_path)
    root = PurePosixPath(folder)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"server returned a file outside {folder!r}: {rel_path!r}") from exc
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe path returned by server: {rel_path!r}")
    return relative


def list_files(server: str, folder: str, timeout: float) -> list[dict]:
    """List files under folder, following subdirectories if present."""
    pending = [folder.strip("/")]
    seen_directories: set[str] = set()
    files: dict[str, dict] = {}

    while pending:
        current = pending.pop()
        if current in seen_directories:
            continue
        seen_directories.add(current)

        query = urlencode({"path": current})
        request = Request(
            f"{server.rstrip('/')}/api/files?{query}",
            headers={"User-Agent": "point-cloud-downloader/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            entries = json.load(response)
        if not isinstance(entries, list):
            raise ValueError(f"unexpected response when listing {current!r}: expected a JSON list")

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("type")
            rel_path = entry.get("rel_path")
            if not isinstance(rel_path, str):
                continue
            if kind == "file":
                safe_relative_path(rel_path, folder)
                files[rel_path] = entry
            elif kind in {"directory", "dir", "folder"}:
                safe_relative_path(rel_path, folder)
                pending.append(rel_path)

    return sorted(files.values(), key=lambda item: item["rel_path"].casefold())


def download_file(
    entry: dict,
    folder: str,
    output: Path,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> tuple[str, str, int]:
    rel_path = entry["rel_path"]
    relative = safe_relative_path(rel_path, folder)
    destination = output.joinpath(*relative.parts)
    partial = destination.with_name(destination.name + ".part")
    expected_size = entry.get("size")
    expected_size = expected_size if isinstance(expected_size, int) and expected_size >= 0 else None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not overwrite:
        if expected_size is None or destination.stat().st_size == expected_size:
            return rel_path, "already present", destination.stat().st_size

    object_url = f"{S3_BUCKET_URL}/{quote(rel_path, safe='/')}"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if partial.exists() and expected_size is not None and offset == expected_size:
            partial.replace(destination)
            return rel_path, "downloaded", offset
        if expected_size is not None and offset > expected_size:
            partial.unlink()
            offset = 0

        headers = {"User-Agent": "point-cloud-downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(object_url, headers=headers)

        try:
            with urlopen(request, timeout=timeout) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)

            actual_size = partial.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"size mismatch for {rel_path}: expected {expected_size} bytes, got {actual_size}"
                )
            partial.replace(destination)
            return rel_path, "downloaded", actual_size
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"failed to download {rel_path} after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all files in the server's point-clouds folder."
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"S3 Sync Manager base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help=f"remote folder to download (default: {DEFAULT_FOLDER})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("point-clouds"),
        help="local output directory (default: ./point-clouds)",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default: 4)")
    parser.add_argument("--timeout", type=float, default=60, help="request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="attempts per file (default: 3)")
    parser.add_argument(
        "--overwrite", action="store_true", help="download files again even if already present"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    args.folder = args.folder.strip("/")
    if not args.folder:
        parser.error("--folder must not be empty")
    return args


def main() -> int:
    args = parse_args()
    try:
        entries = list_files(args.server, args.folder, args.timeout)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not list remote point clouds: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print(f"No files found under {args.folder!r}.")
        return 0

    total_bytes = sum(
        entry.get("size", 0)
        for entry in entries
        if isinstance(entry.get("size"), int) and entry.get("size", 0) >= 0
    )
    print(f"Found {len(entries)} files ({total_bytes / 1_000_000_000:.2f} GB).")
    args.output.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_file,
                entry,
                args.folder,
                args.output,
                args.timeout,
                args.retries,
                args.overwrite,
            ): entry["rel_path"]
            for entry in entries
        }
        for future in as_completed(futures):
            rel_path = futures[future]
            try:
                _, status, size = future.result()
                completed += 1
                print(f"[{completed}/{len(entries)}] {status}: {rel_path} ({size:,} bytes)")
            except Exception as exc:  # Report other files and summarize failures at the end.
                failures.append((rel_path, str(exc)))
                completed += 1
                print(f"[{completed}/{len(entries)}] FAILED: {rel_path}: {exc}", file=sys.stderr)

    if failures:
        print(f"{len(failures)} file(s) failed.", file=sys.stderr)
        return 1
    print(f"Saved files to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
