#!/usr/bin/env python3
"""Finish M-series downloads, verify them, then preprocess every archive."""

from __future__ import annotations

import fcntl
import json
import os
import select
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PREPROCESS = REPO / "preprocess"
RAW = REPO / "challenge_resources/data/raw_zip_videos"
RESULTS = REPO / "challenge_resources/data/zip_embeddings"
LOGS = RESULTS / "m_series_logs"
MANIFEST = PREPROCESS / "download_urls_M01-M10.txt"
ENGINE = REPO / "keyframe_pipeline_global_v9_3/run_pipeline.sh"
PYTHON = REPO / "remote-server/.venv/bin/python"
DOWNLOAD_INTERVAL = 600
PIPELINE_INTERVAL = 600


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def manifest_items() -> list[tuple[str, str]]:
    urls = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip() and not line.startswith("#")]
    items = [(url.rsplit("/", 1)[-1], url) for url in urls]
    if len(items) != 10 or len({name for name, _ in items}) != 10:
        raise RuntimeError("Expected exactly ten distinct M-series URLs")
    return items


def remote_size(url: str) -> int:
    for attempt in range(1, 11):
        try:
            response = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--location", "--head", "--connect-timeout", "30", url],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            values = [
                line.partition(":")[2].strip()
                for line in response.stdout.splitlines()
                if line.lower().startswith("content-length:")
            ]
            size = int(values[-1])
            if size <= 0:
                raise ValueError("empty Content-Length")
            return size
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
            log(f"HEAD attempt {attempt}/10 failed for {url}: {exc}")
            if attempt == 10:
                raise
            time.sleep(30)
    raise AssertionError("unreachable")


def current_bytes(items: list[tuple[str, str]]) -> int:
    return sum((RAW / name).stat().st_size for name, _ in items if (RAW / name).exists())


def counts(items: list[tuple[str, str]], expected: dict[str, int]) -> tuple[int, int]:
    complete = sum(
        (RAW / name).is_file() and (RAW / name).stat().st_size == expected[name]
        for name, _ in items
    )
    return complete, current_bytes(items)


def wait_for_existing_download(pid: int, items: list[tuple[str, str]], expected: dict[str, int]) -> None:
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        while not poller.poll(0):
            complete, downloaded = counts(items, expected)
            log(f"downloads active (PID {pid}): {complete}/10 complete, {downloaded:,}/{sum(expected.values()):,} bytes")
            poller.poll(DOWNLOAD_INTERVAL * 1000)
    finally:
        os.close(pidfd)


def run_download_queue(items: list[tuple[str, str]], expected: dict[str, int]) -> None:
    for attempt in range(1, 6):
        complete, downloaded = counts(items, expected)
        if complete == len(items):
            return
        log(f"resuming downloads (round {attempt}/5): {complete}/10 complete, {downloaded:,} bytes")
        with (LOGS / f"download_round_{attempt}.log").open("w") as output:
            process = subprocess.Popen(
                ["bash", str(PREPROCESS / "resume_zip_downloads.sh"), "--jobs", "2"],
                cwd=PREPROCESS,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                complete, downloaded = counts(items, expected)
                log(f"downloads: {complete}/10 complete, {downloaded:,}/{sum(expected.values()):,} bytes")
                try:
                    process.wait(timeout=DOWNLOAD_INTERVAL)
                except subprocess.TimeoutExpired:
                    pass
            log(f"download round {attempt} exited {process.returncode}")
        if counts(items, expected)[0] == len(items):
            return
        time.sleep(30)
    raise RuntimeError("Downloads remain incomplete after five resume rounds; see download_round_*.log")


def verify_sources(items: list[tuple[str, str]], expected: dict[str, int]) -> dict[str, int]:
    video_counts: dict[str, int] = {}
    for name, _ in items:
        source = RAW / name
        actual = source.stat().st_size
        if actual != expected[name]:
            raise RuntimeError(f"{name}: size {actual:,}, expected {expected[name]:,}")
        with zipfile.ZipFile(source) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"{name}: CRC failure in {bad}")
            videos = [
                info for info in archive.infolist()
                if info.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"))
            ]
        if not videos:
            raise RuntimeError(f"{name}: no videos")
        video_counts[name] = len(videos)
        log(f"verified source {name}: {actual:,} bytes, {len(videos)} videos, full CRC OK")
    return video_counts


def inspect_result(lot: str, expected_videos: int) -> dict:
    archive_path = RESULTS / f"{lot}_results.zip"
    result = subprocess.run(
        [str(PYTHON), "-m", "preprocess", "inspect", str(archive_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{lot}: result validation failed: {result.stderr.strip()}")
    inspection = json.loads(result.stdout)
    if inspection["format_version"] != 3 or inspection["video_count"] != expected_videos:
        raise RuntimeError(f"{lot}: result contains {inspection['video_count']}/{expected_videos} videos")
    return inspection


def run_lot(lot: str, name: str, expected_videos: int) -> dict:
    source = RAW / name
    for attempt in range(1, 4):
        try:
            inspection = inspect_result(lot, expected_videos)
            log(f"{lot}: already complete, {inspection['video_count']} videos")
            return inspection
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass
        command = [
            "bash", str(ENGINE), "--zip", str(source),
            "--work-root", str(RESULTS), "--profile", "disk-rich",
            "--sequential-stages",
            "--transnet-decode-workers", "2",
            "--transnet-prefetch-windows", "64",
            "--prefetch-batches", "2",
        ]
        logfile = LOGS / f"{lot}.attempt{attempt}.log"
        log(f"{lot}: starting full TransNet + embedding + packaging (attempt {attempt}/3); log {logfile}")
        with logfile.open("w") as output:
            process = subprocess.Popen(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                outdir = RESULTS / f"output_{lot}"
                scene_count = len(list(outdir.glob("*/scenes.json")))
                embed_count = len(list(outdir.glob("*/embeddings.npy")))
                log(f"{lot}: process PID {process.pid}; scenes={scene_count}/{expected_videos}, embeddings={embed_count}/{expected_videos}")
                try:
                    process.wait(timeout=PIPELINE_INTERVAL)
                except subprocess.TimeoutExpired:
                    pass
            log(f"{lot}: pipeline exited {process.returncode}")
        try:
            inspection = inspect_result(lot, expected_videos)
            if process.returncode == 0:
                log(f"{lot}: verified {inspection['video_count']} videos, {inspection['keyframe_count']} keyframes")
                return inspection
            log(f"{lot}: complete archive despite nonzero pipeline status; accepting verified result")
            return inspection
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            log(f"{lot}: attempt {attempt} incomplete: {exc}")
            time.sleep(30)
    raise RuntimeError(f"{lot}: incomplete after three pipeline attempts; inspect {LOGS}/{lot}.attempt*.log")


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    lockfile = LOGS / "controller.lock"
    with lockfile.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another M-series controller is already running") from None
        items = manifest_items()
        expected = {name: remote_size(url) for name, url in items}
        log(f"M-series total: {sum(expected.values()):,} bytes across {len(items)} archives")
        if len(sys.argv) == 2:
            wait_for_existing_download(int(sys.argv[1]), items, expected)
        run_download_queue(items, expected)
        video_counts = verify_sources(items, expected)
        summary: dict[str, dict] = {}
        failures: dict[str, str] = {}
        for name, _ in items:
            lot = Path(name).stem.removeprefix("Videos_")
            try:
                summary[lot] = run_lot(lot, name, video_counts[name])
            except Exception as exc:
                failures[lot] = str(exc)
                log(f"FAILED {lot}: {exc}")
            (LOGS / "summary.json").write_text(
                json.dumps({"completed": summary, "failures": failures}, indent=2) + "\n"
            )
        log(f"finished: {len(summary)}/10 lots complete, {len(failures)} failures")
        return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {exc}")
        raise
