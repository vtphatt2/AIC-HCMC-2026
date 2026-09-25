#!/usr/bin/env python3
"""Resume and validate the S, M, and N video-archive preprocessing runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "challenge_resources/data/raw_zip_videos"
RESULTS = REPO / "challenge_resources/data/zip_embeddings"
LOGS = RESULTS / "new_archive_logs"
ENGINE = REPO / "keyframe_pipeline_global_v9_3/run_pipeline.sh"
PYTHON = REPO / "remote-server/.venv/bin/python"
M_LOTS = [f"M{number:02d}" for number in range(1, 11)]
N_LOTS = [f"N{start:03d}-N{start + 9:03d}" for start in range(1, 101, 10)]
LOTS = M_LOTS + ["S01"] + N_LOTS
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def source_path(lot: str) -> Path:
    prefix = "Videos_" if lot.startswith("M") else "Video_"
    return RAW / f"{prefix}{lot}.zip"


def source_video_count(source: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"{source.name}: CRC failure in {bad}")
        count = sum(info.filename.lower().endswith(VIDEO_SUFFIXES) for info in archive.infolist())
    if count == 0:
        raise RuntimeError(f"{source.name}: no videos")
    log(f"source verified: {source.name}, {count} videos, full CRC OK")
    return count


def inspect_result(lot: str, expected_videos: int) -> dict:
    archive = RESULTS / f"{lot}_results.zip"
    result = subprocess.run(
        [str(PYTHON), "-m", "preprocess", "inspect", str(archive)],
        cwd=REPO, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{lot}: invalid result: {result.stderr.strip()}")
    inspection = json.loads(result.stdout)
    if inspection["format_version"] != 3 or inspection["video_count"] != expected_videos:
        raise RuntimeError(
            f"{lot}: result has {inspection['video_count']}/{expected_videos} videos "
            f"in format {inspection['format_version']}"
        )
    return inspection


def wait_for_running_m_controller(pid: int) -> None:
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as command_file:
            command = command_file.read().replace(b"\0", b" ")
        if b"run_M01_M10_unattended.py" not in command:
            raise RuntimeError(f"PID {pid} is not the M-series controller")
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        log(f"waiting for active M-series controller PID {pid}")
        while not poller.poll(60_000):
            log(f"M-series controller PID {pid} still running")
    finally:
        os.close(pidfd)


def run_lot(lot: str, source: Path, expected_videos: int) -> dict:
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
        log(f"{lot}: processing attempt {attempt}/3; details: {logfile}")
        with logfile.open("w") as output:
            process = subprocess.Popen(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                outdir = RESULTS / f"output_{lot}"
                scenes = len(list(outdir.glob("*/scenes.json")))
                embeddings = len(list(outdir.glob("*/embeddings.npy")))
                log(f"{lot}: PID {process.pid}, scenes={scenes}/{expected_videos}, embeddings={embeddings}/{expected_videos}")
                try:
                    process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    pass
            log(f"{lot}: pipeline exit code {process.returncode}")

        try:
            inspection = inspect_result(lot, expected_videos)
            log(f"{lot}: validated {inspection['video_count']} videos, {inspection['keyframe_count']} keyframes")
            return inspection
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            log(f"{lot}: attempt {attempt} incomplete: {exc}")
            if attempt < 3:
                time.sleep(30)
    raise RuntimeError(f"{lot}: incomplete after three attempts; see {LOGS}/{lot}.attempt*.log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-m-pid", type=int, help="Wait for an existing M-series controller")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "controller.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another S/M/N controller is already running") from None

        if args.wait_for_m_pid:
            wait_for_running_m_controller(args.wait_for_m_pid)

        completed: dict[str, dict] = {}
        failures: dict[str, str] = {}
        for lot in LOTS:
            source = source_path(lot)
            try:
                expected = source_video_count(source)
                completed[lot] = run_lot(lot, source, expected)
            except Exception as exc:
                failures[lot] = str(exc)
                log(f"FAILED {lot}: {exc}")
            (LOGS / "summary.json").write_text(
                json.dumps({"completed": completed, "failures": failures}, indent=2) + "\n"
            )
        log(f"finished: {len(completed)}/{len(LOTS)} lots complete, {len(failures)} failures")
        return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {exc}")
        raise
