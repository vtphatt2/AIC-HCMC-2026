"""Offline, verified JPEG fallback for source frames that cannot be seek-decoded.

Decode the original stream sequentially once, select the already indexed PTS,
and publish only pictures whose pre-encoding checksum matches the full frame
map. The archive, selected frame numbers, and embedding vectors are untouched.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.exact_frame_pts import index_path, showinfo_frames
from app.services.readiness_policy import verified_embed_decoder_threads


EXACT_N_FRAME_DIR = (Path(__file__).resolve().parents[3] /
                     "challenge_resources/data/zip_embeddings/exact_n_frames")


def exact_image_path(video_id: str, index, frame_number: int,
                     directory: Path = EXACT_N_FRAME_DIR) -> Path:
    return directory / index_path(video_id, index).stem / f"{frame_number}.jpg"


def build_exact_images(video_id: str, index, frames: list[int], *,
                       directory: Path = EXACT_N_FRAME_DIR,
                       timeout: float = 900) -> list[Path]:
    """Publish exact full-size JPEGs after validating the entire selected batch."""
    if not video_id.startswith("N"):
        raise ValueError("Sequential fallback publication is limited to N videos")
    targets = sorted(set(frames))
    if not targets:
        return []
    table = np.load(index_path(video_id, index), mmap_mode="r", allow_pickle=False)
    if (table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64 or
            not np.array_equal(table[:, 0], np.arange(len(table))) or
            targets[0] < 0 or targets[-1] >= len(table)):
        raise ValueError(f"{video_id}: missing or invalid full decoded-frame map")
    selected = table[targets]
    # A retained picture may share its PTS with an omitted backward-timestamp
    # picture. Select by decoded frame number in that case; the full-map PTS
    # AND pixel checksum below still have to match before any JPEG is exposed.
    unique_pts = len(set(map(int, selected[:, 1]))) == len(targets) and all(
        np.count_nonzero(table[:, 1] == pts) == 1 for pts in selected[:, 1])
    paths = [exact_image_path(video_id, index, frame, directory) for frame in targets]
    if all(path.is_file() and path.stat().st_size for path in paths):
        return paths
    directory.mkdir(parents=True, exist_ok=True)
    source = (f"subfile,,start,{index.data_offset},end,"
              f"{index.data_offset + index.video_size},,:{index.zip_path}")
    selection = ("+".join(f"eq(pts\\,{int(pts)})" for pts in selected[:, 1])
                 if unique_pts else
                 "+".join(f"eq(n\\,{frame})" for frame in targets))
    with tempfile.TemporaryDirectory(dir=directory, prefix=".decode-") as scratch:
        scratch_path = Path(scratch)
        command = [
            "/usr/bin/ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
            "-copyts", "-threads", str(verified_embed_decoder_threads(video_id, index)),
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", f"select='{selection}',showinfo", "-vsync", "0",
            "-filter_threads", "1", "-threads:v", "1",
            "-frames:v", str(len(targets)), "-q:v", "4",
            "-start_number", "0", "-f", "image2", str(scratch_path / "%08d.jpg"),
        ]
        with tempfile.TemporaryFile() as diagnostics:
            process = subprocess.run(command, stdout=subprocess.DEVNULL,
                                     stderr=diagnostics, timeout=timeout)
            diagnostics.seek(0)
            records = showinfo_frames(diagnostics.read())
        if process.returncode or len(records) != len(targets):
            raise RuntimeError(f"{video_id}: sequential JPEG decode returned "
                               f"{len(records)}/{len(targets)} frames, exit {process.returncode}")
        for position, (_, pts, checksum) in enumerate(records):
            if pts != int(selected[position, 1]) or checksum != int(selected[position, 2]):
                raise RuntimeError(f"{video_id}/{targets[position]}: sequential pixel verification failed")
            with Image.open(scratch_path / f"{position:08d}.jpg") as image:
                image.verify()
        # No selected image becomes visible until all identities are checked.
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for position, path in enumerate(paths):
            os.replace(scratch_path / f"{position:08d}.jpg", path)
    return paths
