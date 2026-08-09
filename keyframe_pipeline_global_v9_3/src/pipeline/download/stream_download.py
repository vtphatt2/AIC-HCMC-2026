"""Download a ZIP sequentially while running TransNetV2 on each completed ZIP_STORED video entry.

The downloader appends bytes to --zip. As soon as a complete video entry is present,
its byte range is queued to the single TransNet GPU worker. After the download finishes,
run src/process_video_zip_gpu.py normally: it reuses the generated scenes/keyframes and
continues with the PE-Core embedding phase.
"""
from __future__ import annotations

import argparse
import queue
import re
import shutil
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch

from ..io_utils import atomic_json, safe_video_id
from ..phase1_transnet.decode import decode_transnet_frames, select_keyframes
from ..phase1_transnet.model import load_transnet
from ..video_probe import probe_video
from ..zip_source import LOCAL_HEADER_FIXED_SIZE as LOCAL_HEADER_SIZE
from ..zip_source import VIDEO_EXTENSIONS


@dataclass(frozen=True)
class ReadyEntry:
    name: str
    data_start: int
    data_end: int
    size: int


@dataclass(frozen=True)
class DownloadDone:
    error: str | None = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk-mb", type=int, default=8)
    p.add_argument("--entry-regex", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    return p.parse_args()


class LocalHeaderParser:
    def __init__(self, path: Path, pattern: str | None, q: queue.Queue):
        self.path, self.q = path, q
        self.offset = 0
        self.regex = re.compile(pattern) if pattern else None
        self.seen: set[str] = set()

    def scan(self, available: int):
        with self.path.open("rb") as f:
            while self.offset + 4 <= available:
                f.seek(self.offset)
                sig = f.read(4)
                if sig != b"PK\x03\x04":
                    return  # central directory, or incomplete/unexpected data
                if self.offset + LOCAL_HEADER_SIZE > available:
                    return
                f.seek(self.offset)
                hdr = f.read(LOCAL_HEADER_SIZE)
                fields = struct.unpack("<IHHHHHIIIHH", hdr)
                _, _, flags, compression, _, _, _, comp_size, uncomp_size, name_len, extra_len = fields
                header_end = self.offset + LOCAL_HEADER_SIZE + name_len + extra_len
                if header_end > available:
                    return
                f.seek(self.offset + LOCAL_HEADER_SIZE)
                name = f.read(name_len).decode("utf-8", errors="replace")
                if flags & 0x08:
                    raise RuntimeError(
                        f"ZIP entry {name!r} uses a data descriptor; streaming boundaries are unavailable. "
                        "Let the ZIP finish downloading, then run the normal pipeline."
                    )
                data_end = header_end + comp_size
                if data_end > available:
                    return
                is_video = name.lower().endswith(VIDEO_EXTENSIONS)
                selected = self.regex is None or self.regex.search(name)
                if is_video and selected:
                    if compression != 0:
                        raise RuntimeError(f"Video entry {name!r} is compressed, not ZIP_STORED")
                    if name not in self.seen:
                        self.seen.add(name)
                        self.q.put(ReadyEntry(name, header_end, data_end, uncomp_size))
                self.offset = data_end


def downloader(args, q: queue.Queue):
    try:
        args.zip_path.parent.mkdir(parents=True, exist_ok=True)
        existing = args.zip_path.stat().st_size if args.zip_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        with requests.get(args.url, stream=True, headers=headers, timeout=(30, 120)) as r:
            r.raise_for_status()
            if existing and r.status_code != 206:
                existing = 0
                mode = "wb"
            else:
                mode = "ab" if existing else "wb"
            parser = LocalHeaderParser(args.zip_path, args.entry_regex, q)
            if existing:
                parser.scan(existing)
            total = r.headers.get("Content-Range") or r.headers.get("Content-Length") or "?"
            print(f"download status={r.status_code} resume_at={existing} total={total}", flush=True)
            downloaded = existing
            with args.zip_path.open(mode) as f:
                for chunk in r.iter_content(args.chunk_mb << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    f.flush()
                    downloaded += len(chunk)
                    parser.scan(downloaded)
                    print(f"\rdownloaded {downloaded/1024**2:.1f} MiB", end="", flush=True)
            print()
            parser.scan(downloaded)
        q.put(DownloadDone())
    except BaseException as e:
        q.put(DownloadDone(f"{type(e).__name__}: {e}"))


def process_entry(args, model, item: ReadyEntry):
    vid = safe_video_id(item.name)
    out = args.out_dir / vid
    scenes_path, keys_path = out / "scenes.json", out / "keyframes.json"
    if scenes_path.exists() and keys_path.exists() and not args.overwrite:
        print(f"[TransNet] reuse {item.name}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    source = f"subfile,,start,{item.data_start},end,{item.data_end},,:{args.zip_path.resolve()}"
    t0 = time.perf_counter()
    fps, width, height = probe_video(args.ffprobe_bin, source)
    td = time.perf_counter()
    frames = decode_transnet_frames(args.ffmpeg_bin, source)
    decode_s = time.perf_counter() - td
    ti = time.perf_counter()
    x = torch.from_numpy(frames).to(model.device)
    with torch.inference_mode():
        pred, _ = model.predict_frames(x, quiet=True)
    pred_np = pred.detach().float().cpu().numpy()
    scenes = model.predictions_to_scenes(pred_np, threshold=args.threshold)
    infer_s = time.perf_counter() - ti
    scene_items = [{"start_frame": int(a), "end_frame": int(b)} for a, b in scenes]
    keys = select_keyframes(scenes, fps)
    common = {"zip": str(args.zip_path), "entry": item.name, "fps": fps, "width": width,
              "height": height, "num_frames": int(frames.shape[0])}
    atomic_json(scenes_path, {**common, "threshold": args.threshold,
                             "num_scenes": len(scene_items), "scenes": scene_items})
    atomic_json(keys_path, {**common, "num_scenes": len(scene_items),
                           "num_keyframes": len(keys), "keyframes": keys})
    np.save(out / "transnet_predictions.npy", pred_np)
    atomic_json(out / "transnet_stats.json", {"entry": item.name,
        "decode_seconds": round(decode_s, 3), "inference_seconds": round(infer_s, 3),
        "total_seconds": round(time.perf_counter() - t0, 3), "streamed_during_download": True})
    print(f"[TransNet] {item.name}: {len(scene_items)} scenes, {len(keys)} keyframes", flush=True)
    del frames, x, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if shutil.which(args.ffmpeg_bin) is None or shutil.which(args.ffprobe_bin) is None:
        raise SystemExit("ffmpeg/ffprobe not found")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue(maxsize=2)
    thread = threading.Thread(target=downloader, args=(args, q), daemon=True)
    thread.start()
    model = load_transnet(args.device)
    error = None
    while True:
        item = q.get()
        if isinstance(item, DownloadDone):
            error = item.error
            break
        process_entry(args, model, item)
    thread.join()
    if error:
        raise RuntimeError(error)
    print("Download complete and all streamable TransNet tasks finished.")
