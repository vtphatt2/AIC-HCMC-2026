#!/usr/bin/env python3
"""Download dataset ZIPs, merge a few at a time into one combined archive, and publish each
group as a Kaggle Dataset -- so a Kaggle Notebook can attach it directly (no per-run network
download). Deletes local files as soon as each group's upload succeeds.

Requires: pip install kaggle (separate from kagglehub -- this is the upload-capable CLI/library),
and Kaggle API credentials ($KAGGLE_USERNAME/$KAGGLE_KEY or ~/.kaggle/kaggle.json).

Example (merge 2 links per Kaggle dataset, 3 links -> 2 datasets):
    python tools/build_kaggle_dataset.py \
        --url https://aic-data.ledo.io.vn/Videos_L26_d.zip \
        --url https://aic-data.ledo.io.vn/Videos_L26_e.zip \
        --url https://aic-data.ledo.io.vn/Videos_L28_a.zip \
        --group-size 2 --dataset-title aic-videos

To remove a published dataset later:
    kaggle datasets delete -d <username>/<slug>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import requests

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def download(url: str, out: Path) -> None:
    print(f"  downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        done = 0
        with out.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                print(f"\r    {done / 1024**3:.2f} GiB", end="", flush=True)
    print()


def merge_zips(sources: list[tuple[str, Path]], out: Path) -> int:
    """sources: (label, raw_zip_path) pairs. Video entries only, ZIP_STORED, prefixed by label
    to avoid filename collisions between merged datasets (e.g. both having V001.mp4)."""
    count = 0
    with zipfile.ZipFile(out, "w") as out_zf:
        for label, src_path in sources:
            with zipfile.ZipFile(src_path) as src_zf:
                for info in src_zf.infolist():
                    if not info.filename.lower().endswith(VIDEO_EXTENSIONS):
                        continue
                    out_zf.writestr(
                        f"{label}/{Path(info.filename).name}",
                        src_zf.read(info.filename),
                        compress_type=zipfile.ZIP_STORED,
                    )
                    count += 1
    return count


def upload_to_kaggle(folder: Path, slug: str, title: str, username: str) -> None:
    metadata = {"title": title, "id": f"{username}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    subprocess.run(["kaggle", "datasets", "create", "-p", str(folder), "-q"], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", action="append", required=True, dest="urls", help="Repeatable")
    p.add_argument("--group-size", type=int, default=2, help="Links merged into one Kaggle Dataset (default 2)")
    p.add_argument("--dataset-title", default="aic-videos", help="Base title/slug; group index appended")
    p.add_argument("--username", default=None, help="Kaggle username; default reads $KAGGLE_USERNAME")
    p.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "kaggle_dataset_build")
    p.add_argument("--keep-local", action="store_true", help="Don't delete local files after upload (debugging)")
    args = p.parse_args()

    username = args.username or os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise SystemExit("ERROR: set --username or $KAGGLE_USERNAME")
    if shutil.which("kaggle") is None:
        raise SystemExit("ERROR: kaggle CLI not found -- pip install kaggle")

    groups = [args.urls[i:i + args.group_size] for i in range(0, len(args.urls), args.group_size)]

    for gi, group in enumerate(groups, 1):
        print(f"=== group {gi}/{len(groups)}: {len(group)} link(s) ===", flush=True)
        work = args.work_dir / f"group_{gi}"
        work.mkdir(parents=True, exist_ok=True)

        sources: list[tuple[str, Path]] = []
        for url in group:
            label = Path(url.split("?")[0]).stem  # e.g. Videos_L26_d
            raw = work / f"{label}.raw.zip"
            download(url, raw)
            sources.append((label, raw))

        merged = work / "merged.zip"
        n = merge_zips(sources, merged)
        print(f"  merged {n} video(s) -> {merged} ({merged.stat().st_size / 1024**3:.2f} GiB)", flush=True)

        for _, raw in sources:
            raw.unlink()

        upload_dir = work / "upload"
        upload_dir.mkdir(exist_ok=True)
        shutil.move(str(merged), upload_dir / "merged.zip")

        slug = f"{args.dataset_title}-{gi}"
        print(f"  uploading as {username}/{slug} ...", flush=True)
        upload_to_kaggle(upload_dir, slug, f"{args.dataset_title} batch {gi}", username)
        print(f"  done: {username}/{slug}", flush=True)

        if not args.keep_local:
            shutil.rmtree(work)
            print("  local files removed", flush=True)


if __name__ == "__main__":
    main()
