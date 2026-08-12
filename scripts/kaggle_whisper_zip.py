"""Kaggle notebook: streaming Whisper ASR — one ZIP at a time, minimal disk usage.

=== CELL 1: Install ===
"""
# CELL 1 — Install
# !pip install -q faster-whisper
c
import json, time, zipfile, requests, shutil
from pathlib import Path
from faster_whisper import WhisperModel

# === CELL 2: Load Whisper (minimal disk) ===
# CELL 2 — Load model, use /tmp for cache
import os
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache"

print("Loading large-v3 ...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root="/tmp/whisper_models")
print("Loaded.")

# Free disk: delete pip cache + HF cache on working dir
import shutil
for d in ["/root/.cache/pip", "/kaggle/working/.cache"]:
    if Path(d).exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Freed {d}")

# === CELL 3: Streaming download → extract → transcribe → cleanup ===
# CELL 3 — Pipeline: one ZIP at a time
OUT_DIR = Path("/kaggle/tmp/transcripts")
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR = Path("/kaggle/tmp")
TMP_DIR.mkdir(exist_ok=True)

ZIP_URLS = [
    "https://aic-data.ledo.io.vn/Videos_L26_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_b.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_c.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_d.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_e.zip",
    "https://aic-data.ledo.io.vn/Videos_L28_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L30_a.zip",
]

NEEDED = {
    "L26_V298", "L26_V313", "L26_V337", "L26_V381", "L26_V432", "L26_V491",
    "L28_V001", "L28_V002", "L28_V003", "L28_V004", "L28_V005", "L28_V006",
    "L28_V007", "L28_V008", "L28_V009", "L28_V010", "L28_V011", "L28_V012",
    "L28_V013", "L28_V014", "L28_V015", "L28_V016", "L28_V017", "L28_V018",
    "L28_V019", "L28_V020", "L28_V021", "L28_V022", "L28_V023", "L28_V024",
    "L30_V029", "L30_V039",
}

total_zips = len(ZIP_URLS)
total_ok = total_fail = 0
run_start = time.perf_counter()

for idx, url in enumerate(ZIP_URLS, 1):
    zip_name = url.split("/")[-1]
    zip_path = TMP_DIR / zip_name
    video_dir = TMP_DIR / f"videos_{idx}"

    t0 = time.perf_counter()

    # ── Download ──
    print(f"\n[{idx}/{total_zips}] Downloading {zip_name} ...")
    r = requests.get(url, stream=True, allow_redirects=True, timeout=300)
    if r.status_code != 200:
        print(f"  FAIL: HTTP {r.status_code}")
        continue
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"  {downloaded/1024**2:.0f}/{total/1024**2:.0f} MB ({pct:.0f}%)", end="\r")
        print()
    dl_time = time.perf_counter() - t0
    size_mb = zip_path.stat().st_size / 1024**2
    if size_mb < 0.5:
        print(f"  FAIL: too small ({size_mb:.1f} MB)")
        zip_path.unlink()
        continue
    print(f"  Downloaded {size_mb:.0f} MB in {dl_time:.1f}s")

    # ── Extract needed videos ──
    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        print(f"  FAIL: bad ZIP — {e}")
        zip_path.unlink()
        continue

    video_dir.mkdir(exist_ok=True)
    extracted = []
    with z:
        for name in z.namelist():
            vid = Path(name).stem
            if vid in NEEDED:
                dest = video_dir / Path(name).name
                if not dest.exists():
                    z.extract(name, video_dir)
                    extracted.append(dest)
                    print(f"  Extracted {name}")
    zip_path.unlink()  # free disk space immediately
    print(f"  Extracted {len(extracted)} videos, deleted ZIP")

    # ── Transcribe ──
    for vp in extracted:
        vid = vp.stem
        out_path = OUT_DIR / f"{vid}.jsonl"
        if out_path.exists():
            print(f"  SKIP {vid}")
            continue

        t1 = time.perf_counter()
        segs_iter, info = model.transcribe(str(vp), language="vi", beam_size=5, word_timestamps=True)
        segs = list(segs_iter)
        trans_time = time.perf_counter() - t1

        if not segs:
            print(f"  FAIL {vid} — empty")
            total_fail += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for s in segs:
                f.write(json.dumps({
                    "start_time_ms": round(s.start * 1000),
                    "end_time_ms": round(s.end * 1000),
                    "text": s.text.strip(),
                }, ensure_ascii=False) + "\n")

        dur = segs[-1].end
        xrt = dur / max(trans_time, 0.1)
        print(f"  OK   {vid} — {len(segs)} segs | {dur:.0f}s | {trans_time:.1f}s ({xrt:.1f}x)")
        total_ok += 1

    # ── Cleanup videos ──
    shutil.rmtree(video_dir)
    elapsed = time.perf_counter() - t0
    print(f"  ZIP {idx}/{total_zips} done in {elapsed:.1f}s")

run_elapsed = time.perf_counter() - run_start
print(f"\n{'='*50}")
print(f"ALL DONE — {total_ok} OK, {total_fail} FAIL in {run_elapsed/60:.1f} min")
print(f"Transcripts: {OUT_DIR} ({len(list(OUT_DIR.glob('*.jsonl')))} JSONL files)")

# === CELL 4: Copy results to /kaggle/working for download ===
# CELL 4 — Move transcripts to working dir (JSONL files are tiny, ~500KB total)
# !mkdir -p /kaggle/working/transcripts && cp /kaggle/tmp/transcripts/*.jsonl /kaggle/working/transcripts/
# !cd /kaggle/working && zip -r transcripts_whisper_32videos.zip transcripts/
# from IPython.display import FileLink
# FileLink("transcripts_whisper_32videos.zip")
