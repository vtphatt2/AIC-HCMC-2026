"""Load ONLY the full open_clip CLIP model (never in the same process as
extract_vision_only_vec.py -- see that file's docstring for why). Embed the
SAME keyframes via model.encode_image(), save vectors + param counts for
comparison against the vision-only checkpoint's output.

Instrumented after 2 back-to-back full VM crashes (gpu2, then gpu3) with ZERO
log output between script start and the crash -- the entire
open_clip.create_model_and_transforms() call was one opaque black box, so we
had no idea whether it died during download, CPU weight construction (random
init + ~9GB state dict can transiently coexist at ~18GB), or GPU transfer.
A background thread now prints RSS + system-available memory every 5s
THROUGHOUT that call (not just before/after), so if it crashes again the log
shows the last known-good memory reading right up to the moment of death.
"""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import snapshot_download
from PIL import Image

FULL_CLIP_ID = "hf-hub:timm/PE-Core-bigG-14-448"
KEYFRAME_DIR = Path("/content/pe_core_batch_test")
N_TEST = 5
OUT_DIR = Path("/content/vision_only_compare")
OUT_DIR.mkdir(exist_ok=True)


def read_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1


def read_mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return -1


def read_disk_free_gb(path="/content"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


_stop = threading.Event()


def memory_watcher(tag):
    while not _stop.is_set():
        print(f"[mem-watch:{tag}] rss={read_rss_mb():.0f}MB sys_avail={read_mem_available_mb():.0f}MB "
              f"t={time.perf_counter():.1f}s", flush=True)
        _stop.wait(5)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


print(f"sys_avail_at_start={read_mem_available_mb():.0f}MB disk_free={read_disk_free_gb():.1f}GB", flush=True)

image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_TEST]
print(f"n_test_images={len(image_paths)}", flush=True)

print("=== stage 1: download only (no torch model, minimal RAM) ===", flush=True)
_stop.clear()
watcher = threading.Thread(target=memory_watcher, args=("download",), daemon=True)
watcher.start()
t0 = time.perf_counter()
snapshot_download(repo_id="timm/PE-Core-bigG-14-448")
dl_s = time.perf_counter() - t0
_stop.set()
watcher.join(timeout=1)
print(f"download stage done: {dl_s:.1f}s, disk_free_after={read_disk_free_gb():.1f}GB", flush=True)

print("\n=== stage 2: construct model from local cache (memory-heavy stage) ===", flush=True)
_stop.clear()
watcher = threading.Thread(target=memory_watcher, args=("model_create",), daemon=True)
watcher.start()

t0 = time.perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms(FULL_CLIP_ID, precision="fp32", device="cuda")
model.eval()
load_s = time.perf_counter() - t0

_stop.set()
watcher.join(timeout=1)

total_params = count_params(model)
visual_params = count_params(model.visual)
print(f"full CLIP load: {load_s:.1f}s, total_params={total_params:,}, visual_params={visual_params:,}", flush=True)
print(f"sys_avail_after_load={read_mem_available_mb():.0f}MB gpu_alloc={torch.cuda.memory_allocated()/1e6:.0f}MB", flush=True)

vecs = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    x = preprocess(img).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        vec = model.encode_image(x, normalize=True).float().cpu().numpy()[0]
    vecs.append(vec)
    print(f"  embedded {p.name} shape={vec.shape}", flush=True)

vecs = np.stack(vecs)
np.save(OUT_DIR / "full_clip_visual_vecs.npy", vecs)
meta = {
    "total_params": total_params, "visual_params": visual_params, "load_s": load_s,
    "image_names": [p.name for p in image_paths], "shape": list(vecs.shape),
}
(OUT_DIR / "full_clip_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
print(f"\nSaved to {OUT_DIR}. If vision_only_vecs.npy is also present, comparing now:", flush=True)

vision_only_path = OUT_DIR / "vision_only_vecs.npy"
if vision_only_path.exists():
    vo = np.load(vision_only_path)
    if vo.shape == vecs.shape:
        cos = np.sum(vo * vecs, axis=1)
        print(f"cosine_sim per image: {cos.tolist()}", flush=True)
        print(f"max_abs_diff: {float(np.abs(vo - vecs).max()):.6f}", flush=True)
    else:
        print(f"SHAPE MISMATCH: vision_only={vo.shape} vs full_clip={vecs.shape}", flush=True)
else:
    print("vision_only_vecs.npy not found yet -- run extract_vision_only_vec.py and copy its output here to compare.", flush=True)
