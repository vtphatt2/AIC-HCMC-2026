"""Colab side: watch a directory for incoming packet tar files (sent by
pack_and_send_packets.py) and embed each packet's images as soon as it
arrives, instead of waiting for the whole transfer to finish first.
"""
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

INCOMING_DIR = Path("/content/packets_incoming")
EXTRACT_DIR = Path("/content/pe_core_batch_test")
OUT_DIR = Path("/content/pe_core_batch_test_embeddings")
MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"
BATCH_SIZE = 8
POLL_INTERVAL = 1.0

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = INCOMING_DIR / "manifest.json"
print("waiting for manifest...", flush=True)
while not manifest_path.exists():
    time.sleep(POLL_INTERVAL)
manifest = json.loads(manifest_path.read_text())
n_packets = manifest["n_packets"]
n_images_total = manifest["n_images"]
print(f"manifest: {n_images_total} images, {n_packets} packets", flush=True)

t0 = time.perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID, precision="fp32", device="cuda")
model.eval()
load_s = time.perf_counter() - t0
print(f"model loaded in {load_s:.1f}s", flush=True)

processed = set()
n_embedded = 0
batch_imgs, batch_names = [], []


def flush_batch():
    global n_embedded
    if not batch_imgs:
        return
    batch = torch.stack(batch_imgs).to("cuda")
    with torch.inference_mode():
        feats = model.encode_image(batch, normalize=True)
    vecs = feats.detach().float().cpu().numpy()
    for name, vec in zip(batch_names, vecs):
        np.save(OUT_DIR / f"{name}.npy", vec.astype("float32"))
    n_embedded += len(batch_names)
    batch_imgs.clear()
    batch_names.clear()


t1 = time.perf_counter()
while len(processed) < n_packets:
    made_progress = False
    for entry in manifest["packets"]:
        idx = entry["index"]
        if idx in processed:
            continue
        done_marker = INCOMING_DIR / f"{entry['file']}.done"
        if not done_marker.exists():
            continue
        packet_file = INCOMING_DIR / entry["file"]
        with tarfile.open(packet_file, "r") as tar:
            names = tar.getnames()
            tar.extractall(EXTRACT_DIR, filter="data")
        for name in names:
            pil = Image.open(EXTRACT_DIR / name).convert("RGB")
            batch_imgs.append(preprocess(pil))
            batch_names.append(Path(name).stem)
            if len(batch_imgs) == BATCH_SIZE:
                flush_batch()
        processed.add(idx)
        made_progress = True
        elapsed = time.perf_counter() - t1
        print(f"packet {idx} done, embedded {n_embedded}/{n_images_total}, elapsed={elapsed:.1f}s", flush=True)
    if not made_progress:
        time.sleep(POLL_INTERVAL)

flush_batch()
process_s = time.perf_counter() - t1
total_s = time.perf_counter() - t0
summary = {
    "load_s": load_s,
    "process_s": process_s,
    "total_s": total_s,
    "n_embedded": n_embedded,
    "n_expected": n_images_total,
}
print(json.dumps(summary, indent=2), flush=True)
Path("/content/packet_pipeline_summary.json").write_text(json.dumps(summary, indent=2))
