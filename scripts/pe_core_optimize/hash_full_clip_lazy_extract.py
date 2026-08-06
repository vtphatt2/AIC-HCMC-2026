"""Side B of the split weight-verification: extract ONLY the vision tower
from the full PE-Core safetensors checkpoint via the fake-text-branch +
safetensors-lazy-load trick (mirrored from D:\\Downloads\\textencoderextract.py,
which used fake-vision + real-text to pull just the text tower; here it's
fake-text + real-vision). Avoids open_clip.create_model_and_transforms()'s
full materialization, suspected root cause of 2 back-to-back full VM crashes
earlier today. Hashes model.visual's state_dict + embeds the same 5 test
images used by hash_vision_only_checkpoint.py. Run on a SEPARATE GPU session
from that script so the two ~7.5GB checkpoints never coexist in one process.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

FULL_MODEL_ID = "timm/PE-Core-bigG-14-448"
TIMM_ARCH_NAME = "vit_pe_core_gigantic_patch14_448"
KEYFRAME_DIR = Path("/content/pe_core_batch_test")
N_TEST = 5
OUT_DIR = Path("/content/weight_verify")
OUT_DIR.mkdir(exist_ok=True)


def read_mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return -1


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


_stop = threading.Event()


def memory_watcher(tag):
    while not _stop.is_set():
        print(f"[mem-watch:{tag}] sys_avail={read_mem_available_mb():.0f}MB t={time.perf_counter():.1f}s", flush=True)
        _stop.wait(5)


print(f"sys_avail_at_start={read_mem_available_mb():.0f}MB", flush=True)

print("\n=== stage 1: build fake-text / real-vision architecture (CPU, no weights) ===", flush=True)
t0 = time.perf_counter()
full_cfg = open_clip.get_model_config(f"hf-hub:{FULL_MODEL_ID}")
embed_dim = full_cfg["embed_dim"]
real_vision_cfg = full_cfg["vision_cfg"]
fake_text_cfg = {"context_length": 1, "vocab_size": 2, "width": 8, "heads": 1, "layers": 1}
model = open_clip.CustomTextCLIP(embed_dim=embed_dim, vision_cfg=real_vision_cfg, text_cfg=fake_text_cfg)
model.eval()
print(f"built in {time.perf_counter()-t0:.1f}s, visual_params={sum(p.numel() for p in model.visual.parameters()):,}, "
      f"sys_avail={read_mem_available_mb():.0f}MB", flush=True)

print("\n=== stage 2: download full checkpoint (disk only, proven safe) ===", flush=True)
_stop.clear()
watcher = threading.Thread(target=memory_watcher, args=("download",), daemon=True)
watcher.start()
t0 = time.perf_counter()
weight_path = hf_hub_download(repo_id=FULL_MODEL_ID, filename="open_clip_model.safetensors")
_stop.set()
watcher.join(timeout=1)
print(f"download done in {time.perf_counter()-t0:.1f}s, sys_avail={read_mem_available_mb():.0f}MB", flush=True)

print("\n=== stage 3: lazy-load ONLY visual.* tensors ===", flush=True)
_stop.clear()
watcher = threading.Thread(target=memory_watcher, args=("lazy_load",), daemon=True)
watcher.start()
t0 = time.perf_counter()
visual_state = {}
with safe_open(weight_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    visual_keys = [k for k in all_keys if k.startswith("visual.")]
    print(f"total keys: {len(all_keys)}, visual.* keys: {len(visual_keys)}", flush=True)
    for i, k in enumerate(visual_keys):
        visual_state[k[len("visual."):]] = f.get_tensor(k)
        if i % 100 == 0:
            print(f"  loaded {i}/{len(visual_keys)}, sys_avail={read_mem_available_mb():.0f}MB", flush=True)
_stop.set()
watcher.join(timeout=1)
print(f"lazy-load done in {time.perf_counter()-t0:.1f}s, sys_avail={read_mem_available_mb():.0f}MB", flush=True)

print("\n=== stage 4: load into model.visual ===", flush=True)
try:
    missing, unexpected = model.visual.load_state_dict(visual_state, strict=True)
    print("strict=True load succeeded, no missing/unexpected keys", flush=True)
except Exception as e:
    print(f"strict=True FAILED ({e}), retrying with strict=False", flush=True)
    result = model.visual.load_state_dict(visual_state, strict=False)
    print(f"missing keys: {result.missing_keys}", flush=True)
    print(f"unexpected keys: {result.unexpected_keys}", flush=True)
del visual_state

print("\n=== stage 5: hash model.visual state_dict ===", flush=True)
t0 = time.perf_counter()
state = model.visual.state_dict()
by_key, by_content = [], []
for i, (k, v) in enumerate(state.items()):
    h = tensor_hash(v)
    by_key.append({"key": k, "shape": list(v.shape), "hash": h})
    by_content.append((list(v.shape), h))
    if i % 100 == 0:
        print(f"  hashed {i}/{len(state)}  {time.perf_counter()-t0:.1f}s", flush=True)
by_content_sorted = sorted(by_content, key=lambda x: (str(x[0]), x[1]))
combined_hash = hashlib.sha256(json.dumps(by_content_sorted).encode()).hexdigest()
print(f"hashing done in {time.perf_counter()-t0:.1f}s, combined_content_hash={combined_hash}", flush=True)

(OUT_DIR / "full_clip_visual_hashes.json").write_text(json.dumps({
    "by_key": by_key, "by_content_sorted": by_content_sorted,
    "combined_content_hash": combined_hash, "n_tensors": len(state),
}))
print(f"saved hashes to {OUT_DIR / 'full_clip_visual_hashes.json'}", flush=True)

print("\n=== stage 6: embed test images (GPU) ===", flush=True)
model = model.to("cuda")
pcfg = timm.get_pretrained_cfg(f"{TIMM_ARCH_NAME}.fb").to_dict()
data_cfg = timm.data.resolve_data_config({}, pretrained_cfg=pcfg)
transform = timm.data.create_transform(**data_cfg)

image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_TEST]
vecs = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    x = transform(img).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        vec = model.encode_image(x, normalize=True).float().cpu().numpy()[0]
    vecs.append(vec)
    print(f"  embedded {p.name} shape={vec.shape}", flush=True)

vecs = np.stack(vecs)
np.save(OUT_DIR / "full_clip_visual_embed_vecs.npy", vecs)
(OUT_DIR / "full_clip_visual_embed_meta.json").write_text(json.dumps({
    "image_names": [p.name for p in image_paths], "shape": list(vecs.shape),
}))
print(f"\nDONE. Download {OUT_DIR} and compare against hash_vision_only_checkpoint.py's output.", flush=True)
