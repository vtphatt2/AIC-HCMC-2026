"""Side A of the split weight-verification: load ONLY the vision-only timm
checkpoint, hash its state_dict (per-tensor, permutation-invariant), and
embed the same 5 test images. Saves tiny artifacts only (hash JSON + small
.npy) -- meant to run on a SEPARATE GPU session from hash_full_clip_lazy_extract.py
so the two ~7.5GB checkpoints never coexist in the same process/machine.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

VISION_ONLY_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"
KEYFRAME_DIR = Path("/content/pe_core_batch_test")
N_TEST = 5
OUT_DIR = Path("/content/weight_verify")
OUT_DIR.mkdir(exist_ok=True)


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


print("=== load vision-only timm checkpoint ===", flush=True)
t0 = time.perf_counter()
model = timm.create_model(VISION_ONLY_ID, pretrained=True).eval()
print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

state = model.state_dict()
print(f"state_dict tensors: {len(state)}", flush=True)

print("\n=== hashing (per-tensor, may take a bit for 1.88B params) ===", flush=True)
t0 = time.perf_counter()
by_key = []
by_content = []
for i, (k, v) in enumerate(state.items()):
    h = tensor_hash(v)
    by_key.append({"key": k, "shape": list(v.shape), "hash": h})
    by_content.append((list(v.shape), h))
    if i % 50 == 0:
        print(f"  hashed {i}/{len(state)}  {time.perf_counter()-t0:.1f}s", flush=True)
by_content_sorted = sorted(by_content, key=lambda x: (str(x[0]), x[1]))
combined_hash = hashlib.sha256(json.dumps(by_content_sorted).encode()).hexdigest()
print(f"hashing done in {time.perf_counter()-t0:.1f}s, combined_content_hash={combined_hash}", flush=True)

(OUT_DIR / "vision_only_hashes.json").write_text(json.dumps({
    "by_key": by_key,
    "by_content_sorted": by_content_sorted,
    "combined_content_hash": combined_hash,
    "n_tensors": len(state),
}))
print(f"saved hashes to {OUT_DIR / 'vision_only_hashes.json'}", flush=True)

print("\n=== embed test images ===", flush=True)
data_cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_cfg)
image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_TEST]
vecs = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.inference_mode():
        raw = model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        vec = F.normalize(raw, dim=-1).float().numpy()[0]
    vecs.append(vec)
    print(f"  embedded {p.name} shape={vec.shape}", flush=True)

vecs = np.stack(vecs)
np.save(OUT_DIR / "vision_only_embed_vecs.npy", vecs)
(OUT_DIR / "vision_only_embed_meta.json").write_text(json.dumps({
    "image_names": [p.name for p in image_paths], "shape": list(vecs.shape),
}))
print(f"\nDONE. Download {OUT_DIR} and compare against hash_full_clip_lazy_extract.py's output.", flush=True)
