"""Load ONLY the vision-only timm checkpoint (no text tower, no full CLIP
model in the same process -- loading both in one process is what likely
crashed the Colab VM last time). Embed a handful of real keyframes, save
vectors + param count for later comparison against the full-CLIP .visual
output (extract_full_clip_visual_vec.py), computed in a SEPARATE process/
session so the two ~8-9GB models are never resident at the same time.
"""
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image

VISION_ONLY_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"
KEYFRAME_DIR = Path("/content/pe_core_batch_test")
N_TEST = 5
OUT_DIR = Path("/content/vision_only_compare")
OUT_DIR.mkdir(exist_ok=True)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_TEST]
print(f"n_test_images={len(image_paths)}", flush=True)

t0 = time.perf_counter()
model = timm.create_model(VISION_ONLY_ID, pretrained=True).to("cuda").eval()
load_s = time.perf_counter() - t0
params = count_params(model)
print(f"vision-only load: {load_s:.1f}s, params={params:,}", flush=True)

data_cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**data_cfg)
print(f"data config: {data_cfg}", flush=True)

vecs = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    x = transform(img).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        raw = model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        vec = F.normalize(raw, dim=-1).float().cpu().numpy()[0]
    vecs.append(vec)
    print(f"  embedded {p.name} shape={vec.shape}", flush=True)

import numpy as np
vecs = np.stack(vecs)
np.save(OUT_DIR / "vision_only_vecs.npy", vecs)
meta = {"params": params, "load_s": load_s, "image_names": [p.name for p in image_paths], "shape": list(vecs.shape)}
(OUT_DIR / "vision_only_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
print(f"\nSaved to {OUT_DIR} -- download this whole directory, then run "
      f"extract_full_clip_visual_vec.py in a SEPARATE session/process before comparing.", flush=True)
