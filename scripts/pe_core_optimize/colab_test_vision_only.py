"""Verify timm's vision-only PE-Core checkpoint (no text tower) against the
full open_clip CLIP model on the same image, before trusting it as a
drop-in replacement:
 - does parameter count match model.visual's?
 - does output shape match the 1280-dim shared embedding space?
 - do the actual embedding values match (same weights, or just same arch)?
 - how much smaller/faster is load time without the text tower?
"""
import time

import numpy as np
import open_clip
import timm
import torch
import torch.nn.functional as F
from PIL import Image

VISION_ONLY_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"
FULL_CLIP_ID = "hf-hub:timm/PE-Core-bigG-14-448"
TEST_IMAGE = "/content/pe_core_batch_test/000020.jpg"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


print("=== Loading timm vision-only checkpoint ===", flush=True)
t0 = time.perf_counter()
vision_model = timm.create_model(VISION_ONLY_ID, pretrained=True).to("cuda").eval()
vision_load_s = time.perf_counter() - t0
vision_params = count_params(vision_model)
print(f"vision-only load: {vision_load_s:.1f}s, params={vision_params:,}", flush=True)

data_cfg = timm.data.resolve_data_config({}, model=vision_model)
vision_transform = timm.data.create_transform(**data_cfg)
print(f"timm data config: {data_cfg}", flush=True)

img = Image.open(TEST_IMAGE).convert("RGB")
vision_input = vision_transform(img).unsqueeze(0).to("cuda")

with torch.inference_mode():
    vision_raw_out = vision_model(vision_input)
    if isinstance(vision_raw_out, (tuple, list)):
        print(f"vision model returned {len(vision_raw_out)} outputs, shapes={[o.shape for o in vision_raw_out]}", flush=True)
        vision_raw_out = vision_raw_out[0]
    print(f"vision-only raw output shape: {tuple(vision_raw_out.shape)}", flush=True)
    vision_out = F.normalize(vision_raw_out, dim=-1).float().cpu().numpy()

del vision_model
torch.cuda.empty_cache()

print("=== Loading full open_clip CLIP model for comparison ===", flush=True)
t1 = time.perf_counter()
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(FULL_CLIP_ID, precision="fp32", device="cuda")
clip_model.eval()
clip_load_s = time.perf_counter() - t1
clip_total_params = count_params(clip_model)
clip_visual_params = count_params(clip_model.visual)
print(f"full CLIP load: {clip_load_s:.1f}s, total_params={clip_total_params:,}, visual_params={clip_visual_params:,}", flush=True)

clip_input = clip_preprocess(img).unsqueeze(0).to("cuda")
with torch.inference_mode():
    clip_out = clip_model.encode_image(clip_input, normalize=True).float().cpu().numpy()
print(f"full CLIP encode_image output shape: {clip_out.shape}", flush=True)

print("\n=== Comparison ===", flush=True)
print(f"param count match (vision-only vs model.visual): {vision_params == clip_visual_params} ({vision_params:,} vs {clip_visual_params:,})", flush=True)
print(f"output shape match: {vision_out.shape == clip_out.shape} ({vision_out.shape} vs {clip_out.shape})", flush=True)
if vision_out.shape == clip_out.shape:
    max_abs_diff = float(np.abs(vision_out - clip_out).max())
    cos_sim = float((vision_out @ clip_out.T).item())
    print(f"max_abs_diff={max_abs_diff:.6f} cosine_similarity={cos_sim:.6f}", flush=True)
    print(f"identical (atol=1e-4): {np.allclose(vision_out, clip_out, atol=1e-4)}", flush=True)
else:
    print("SHAPE MISMATCH -- vision-only checkpoint output is not directly comparable, would need manual projection or different pooling to align.", flush=True)

print(f"\nload_time_saved_s = {clip_load_s - vision_load_s:.1f}s (vision-only vs full CLIP)", flush=True)
