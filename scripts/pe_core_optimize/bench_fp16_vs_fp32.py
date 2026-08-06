"""Stage 1: sweep FP32 baseline, FP32+torch.compile, FP32+ONNX (best-effort,
time-boxed), and FP16 on a 120-image subset. Report cosine similarity +
Recall@10-overlap against the FP32 baseline (computed fresh here) + throughput
for each variant, in one table -- before deciding whether the custom Triton
kernel (Stage 2/3) is worth pursuing.

Loader switched from open_clip.create_model_and_transforms(full CLIP) to
timm.create_model(vision-only checkpoint) after 2 back-to-back full VM
crashes today loading the full model (real vision + real text random-init +
torch.load()-style full checkpoint materialization, suspected ~18GB CPU RAM
peak on a free-tier VM). verify_vision_weights_exact.py + hash_vision_only_checkpoint.py
/ hash_full_clip_lazy_extract.py confirmed the vision-only checkpoint is
byte-identical (sha256 match) to the full checkpoint's visual branch, and
cosine=1.0 / max_abs_diff=2e-07 on real images -- so this loses nothing and
removes the crash risk entirely (proven safe 3x today, ~59s on a cached
download, no text tower ever constructed).
"""
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
N_IMAGES = 120
BATCH_SIZE = 8

image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_IMAGES]
print(f"n_images={len(image_paths)}", flush=True)

gt_vecs = None
results = []


def load_vision_only(dtype=torch.float32):
    model = timm.create_model(VISION_ONLY_ID, pretrained=True).to("cuda", dtype=dtype).eval()
    data_cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**data_cfg)
    return model, preprocess


def encode_with(model):
    def _encode(batch):
        raw = model(batch)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return F.normalize(raw, dim=-1)
    return _encode


def embed_all(encode_fn, preprocess, device, dtype):
    """encode_fn(batch_tensor) -> normalized embeddings tensor on GPU."""
    n_total = len(image_paths)
    n_batches_total = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    vecs = []
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, n_total, BATCH_SIZE)):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device, dtype=dtype)
        with torch.inference_mode():
            feats = encode_fn(batch)
        vecs.append(feats.float().cpu().numpy())
        elapsed = time.perf_counter() - t0
        done = min(start + BATCH_SIZE, n_total)
        print(f"  batch {i+1}/{n_batches_total}: {done}/{n_total} images, {elapsed:.1f}s elapsed, {done/elapsed:.2f} img/s so far", flush=True)
    elapsed = time.perf_counter() - t0
    return np.concatenate(vecs, axis=0), elapsed


def recall_at_10_overlap(ground_truth, variant):
    gt_sim = ground_truth @ ground_truth.T
    var_sim = variant @ ground_truth.T
    n = ground_truth.shape[0]
    overlaps = []
    for i in range(n):
        gt_top10 = set(np.argsort(-gt_sim[i])[:10])
        var_top10 = set(np.argsort(-var_sim[i])[:10])
        overlaps.append(len(gt_top10 & var_top10) / 10.0)
    return float(np.mean(overlaps))


def compare_to_gt(vecs, label, elapsed_s, extra=None):
    cos = np.sum(vecs * gt_vecs, axis=1)  # both L2-normalized -> dot == cosine
    r10 = recall_at_10_overlap(gt_vecs, vecs)
    row = {
        "variant": label,
        "elapsed_s": elapsed_s,
        "img_per_s": N_IMAGES / elapsed_s,
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_p1": float(np.percentile(cos, 1)),
        "recall10_overlap": r10,
    }
    if extra:
        row.update(extra)
    results.append(row)
    print(json.dumps(row, indent=2), flush=True)
    save_results()
    return row


def save_results():
    # Write after every sub-stage, not just at the end -- a VM-level crash
    # must not lose earlier results.
    Path("/content/stage1_sweep_results.json").write_text(json.dumps(results, indent=2))


# --- 1a. FP32 baseline (also serves as ground truth) ---
print("=== 1a. FP32 baseline ===", flush=True)
model, preprocess = load_vision_only(torch.float32)
gt_vecs, fp32_s = embed_all(encode_with(model), preprocess, "cuda", torch.float32)
results.append({
    "variant": "fp32_baseline", "elapsed_s": fp32_s, "img_per_s": N_IMAGES / fp32_s,
    "cosine_mean": 1.0, "cosine_min": 1.0, "cosine_p1": 1.0, "recall10_overlap": 1.0,
})
print(f"fp32 baseline: {fp32_s:.1f}s ({N_IMAGES/fp32_s:.2f} img/s)", flush=True)
np.save("/content/gt_vecs_120.npy", gt_vecs)
save_results()
del model
torch.cuda.empty_cache()

# --- 1b. FP32 + torch.compile ---
print("=== 1b. FP32 + torch.compile ===", flush=True)
try:
    model, preprocess = load_vision_only(torch.float32)
    # "max-autotune" benchmarks many candidate kernel configs during compile,
    # which spiked memory enough to crash the whole Colab VM (not just a
    # python-catchable OOM) on a 2B-param model + free-tier T4 15GB VRAM.
    # "default" still fuses ops but skips the heavy autotune search.
    compiled = torch.compile(model, mode="default")
    compiled_encode = encode_with(compiled)

    # test small first (CLAUDE.md rule) -- compile warmup on 10 images before the full 120
    warm_batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in image_paths[:10]]).to("cuda")
    t_warm = time.perf_counter()
    with torch.inference_mode():
        _ = compiled_encode(warm_batch)
    print(f"compile warmup (10 imgs, incl. autotune): {time.perf_counter()-t_warm:.1f}s", flush=True)

    vecs, s = embed_all(compiled_encode, preprocess, "cuda", torch.float32)
    compare_to_gt(vecs, "fp32_torch_compile", s)
    del model, compiled
    torch.cuda.empty_cache()
except Exception as e:
    print(f"torch.compile FAILED: {e}", flush=True)
    results.append({"variant": "fp32_torch_compile", "error": str(e)})
    save_results()
    torch.cuda.empty_cache()

# --- 1c. FP32 + ONNX Runtime (best-effort export attempt, time-boxed) ---
print("=== 1c. FP32 + ONNX (best-effort export attempt) ===", flush=True)
try:
    model, preprocess = load_vision_only(torch.float32)
    dummy = torch.randn(1, 3, 448, 448, device="cuda")
    t_export = time.perf_counter()
    torch.onnx.export(
        model, dummy, "/content/pe_core_visual.onnx",
        input_names=["image"], output_names=["embedding"], opset_version=17,
    )
    print(f"ONNX export succeeded: {time.perf_counter()-t_export:.1f}s -- inference loop not yet wired, follow up if this path is worth pursuing", flush=True)
    results.append({"variant": "fp32_onnx", "note": "export succeeded, inference loop not yet implemented"})
    save_results()
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"ONNX export FAILED (real risk noted in the plan, skipping per time-box): {e}", flush=True)
    results.append({"variant": "fp32_onnx", "error": str(e)[:500]})
    save_results()
    torch.cuda.empty_cache()

# --- 1d. FP16 ---
print("=== 1d. FP16 ===", flush=True)
model, preprocess = load_vision_only(torch.float16)
vecs, s = embed_all(encode_with(model), preprocess, "cuda", torch.float16)
compare_to_gt(vecs, "fp16", s)
del model
torch.cuda.empty_cache()

print("\n=== FINAL TABLE ===", flush=True)
print(json.dumps(results, indent=2), flush=True)
Path("/content/stage1_sweep_results.json").write_text(json.dumps(results, indent=2))
