"""Stage 1c redo: ONNX export + actual ONNX Runtime inference benchmark,
isolated into its own process (the original attempt shared a process with
1b's torch.compile step and hit a CUDA OOM from ~14GB of leftover VRAM that
torch.compile never released -- not a CPU-vs-GPU build issue, just process
isolation). That original attempt also never got past export() to actually
benchmark ONNX Runtime inference speed, which is the actually interesting
number.

Export traces on CPU (dummy input stays on CPU) -- tracing doesn't need to be
fast and this removes ANY GPU memory risk from the export step entirely.
Inference then runs through onnxruntime's CUDAExecutionProvider to measure
real GPU inference speed, compared against the FP32/FP16 baselines already
measured in bench_fp16_vs_fp32.py (0.78 img/s / 3.07 img/s).
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnx import version_converter
from PIL import Image

VISION_ONLY_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"
KEYFRAME_DIR = Path("/content/pe_core_batch_test")
N_IMAGES = 120
BATCH_SIZE = 8
ONNX_PATH = "/content/pe_core_visual.onnx"
ONNX_FIXED_PATH = "/content/pe_core_visual_fixed.onnx"


def read_mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return -1


_stop = threading.Event()


def memory_watcher(tag):
    while not _stop.is_set():
        print(f"[mem-watch:{tag}] sys_avail={read_mem_available_mb():.0f}MB t={time.perf_counter():.1f}s", flush=True)
        _stop.wait(5)


image_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))[:N_IMAGES]
print(f"n_images={len(image_paths)}, sys_avail_at_start={read_mem_available_mb():.0f}MB", flush=True)
print(f"onnxruntime providers available: {ort.get_available_providers()}", flush=True)

print("\n=== step 1: load vision-only model on CPU, build preprocess ===", flush=True)
t0 = time.perf_counter()
model = timm.create_model(VISION_ONLY_ID, pretrained=True).eval()
data_cfg = timm.data.resolve_data_config({}, model=model)
preprocess = timm.data.create_transform(**data_cfg)
print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

print("\n=== step 2: export to ONNX (CPU tracing, zero GPU memory risk) ===", flush=True)
print("dynamo=True (torch.export-based exporter) -- dynamo=False's legacy tracer hits a real "
      "incompatibility: timm's EVA attention passes is_causal as a traced Tensor into "
      "F.scaled_dot_product_attention, which requires a Python bool.", flush=True)
dummy = torch.randn(1, 3, 448, 448)  # CPU tensor -- traces on CPU, no CUDA involvement at all
_stop.clear()
watcher = threading.Thread(target=memory_watcher, args=("onnx_export",), daemon=True)
watcher.start()
t0 = time.perf_counter()
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["image"], output_names=["embedding"], opset_version=18,
    dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
    dynamo=True,
)
_stop.set()
watcher.join(timeout=1)
print(f"export done in {time.perf_counter()-t0:.1f}s, sys_avail={read_mem_available_mb():.0f}MB", flush=True)
del model
import gc
gc.collect()

print("\n=== step 2b: fix malformed opset via ONNX's official version_converter ===", flush=True)
print("The exporter's own internal opset downgrade left a Split node with a 'num_outputs' "
      "attribute (only valid in opset 18) inside a graph declared as opset 17 -- onnxruntime "
      "rejects it either way (confirmed: same error at opset_version=18 too, since the exporter's "
      "declared opset stayed 17 regardless of what was requested). Running onnx's own "
      "version_converter (not torch's internal one) to properly rewrite it. Graph-only operation, "
      "does not need the 6.5GB of external weight data loaded.", flush=True)
t0 = time.perf_counter()
graph_model = onnx.load(ONNX_PATH, load_external_data=False)
print(f"loaded graph, declared opset: {[(o.domain, o.version) for o in graph_model.opset_import]}", flush=True)
fixed_model = version_converter.convert_version(graph_model, 17)
onnx.save(fixed_model, ONNX_FIXED_PATH, save_as_external_data=False)
print(f"conversion done in {time.perf_counter()-t0:.1f}s", flush=True)

print("\n=== step 3: load ONNX Runtime session (CUDA execution provider) ===", flush=True)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
print(f"using providers: {providers}", flush=True)
t0 = time.perf_counter()
session = ort.InferenceSession(ONNX_FIXED_PATH, providers=providers)
print(f"session created in {time.perf_counter()-t0:.1f}s, actual provider: {session.get_providers()}", flush=True)

print("\n=== step 4: embed 120 images, compare to FP32 ground truth ===", flush=True)
gt_path = Path("/content/gt_vecs_120.npy")
gt_vecs = np.load(gt_path) if gt_path.exists() else None
if gt_vecs is None:
    print("WARNING: gt_vecs_120.npy not found -- skipping cosine/recall comparison, speed-only run", flush=True)


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


input_name = session.get_inputs()[0].name
n_total = len(image_paths)
n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
vecs = []
t0 = time.perf_counter()
for i, start in enumerate(range(0, n_total, BATCH_SIZE)):
    batch_paths = image_paths[start:start + BATCH_SIZE]
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).numpy()
    raw = session.run(None, {input_name: batch})[0]
    normed = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    vecs.append(normed)
    elapsed = time.perf_counter() - t0
    done = min(start + BATCH_SIZE, n_total)
    print(f"  batch {i+1}/{n_batches}: {done}/{n_total} images, {elapsed:.1f}s elapsed, {done/elapsed:.2f} img/s so far", flush=True)
elapsed = time.perf_counter() - t0
vecs = np.concatenate(vecs, axis=0)

result = {"variant": "fp32_onnx_runtime", "elapsed_s": elapsed, "img_per_s": N_IMAGES / elapsed, "providers": session.get_providers()}
if gt_vecs is not None:
    cos = np.sum(vecs * gt_vecs, axis=1)
    result.update({
        "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()),
        "cosine_p1": float(np.percentile(cos, 1)),
        "recall10_overlap": recall_at_10_overlap(gt_vecs, vecs),
    })

print("\n=== RESULT ===", flush=True)
print(json.dumps(result, indent=2), flush=True)
Path("/content/onnx_result.json").write_text(json.dumps(result, indent=2))
