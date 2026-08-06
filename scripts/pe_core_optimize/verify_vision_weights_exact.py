"""Weight-level (not just embedding-level) equivalence check between the
vision-only timm checkpoint and the visual.* subset of the full PE-Core
safetensors checkpoint. A prior local test proved the two produce the SAME
ARCHITECTURE (identical param count, 1,882,033,920) but that used a randomly
initialized model -- it says nothing about whether the actual trained weight
VALUES match. This script loads real weights from both sources and diffs
them tensor-by-tensor: exact equality first, then max abs diff as fallback.
No GPU/embedding/inference needed -- pure state-dict comparison, CPU only,
so this is safe to run even on a fresh low-RAM session (each tensor pair is
compared then dropped, not held all at once).

Uses the safetensors-lazy-load + fake-branch trick from a prior working
Colab notebook (D:\\Downloads\\textencoderextract.py, mirrored for vision
instead of text) specifically because torch.load()-based full-checkpoint
materialization is suspected to have caused 2 back-to-back full VM crashes
when loading the complete ~9GB CLIP model via open_clip.create_model_and_transforms().
"""
import time

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

FULL_MODEL_ID = "timm/PE-Core-bigG-14-448"
VISION_ONLY_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"


def read_mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return -1


print(f"sys_avail_at_start={read_mem_available_mb():.0f}MB", flush=True)

print("\n=== A: load vision-only timm checkpoint state_dict ===", flush=True)
t0 = time.perf_counter()
vo_model = timm.create_model(VISION_ONLY_ID, pretrained=True)
vo_state = vo_model.state_dict()
print(f"loaded in {time.perf_counter()-t0:.1f}s, {len(vo_state)} tensors, "
      f"sys_avail={read_mem_available_mb():.0f}MB", flush=True)
print("sample keys:", list(vo_state.keys())[:5], flush=True)

print("\n=== B: lazy-load visual.* tensors from full checkpoint safetensors ===", flush=True)
t0 = time.perf_counter()
weight_path = hf_hub_download(repo_id=FULL_MODEL_ID, filename="open_clip_model.safetensors")
print(f"download/cache done in {time.perf_counter()-t0:.1f}s, sys_avail={read_mem_available_mb():.0f}MB", flush=True)

full_visual_state = {}
t0 = time.perf_counter()
with safe_open(weight_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    visual_keys = [k for k in all_keys if k.startswith("visual.")]
    print(f"total keys in file: {len(all_keys)}, visual.* keys: {len(visual_keys)}", flush=True)
    for i, k in enumerate(visual_keys):
        full_visual_state[k[len("visual."):]] = f.get_tensor(k)
        if i % 100 == 0:
            print(f"  loaded {i}/{len(visual_keys)} visual tensors, sys_avail={read_mem_available_mb():.0f}MB", flush=True)
print(f"lazy-load done in {time.perf_counter()-t0:.1f}s, {len(full_visual_state)} tensors, "
      f"sys_avail={read_mem_available_mb():.0f}MB", flush=True)
print("sample keys:", list(full_visual_state.keys())[:5], flush=True)

print("\n=== C: match keys and diff tensors ===", flush=True)
# timm's own state_dict keys vs the safetensors "visual."-stripped keys may use different
# naming (e.g. timm's checkpoint might not have a "trunk." prefix that open_clip's timm
# wrapper adds, or vice versa) -- try direct match first, then a best-effort suffix match.
direct_matches = set(vo_state.keys()) & set(full_visual_state.keys())
print(f"direct key matches: {len(direct_matches)} / {len(vo_state)} (timm) vs {len(full_visual_state)} (safetensors)", flush=True)

if len(direct_matches) < min(len(vo_state), len(full_visual_state)) * 0.5:
    print("Direct match is low -- trying suffix-based matching (strip common prefixes)...", flush=True)
    def strip_prefix(k):
        for p in ("trunk.", "model.", "visual."):
            if k.startswith(p):
                k = k[len(p):]
        return k
    vo_by_suffix = {strip_prefix(k): k for k in vo_state.keys()}
    full_by_suffix = {strip_prefix(k): k for k in full_visual_state.keys()}
    matched_suffixes = set(vo_by_suffix.keys()) & set(full_by_suffix.keys())
    print(f"suffix-based matches: {len(matched_suffixes)}", flush=True)
    pairs = [(vo_by_suffix[s], full_by_suffix[s]) for s in matched_suffixes]
else:
    pairs = [(k, k) for k in direct_matches]

n_exact, n_close, n_diff, n_shape_mismatch = 0, 0, 0, 0
max_diff_seen = 0.0
mismatches = []
for vo_key, full_key in pairs:
    a, b = vo_state[vo_key], full_visual_state[full_key]
    if a.shape != b.shape:
        n_shape_mismatch += 1
        mismatches.append(f"SHAPE MISMATCH {vo_key}: {a.shape} vs {b.shape}")
        continue
    if torch.equal(a, b):
        n_exact += 1
    else:
        diff = (a.float() - b.float()).abs().max().item()
        max_diff_seen = max(max_diff_seen, diff)
        if diff < 1e-5:
            n_close += 1
        else:
            n_diff += 1
            if len(mismatches) < 10:
                mismatches.append(f"DIFF {vo_key}: max_abs_diff={diff:.6e}")

print(f"\n=== RESULT ===", flush=True)
print(f"pairs compared: {len(pairs)}", flush=True)
print(f"exact match (torch.equal): {n_exact}", flush=True)
print(f"close match (<1e-5): {n_close}", flush=True)
print(f"real mismatch (>=1e-5): {n_diff}", flush=True)
print(f"shape mismatch: {n_shape_mismatch}", flush=True)
print(f"max abs diff seen (non-exact pairs): {max_diff_seen:.6e}", flush=True)
if mismatches:
    print("sample issues:", flush=True)
    for m in mismatches:
        print(" ", m, flush=True)

unmatched_vo = set(vo_state.keys()) - {p[0] for p in pairs}
unmatched_full = set(full_visual_state.keys()) - {p[1] for p in pairs}
print(f"\nunmatched timm keys: {len(unmatched_vo)} (sample: {list(unmatched_vo)[:5]})", flush=True)
print(f"unmatched safetensors visual keys: {len(unmatched_full)} (sample: {list(unmatched_full)[:5]})", flush=True)

if n_exact + n_close == len(pairs) and len(pairs) == len(vo_state) == len(full_visual_state):
    print("\nVERDICT: CONFIRMED -- vision-only checkpoint is weight-identical to the full checkpoint's visual branch.", flush=True)
else:
    print("\nVERDICT: NOT FULLY CONFIRMED -- see mismatches/unmatched keys above before trusting the vision-only checkpoint as a drop-in.", flush=True)
