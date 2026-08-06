"""Stage 2a: de-risk the "emulated FP32 via 3x FP16 Tensor Core matmul" idea
using PLAIN PyTorch ops first -- no Triton kernel yet. torch.matmul on fp16
CUDA tensors already dispatches to cuBLAS Tensor Core paths, so this tests
both correctness (does the 3-term hi/lo decomposition actually recover
fp32-like accuracy?) and the ~2.7x theoretical speed ceiling (3 fp16 matmuls
vs 1 fp32 matmul) with zero custom-kernel risk. Only write the fused Triton
kernel (Stage 2b) if this passes both checks -- a hand-fused kernel only
saves the overhead of 3 separate launches + intermediate writes; it can't
fix a numerically-wrong or fundamentally-not-faster idea.

Shape matches PE-Core-bigG-14-448's real vision tower: hidden_dim=1536,
1024 image tokens (448/14=32, 32*32=1024 patches), batch=8 (matches our
embed batch_size).
"""
import time

import torch

torch.manual_seed(0)
DEVICE = "cuda"
BATCH, TOKENS, HIDDEN = 8, 1024, 1536
M = BATCH * TOKENS  # flatten batch+seq for a plain 2D matmul, like a Linear layer sees

X = torch.randn(M, HIDDEN, device=DEVICE, dtype=torch.float32)
W = torch.randn(HIDDEN, HIDDEN, device=DEVICE, dtype=torch.float32)


def split_hi_lo(t: torch.Tensor):
    hi = t.half()
    lo = (t - hi.float()).half()
    return hi, lo


def emulated_matmul(X32: torch.Tensor, W32: torch.Tensor) -> torch.Tensor:
    Xh, Xl = split_hi_lo(X32)
    Wh, Wl = split_hi_lo(W32)
    # 3-term decomposition (plan drops the Xl@Wl term -- smallest magnitude).
    # fp16 @ fp16 with fp32 accumulation is torch.matmul's native Tensor Core path.
    acc = torch.zeros(M, HIDDEN, device=DEVICE, dtype=torch.float32)
    acc += torch.matmul(Xh, Wh, out=None).float()
    acc += torch.matmul(Xh, Wl, out=None).float()
    acc += torch.matmul(Xl, Wh, out=None).float()
    return acc


# --- correctness ---
ref = torch.matmul(X, W)
emu = emulated_matmul(X, W)
max_abs_err = (ref - emu).abs().max().item()
rel_err = ((ref - emu).abs() / ref.abs().clamp_min(1e-6)).mean().item()
print(f"max_abs_err={max_abs_err:.6e} mean_rel_err={rel_err:.6e} target<1e-5", flush=True)
print(f"correctness pass (max_abs_err<1e-5): {max_abs_err < 1e-5}", flush=True)

# --- speed (warmup both paths first -- CUDA/cuBLAS kernel selection has first-call overhead) ---
N_WARMUP, N_ITERS = 5, 50

for _ in range(N_WARMUP):
    _ = torch.matmul(X, W)
    _ = emulated_matmul(X, W)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N_ITERS):
    _ = torch.matmul(X, W)
torch.cuda.synchronize()
fp32_s = (time.perf_counter() - t0) / N_ITERS

t0 = time.perf_counter()
for _ in range(N_ITERS):
    _ = emulated_matmul(X, W)
torch.cuda.synchronize()
emu_s = (time.perf_counter() - t0) / N_ITERS

print(f"fp32 matmul: {fp32_s*1000:.3f}ms/call", flush=True)
print(f"emulated (3x fp16) matmul: {emu_s*1000:.3f}ms/call", flush=True)
print(f"speedup: {fp32_s/emu_s:.2f}x (theoretical ceiling ~2.7x)", flush=True)
print(f"\nSTAGE 2 DECISION: {'proceed to fused Triton kernel (2b)' if max_abs_err < 1e-5 and emu_s < fp32_s else 'STOP -- fails correctness or speed on this isolated shape'}", flush=True)
