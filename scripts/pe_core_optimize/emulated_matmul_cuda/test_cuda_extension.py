"""Test the hand-written WMMA CUDA extension. Per CLAUDE.md (test small
before big): micro-scale (16x16x16) correctness gate FIRST, then check the
compiled .so for actual hmma.sync instructions via cuobjdump -sass, and only
run the full 8192x1536x1536 benchmark if the micro gate passes. Mirrors
test_triton_micro_debug.py's Part B/C, this time for a real CUDA kernel
instead of Triton, to isolate whether Triton itself (not emitting mma.sync
at all, confirmed via PTX dump) was the broken link, or whether the T4/Colab
environment can't use Tensor Cores at all regardless of how the kernel is written.
"""
import glob
import subprocess
import time

import torch
import emulated_matmul_cuda

DEVICE = "cuda"


def split_hi_lo(t: torch.Tensor):
    hi = t.half()
    lo = (t - hi.float()).half()
    return hi, lo


def run_emulated(X, W):
    Xh, Xl = split_hi_lo(X)
    Wh, Wl = split_hi_lo(W)
    return emulated_matmul_cuda.emulated_matmul(Xh.contiguous(), Xl.contiguous(), Wh.contiguous(), Wl.contiguous())


print("=== micro-scale (16x16x16) correctness gate ===", flush=True)
torch.manual_seed(0)
Mm, Km, Nm = 16, 16, 16
Xm = torch.randn(Mm, Km, device=DEVICE, dtype=torch.float32)
Wm = torch.randn(Km, Nm, device=DEVICE, dtype=torch.float32)
true_micro = torch.matmul(Xm, Wm)
out_micro = run_emulated(Xm, Wm)
err_micro = (out_micro - true_micro).abs().max().item()
print(f"micro max_abs_err vs true fp32: {err_micro:.6e} (target <1e-5)", flush=True)
micro_pass = err_micro < 1e-5
print(f"MICRO GATE: {'PASS' if micro_pass else 'FAIL'}", flush=True)

if not micro_pass:
    print("\nSTOPPING per CLAUDE.md 'test small before big' -- micro gate failed, not running full shape.", flush=True)
    raise SystemExit(1)

print("\n=== SASS check: does the compiled kernel actually use Tensor Cores? ===", flush=True)
so_files = glob.glob("**/emulated_matmul_cuda*.so", recursive=True) + \
    glob.glob("/usr/local/lib/python*/dist-packages/emulated_matmul_cuda*.so") + \
    glob.glob("/root/.cache/torch_extensions/**/*.so", recursive=True)
so_path = emulated_matmul_cuda.__file__
print(f"module file: {so_path}", flush=True)
try:
    sass = subprocess.run(["cuobjdump", "-sass", so_path], capture_output=True, text=True, timeout=60).stdout
    n_hmma = sass.count("HMMA")
    n_ffma = sass.count("FFMA")
    print(f"SASS instruction counts: HMMA={n_hmma}, FFMA={n_ffma}", flush=True)
    if n_hmma > 0:
        print("=> Tensor Core (HMMA) instructions CONFIRMED present. Hand-written WMMA succeeds where Triton didn't.", flush=True)
    else:
        print("=> NO HMMA instructions found even in hand-written WMMA kernel -- suggests an environment-level issue, not Triton-specific.", flush=True)
except Exception as e:
    print(f"cuobjdump check failed/unavailable: {e}", flush=True)

print("\n=== full shape (8192x1536x1536): correctness + speed vs cuBLAS ===", flush=True)
M, K, N = 8192, 1536, 1536
X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
Xh, Xl = split_hi_lo(X)
Wh, Wl = split_hi_lo(W)

true_ref = torch.matmul(X, W)
out_full = emulated_matmul_cuda.emulated_matmul(Xh, Xl, Wh, Wl)
err_full = (out_full - true_ref).abs().max().item()
print(f"full-shape max_abs_err vs true fp32: {err_full:.6e}", flush=True)

N_WARMUP, N_ITERS = 5, 20
for _ in range(N_WARMUP):
    _ = torch.matmul(X, W)
    _ = torch.matmul(X.half(), W.half())
    _ = emulated_matmul_cuda.emulated_matmul(Xh, Xl, Wh, Wl)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N_ITERS):
    _ = torch.matmul(X, W)
torch.cuda.synchronize()
fp32_s = (time.perf_counter() - t0) / N_ITERS

t0 = time.perf_counter()
for _ in range(N_ITERS):
    _ = torch.matmul(X.half(), W.half())
torch.cuda.synchronize()
fp16_s = (time.perf_counter() - t0) / N_ITERS

t0 = time.perf_counter()
for _ in range(N_ITERS):
    _ = emulated_matmul_cuda.emulated_matmul(Xh, Xl, Wh, Wl)
torch.cuda.synchronize()
emulated_s = (time.perf_counter() - t0) / N_ITERS

print(f"cuBLAS fp32:        {fp32_s*1000:.3f}ms/call", flush=True)
print(f"cuBLAS fp16:        {fp16_s*1000:.3f}ms/call", flush=True)
print(f"WMMA emulated (4x): {emulated_s*1000:.3f}ms/call", flush=True)
print(f"emulated vs fp32 speedup: {fp32_s/emulated_s:.2f}x", flush=True)
print(f"emulated vs fp16 ratio: {emulated_s/fp16_s:.2f}x slower" if emulated_s > fp16_s else f"emulated vs fp16: {fp16_s/emulated_s:.2f}x faster", flush=True)

print("\n=== DONE ===", flush=True)
