"""Root-cause debug for test_minimal_triton_matmul.py's two anomalies:
(1) "correctness" error of 6.3e-2 vs cuBLAS fp16, (2) 24x slowdown vs cuBLAS fp16.

Part A -- is the correctness "bug" actually just fp16-output-rounding (already
established in Stage 2a: torch.matmul(fp16,fp16) rounds its OUTPUT to fp16
before any .float() cast, even though Tensor Cores accumulate in fp32
internally)? Test: compare both Triton and cuBLAS-fp16 against a TRUE fp32
reference. If Triton is closer to true fp32 than cuBLAS-fp16 is, the "bug" is
just that our fp16 reference was already rounded -- not a stride/layout bug.

Part B -- micro-scale (16x16x16) element-wise diff to catch any real
stride/masking bug directly (independent of Part A's precision question).

Part C -- PTX dump: does Triton actually emit Tensor Core instructions
(mma/hmma) or fall back to plain fma? Explains the speed gap or doesn't.

Part D -- try a couple different num_warps/num_stages configs (T4-appropriate)
on the full shape to see if the default was just badly tuned.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def minimal_matmul_kernel(
    xh_ptr, wh_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    xh_ptrs = xh_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    wh_ptrs = wh_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        w_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
        xh = tl.load(xh_ptrs, mask=x_mask, other=0.0)
        wh = tl.load(wh_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(xh, wh, acc)
        xh_ptrs += BLOCK_K * stride_xk
        wh_ptrs += BLOCK_K * stride_wk
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def run_triton(Xh, Wh, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2):
    M, K = Xh.shape
    _, N = Wh.shape
    out = torch.empty((M, N), device=Xh.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel = minimal_matmul_kernel[grid](
        Xh, Wh, out, M, N, K,
        Xh.stride(0), Xh.stride(1), Wh.stride(0), Wh.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, kernel


DEVICE = "cuda"
torch.manual_seed(0)

print("=== Part A: is the 6.3e-2 'error' just fp16-output-rounding in the reference? ===", flush=True)
M, K, N = 8192, 1536, 1536
X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
Xh, Wh = X.half().contiguous(), W.half().contiguous()

true_fp32_ref = torch.matmul(X, W)  # never touches fp16
cublas_fp16_ref = torch.matmul(Xh, Wh).float()  # rounded to fp16 mid-computation
triton_out, _ = run_triton(Xh, Wh)

err_cublas_vs_true = (cublas_fp16_ref - true_fp32_ref).abs().max().item()
err_triton_vs_true = (triton_out - true_fp32_ref).abs().max().item()
err_triton_vs_cublas = (triton_out - cublas_fp16_ref).abs().max().item()
print(f"cuBLAS-fp16 vs TRUE fp32:  max_abs_err = {err_cublas_vs_true:.6e}", flush=True)
print(f"Triton      vs TRUE fp32:  max_abs_err = {err_triton_vs_true:.6e}", flush=True)
print(f"Triton      vs cuBLAS-fp16: max_abs_err = {err_triton_vs_cublas:.6e}  (this is the number that looked scary)", flush=True)
if err_triton_vs_true < err_cublas_vs_true:
    print("=> CONFIRMED: Triton is MORE accurate than cuBLAS-fp16 vs true fp32. "
          "The 'bug' is that cuBLAS's own fp16 output rounding was the wrong reference to diff against. No stride bug.", flush=True)
else:
    print("=> Triton is NOT closer to true fp32 than cuBLAS-fp16 is -- real bug still possible, needs Part B.", flush=True)

print("\n=== Part B: micro-scale (16x16x16) element-wise check ===", flush=True)
Mm, Km, Nm = 16, 16, 16
Xm = torch.arange(Mm * Km, device=DEVICE, dtype=torch.float32).reshape(Mm, Km) % 5 + 1
Wm = torch.arange(Km * Nm, device=DEVICE, dtype=torch.float32).reshape(Km, Nm) % 5 + 1
Xmh, Wmh = Xm.half().contiguous(), Wm.half().contiguous()
true_micro = torch.matmul(Xm, Wm)
triton_micro, _ = run_triton(Xmh, Wmh, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
diff = (triton_micro - true_micro).abs()
print(f"micro max_abs_err vs true fp32: {diff.max().item():.6e}", flush=True)
if diff.max().item() > 1.0:
    bad = (diff > 1.0).nonzero()
    print(f"MISMATCHED CELLS (row,col): {bad[:10].tolist()} ... total={len(bad)}", flush=True)
    print("row 0 triton:", triton_micro[0, :8].tolist(), flush=True)
    print("row 0 true:  ", true_micro[0, :8].tolist(), flush=True)
else:
    print("=> micro-scale matches (within fp16 precision) -- no stride/masking bug.", flush=True)

print("\n=== Part C: does Triton emit Tensor Core instructions? ===", flush=True)
_, kernel = run_triton(Xh, Wh)
try:
    ptx = kernel.asm["ptx"]
    n_mma = ptx.count("mma.sync") + ptx.count("hmma")
    n_fma = ptx.count("fma.rn.f32")
    print(f"PTX instruction counts: mma/hmma={n_mma}, fma.rn.f32={n_fma}", flush=True)
    if n_mma == 0:
        print("=> NO Tensor Core instructions emitted -- kernel is running on CUDA cores only. This explains the slowdown.", flush=True)
    else:
        print("=> Tensor Core instructions present -- slowdown is NOT from missing TC usage, likely tiling/occupancy.", flush=True)
except Exception as e:
    print(f"could not inspect PTX ({e}); trying kernel.asm.keys()", flush=True)
    try:
        print(list(kernel.asm.keys()), flush=True)
    except Exception as e2:
        print(f"asm inspection unavailable: {e2}", flush=True)

print("\n=== Part D: num_warps/num_stages sweep on full shape ===", flush=True)
import time
configs = [
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2),
    dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, num_warps=4, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=4),
]
t0 = time.perf_counter()
for _ in range(5):
    _ = torch.matmul(Xh, Wh)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _ = torch.matmul(Xh, Wh)
torch.cuda.synchronize()
cublas_s = (time.perf_counter() - t0) / 20
print(f"cuBLAS fp16 baseline: {cublas_s*1000:.3f}ms/call", flush=True)

for cfg in configs:
    try:
        for _ in range(5):
            _ = run_triton(Xh, Wh, **cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = run_triton(Xh, Wh, **cfg)
        torch.cuda.synchronize()
        s = (time.perf_counter() - t0) / 20
        print(f"{cfg}: {s*1000:.3f}ms/call ({s/cublas_s:.2f}x cuBLAS-fp16)", flush=True)
    except Exception as e:
        print(f"{cfg}: FAILED ({e})", flush=True)

print("\n=== DONE ===", flush=True)
