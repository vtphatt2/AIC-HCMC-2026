"""Sanity check / isolation test: does the 178x slowdown come from the
"emulate 4-term" logic, or from something wrong with the Triton/grid setup
on T4 in general? Same BLOCK_M/BLOCK_N/BLOCK_K as the failed emulated kernel,
but ONLY 1 tl.dot(Xh, Wh) per K-iteration, single fp16 inputs, no hi/lo split.

- If this is fast (near cuBLAS fp16): the problem is specific to the 4-term
  emulation logic (register pressure from 4 tiles + accumulator, or the
  4x tl.dot calls per iteration).
- If this is ALSO abnormally slow: the problem is in the Triton/grid
  config/driver on T4 generally -- "Emulated FP32 via Triton on T4" is dead,
  no more time should be spent on it.
"""
import time

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

        # ONLY ONE dot per iteration -- this is the isolation variable.
        acc = tl.dot(xh, wh, acc)

        xh_ptrs += BLOCK_K * stride_xk
        wh_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def minimal_matmul_triton(Xh: torch.Tensor, Wh: torch.Tensor) -> torch.Tensor:
    M, K = Xh.shape
    K2, N = Wh.shape
    assert K == K2
    out = torch.empty((M, N), device=Xh.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32  # same as the failed emulated kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    minimal_matmul_kernel[grid](
        Xh, Wh, out,
        M, N, K,
        Xh.stride(0), Xh.stride(1),
        Wh.stride(0), Wh.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = "cuda"
    M, K, N = 8192, 1536, 1536  # same full shape as before

    X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    Xh, Wh = X.half().contiguous(), W.half().contiguous()

    print("=== correctness (vs fp16 cuBLAS matmul, not fp32 -- this kernel has no hi/lo split) ===", flush=True)
    ref_fp16 = torch.matmul(Xh, Wh).float()
    triton_out = minimal_matmul_triton(Xh, Wh)
    err = (ref_fp16 - triton_out).abs().max().item()
    print(f"max_abs_err vs plain fp16 matmul: {err:.6e} (should be ~0, same math)", flush=True)

    print("=== speed: minimal triton (1x dot) vs cuBLAS fp16 vs cuBLAS fp32 ===", flush=True)
    N_WARMUP, N_ITERS = 5, 20
    for _ in range(N_WARMUP):
        _ = torch.matmul(X, W)
        _ = torch.matmul(Xh, Wh)
        _ = minimal_matmul_triton(Xh, Wh)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = torch.matmul(X, W)
    torch.cuda.synchronize()
    fp32_s = (time.perf_counter() - t0) / N_ITERS

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = torch.matmul(Xh, Wh)
    torch.cuda.synchronize()
    fp16_cublas_s = (time.perf_counter() - t0) / N_ITERS

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = minimal_matmul_triton(Xh, Wh)
    torch.cuda.synchronize()
    triton_s = (time.perf_counter() - t0) / N_ITERS

    print(f"cuBLAS fp32 matmul:        {fp32_s*1000:.3f}ms/call", flush=True)
    print(f"cuBLAS fp16 matmul:        {fp16_cublas_s*1000:.3f}ms/call", flush=True)
    print(f"minimal triton (1x dot):   {triton_s*1000:.3f}ms/call", flush=True)
    print(f"triton vs cuBLAS-fp16 ratio: {triton_s/fp16_cublas_s:.2f}x slower", flush=True)
    print(f"triton vs cuBLAS-fp32 speedup: {fp32_s/triton_s:.2f}x", flush=True)

    verdict = (
        "PASS -- minimal kernel is close to cuBLAS fp16, problem is specific to the 4-term emulation logic"
        if triton_s < fp16_cublas_s * 3
        else "FAIL -- minimal kernel is ALSO abnormally slow, problem is generic Triton/grid/driver config on T4, not the emulation logic"
    )
    print(f"\nVERDICT: {verdict}", flush=True)
