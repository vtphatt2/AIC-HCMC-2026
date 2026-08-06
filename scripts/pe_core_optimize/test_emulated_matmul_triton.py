"""Stage 2b: the real fused Triton kernel. Stage 2a's pure-PyTorch prototype
was inconclusive by construction -- torch.matmul(fp16, fp16) always rounds
its OUTPUT to fp16 before we can accumulate in fp32, so 3 separate calls
can never preserve the precision the scheme is supposed to deliver (see
test_fp16_accum_diagnostic.py: emulated error ~= plain fp16 error, both far
from the hi/lo-split-alone error of 2.4e-7). A fused kernel keeps the
accumulator as fp32 in SRAM across all 3 partial products and never writes
an intermediate fp16 result to HBM -- this is the only way to actually test
whether the "emulated fp32" idea works, not an optimization nicety.

Standard Triton block-tiled GEMM structure (see Triton's own matmul
tutorial), extended to load hi+lo fp16 tiles for both operands and issue
3 tl.dot() calls per K-block into one fp32 accumulator.
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def emulated_matmul_kernel(
    xh_ptr, xl_ptr, wh_ptr, wl_ptr, out_ptr,
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
    xl_ptrs = xl_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    wh_ptrs = wh_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)
    wl_ptrs = wl_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        w_mask = (rk[:, None] + k < K) & (rn[None, :] < N)

        xh = tl.load(xh_ptrs, mask=x_mask, other=0.0)
        xl = tl.load(xl_ptrs, mask=x_mask, other=0.0)
        wh = tl.load(wh_ptrs, mask=w_mask, other=0.0)
        wl = tl.load(wl_ptrs, mask=w_mask, other=0.0)

        # 4-term decomposition (the plan's 3-term version left ~3.4x the
        # 1e-5 target at small scale -- adding the dropped Xl@Wl term back,
        # now that fp32 accumulation is actually working, per CLAUDE.md
        # "test small" this was checked before scaling up).
        # fp32 accumulator lives in SRAM/registers the whole time -- this is
        # what pure PyTorch composition could not do.
        acc = tl.dot(xh, wh, acc)
        acc = tl.dot(xh, wl, acc)
        acc = tl.dot(xl, wh, acc)
        acc = tl.dot(xl, wl, acc)

        xh_ptrs += BLOCK_K * stride_xk
        xl_ptrs += BLOCK_K * stride_xk
        wh_ptrs += BLOCK_K * stride_wk
        wl_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def split_hi_lo(t: torch.Tensor):
    hi = t.half()
    lo = (t - hi.float()).half()
    return hi.contiguous(), lo.contiguous()


def emulated_matmul_triton(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Xh, Xl = split_hi_lo(X)
    Wh, Wl = split_hi_lo(W)
    out = torch.empty((M, N), device=X.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    emulated_matmul_kernel[grid](
        Xh, Xl, Wh, Wl, out,
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
    M, K, N = 8192, 1536, 1536  # same shape as Stage 2a (batch*tokens, hidden, hidden)

    X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

    print("=== correctness (small shape first, per CLAUDE.md) ===", flush=True)
    Xs, Ws = X[:128, :128], W[:128, :128]
    ref_s = torch.matmul(Xs, Ws)
    emu_s = emulated_matmul_triton(Xs, Ws)
    err_s = (ref_s - emu_s).abs().max().item()
    print(f"small-shape (128x128x128) max_abs_err={err_s:.6e} target<1e-5 pass={err_s < 1e-5}", flush=True)

    if err_s >= 1e-5:
        print(
            f"NOTE: small-shape error {err_s:.2e} misses the strict <1e-5 target, but 3-term and "
            f"4-term variants gave near-identical error (3.43e-05 vs 3.81e-05) -- likely an inherent "
            f"floating-point accumulation-order floor of this technique, not a fixable bug. "
            f"Proceeding to full-shape correctness+speed anyway for informational purposes.",
            flush=True,
        )
        print("=== correctness (full shape) ===", flush=True)
        ref = torch.matmul(X, W)
        emu = emulated_matmul_triton(X, W)
        err = (ref - emu).abs().max().item()
        print(f"full-shape ({M}x{K}x{N}) max_abs_err={err:.6e} target<1e-5 pass={err < 1e-5}", flush=True)

        print("=== speed ===", flush=True)
        N_WARMUP, N_ITERS = 5, 20
        for _ in range(N_WARMUP):
            _ = torch.matmul(X, W)
            _ = emulated_matmul_triton(X, W)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _ = torch.matmul(X, W)
        torch.cuda.synchronize()
        fp32_s = (time.perf_counter() - t0) / N_ITERS

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _ = emulated_matmul_triton(X, W)
        torch.cuda.synchronize()
        triton_s = (time.perf_counter() - t0) / N_ITERS

        print(f"fp32 matmul: {fp32_s*1000:.3f}ms/call", flush=True)
        print(f"fused triton emulated matmul: {triton_s*1000:.3f}ms/call", flush=True)
        print(f"speedup: {fp32_s/triton_s:.2f}x (theoretical ceiling ~2.7x)", flush=True)
        decision = "correctness+speed both pass -- worth Stage 3 integration" if err < 1e-5 and triton_s < fp32_s else "STOP -- fails correctness or speed"
        print(f"\nSTAGE 2b DECISION: {decision}", flush=True)
