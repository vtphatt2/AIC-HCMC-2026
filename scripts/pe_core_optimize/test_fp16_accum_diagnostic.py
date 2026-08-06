"""Diagnose why the Stage 2a emulated matmul had 6.3e-2 abs error (target
<1e-5): hypothesis is torch.matmul(fp16, fp16) rounds its OUTPUT to fp16
storage before we ever call .float() on it, discarding whatever fp32
precision the Tensor Core accumulated internally. Test whether disabling
cuBLAS's reduced-precision reduction flag changes anything, and separately
measure the error contributed by hi/lo splitting ALONE (no matmul, just
reconstructing a random fp32 tensor from its hi+lo fp16 parts) to isolate
splitting-error from matmul-output-rounding-error.
"""
import torch

torch.manual_seed(0)
DEVICE = "cuda"

# --- isolate: how much error comes from hi/lo splitting alone (no matmul)? ---
x = torch.randn(10000, device=DEVICE, dtype=torch.float32)
x_hi = x.half()
x_lo = (x - x_hi.float()).half()
x_reconstructed = x_hi.float() + x_lo.float()
split_err = (x - x_reconstructed).abs().max().item()
print(f"hi/lo split-and-reconstruct max abs error (no matmul at all): {split_err:.6e}", flush=True)

# --- isolate: does allow_fp16_reduced_precision_reduction matter? ---
M, K, N = 64, 1536, 1536
X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
W = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
ref = torch.matmul(X, W)

for flag in (True, False):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
    Xh, Xl = X.half(), (X - X.half().float()).half()
    Wh, Wl = W.half(), (W - W.half().float()).half()
    acc = torch.matmul(Xh, Wh).float() + torch.matmul(Xh, Wl).float() + torch.matmul(Xl, Wh).float()
    err = (ref - acc).abs().max().item()
    print(f"allow_fp16_reduced_precision_reduction={flag}: max_abs_err={err:.6e}", flush=True)

# --- isolate: what if we accumulate the 4th (dropped) term too? ---
Xh, Xl = X.half(), (X - X.half().float()).half()
Wh, Wl = W.half(), (W - W.half().float()).half()
acc4 = (
    torch.matmul(Xh, Wh).float()
    + torch.matmul(Xh, Wl).float()
    + torch.matmul(Xl, Wh).float()
    + torch.matmul(Xl, Wl).float()
)
err4 = (ref - acc4).abs().max().item()
print(f"4-term (incl. dropped Xl@Wl): max_abs_err={err4:.6e}", flush=True)

# --- for reference: plain fp16 matmul (no emulation at all) error, as a sanity ceiling ---
plain_fp16_err = (ref - torch.matmul(X.half(), W.half()).float()).abs().max().item()
print(f"plain fp16 matmul (no emulation): max_abs_err={plain_fp16_err:.6e}", flush=True)
