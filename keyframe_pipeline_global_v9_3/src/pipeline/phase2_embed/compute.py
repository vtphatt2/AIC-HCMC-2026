"""AMP/TF32 policy. PE-Core parameters always stay FP32; these only control compute dtype."""
from __future__ import annotations

import torch


def resolve_amp_mode(args) -> str:
    """Resolve v6 AMP policy without changing model parameter dtype."""
    # Backward compatibility with v4 launchers/users.
    if args.precision is not None:
        compat = {"fp32": "off", "fp16": "fp16", "bf16": "bf16"}[args.precision]
        print(f"[compat] --precision {args.precision} -> --amp {compat}; model weights remain FP32 in v6", flush=True)
        requested = compat
    else:
        requested = args.amp

    if not args.device.startswith("cuda"):
        if requested != "off":
            print(f"[warning] AMP {requested} requested on {args.device}; using FP32", flush=True)
        return "off"

    if requested == "auto":
        # torch.cuda.is_bf16_supported() can return True on pre-Ampere GPUs (e.g. Turing/T4)
        # where bf16 has no Tensor Core acceleration and runs via slow software emulation --
        # nearly as slow as FP32. Actual bf16 Tensor Core support needs compute capability >= 8.0.
        if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
            return "bf16"
        return "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 autocast was requested but this CUDA GPU/PyTorch build does not support BF16. "
            "Use --amp auto (recommended), --amp fp16, or --amp off."
        )
    return requested


def configure_cuda_math(args) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    # 'high' permits TF32-style fast FP32 matmul where supported; 'highest' keeps strict FP32.
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cudnn.benchmark = True
