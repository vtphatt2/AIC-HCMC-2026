"""Load the PE-Core vision encoder from Hugging Face, Kaggle, or a local checkpoint dir."""
from __future__ import annotations

import os
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file as safetensors_load_file
from timm.data import create_transform, resolve_model_data_config

from .compute import configure_cuda_math, resolve_amp_mode
from .preprocess import build_encoder_preprocess_plan

try:
    import kagglehub
except Exception:
    kagglehub = None

DEFAULT_MODEL_ID = "hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"
DEFAULT_MODEL_SOURCE = "huggingface"
DEFAULT_KAGGLE_MODEL = "meowluvmatcha/lufina/transformers/default"
DEFAULT_MODEL_ARCH = "vit_pe_core_gigantic_patch14_448"


def _resolve_local_model_dir(args) -> Path:
    if args.model_source == "kaggle":
        if kagglehub is None:
            raise RuntimeError("kagglehub is not installed; install it or choose another --model-source")
        if args.kaggle_cache_dir is not None:
            os.environ["KAGGLEHUB_CACHE"] = str(args.kaggle_cache_dir)
        model_path = Path(kagglehub.model_download(args.kaggle_model))
        print(f"Kaggle model cached at: {model_path}", flush=True)
        return model_path
    if args.model_source == "local":
        if args.model_dir is None:
            raise ValueError("--model-dir is required when --model-source local")
        if not args.model_dir.is_dir():
            raise FileNotFoundError(args.model_dir)
        return args.model_dir
    raise RuntimeError(f"unsupported local model source: {args.model_source}")


def _find_checkpoint(model_dir: Path) -> Path:
    """Find a supported checkpoint recursively (fixes nested Kaggle model layouts)."""
    names = ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt", "checkpoint.pth")
    candidates: list[Path] = []
    for priority, name in enumerate(names):
        for candidate in model_dir.rglob(name):
            if candidate.is_file():
                candidates.append(candidate)
        if candidates:
            # Prefer PE-Core vision-only directory if present, then the shallowest path.
            candidates.sort(key=lambda x: ("pecore_vision_only" not in str(x).lower(), len(x.parts), str(x)))
            chosen = candidates[0]
            print(f"Resolved checkpoint: {chosen}", flush=True)
            return chosen
    preview = ", ".join(str(x.relative_to(model_dir)) for x in list(model_dir.rglob("*"))[:30])
    raise FileNotFoundError(f"No supported checkpoint found recursively under {model_dir}; sample entries: {preview}")


def _load_local_timm_model(model_dir: Path, model_arch: str):
    checkpoint = _find_checkpoint(model_dir)
    model = timm.create_model(model_arch, pretrained=False)
    if checkpoint.suffix == ".safetensors":
        state = safetensors_load_file(str(checkpoint))
    else:
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[warning] unexpected keys while loading local model: {len(unexpected)}", flush=True)
    if missing:
        print(f"[warning] missing keys while loading local model: {len(missing)}", flush=True)
    return model


def load_image_encoder(args):
    if args.local_files_only and args.model_source == "huggingface":
        os.environ["HF_HUB_OFFLINE"] = "1"

    amp_mode = resolve_amp_mode(args)
    configure_cuda_math(args)

    if args.model_source == "huggingface":
        print(f"Loading vision-only {args.model_id} on {args.device} ...", flush=True)
        model = timm.create_model(args.model_id, pretrained=True)
    else:
        source_desc = args.kaggle_model if args.model_source == "kaggle" else str(args.model_dir)
        print(f"Loading vision-only {args.model_source}:{source_desc} on {args.device} ...", flush=True)
        model_dir = _resolve_local_model_dir(args)
        model = _load_local_timm_model(model_dir, args.model_arch)

    data_config = resolve_model_data_config(model)
    preprocess = create_transform(**data_config, is_training=False)
    preprocess_plan = build_encoder_preprocess_plan(data_config) if args.embed_ffmpeg_preprocess else None

    # v6 invariant: parameters stay FP32. autocast controls compute dtype only.
    model = model.eval().to(device=args.device, dtype=torch.float32)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a PyTorch build with torch.compile")
        print("Compiling PE-Core with torch.compile(mode='reduce-overhead') ...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")

    device_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device
    print(
        f"PE-Core compute: weights=fp32 amp={amp_mode} tf32={bool(args.tf32)} "
        f"compile={bool(args.compile)} device={device_name}", flush=True,
    )
    if preprocess_plan is not None:
        print(
            f"PE-Core ffmpeg preprocess: resize={preprocess_plan.resize_w}x{preprocess_plan.resize_h} "
            f"crop={preprocess_plan.output_w}x{preprocess_plan.output_h} "
            f"interp={preprocess_plan.interpolation}",
            flush=True,
        )
    else:
        print("PE-Core ffmpeg preprocess: unsupported model transform; using legacy PIL/timm path", flush=True)
    return model, preprocess, preprocess_plan, amp_mode
