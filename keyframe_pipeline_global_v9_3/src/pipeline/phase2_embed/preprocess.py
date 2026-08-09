"""Encoder-aware ffmpeg resize/crop plan, and the uint8->normalized-float steps around it.

`build_encoder_preprocess_plan` turns a timm `resolve_model_data_config(model)` dict into an
ffmpeg filter that resizes + center-crops selected keyframes to exactly the encoder's expected
input, so ffmpeg emits `output_h x output_w x 3` uint8 RGB directly (no full-resolution frame or
PIL resize in Python). `fast_preprocess_rgb` just reshapes those raw bytes to CHW; the actual
float cast + mean/std normalization happens later, on GPU, right before the encoder forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EncoderPreprocessPlan:
    output_h: int
    output_w: int
    resize_h: int
    resize_w: int
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    ffmpeg_filter: str


def build_encoder_preprocess_plan(data_config: dict) -> EncoderPreprocessPlan | None:
    """Build an ffmpeg fast path from the common timm inference data config."""
    input_size = tuple(data_config.get("input_size", ()))
    if len(input_size) != 3 or int(input_size[0]) != 3:
        return None
    output_h, output_w = int(input_size[1]), int(input_size[2])
    if output_h <= 0 or output_w <= 0 or output_h != output_w:
        return None

    crop_mode = str(data_config.get("crop_mode", "center")).lower()
    if crop_mode not in {"center", ""}:
        return None

    crop_pct = data_config.get("crop_pct", 1.0)
    if isinstance(crop_pct, (tuple, list)):
        if len(crop_pct) != 2 or float(crop_pct[0]) != float(crop_pct[1]):
            return None
        crop_pct = float(crop_pct[0])
    crop_pct = float(crop_pct or 1.0)
    if crop_pct <= 0:
        return None
    resize_h = max(output_h, int(math.floor(output_h / crop_pct)))
    resize_w = max(output_w, int(math.floor(output_w / crop_pct)))

    interpolation = str(data_config.get("interpolation", "bicubic")).lower()
    ffmpeg_flags = {
        "bicubic": "bicubic",
        "bilinear": "bilinear",
        "nearest": "neighbor",
        "lanczos": "lanczos",
    }.get(interpolation)
    if ffmpeg_flags is None:
        return None

    mean = tuple(float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
        return None

    # Standard timm inference: resize shortest side to input/crop_pct, preserve aspect,
    # then center crop. FFmpeg crop centers by default when x/y are omitted.
    r = resize_w
    scale = (
        f"scale=w='if(gte(iw,ih),-2,{r})':"
        f"h='if(gte(iw,ih),{r},-2)':flags={ffmpeg_flags}"
    )
    crop = f"crop={output_w}:{output_h}"
    return EncoderPreprocessPlan(
        output_h=output_h, output_w=output_w,
        resize_h=resize_h, resize_w=resize_w,
        interpolation=interpolation, mean=mean, std=std,
        ffmpeg_filter=f"{scale},{crop}",
    )


def fast_preprocess_rgb(raw: bytes, plan: EncoderPreprocessPlan) -> torch.Tensor:
    """Return contiguous CHW uint8; cast/normalize happens after async H2D copy."""
    array = np.frombuffer(raw, dtype=np.uint8).reshape(plan.output_h, plan.output_w, 3)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def normalize_uint8_cpu(tensor: torch.Tensor, plan: EncoderPreprocessPlan) -> torch.Tensor:
    out = tensor.float().div_(255.0)
    mean = torch.tensor(plan.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(plan.std, dtype=torch.float32).view(3, 1, 1)
    return out.sub_(mean).div_(std)
