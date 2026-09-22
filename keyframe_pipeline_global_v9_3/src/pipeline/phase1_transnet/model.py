"""TransNetV2 loading, GPU batch inference, and per-video finalization (scenes/keyframes.json)."""
from __future__ import annotations

import numpy as np
import torch
from transnetv2_pytorch import TransNetV2

from ..io_utils import atomic_json
from ..keyframe_selection import selection_metadata
from .decode import TRANSNET_H, TRANSNET_W, TransNetDecodedVideo, select_keyframes


def load_transnet(device: str) -> TransNetV2:
    print(f"Loading TransNetV2 on {device} ...", flush=True)
    model = TransNetV2()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _to_numpy_prediction(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _transnet_predict_raw_batch(model, batch_np: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run one native TransNet batch [B,100,27,48,3].

    transnetv2-pytorch's predict_raw() requires a torch.uint8 Tensor, not numpy.
    V9.1 passed numpy here, which only failed once the first global batch finally ran.
    """
    if batch_np.dtype != np.uint8 or batch_np.ndim != 5 or list(batch_np.shape[2:]) != [TRANSNET_H, TRANSNET_W, 3]:
        raise ValueError(f"bad TransNet batch: shape={batch_np.shape} dtype={batch_np.dtype}")
    batch_t = torch.from_numpy(np.ascontiguousarray(batch_np)).to(device=device, dtype=torch.uint8, non_blocking=True)
    with torch.inference_mode():
        if hasattr(model, "predict_raw"):
            single, many = model.predict_raw(batch_t)
        else:
            single, many = model(batch_t)
            single = torch.sigmoid(single)
            if isinstance(many, dict):
                many = many["many_hot"]
            many = torch.sigmoid(many)
    return _to_numpy_prediction(single), _to_numpy_prediction(many)


def _finalize_transnet_video(
    args,
    model,
    item: TransNetDecodedVideo,
    predictions_np: np.ndarray,
    inference_seconds: float,
    batch_count: int,
    batch_size_sum: int,
    num_frames_override: int | None = None,
) -> None:
    video_out = args.out_dir / item.video_id
    video_out.mkdir(parents=True, exist_ok=True)
    scenes_path = video_out / "scenes.json"
    keyframes_path = video_out / "keyframes.json"
    scenes = model.predictions_to_scenes(predictions_np, threshold=args.threshold)
    scene_items = [
        {"start_frame": int(start), "end_frame": int(end)}
        for start, end in scenes
    ]
    selected = select_keyframes(
        scenes, item.fps,
        strategy=args.keyframe_strategy,
        keyframes_per_second=args.keyframes_per_second,
        min_keyframes_per_scene=args.min_keyframes_per_scene,
        max_keyframes_per_scene=args.max_keyframes_per_scene,
    )
    num_frames = int(num_frames_override if num_frames_override is not None else len(item.frames))
    atomic_json(scenes_path, {
        "zip": str(args.zip_path), "entry": item.entry.name,
        "fps": item.fps, "width": item.width, "height": item.height,
        "num_frames": num_frames, "threshold": args.threshold,
        "num_scenes": len(scene_items), "scenes": scene_items,
    })
    atomic_json(keyframes_path, {
        "zip": str(args.zip_path), "entry": item.entry.name,
        "fps": item.fps, "width": item.width, "height": item.height,
        "num_frames": num_frames, "num_scenes": len(scene_items),
        "selection": selection_metadata(
            strategy=args.keyframe_strategy,
            keyframes_per_second=args.keyframes_per_second,
            min_keyframes_per_scene=args.min_keyframes_per_scene,
            max_keyframes_per_scene=args.max_keyframes_per_scene,
        ),
        "num_keyframes": len(selected), "keyframes": selected,
    })
    if args.save_transnet_predictions:
        np.save(video_out / "transnet_predictions.npy", predictions_np)
    avg_batch = (batch_size_sum / batch_count) if batch_count else 0.0
    atomic_json(video_out / "transnet_stats.json", {
        "entry": item.entry.name,
        "mode": "global",
        "num_frames": num_frames,
        "decode_seconds": round(item.decode_seconds, 3),
        "decode_frames_per_second": round(num_frames / max(item.decode_seconds, 1e-9), 3),
        "inference_seconds": round(inference_seconds, 3),
        "inference_frames_per_second": round(num_frames / max(inference_seconds, 1e-9), 3),
        "transnet_batches": batch_count,
        "average_transnet_batch": round(avg_batch, 3),
        "configured_transnet_batch_size": args.transnet_batch_size,
    })
    print(
        f"  [done] {item.entry.name}: {num_frames} frames -> {len(scene_items)} scenes -> "
        f"{len(selected)} keyframes | decode {num_frames/max(item.decode_seconds,1e-9):.1f} fps | "
        f"TransNet {num_frames/max(inference_seconds,1e-9):.1f} fps",
        flush=True,
    )
