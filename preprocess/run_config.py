"""Load one declarative ZIP-native preprocessing run from JSON."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any


PATH_FIELDS = {
    "zip_path", "work_root", "archive", "transcripts", "transcript_metadata",
    "transcript_raw_dir", "transcript_output_dir", "transcript_state_dir",
}
RUN_CONFIG_FIELDS = {
    "url", "zip", "work_root", "archive", "profile", "device", "batch_size",
    "prefetch_batches", "transnet_batch_size", "transnet_decode_workers",
    "transnet_prefetch_windows", "keyframe_strategy", "keyframes_per_second",
    "min_keyframes_per_scene", "max_keyframes_per_scene", "limit",
    "sequential_stages", "delete_source_before_package", "local_files_only",
    "transcripts", "transcript_metadata", "transcript_raw_dir",
    "transcript_output_dir", "transcript_state_dir", "transcript_model",
    "transcript_concurrency", "transcript_rpm", "transcript_tpm",
    "transcript_video_id_regex", "transcript_overwrite",
}


def apply_run_config(args: Namespace) -> Namespace:
    """Overlay validated JSON fields onto parsed run defaults.

    Relative paths intentionally resolve from the config's directory, making a
    run definition portable as one self-contained file.
    """
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    if args.url is not None or args.zip_path is not None:
        raise ValueError("--config cannot be combined with --url or --zip")

    path = Path(config_path).expanduser().resolve()
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"run config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON run config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("run config must be a JSON object")
    unknown = sorted(set(payload) - RUN_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unsupported run config fields: {', '.join(unknown)}")
    if "zip" in payload:
        payload = {**payload, "zip_path": payload["zip"]}
        del payload["zip"]

    values = vars(args).copy()
    for name, value in payload.items():
        if name in PATH_FIELDS and value is not None:
            candidate = Path(value).expanduser()
            value = candidate if candidate.is_absolute() else path.parent / candidate
        values[name] = value
    values["config"] = path
    return Namespace(**values)
