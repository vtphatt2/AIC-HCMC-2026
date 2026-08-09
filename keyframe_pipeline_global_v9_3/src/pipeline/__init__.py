"""Keyframe pipeline: TransNetV2 scene/keyframe detection + PE-Core embedding.

Package layout:
    io_utils            atomic JSON write, video-id sanitizing (shared)
    zip_source          ZIP video listing + ffmpeg subfile:// materialization (shared)
    video_probe          ffprobe wrapper (fps/width/height) (shared)
    cli                   argument parsing + end-to-end entrypoint (shared)
    phase1_transnet/        Phase 1: TransNetV2 scene/keyframe detection
        decode                frame decode (sequential + streaming), keyframe selection
        model                  TransNetV2 loading and GPU batch inference
        phase                   orchestration (sequential and streaming-global modes)
    phase2_embed/            Phase 2: PE-Core embedding
        compute                AMP/TF32 policy
        preprocess              encoder-aware ffmpeg resize/crop plan
        model                    PE-Core loading (Hugging Face / Kaggle / local)
        decode                    selected-keyframe ffmpeg decode (sequential/seek)
        producer                   CPU producer thread -> GPU batch queue
        phase                       orchestration (GPU consumer, per-video finalize)
    download/                 standalone download-while-decoding tool
        stream_download          streams a ZIP down while running Phase 1 on completed entries
"""
