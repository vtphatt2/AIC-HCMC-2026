"""Phase 2: PE-Core keyframe embedding.

    compute      AMP/TF32 policy
    preprocess   Encoder-aware ffmpeg resize/crop plan
    model        PE-Core loading (Hugging Face / Kaggle / local)
    decode       Selected-keyframe ffmpeg decode (sequential/seek)
    producer     CPU producer thread -> GPU batch queue
    phase        Orchestration: GPU consumer loop, per-video finalize
"""
