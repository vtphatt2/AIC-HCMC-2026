# Changelog

Newest first.

## v9.3 — encoder-aware keyframe decode / preprocess
- FFmpeg drops non-keyframes before stdout; only requested keyframes cross the pipe.
- Encoder resize/crop/mean/std are derived from `resolve_model_data_config(model)` for standard timm transforms; FFmpeg does the resize + center-crop. Falls back to legacy PIL preprocessing for nonstandard transforms.
- Selected frames stay `uint8 CHW` on CPU; batches are pinned, copied async to CUDA, then cast + normalized on GPU (~4x less H2D traffic than float32 batches).
- New adaptive `--embed-decode-mode auto|sequential|seek` (groups nearby keyframes and seeks when estimated savings are high enough).
- Embedding dim now read from `model.num_features` instead of hard-coded.
- New flags: `--[no-]embed-ffmpeg-preprocess`, `--embed-decode-mode`, `--embed-seek-gap-seconds`, `--embed-seek-min-savings`, `--embed-seek-max-groups`.

## v9.2 — true streaming TransNet
- Global TransNet batch mode no longer waits for a full video: ffmpeg streams frames and emits temporal windows as they become available (first window ~75 decoded frames, then every 50).
- Fixed `predict_raw()` input to `torch.uint8 [B,T,27,48,3]` on the selected device.
- Added bounded `--transnet-prefetch-windows` queue and `--transnet-batch-timeout-ms` partial-batch flush.
- `--transnet-active-videos` / `--transnet-prefetch-videos` are now legacy no-ops in streaming-global mode.

## v9.1 — Colab venv hotfix
- If `ensurepip` fails on Colab, retry venv creation with `--without-pip --system-site-packages`.
- Verify pip is available before installing deps; `runtime_env.sh` honors the `.runtime_python` file written by `setup.sh`.

## v9 — global TransNet batching
- TransNet windows batch across multiple videos on the GPU instead of one-video-at-a-time (`--transnet-mode global`, default).
- Kept exact 25-left / 50-output / 25-right padding semantics per 100-frame window.
- Adds parallel low-resolution decode workers (`--transnet-decode-workers`).
- `--transnet-mode sequential` keeps the old per-video path for A/B correctness checks (see `tools/compare_transnet_outputs.py`).

## v8 — workspace safety + telemetry
- Safe persistent workspace resolver: `--work-root`, then `/kaggle/working`, `/content`, writable `/workspace`, else `$HOME/workspace`.
- Preflight write tests on work root / source / output / archive before any GPU work starts.
- `run_many.sh` added for sequential multi-dataset runs (`--url` repeatable or `--urls-file`).
- `package_results.sh` requires an explicit output dir when more than one `output_*` exists.
- Adds `performance_summary.json` (end-to-end keyframes/s, GPU img/s, CPU preprocess keyframes/s, wall time, TransNet timing).

## v7 — project-local venv
- `setup.sh` creates a PEP-668-safe venv with `--system-site-packages` (reuses preinstalled CUDA PyTorch), auto-installs deps and Torch if missing.
- `runtime_env.sh` added; every launcher auto-selects `.venv/bin/python`, then `python3`, then `python`. No manual venv activation needed.
- `install.sh` becomes a compatibility shim delegating to `setup.sh`.

## v6 — multi-GPU sharding
- Deterministic video sharding: `--num-shards`, `--shard-index`; both TransNet and PE-Core phases are sharded.
- `run_kaggle_t4x2.sh` added: two independent single-GPU processes (no DDP), tuned defaults for Kaggle T4 x2.
- Shard-specific run summaries avoid multi-process write races.

## v5 — mixed precision + Colab
- PE-Core weights always stay FP32; true inference autocast via `--amp auto|off|bf16|fp16` (`auto` prefers BF16, else FP16 on CUDA).
- `--tf32` / `--no-tf32` control added.
- Optional `--compile` (`torch.compile(mode="reduce-overhead")`).
- Fixed recursive checkpoint resolution for nested Kaggle model layouts.
- `colab` storage profile + `run_colab.sh` launcher added.
- `check_platform.py` and `compare_embeddings.py` utilities added.
