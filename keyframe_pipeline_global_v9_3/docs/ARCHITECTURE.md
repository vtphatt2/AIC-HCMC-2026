# Architecture

How the two phases actually schedule work on CPU and GPU. Source of truth is the code;
this document explains the *shape* of the mechanism so you don't have to reconstruct it
from the threading/queue code every time.

## Overview

```
ZIP (video archive)
   │
   ├── Phase 1: TransNet  ──▶ scenes.json + keyframes.json   (per video)
   │
   └── Phase 2: PE-Core    ──▶ embeddings.npy                (per video, needs Phase 1 output)
```

Both phases decouple **decode/CPU work** from **GPU compute** using a producer/consumer
pattern over a bounded queue, so ffmpeg decoding for the *next* unit of work overlaps GPU
inference on the *current* one. They differ in what "unit of work" and "consumer" mean.

---

## Phase 1 — TransNet scene detection (`src/pipeline/phase1_transnet/phase.py`)

Two schedulers, selected by `--transnet-mode`:

### `global` (default) — streaming, cross-video GPU batching

File: `_run_transnet_phase_global`, decode side in `src/pipeline/phase1_transnet/decode.py::_stream_transnet_entry`.

```
        entry queue                 event queue (maxsize=prefetch_windows)
todo ──▶ [video, video, ...] ──▶  N decode workers ──▶ [StreamStart | Window | StreamDone | DecodeFailure]
                                   (ffmpeg per video,                    │
                                    --transnet-decode-workers            ▼
                                    threads pulling from                main thread:
                                    the entry queue)                    GPU batcher loop
```

- Each decode worker thread runs ffmpeg on one video at a time (pulling the next video off
  a shared task queue once its current one finishes), reading stdout **incrementally** and
  emitting a `TransNetWindowItem` event as soon as a full 100-frame window is ready — it does
  **not** wait for the whole video to decode. A window is 25 frames left-context + 50 frames
  of actual output + 25 frames right-context, matching TransNetV2's padding semantics exactly.
  First window ships after ~75 decoded frames.
- All workers push into one shared `event_q` (bounded, `maxsize = max(--transnet-prefetch-windows, --transnet-batch-size)`).
  This is where cross-video mixing happens: windows from video A and video B interleave in
  the same queue purely based on decode speed, no explicit round-robin.
- The **main thread** is the GPU batcher: it drains `event_q`, accumulates `TransNetWindowItem`s
  from *any* video into a `pending` list, and runs a GPU forward pass as soon as
  `len(pending) >= --transnet-batch-size`. A single GPU batch can therefore contain windows
  from several different videos at once — that's the "global" in global batching.
- If the queue goes quiet for `--transnet-batch-timeout-ms`, the batcher flushes whatever is
  pending (`process_pending(force=True)`) instead of leaving the GPU idle waiting for a full
  batch. This is a latency/throughput trade-off knob, not a correctness one.
- Per-video state (`states[video_id]`) tracks how many of its windows have come back from the
  GPU; a video finalizes (`maybe_finalize`) only once every expected window for it has been
  processed, regardless of what other videos were mixed into the same batches.
- `transnet_performance_summary.json` (written to `--out-dir`) reports the real fill rate:
  `average_batch_size` / `batch_fill_ratio` tell you whether decode workers are actually
  keeping the queue fed, and `gpu_idle_wait_seconds` vs `wall_seconds` tells you how much of
  the phase the GPU spent waiting. See `scripts/bench_batch_size.sh` / manual runs for how to
  read these — low fill ratio means decode-bound (more `--transnet-decode-workers`, up to CPU
  core count), not a batch-size problem.

### `sequential` (legacy v8 path)

File: `_run_transnet_phase_sequential`. One video fully decoded (`_decode_transnet_entry`,
blocks until ffmpeg exits), then inferred, then the next video. No overlap, no cross-video
batching. Kept only as an A/B correctness/performance reference
(`tools/compare_transnet_outputs.py`), not for production runs.

---

## Phase 2 — PE-Core embedding (`src/pipeline/phase2_embed/phase.py`, `src/pipeline/phase2_embed/producer.py`)

Single scheduler, always producer/consumer (no legacy sequential mode):

```
                     batch_queue (maxsize = --prefetch-batches)
producer thread ──▶ [TensorBatch | VideoEnd | ProducerFailure] ──▶ main thread (GPU consumer)
(embedding_producer)                                                (run_embedding_phase)
```

- **Producer** (`embedding_producer`, one background thread): iterates videos **sequentially**
  (only one video's ffmpeg decode running at a time — unlike Phase 1, there's no concurrent
  multi-video decode here). For each video it decodes only the *requested* keyframes
  (`_decode_selected_frames`, adaptive `--embed-decode-mode auto|sequential|seek`), resizes/crops
  per the encoder's own preprocessing config (`--embed-ffmpeg-preprocess`, offloaded into the
  ffmpeg filter graph so the GPU/CPU-normalize path doesn't have to touch full-resolution frames),
  and accumulates decoded tensors into a batch.
- The accumulator **intentionally survives video boundaries**: when video A ends with, say,
  9 keyframes accumulated and `--batch-size` is 64, the producer does *not* flush a partial
  batch — it keeps accumulating into video B's keyframes until it reaches 64, then flushes one
  full `TensorBatch` that spans both videos. This is what keeps GPU batches consistently full
  even for a video with few keyframes, at the cost of *not* decoding multiple videos concurrently
  (that concurrency isn't needed here because decode is fast relative to keyframe count; Phase 1
  needs it because a single video already has thousands of frames to decode).
- Each `TensorBatch` carries `video_ids`/`positions` per row, since one batch can contain rows
  from more than one video — the consumer scatters results back per-video by that mapping.
- **Consumer** (main thread, `run_embedding_phase`): pulls batches off `batch_queue` (bounded by
  `--prefetch-batches`, so the producer can prepare that many batches ahead before blocking —
  this is the overlap knob), runs the PE-Core forward pass under `--amp`/`--tf32`, normalizes,
  and scatters rows into a per-video `np.memmap` (`embeddings.partial.npy`). A video finalizes
  (renamed to `embeddings.npy`) once its `VideoEnd` marker has arrived *and* every expected row
  has been written — which may happen after a batch that also contains the next video's rows.
- `--prefetch-batches` is the CPU/GPU overlap depth: `1` means the producer can prepare one
  batch ahead while the GPU consumes the current one; higher values buffer more decode work
  ahead of GPU consumption (more RAM for pinned tensors, smoother against decode-speed jitter).
- Embed-phase throughput/overlap diagnostics are in `[embed metrics]` (stdout) and per-video
  `embedding_stats.json` (`timing_seconds.gpu_embedding`, `num_keyframes`) — this is what
  `scripts/bench_batch_size.sh` reads to compare `--batch-size` values.

---

## What's *not* parallel

- **Multiple GPUs**: neither phase shards across GPUs by itself. `--num-shards`/`--shard-index`
  exist in `src/pipeline/cli.py` for deterministic dataset splitting, but `run_pipeline.sh` always
  runs a single process on one `--device`. Multi-GPU means running one process per GPU with a
  different `--shard-index` — see `scripts/run_kaggle_t4x2.sh` for the pattern (two independent
  processes, no DDP/NCCL).
- **Multiple datasets**: `scripts/run_many.sh` runs `run_pipeline.sh` once per URL, one dataset
  fully to completion (both phases) before starting the next. No cross-dataset overlap.
- **Phase 1 and Phase 2 of the same dataset**: strictly sequential — Phase 2 needs Phase 1's
  `scenes.json`/`keyframes.json` on disk first.
