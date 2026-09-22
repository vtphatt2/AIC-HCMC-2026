# Keyframe Pipeline (v9.3)

Turns a ZIP of videos into per-video scene/keyframe metadata (TransNetV2) and PE-Core vision
embeddings for each keyframe. Built for CCTV/AIC-style CFR video on a single CUDA GPU
(with an optional multi-GPU shard mode).

- **Phase 1 — TransNetV2**: scene boundaries + selected keyframes per video, GPU-batched across
  videos (`--transnet-mode global`, default).
- **Phase 2 — PE-Core embedding**: FFmpeg decodes only the requested keyframes, resized/cropped
  from the encoder's own preprocessing config, then batched onto the GPU for embedding.

Both phases overlap CPU decode and GPU compute via producer/consumer queues and can mix work from
multiple videos into one GPU batch. They also overlap each other by default: as soon as TransNet
publishes one video's metadata, PE-Core may consume that video while TransNet continues with the
rest of the ZIP. See **`docs/ARCHITECTURE.md`** for the full mechanism and VRAM trade-off before
tuning `--transnet-*`/`--prefetch-batches`/`--batch-size` on a new machine.

`src/process_video_zip_gpu.py` and `src/stream_download_transnet.py` are thin CLI entrypoints
(launchers below call these, not repo-root scripts); the actual implementation lives in
`src/pipeline/`:

```
src/
  process_video_zip_gpu.py       # entrypoint: --phase transnet|embed|all
  stream_download_transnet.py    # entrypoint: download-while-decoding
  pipeline/
    cli.py, io_utils.py, video_probe.py, zip_source.py   # shared
    phase1_transnet/   decode.py, model.py, phase.py     # Phase 1 (TransNetV2)
    phase2_embed/      compute.py, preprocess.py, model.py, decode.py, producer.py, phase.py  # Phase 2 (PE-Core)
    download/          stream_download.py                # download-while-decoding tool
```

See each module's docstring for what it owns, and `docs/ARCHITECTURE.md` for how Phase 1/2
scheduling and batching work.

## Setup

```bash
bash setup.sh      # creates .venv (--system-site-packages) and installs deps + ffmpeg/aria2
```

Every launcher auto-picks the right Python (`.venv` → `python3` → `python`) via `runtime_env.sh`.
No manual `source .venv/bin/activate` needed.

The PE-Core checkpoint downloads from Hugging Face by default (`--model-source huggingface`,
`hf_transfer` installed by `setup.sh` for multi-connection speed). A free HF token raises the
anonymous rate-limit tier -- `export HF_TOKEN=... HF_HUB_ENABLE_HF_TRANSFER=1` before running.
Not required for a public model like this one to work, just to go faster.

## Quick start

```bash
bash run_pipeline.sh --url https://host/Videos_L28_a.zip \
  --profile balanced --batch-size 64 --prefetch-batches 3
```

Or with an already-downloaded ZIP: `bash run_pipeline.sh --zip /path/videos.zip --profile balanced`.

Full option list: `bash run_pipeline.sh --help`.

Inter-stage streaming is enabled by default (`--parallel-stages`). Use `--sequential-stages` only
for an A/B baseline or when GPU memory cannot hold both model processes at once.

Keyframe count defaults to the existing `tiered` scene policy (1/3/5 frames for
scenes of <=3s, <=10s, and >10s). Duration-linear sampling is available with:

```bash
bash run_pipeline.sh --zip /path/Video_N001-N010.zip \
  --keyframe-strategy linear --keyframes-per-second 0.3 \
  --min-keyframes-per-scene 1 --max-keyframes-per-scene 20
```

Linear counts use `ceil(scene_seconds * rate)` and are clamped per scene. Frames
remain evenly spaced inside each detected scene. The configuration is recorded
in `keyframes.json`; changing it invalidates stale keyframes and embeddings.

### Storage profiles

| Situation | Profile | ZIP | HF cache | Temp |
|---|---|---|---|---|
| 15G `/dev/shm`, ~17G disk | `balanced` (default) | disk | shm | shm |
| Large `/dev/shm` | `ram-rich` | shm | shm | shm |
| Large disk, small shm | `disk-rich` | disk | disk | disk |
| Google Colab | `colab` | `/content` | `/content` | `/content` |

Any location can be overridden with `--source-dir`, `--hf-cache-dir`, `--temp-dir`. Compute knobs
(`--amp`, `--tf32`, `--compile`, `--batch-size`, `--prefetch-batches`, `--device`) are independent
of storage.

### Output

`run_pipeline.sh` writes to `--out-dir` (auto-derived from the ZIP name) and packages a result ZIP
via `package_results.sh`:

```
<out-dir>/<video_id>/scenes.json
<out-dir>/<video_id>/keyframes.json
<out-dir>/<video_id>/embeddings.npy      # embeddings[i] <-> keyframes[i]
<out-dir>/performance_summary.json
<result>.zip
  phase1_transnet/<video_id>/{scenes,keyframes}.json
  phase2_embeddings/<video_id>/embeddings.npy
  manifest.json
```

## Before running on a new rented server

Disk/RAM/`/dev/shm` size varies a lot between providers (and even between rentals from the same
provider) — `run_pipeline.sh` self-defends against the two failure modes this has actually caused
(source ZIP disk exhaustion, `/dev/shm` too small for the PE-Core cache: both now fail loud or
fall back automatically instead of crashing partway through an unattended run). What is **not**
automatic is performance tuning — batch size and decode worker count depend on the specific GPU/
CPU and are deliberately not auto-tuned at runtime (see `docs/ARCHITECTURE.md` for why). Checklist
for a server you haven't used before:

```bash
bash setup.sh                                   # also bundles TransNet's weights (no separate download)
bash tools/check_machine.sh && nproc            # see real RAM/disk/GPU/CPU cores
bash scripts/fetch_hf_model_fast.sh             # pre-fetch PE-Core once, into ./hf_model
bash scripts/test_multi_link_fake.sh --count 2  # sanity-check storage end-to-end, ~minutes
python tools/make_real_sample_zip.py --url <a real dataset URL> --out sample.zip --videos 5
bash scripts/bench_batch_size.sh --zip ./sample.zip --out-dir ./bench_out \
  --model-source local --model-dir ./hf_model   # pick --batch-size
```

Check `docs/HARDWARE_NOTES.md` for known gotchas on this GPU/provider before that — and add a
section there afterward if you find something new.

**Pre-fetching PE-Core once with `--model-source local --model-dir ./hf_model`** (instead of
letting `--model-source huggingface`/`kaggle` download it lazily on first use) removes the only
runtime download left in the pipeline — nothing downloads mid-job once the ZIP source itself is
fetched. TransNet needs no equivalent step; its weights install with the `transnetv2-pytorch` pip
package in `setup.sh`. If HF/Kaggle are both slow from a given rental (seen: ~13MB/s to both even
with aria2c multi-connection — a route/bandwidth ceiling from that specific rental, not a
provider problem), see `docs/HARDWARE_NOTES.md`; `scripts/fetch_kaggle_model_fast.sh` and
`scripts/fetch_model_url.sh` (any plain HTTP(S) URL, e.g. your own object storage) are the other
options, but neither beats a slow *route*, only a slow *client*.

Then run the real job with `--profile` matching `check_machine.sh`'s numbers (table below),
`--transnet-decode-workers` around the real `nproc` (not the default `4`), `--batch-size` from the
bench run, and `--model-source local --model-dir ./hf_model` (or wherever you pre-fetched it) so
the real job never triggers a download. Pick a rental with enough disk for PE-Core (~8G, fixed,
doesn't grow) plus one source ZIP at a time (~6-7G, auto-cleaned between datasets) plus OS/venv
overhead — comfortably over ~20G total, more if you'd rather not think about it precisely.
`test_multi_link_fake.sh` is worth rerunning whenever the server "shape" changes (new
provider/instance type) — not before every job on a server you've already validated.

If PE-Core downloads slowly despite `HF_HUB_ENABLE_HF_TRANSFER=1` (stale venv missing the
`hf_transfer` package, or a throttled route to HF's CDN from your region), fetch it directly via
aria2c against HF's public resolve URL instead:
`bash scripts/fetch_hf_model_fast.sh` (writes to `./hf_model`; then pass `--model-source local
--model-dir ./hf_model --model-arch vit_pe_core_gigantic_patch14_448`).

## Other launchers (`scripts/`)

All forward to (or share code with) `run_pipeline.sh` / `process_video_zip_gpu.py`:

- `run_colab.sh` — Colab convenience wrapper (`--profile colab`, sane batch defaults).
- `run_many.sh` — run multiple datasets sequentially (`--url` repeated or `--urls-file`).
- `run_kaggle_t4x2.sh` — shards one ZIP across 2 GPUs (e.g. Kaggle T4x2), no DDP.
- `run_ram_pipeline.sh` — streams download + TransNet concurrently, ZIP/cache live in RAM.
- `run_embed_from_disk_zip_in_ram.sh` — copy an existing ZIP into RAM, skip TransNet, embed only.
- `run_embed_only.sh` — run Phase 2 only, against an existing ZIP + out-dir.
- `run_transnet_global.sh` — download a ZIP from a URL and run Phase 1 only
  (`--transnet-mode global`), for quick TransNet-only smoke tests
  (`bash scripts/run_transnet_global.sh --url ... --limit 5`).
- `bench_batch_size.sh` — sweep `--batch-size` for the embed phase on a small
  `--limit` subset, reporting GPU-util average and images/s per size, to pick
  a batch size by evidence instead of guessing
  (`bash scripts/bench_batch_size.sh --zip ./source.zip --out-dir ./bench_out`).
- `test_resilience.sh` — fast (seconds-to-minutes) checks for the disk-space
  preflight and multi-dataset ZIP cleanup, so you don't have to run a full
  multi-hour unattended job to find out they broke
  (`bash scripts/test_resilience.sh disk-preflight --url ...` /
  `bash scripts/test_resilience.sh cleanup --url ... --url ...`).
- `make_fake_zip.sh` / `test_multi_link_fake.sh` — generate tiny synthetic
  videos (ffmpeg testsrc), serve them from a local HTTP server, and run
  `run_many.sh` over those fake URLs end-to-end: no real dataset, no real
  network dependency, but the real download/preflight/decode/TransNet/embed
  code paths (`bash scripts/test_multi_link_fake.sh --count 3
  --continue-on-error`). Still loads the real PE-Core model (one-time
  download cost).

Each supports `--help`.

## Diagnostics (`tools/`)

- `check_machine.sh` / `check_platform.py` — RAM/disk/GPU/AMP capability report.
- `make_real_sample_zip.py` — build a small real-video sample ZIP for `bench_batch_size.sh`
  without downloading a full multi-GB dataset ZIP: streams just enough of the remote ZIP to
  capture N complete video entries, repackages them into a fresh small ZIP, and stops the
  download early (`python tools/make_real_sample_zip.py --url https://host/Videos_X.zip --out
  sample.zip --videos 5`).
- `compare_transnet_outputs.py` — A/B check `sequential` vs `global` TransNet mode.
- `compare_embeddings.py` — cosine-similarity diff between two embedding output dirs (e.g. FP32
  baseline vs AMP/TF32 candidate).
- `gpu_stress_test.py` — load the real TransNet or PE-Core model and run its GPU forward pass
  against synthetic random input (no ZIP/disk needed) to isolate GPU-side bugs (OOM ceiling, VRAM
  leak across iterations) from decode/disk issues in minutes:
  `python tools/gpu_stress_test.py transnet --ramp --iters 20` finds the largest batch size before
  OOM; `python tools/gpu_stress_test.py embed --batch-size 64 --iters 200` checks for VRAM growth
  (`alloc_drift_mb`) over many iterations at a fixed size. Run it twice back-to-back as separate
  processes to confirm VRAM is fully released between them (the run_many.sh dataset-to-dataset
  scenario).

## Reference

- `docs/ARCHITECTURE.md` — how Phase 1/Phase 2 scheduling, batching, and CPU/GPU overlap
  actually work.
- `docs/PROFILES.md` — storage profile cheat sheet.
- `docs/HARDWARE_NOTES.md` — per-GPU/instance-type gotchas found in practice (T4, RTX 5060 Ti,
  Vast.ai rentals, ...); append to it when a new machine type turns up something new.
- `docs/CHANGELOG.md` — condensed v5 → v9.3 history.
