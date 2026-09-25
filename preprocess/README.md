# Preprocess: ZIP-native video and transcript pipeline

This directory owns the public preprocessing command:

```bash
python -m preprocess ...
```

It converts one organizer video ZIP into a compact, challenge-compatible
`*_results.zip` without extracting the whole ZIP or persisting frame images.
It can concurrently collect YouTube captions from media-info metadata and
clean them with Gemini.
v
Archive names are normalized before output paths are made: historical
\`Videos_L26_c.zip\` becomes \`L26_c\`, while current
\`Video_N001-N010.zip\` becomes \`N001-N010\`. The latter therefore defaults to
\`<work-root>/N001-N010_results.zip\`.

The GPU implementation is deliberately isolated in
`../keyframe_pipeline_global_v9_3/`. This package is the stable facade:
argument validation, concurrent orchestration, transcript tools, and final
archive validation.

## Contents

- [What the system does](#what-the-system-does)
- [Requirements and one-time setup](#requirements-and-one-time-setup)
- [Quick start](#quick-start)
- [Run configuration](#run-configuration)
- [Transcript workflow](#transcript-workflow)
- [Output contract](#output-contract)
- [Runbook](#runbook)
- [Troubleshooting and recovery](#troubleshooting-and-recovery)
- [Legacy transcript utilities](#legacy-transcript-utilities)

## What the system does

```text
Videos_Lxx_y.zip (local path or URL)
       │
       ├── ZIP_STORED entry: FFmpeg reads the MP4 directly by byte range
       └── compressed entry: materialize one MP4 temporarily, then delete it
       │
       ▼
TransNetV2: decode → scene cuts → selected-frame metadata
       │ atomic per-video JSON
       ▼
ready queue ──► PE-Core: decode selected frames → embed batches
       │
       ▼
package: scenes.json + keyframes.json + embeddings.npy → *_results.zip

media-info JSON/ZIP ──► YouTube captions ──► raw JSONL ──► Gemini-cleaned JSONL
       (this entire branch starts in parallel with the video branch)
```

### Concurrency model

- TransNet uses multiple FFmpeg decode workers and global GPU batches.
- As soon as one video publishes valid scene/keyframe metadata atomically,
  PE-Core begins decoding and embedding that video. It does **not** wait for
  TransNet to finish the complete lot.
- PE-Core overlaps CPU decode/preprocessing of the next batch with GPU inference
  of the current batch.
- Caption collection and Gemini cleaning also form a producer/consumer flow:
  each atomically published raw JSONL can be cleaned while captions for other
  videos are still downloading.
- The final result package is the only unavoidable barrier.

The default is therefore the lowest-idle-time mode. Use sequential stages only
when the GPU cannot retain both TransNetV2 and PE-Core processes at once.

## Requirements and one-time setup

Run all commands from the repository root.

### Video engine

The video engine needs CUDA-capable PyTorch, FFmpeg, FFprobe, and optionally
aria2. Its setup script creates its own environment:

```bash
bash keyframe_pipeline_global_v9_3/setup.sh
bash keyframe_pipeline_global_v9_3/tools/check_machine.sh
```

`setup.sh` installs/checks FFmpeg and aria2 where possible, creates
`keyframe_pipeline_global_v9_3/.venv`, and records the Python interpreter for
the engine launchers. Do not mix this environment with a system Python install.

For a new GPU/server, read
[`../keyframe_pipeline_global_v9_3/docs/HARDWARE_NOTES.md`](../keyframe_pipeline_global_v9_3/docs/HARDWARE_NOTES.md)
and use a small smoke run before a full lot.

### Transcript tools

Video-only runs use only the standard library in this facade. Caption collection
and Gemini cleaning require a small separate environment:

```bash
uv venv --python 3.12 preprocess/.venv
source preprocess/.venv/bin/activate
uv pip install -r preprocess/requirements.txt
export GEMINI_API_KEY='your-key'
```

Use `preprocess/.venv/bin/python -m preprocess` for every command that has a
`--transcript-*` option. The key is read only from the environment; never put
it in a command line, config file, or repository file.

Optional variables for faster model download:

```bash
export HF_TOKEN='optional-hugging-face-token'
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Quick start

### Recommended: a JSON configuration per lot

Copy [`run.example.json`](run.example.json), set the source and output paths,
then keep the resulting file beside the run log. The full run is reproducible
with one command:

```bash
cp preprocess/run.example.json /data/aic-runs/N001-N010.json
preprocess/.venv/bin/python -m preprocess run --config /data/aic-runs/N001-N010.json
```

`zip` is a local source; use `url` instead for a downloaded source. Paths inside
the JSON may be absolute or relative to the config file. The JSON keys use the
same snake_case names as the CLI flags, for example
`keyframe_strategy`, `transnet_decode_workers`, and
`transcript_metadata`. `--dry-run` can be appended to inspect the resolved
commands. Do not put `GEMINI_API_KEY` in this file.

`max_keyframes_per_scene: 0` means **unlimited**. It should only be used when
the expected embedding cost for long scenes is acceptable.

### Video only, local ZIP

```bash
python3 -m preprocess run \
  --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a
```

### Video only, download source ZIP

```bash
python3 -m preprocess run \
  --url 'https://host.example/Videos_L30_a.zip' \
  --work-root /data/aic-preprocess/L30_a
```

The archive defaults to:

```text
<work-root>/L30_a_results.zip
```

### Video plus transcript collection and cleaning

```bash
preprocess/.venv/bin/python -m preprocess run \
  --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a \
  --transcript-metadata challenge_resources/data/zip_embeddings/media-info-aic25-b1.zip \
  --transcript-concurrency 8
```

This starts the video and transcript branches at the same time. For
`Videos_L26_c.zip`, the metadata filter is inferred as `^L26_V2\\d{2}$`;
only that block is collected.

### Validate without running

```bash
# Print the exact video/transcript subprocesses; no GPU, download, or Gemini call.
python3 -m preprocess run \
  --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a \
  --dry-run

# Validate an existing result archive without extracting it.
python3 -m preprocess inspect /data/aic-preprocess/L30_a/L30_a_results.zip
```

## Run configuration

### Facade options

| Option | Default | Meaning |
| --- | --- | --- |
| `--url URL` / `--zip PATH` | required, exactly one | Source organizer ZIP. URL downloads a complete ZIP once; local ZIP is used in place. |
| `--work-root PATH` | `data/zip-preprocess` | Persistent working root and default final archive location. Use a dedicated directory per lot. |
| `--archive PATH` | `<work-root>/<lot>_results.zip` | Explicit final result archive path. |
| `--profile` | `balanced` | `balanced`, `ram-rich`, `disk-rich`, or `colab`; controls engine storage defaults. |
| `--device DEVICE` | `cuda` | Torch device passed to the engine, normally `cuda`. |
| `--batch-size N` | 64 | PE-Core global GPU batch size. Lower this first after CUDA OOM. |
| `--prefetch-batches N` | 3 | CPU-prepared embedding batches kept ahead of GPU inference. |
| `--transnet-batch-size N` | 16 | Number of 100-frame TransNet windows per GPU forward. |
| `--transnet-decode-workers N` | 4 | Concurrent FFmpeg decoders for TransNet. Tune to available CPU/IO. |
| `--transnet-prefetch-windows N` | 256 | Decoded TransNet windows buffered ahead of GPU work. |
| `--keyframe-strategy tiered\|linear` | `tiered` | Per-scene keyframe count policy. `tiered` preserves the existing 1/3/5 rule; `linear` scales with scene duration. |
| `--keyframes-per-second N` | 0.3 | Linear rate: `ceil(scene_seconds × N)`. |
| `--min-keyframes-per-scene N` | 1 | Lower bound for each scene in linear mode. |
| `--max-keyframes-per-scene N` | 20 | Upper bound for each scene in linear mode; `0` disables the cap. |
| `--limit N` | unset | Process only the first N videos; use for smoke tests. |
| `--sequential-stages` | off | Disable per-video TransNet → PE-Core streaming to reduce concurrent VRAM demand. |
| `--delete-source-before-package` | off | Delete only a **downloaded** source ZIP after embedding, before packaging. Do not use when preserving source is required. |
| `--local-files-only` | off | Fail instead of downloading PE-Core assets; use a pre-populated local cache. |
| `--dry-run` | off | Print commands only. |

The facade intentionally exposes safe, frequently used controls. For advanced
engine configuration (model source, cache paths, AMP/TF32, aria2 selection,
connections, decoder seeking, or `--keep-temp`), call the engine directly:

```bash
bash keyframe_pipeline_global_v9_3/run_pipeline.sh --help
```

Important advanced controls include:

| Engine option | Use when |
| --- | --- |
| `--download-engine auto|aria2|python` | `auto` uses aria2 when installed; choose `aria2` to require it. |
| `--connections N` | Set aria2 parallel connections; engine default is 16. |
| `--model-source huggingface|kaggle|local` | Use `local` with `--model-dir` to avoid model download during an unattended run. |
| `--hf-cache-dir PATH`, `--temp-dir PATH`, `--source-dir PATH` | Override profile storage placement. |
| `--amp auto|off|fp16|bf16`, `--tf32`, `--compile` | Tune GPU execution only after a small validated benchmark. |
| `--keep-temp` | Preserve temporary materials for diagnosis; normally they are removed. |

### Profile selection

| Machine condition | Profile | Intended placement |
| --- | --- | --- |
| Normal persistent disk and at least 12 GiB free `/dev/shm` | `balanced` | ZIP on disk, model cache/temp in shared memory. |
| Large shared memory | `ram-rich` | ZIP, cache, and temp in shared memory. |
| Small shared memory, sufficient disk | `disk-rich` | Everything on persistent disk. |
| Google Colab | `colab` | Persistent paths under `/content`. |

The engine validates writable paths and avoids an undersized `/dev/shm` cache.
Choose `disk-rich` on a small-memory VM instead of forcing RAM paths.

### Recommended per-run configurations

```bash
# 1. Smoke test: validates source, model, GPU, and output contract.
python3 -m preprocess run --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/smoke-L30_a --limit 2

# 2. Standard one-GPU run.
python3 -m preprocess run --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a \
  --profile balanced --batch-size 64 \
  --transnet-decode-workers 4 --transnet-prefetch-windows 256

# 3. Low-VRAM run: reduce memory pressure, sacrificing overlap/throughput.
python3 -m preprocess run --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a \
  --profile disk-rich --batch-size 16 --prefetch-batches 1 \
  --transnet-batch-size 4 --transnet-decode-workers 2 \
  --sequential-stages

# 4. Linear scene sampling: about 3 frames per 10 seconds, no per-scene cap.
python3 -m preprocess run --zip /data/Video_N001-N010.zip \
  --work-root /data/aic-preprocess/N001-N010 \
  --keyframe-strategy linear --keyframes-per-second 0.3 \
  --min-keyframes-per-scene 1 --max-keyframes-per-scene 0

# 5. Downloaded source, preserve all evidence for debugging.
bash keyframe_pipeline_global_v9_3/run_pipeline.sh \
  --url 'https://host.example/Videos_L30_a.zip' \
  --work-root /data/aic-preprocess/L30_a \
  --profile disk-rich --download-engine aria2 --connections 16 --keep-temp
```

Never reuse the same `--work-root` concurrently for two lots. Use one
directory per lot and one final archive path per lot.

The tiered policy selects 1 frame for scenes up to 3 seconds, 3 frames for
scenes up to 10 seconds, and 5 frames for longer scenes. The linear policy
computes `ceil(duration × keyframes-per-second)`, clamps it to the configured
minimum/maximum (or has no upper bound when maximum is `0`), and positions
frames evenly at sub-interval centers. Selection
settings are stored in every `keyframes.json`. Reusing a work root with a
different policy automatically invalidates stale keyframe and embedding files.

## Transcript workflow

### Inputs and outputs

The collector reads organizer media-info as a directory, JSON file, or ZIP
directly; it does not extract the whole metadata ZIP. It recognizes
`watch_url`, `video_link`, or `url` containing a supported YouTube URL.

```text
<work-root>/
├── transcripts_raw/<lot>/<video_id>.jsonl    # source captions; atomic publish
├── transcripts_raw/<lot>/collection_failures.ndjson
├── transcripts_raw/<lot>/missing_transcript_ids.txt
├── clean_transcript/<video_id>.jsonl          # Gemini-cleaned output
└── .clean_transcript_state/                    # resumable per-video state
```

Raw and cleaned files are separate. A successful clean preserves timing and
metadata; only the text field is changed.

### Transcript options on `run`

| Option | Meaning |
| --- | --- |
| `--transcript-metadata PATH` | Collect captions from media-info then clean them; mutually exclusive with `--transcripts`. |
| `--transcripts PATH` | Clean existing JSONL file/directory in parallel with video work. |
| `--transcript-raw-dir PATH` | Override collector raw output directory. |
| `--transcript-output-dir PATH` | Override cleaned JSONL output directory. |
| `--transcript-state-dir PATH` | Override persistent Gemini checkpoint directory. |
| `--transcript-model NAME` | Gemini model; default `gemini-3.5-flash-lite`. |
| `--transcript-concurrency N` | Concurrent caption requests and Gemini jobs; default 8. |
| `--transcript-rpm N`, `--transcript-tpm N` | Proactive Gemini limits; 0 disables each limit. |
| `--transcript-video-id-regex REGEX` | Explicit metadata filter; otherwise derived from ZIP lot name. |
| `--transcript-overwrite` | Recollect/reclean existing files. Use only when intentionally replacing output. |

### Standalone transcript commands

```bash
# Collect only. Nonzero exit means at least one video failed, while successful
# captions remain written.
preprocess/.venv/bin/python -m preprocess collect-transcripts media-info.zip \
  --output-dir /data/transcripts/raw --concurrency 8

# Clean JSONL already available on disk.
preprocess/.venv/bin/python -m preprocess clean-transcripts /data/transcripts/raw \
  --output-dir /data/transcripts/clean \
  --state-dir /data/transcripts/.state \
  --concurrency 8 --rpm 0 --tpm 0

# Convert prior [HH:MM:SS] TXT transcripts to the JSONL contract.
preprocess/.venv/bin/python -m preprocess transcripts normalize /data/legacy-txt \
  --output-dir /data/transcripts/raw

# Collection + cleaning without video embedding.
preprocess/.venv/bin/python -m preprocess transcripts collect-clean media-info.zip \
  --raw-dir /data/transcripts/raw \
  --output-dir /data/transcripts/clean \
  --state-dir /data/transcripts/.state
```

### Missing YouTube captions

There is no automatic ASR/Whisper fallback. The system must not invent
transcript text. When a video has no caption, captions are disabled, the video
is unavailable, or its metadata has no usable YouTube URL:

- `collection_failures.ndjson` records `video_id`, YouTube ID, and the error.
- `missing_transcript_ids.txt` contains one failed `video_id` per line,
  sorted and deduplicated; this is the input list for a later ASR/retry job.
- Other videos continue; a completed video publishes its JSONL atomically.
- If a later run has no failures, both diagnostic files are removed so they
  cannot be mistaken for current work.

Existing timestamped TXT can be used intentionally through `normalize`; it is
not silently substituted for a missing YouTube caption.

## Output contract

The result archive contains no `.mp4`, JPG, PNG, or other media artifacts.
Its manifest is version 3 for newly generated output.

```text
L30_a_results.zip
├── manifest.json
├── phase1_transnet/<video_id>/scenes.json
├── phase1_transnet/<video_id>/keyframes.json
└── phase2_embeddings/<video_id>/embeddings.npy
```

For each video:

- `scenes.json` records scene segments.
- `keyframes.json` records selected frames and timestamps.
- `embeddings.npy` is a two-dimensional matrix; row `i` maps to keyframe
  `i` in the corresponding keyframe metadata.
- `manifest.json` lists every video artifact and its keyframe count.

`python -m preprocess inspect ARCHIVE` verifies ZIP CRC, rejects media
artifacts, validates every manifest reference, verifies keyframe counts, and
checks all NPY shapes/dimensions without extracting the archive.

## Runbook

### Before every new lot

1. Update the repository and inspect local changes.

   ```bash
   git status --short --branch
   git pull --ff-only origin main
   ```

2. Confirm machine and engine setup.

   ```bash
   bash keyframe_pipeline_global_v9_3/tools/check_machine.sh
   nvidia-smi
   df -h /data /dev/shm
   ```

3. Copy the JSON example, set a unique work root, check the source name, and
   print the planned command.

   ```bash
   python3 -m preprocess run --config /data/aic-runs/N001-N010.json --dry-run
   ```

4. Run a two-video smoke test using a separate work root. Inspect its archive.

   ```bash
   python3 -m preprocess run --zip /data/Videos_L30_a.zip \
     --work-root /data/aic-preprocess/smoke-L30_a --limit 2
   python3 -m preprocess inspect /data/aic-preprocess/smoke-L30_a/L30_a_results.zip
   ```

5. If using transcripts, activate/use the transcript environment and check
   `GEMINI_API_KEY` is present without printing it:

   ```bash
   test -n "${GEMINI_API_KEY:-}" && echo 'GEMINI_API_KEY is set'
   ```

### Start an unattended full run

Use `tmux` or equivalent so an SSH disconnect does not end the job:

```bash
tmux new -s preprocess-L30a
cd /path/to/AIC-HCMC-2026

preprocess/.venv/bin/python -m preprocess run \
  --zip /data/Videos_L30_a.zip \
  --work-root /data/aic-preprocess/L30_a \
  --profile balanced --batch-size 64 --transnet-decode-workers 4 \
  --transcript-metadata challenge_resources/data/zip_embeddings/media-info-aic25-b1.zip \
  --transcript-concurrency 8 \
  2>&1 | tee /data/aic-preprocess/L30_a/run.log
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t preprocess-L30a`.
Avoid `--delete-source-before-package` unless disk pressure is understood and
the downloaded source archive is recoverable.

### During the run

- Watch the terminal/log for TransNet and PE-Core progress and GPU errors.
- Use `nvidia-smi` to confirm expected GPU activity.
- Check capacity with `df -h /data /dev/shm`.
- Do not delete files under the active work root, raw transcript directory, or
  state directory.
- A stop with `Ctrl-c` returns exit code 130. Preserve the work root for
  diagnosis/resume behavior rather than deleting it immediately.

### After a successful run

```bash
RESULT=/data/aic-preprocess/L30_a/L30_a_results.zip
python3 -m preprocess inspect "$RESULT"
unzip -l "$RESULT" | head -30

# If transcripts were enabled:
cat /data/aic-preprocess/L30_a/transcripts_raw/L30_a/missing_transcript_ids.txt \
  2>/dev/null || echo 'No missing YouTube transcripts recorded.'
```

Keep the result archive, run log, and transcript diagnostics. Only clean
temporary/work files after validation and any required upload/ingestion finish.

## Troubleshooting and recovery

| Symptom | Action |
| --- | --- |
| CUDA out of memory | First lower `--batch-size`, then `--prefetch-batches` and `--transnet-batch-size`; use `--sequential-stages` if concurrent models still exceed VRAM. |
| `/dev/shm` too small | Use `--profile disk-rich` or pass explicit cache/temp directories to the engine. |
| Source URL download is slow | Ensure aria2 is installed; run the engine directly with `--download-engine aria2 --connections N`. |
| Model download fails/rate-limited | Set optional `HF_TOKEN`, prefetch the model, then run engine with `--model-source local --model-dir PATH`. |
| `--local-files-only` fails | Populate the engine model cache first, or omit this flag. |
| Result inspection fails | Keep work root and log; rerun `inspect` to get the first invalid artifact. Never upload a ZIP that fails inspection. |
| Some captions fail | Read `collection_failures.ndjson`; use `missing_transcript_ids.txt` for targeted retry/ASR. Other completed transcript files remain valid. |
| Gemini limit/API error | Lower `--transcript-concurrency` and set provider-compatible `--transcript-rpm`/`--transcript-tpm`; preserve state dir to resume clean work. |
| A transcript run finds old raw JSONL | It skips existing raw files by default. Use `--transcript-overwrite` only when replacement is intentional. |

Exit status: 0 means every selected stage completed and the video result passed
validation; 1 means a stage failed or output validation failed; 130 means user
interrupted the facade. Standalone transcript collection returns nonzero if any
video fails, even though successful raw JSONL files are retained.

## Legacy transcript utilities

`preprocess/transcript/` remains only for existing timestamped TXT,
sentence-index, and transcript-banner workflows. It is not part of the ZIP
video output contract and expects legacy keyframe/image layouts. The old
extracted-video pipeline, per-frame image renderer, legacy PE-Core encoder, and
Kaggle uploader have been removed; do not use stale `preprocess.batch` or
`preprocess.pecore` commands.

For complete low-level engine design and hardware tuning, see:

- [GPU engine README](../keyframe_pipeline_global_v9_3/README.md)
- [Engine architecture](../keyframe_pipeline_global_v9_3/docs/ARCHITECTURE.md)
- [Engine profiles](../keyframe_pipeline_global_v9_3/docs/PROFILES.md)
