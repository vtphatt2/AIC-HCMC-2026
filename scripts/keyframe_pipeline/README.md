# Keyframe pipeline

Video &rarr; shot-boundary scenes &rarr; keyframes &rarr; embedding vectors, split across
a CPU-strong local machine and a Colab GPU (T4). This is the pipeline that came out of the
experiments in `../../REPORT.md` — decode locally (fast CPU, no GPU needed for decoding),
run the two GPU-bound steps (TransNetV2, PE-Core) on Colab.

No quantization, no custom Triton/CUDA kernels, no ONNX — plain PyTorch inference on Colab.
The speedups here are all from *where* work happens and *how data moves*, not from lower
precision math (see `../../REPORT.md` for why the custom-kernel route was abandoned).

## Stages

```
1. local_extract_frames.py    (local, CPU)  video.mp4        -> low-res frames/*.npz + meta.json
2. colab_run_transnet.py      (Colab, GPU)  frames/          -> scenes.json
3. select_keyframes.py        (local, CPU)  scenes.json      -> keyframes.json
4. local_extract_keyframes.py (local, CPU)  video.mp4 + keyframes.json -> keyframes/*.jpg
5. colab_embed_pecore.py      (Colab, GPU)  keyframes/*.jpg  -> vectors/*.npy
```

Steps 1, 3, 4 need only `ffmpeg`/`ffprobe` on PATH and numpy — no GPU, no torch. Steps 2
and 5 must run on the Colab VM (need `torch`, `transnetv2_pytorch` / `open_clip`, a GPU).
Transfer chunk files (step 1&rarr;2) and keyframe JPEGs (step 4&rarr;5) to Colab with `scp`;
tar many-small-file transfers first (`../../REPORT.md`: 829s for 586 loose files vs 9s for
one tar of the same content).

### 1. Decode + downscale for shot detection (local)

```
python local_extract_frames.py --video path/to/video.mp4 --out-dir OUT/frames
```

Single ffmpeg pass, scaled to 48x27 (TransNetV2 input size), piped straight from ffmpeg's
stdout (no intermediate files). Consecutive near-duplicate frames are collapsed
(`--dedup-threshold`, default 1.0 mean-abs-diff) and restored losslessly on the Colab side
via `np.repeat` — this only shrinks the transfer payload, TransNetV2 still sees every frame.
Assumes constant frame rate (true for this corpus): frame index alone is enough downstream,
so unlike the earlier "merged Pass 1/2" experiments this does not need a separate PTS scan.

Measured: 28498 frames in ~57-77s (machine dependent; see `../../REPORT.md` exp #6).

**Test small first**: run with `--end 200` and confirm `num_frames == 200` in the printed
meta.json before pointing it at a full video.

### 2. Shot-boundary detection (Colab)

```
python3 colab_run_transnet.py --frames-dir FRAMES --out-dir OUT/scenes
```

Unchanged from the original experiments — reconcatenates the chunked `.npz` files,
re-expands dedup runs, runs TransNetV2, writes `scenes.json` (`start_frame`/`end_frame`
pairs, frame indices are absolute).

### 3. Keyframe selection (local)

```
python select_keyframes.py --scenes SCENES/scenes.json --out-dir OUT/keyframes_index
```

Applies the duration-based ladder from `../../../AIC-HCMC-2026/docs/keyframe_selection.md`:

| scene duration | keyframes | sampling |
|---|---|---|
| &le; 3s | 1 | 50% |
| &le; 10s | 3 | 1/6, 3/6, 5/6 |
| &gt; 10s | 5 | 1/10, 3/10, .., 9/10 |

Pure arithmetic on `scenes.json`, no decode, no GPU. Verified against the reference
`data/transnet-scenes/L01_V002.json` (274 scenes): produces exactly 586 keyframes, matching
`../../REPORT.md`.

### 4. Full-resolution keyframe extraction (local)

```
python local_extract_keyframes.py --video path/to/video.mp4 --keyframes KEYFRAMES/keyframes.json --out-dir OUT/keyframe_jpgs
```

One ffmpeg decode pass for the *whole* keyframe list: the frame list is split into
&le;80-frame groups, each its own `split` branch + `select` filter, each branch mapped to
its own output pattern — same "single decode, N select branches" trick as the experiments'
merged passes, staying under the ~100-200 chained `eq(n,X)` condition limit some local
ffmpeg builds hit (see `../../CLAUDE.md`). Measured: 586 frames in ~22s (exp #5).

**Test small first**: run against a keyframes.json trimmed to ~15 entries and confirm the
output count matches before running the full list — this is the exact filter-graph shape
(`-filter_complex` + multiple `-map`) that caused the 33GB garbage-output incident in
`../../REPORT.md` when a `-map` ended up in the wrong position.

### 5. Embedding (Colab)

```
python3 colab_embed_pecore.py --images-dir KEYFRAME_JPGS --out-dir OUT/vectors [--batch-size 32]
```

Unchanged from the original experiments: full PE-Core-bigG-14-448 CLIP checkpoint, fp32,
`normalize=True`, one `.npy` vector (dim 1280) per image, filename-matched. `../../REPORT.md`
also validated a vision-only checkpoint (skips loading the unused text tower) and FP16
(3.9x faster, Recall@10 unaffected on the tested set) — kept out of this script for now
since the full/fp32 combination is the one already wired to match
`remote-server/app/services/text_encoder.py`'s convention; swap `MODEL_ID`/`precision` in
one place if the extra speed is worth it later.

Batch size does not matter here — PE-Core-bigG-14-448 fp32 on a T4 is compute-bound, not
I/O or batching-bound (`../../REPORT.md` task 4).

## Processing videos straight out of a zip archive (no extraction)

The raw video corpus ships as zips of many videos (e.g. `data/raw_zip/Videos_L30_a.zip`:
96 videos, 4.1GB). Extracting the whole archive first wastes disk for no reason, and won't
scale to the much larger archives still to come — some of which may not even fit on disk
fully extracted.

`zip_source.py` resolves one video entry inside a zip to an ffmpeg `subfile` URL
(`subfile,,start,N,end,M,,:archive.zip`) that ffmpeg reads as if it were a standalone file,
directly out of the archive. Pass that URL as `--video` to `local_extract_frames.py` /
`local_extract_keyframes.py` instead of a real path — **zero bytes are ever extracted to
disk**, not even to a temp file. Verified pixel-exact (max_abs_diff=0) against
extract-then-decode on a real archive, and 3162 frames decoded in 1.6s straight out of a
4.1GB zip.

```
python zip_source.py --zip archive.zip --list                              # see what's in it
url=$(python zip_source.py --zip archive.zip --entry video/L30_V001.mp4)
python local_extract_frames.py --video "$url" --out-dir OUT/frames
```

This only works when the entry is **ZIP_STORED** (uncompressed) — true here because the
videos are already h264/mp4-compressed, so the zip tool that made these archives didn't
bother compressing them again (`zip_source.py --list` flags any entry where this doesn't
hold). If a future archive *does* use DEFLATE, this trick doesn't apply — a compressed
entry must be decompressed sequentially from byte 0, so there's no byte-range to point
ffmpeg at. Fall back to extracting just that one entry to a temp file, processing it, then
deleting it (`zipfile.ZipFile.extract()` + cleanup) — still only ever holds one video's
worth of disk space regardless of the archive's total size, which is the same property that
makes the subfile trick worth having in the first place: memory/disk usage stays flat as
archives get bigger, since nothing is ever proportional to the whole archive.

## Not part of this pipeline

`../pe_core_optimize/` holds the Triton/CUDA custom-kernel experiments ("Emulated FP32") and
the ONNX/torch.compile sweeps from `../../Optimize-Plan.md`. Both dead ends (custom kernels
ended up slower than plain fp32; torch.compile and ONNX gave no win on T4) — kept for
reference, not part of the pipeline above. `../../REPORT.md` has the full experiment log and
numbers this README summarizes.
