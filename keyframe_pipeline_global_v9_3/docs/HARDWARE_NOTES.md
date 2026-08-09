# Hardware notes

Per-GPU/instance-type gotchas found in practice, kept here so the next run on
a given machine type doesn't have to rediscover them. Append a section
whenever a new GPU/provider combination turns up something that isn't
already covered by the general checklist in the README ("Before running on
a new rented server").

Format per entry: what's different about this hardware, and what to set/watch
because of it. Skip anything the pipeline already auto-detects/self-defends
against (see README) -- only note what still needs a human decision.

## Tesla T4 (Colab free tier, Kaggle free tier)

- Turing (compute capability 7.5): **no bf16 Tensor Core support.** `--amp auto`
  correctly falls back to `fp16` (fixed -- older versions of this repo could
  pick bf16 here, which runs via slow software emulation, nearly as slow as
  FP32). If you ever force `--amp bf16` explicitly on a T4, expect FP32-ish
  speed, not a crash.
- Colab free tier: **2 vCPUs.** Set `--transnet-decode-workers 2` (the
  default of 4 oversubscribes and doesn't help). Low GPU utilization on
  TransNet phase (`batch_fill_ratio` in `transnet_performance_summary.json`
  staying low, e.g. <20%) on this tier is expected/CPU-decode-bound, not a
  bug -- see `docs/ARCHITECTURE.md`.
- `/dev/shm` observed as low as **5.7G** on a Colab instance -- smaller than
  the PE-Core checkpoint (~7-8G). `--profile balanced` now checks for this
  and falls back to disk automatically; you'll see `[profile] /dev/shm has
  less than 12G free...` in the log, which is expected here, not an error.
- Kaggle: model cache can be near-instant if the checkpoint is attached as a
  Kaggle Dataset input (`/kaggle/input/...`) instead of downloaded via
  kagglehub -- no action needed, just don't be surprised it skips the
  download step entirely.

## RTX 5060 Ti (seen on Vast.ai)

**Comes in two VRAM variants (8G/16G) -- don't assume, always check `nvidia-smi`.** Two different
Vast.ai rentals with the *same* GPU model had very different surrounding specs:

| | Rental A | Rental B |
|---|---|---|
| `nproc` | 48 | 48 |
| RAM | 109G | 125G (89G available) |
| Disk | 17G | 63G |
| `/dev/shm` | 13G | 7G |
| VRAM (`nvidia-smi`) | 16311 MiB total | 16311 MiB total, only 13782 MiB free |

Same GPU, same core count, but disk/shm/free-VRAM are not implied by any of that -- confirms the
"General: Vast.ai" note below isn't theoretical. A third guess made before checking anything said
24G VRAM/11 vCPU/28G RAM -- also wrong. Always run `check_machine.sh` + `nproc`, never go by the
listing.

- **Rental B had ~2.5G of VRAM already in use before running anything** (13782/16311 MiB free).
  Check `nvidia-smi` for other processes before assuming the full card is available -- could be a
  shared/passthrough instance or a leftover process from a prior session.
- Rental B's `/dev/shm` (7G) is under this repo's 12G model-cache threshold -- `--profile balanced`
  falls back to disk automatically (fine here, disk had 63G free).

- Blackwell, compute capability `(12, 0)`: bf16 Tensor Cores available,
  `--amp auto` picks `bf16` correctly (confirmed via `check_machine.sh`'s AMP
  capability report). Verified working with `torch==2.12.0+cu130` -- a
  trivial CUDA op and `get_device_capability()` both worked, so this specific
  torch build does have kernels for it. Still worth the same check on a
  fresh rental in case an older cached wheel gets installed instead
  (`pip install torch` in `setup.sh` has no version pin).
- 16G VRAM is moderate, not generous -- run `scripts/bench_batch_size.sh`
  rather than assuming a large `--batch-size` is safe, regardless of how
  much disk/RAM the rental has.
- Disk was tight on rental A (17G vs ~6G source ZIPs) but not rental B (63G)
  -- relies on `run_many.sh`'s default ZIP auto-cleanup and
  `run_pipeline.sh`'s disk preflight on any rental; don't disable either
  (`--keep-source-zips`) without checking the real disk size first.
- `/dev/shm` is **not** implied by RAM size -- 13G on 109G RAM (rental A,
  ~12% -- would be ~54G under the naive "50% of RAM" rule of thumb), 7G on
  125G RAM (rental B, ~6%). Vast.ai containers set `--shm-size`
  independently of host RAM. Check the real number (`df -h /dev/shm`) per
  rental; this repo's `--profile balanced` needs >=12G there for the model
  cache or it falls back to disk (self-handled, but good to know which path
  you're on before benchmarking).
- With `nproc`=48, `--transnet-decode-workers 16` is a reasonable starting
  point (no need to max out core count -- diminishing returns past a point,
  and leaves room for the embed-phase producer thread + OS). Watch
  `batch_fill_ratio` in `transnet_performance_summary.json`; raise if still
  low.
- Model download on rental B capped at ~13MB/s via aria2c (16 connections)
  regardless of source -- `scripts/fetch_hf_model_fast.sh` and
  `scripts/fetch_kaggle_model_fast.sh` gave the *same* speed. When two
  different CDN providers both cap at the same number under multi-connection,
  that's the rental's real network ceiling, not a client-side or
  provider-side problem worth chasing further. A much faster (~30s) Kaggle
  download experienced previously was almost certainly from running directly
  on a Kaggle Notebook with the checkpoint mounted via `/kaggle/input/` (zero
  transfer) -- not reproducible off-platform.

## General: Vast.ai / similar rental marketplaces

Disk and `/dev/shm` size are set per-listing, not by a fixed formula from
GPU/RAM -- two rentals with the same GPU can differ. Always run the
checklist (`tools/check_machine.sh`, `scripts/test_multi_link_fake.sh`) on a
new rental before a real job, even if the GPU model itself is already listed
above.
