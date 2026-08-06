# Data Processing

Experiments and packaged pipeline for the AIC video keyframe &rarr; embedding pipeline
(local CPU decode + Colab GPU inference).

- **`scripts/keyframe_pipeline/README.md`** — the packaged, working pipeline: run this.
- **`REPORT.md`** — full experiment log (what was tried, timings, bugs found/fixed) that the
  pipeline above is distilled from.
- **`Optimize-Plan.md`** — the original custom-Triton-kernel plan; abandoned, see REPORT.md's
  "Optimize-Plan" section for why.
- **`scripts/pe_core_optimize/`** — the custom-kernel/quantization exploration scripts
  (Triton, CUDA WMMA, ONNX, torch.compile). Dead ends, kept for reference only.
- **`CLAUDE.md`** — operational rules learned the hard way (ffmpeg pitfalls, Colab SSH
  quirks) — applies to any future work in this repo.
