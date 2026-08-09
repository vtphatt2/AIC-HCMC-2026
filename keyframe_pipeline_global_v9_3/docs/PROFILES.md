# Profiles cheat sheet

| Situation | Profile | ZIP | HF cache | Temp |
|---|---|---|---|---|
| 15G `/dev/shm`, ~17G disk | `balanced` | disk | shm | shm |
| Large `/dev/shm` | `ram-rich` | shm | shm | shm |
| Large disk, small shm | `disk-rich` | disk | disk | disk |
| Google Colab | `colab` | `/content` | `/content` | `/content` |

All locations can be overridden independently with `--source-dir`, `--hf-cache-dir`, and `--temp-dir`.

`balanced` and `ram-rich` only use `/dev/shm` for the HF/Kaggle model cache if it has at least
12G free (PE-Core checkpoints run ~7-8G; `/dev/shm` size varies a lot across Colab/Kaggle/cloud
instances and is sometimes well under that). `balanced` silently falls back to disk in that case;
`ram-rich` hard-errors instead, since it requires the ZIP + cache + temp to all fit in RAM.

Compute knobs are independent of storage: `--amp`, `--tf32`, `--compile`, `--batch-size`, `--prefetch-batches`, `--device`.
