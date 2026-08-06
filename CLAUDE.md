# Data Processing — working notes

## Test small before running big

Before running any new or modified command/script against the full dataset (full video, all
586+ keyframes, full 28498-frame decode, etc.), first run it at a tiny bounded scale (a handful
of frames, `-frames:v N`, a short clip) and verify the output matches expectations (byte count,
frame count, file count) before scaling up.

**Why:** a single mistake in an ffmpeg `-filter_complex`/`-map` command silently consumed ~177GB
through a pipe and crashed a Colab GPU session, and separately wrote 33GB of garbage output to
local disk — both from the exact same root cause (a stray `-map 0:v:0` before `-filter_complex`
leaking the original full-res stream into an output that should only have received a small
filtered branch). A 5-frame test would have caught it in seconds instead of after a crash/cleanup.

**How to apply:** for any ffmpeg command that pipes raw video, extracts many frames, or uses
`-filter_complex`/`split`, add a cheap sanity check first — cap frames with `-frames:v N`, compute
the expected output byte/file count, and assert it matches before trusting the command on the
full video. This applies on both the local machine and Colab.

## Every long-running script must print progress

Any script that loops over many items (frames, images, batches) taking more than ~30s total must
print periodic progress with `flush=True` — not just a single summary line at the end. Default to
printing **every iteration/batch** when each one takes more than ~2-3s (this is the common case for
GPU embed/inference loops) — only batch the print to every 5-10 iterations for loops with many fast
(<1s) iterations, where per-iteration printing itself would be the overhead.

**Why:** Colab SSH sessions run detached (`nohup ... &`), and Python fully buffers stdout when it's
not a terminal — so without periodic flushed prints, logs stay empty for the entire run (minutes to
tens of minutes) and there's no way to tell a slow-but-healthy process from a stuck one without
opening a second SSH connection (which Colab's proxy doesn't allow — only one connection at a time).

**How to apply:** launch long Python scripts with `python3 -u` (unbuffered) AND print progress
inside the loop itself (not only a final per-phase summary), e.g. every 5 batches:
`print(f"{n_done}/{n_total} ... {elapsed:.1f}s", flush=True)`. When streaming output back over SSH
in real time (not just to a log file), pipe through `tee`: `... 2>&1 | tee out.log` — this both
saves the log on Colab and streams it back live over the same connection, rather than fully
redirecting away from the session with `> file 2>&1`.

## Launching detached background jobs on Colab: use `setsid -f`, not bare `setsid` or `nohup`

`nohup` does NOT detach a process from its session — it only ignores SIGHUP — so the SSH connection
stays occupied for the full job duration regardless (confirmed: held connection for 24+ minutes on a
long job). Bare `setsid cmd &` works when `setsid` is the very first command of the backgrounded job
(`ssh host "setsid cmd < /dev/null > log 2>&1 & echo done; sleep 1"` — confirmed: SSH returns in ~2-4s
here). But when `setsid` is NOT the first command (e.g. preceded by `cd dir &&`), it silently fails to
detach — likely because the calling process ends up as the backgrounded subshell's process-group
leader, and `setsid()` returns EPERM for a process that's already a group leader. **Always use
`setsid -f` (force fork)** instead of bare `setsid` — this avoids the edge case regardless of what
precedes it in the command chain. Confirmed pattern that reliably frees the connection in ~2-4s:
```
ssh host "export FOO=bar; cd /content && setsid -f python3 -u script.py < /dev/null > log 2>&1 & echo LAUNCHED; sleep 2"
```
Verify with a second `ssh host "cat log"` call immediately after — if it returns without an
"already-active SSH session" error and shows partial progress, the job is genuinely detached.

## Non-interactive Colab SSH needs `LD_LIBRARY_PATH` set manually or `torch.cuda.is_available()` is falsely False

`ssh host "python3 -c '...'"` runs a non-interactive, non-login shell, which does NOT source the
profile script where Colab sets `LD_LIBRARY_PATH` to include the NVIDIA driver libs. Without it,
PyTorch can't find `libnvidia-ml.so` and `torch.cuda.is_available()` returns `False` / `nvidia-smi`
errors with "couldn't find libnvidia-ml.so" — even though the GPU is fully attached and working
(`/dev/nvidia0` present, driver files present at `/usr/lib64-nvidia/`). This looks exactly like "no
GPU on this session" but isn't.

**Why:** wasted time concluding a freshly-reattached Colab session had no GPU, when it was only an
`ssh`-non-interactive-shell environment issue — confirmed by checking `/dev/nvidia*` and
`find / -iname libnvidia-ml*` directly, and by checking `torch.version.cuda` (present = GPU build of
torch, just can't see the driver) before assuming GPU absence from `torch.cuda.is_available()` alone.

**How to apply:** before concluding a Colab session has no GPU, always check `ls /dev/nvidia*` and
`find / -iname 'libnvidia-ml*'` directly, and prefix Python GPU commands with
`export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH;` — e.g.
`ssh host "export LD_LIBRARY_PATH=/usr/lib64-nvidia:\$LD_LIBRARY_PATH; python3 -c '...'"`. Bake this
export into every `setsid -f` launch command on a freshly-attached/reattached session too.
