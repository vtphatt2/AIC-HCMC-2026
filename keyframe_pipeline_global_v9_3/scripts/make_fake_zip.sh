#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/make_fake_zip.sh OUTPUT.zip [num_videos] [duration_seconds]

Generates tiny synthetic CFR videos (ffmpeg testsrc, low-res) and packs them
into a ZIP the pipeline can process for real (real ffmpeg decode, real
TransNet/PE-Core inference) without needing any real dataset or network
access. Entries are stored uncompressed (ZIP_STORED), matching how real
datasets are packed, so the pipeline's subfile:// fast path is exercised too.

Defaults: 3 videos, 5 seconds each (~a few hundred KB total, seconds to build).
USAGE
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
OUT="${1:?output.zip path required}"
NUM="${2:-3}"
DUR="${3:-5}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found on PATH" >&2; exit 1; }
command -v zip >/dev/null 2>&1 || { echo "ERROR: zip not found on PATH (apt/brew install zip)" >&2; exit 1; }

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

for i in $(seq 1 "$NUM"); do
  name=$(printf "FAKE_V%03d.mp4" "$i")
  # Scene cut every ~2s via changing testsrc seed, so TransNet has something to detect.
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x240:rate=25:duration=$DUR" \
    -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$WORKDIR/$name"
done

rm -f "$OUT"
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
( cd "$WORKDIR" && zip -q -0 -X "$OUT_ABS" ./*.mp4 )

echo "Wrote $OUT: $NUM video(s), ${DUR}s each, $(du -h "$OUT" | cut -f1)"
