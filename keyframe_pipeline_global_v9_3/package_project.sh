#!/usr/bin/env bash
# Zip only git-tracked project files (source, scripts, docs) for upload to a GPU server.
set -euo pipefail
cd "$(dirname "$0")"

out="${1:-keyframe_pipeline_global_v9_3.zip}"
git archive --worktree-attributes --format=zip -o "$out" HEAD
echo "Wrote $out ($(du -h "$out" | cut -f1))"
