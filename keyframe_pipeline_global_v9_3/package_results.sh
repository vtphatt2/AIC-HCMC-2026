#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/runtime_env.sh"
if [[ $# -ge 1 ]]; then
  OUT_DIR="$1"
  ARCHIVE="${2:-$(dirname "$OUT_DIR")/$(basename "$OUT_DIR" | sed 's/^output_//')_results.zip}"
else
  if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d /kaggle/working ]]; then WR=/kaggle/working
  elif [[ -d /content && -w /content ]]; then WR=/content
  elif [[ -d /workspace && -w /workspace ]]; then WR=/workspace
  else WR="${HOME:-$PWD}/workspace"; fi
  mapfile -t matches < <(find "$WR" -maxdepth 1 -type d -name 'output_*' | sort)
  if (( ${#matches[@]} == 0 )); then echo "ERROR: no output_* directories found under $WR" >&2; exit 1; fi
  if (( ${#matches[@]} > 1 )); then
    echo "ERROR: multiple outputs found; choose explicitly:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    echo "Usage: bash package_results.sh OUTPUT_DIR [ARCHIVE]" >&2; exit 2
  fi
  OUT_DIR="${matches[0]}"; ARCHIVE="$WR/$(basename "$OUT_DIR" | sed 's/^output_//')_results.zip"
fi
[[ -d "$OUT_DIR" ]] || { echo "ERROR: output directory not found: $OUT_DIR" >&2; exit 1; }
"$PYTHON_BIN" - "$OUT_DIR" "$ARCHIVE" <<'PY'
from pathlib import Path
import json, sys, zipfile
out=Path(sys.argv[1]).resolve(); archive=Path(sys.argv[2]).resolve()
archive.parent.mkdir(parents=True, exist_ok=True); archive.unlink(missing_ok=True)
videos=[]
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zf:
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        sc=d/'scenes.json'; kf=d/'keyframes.json'; em=d/'embeddings.npy'
        if not (sc.exists() and kf.exists() and em.exists()): continue
        vid=d.name; zf.write(sc,f'phase1_transnet/{vid}/scenes.json'); zf.write(kf,f'phase1_transnet/{vid}/keyframes.json'); zf.write(em,f'phase2_embeddings/{vid}/embeddings.npy')
        data=json.loads(kf.read_text(encoding='utf-8'))
        videos.append({'video_id':vid,'video':data.get('video'),'num_keyframes':data.get('num_keyframes'),'scenes':f'phase1_transnet/{vid}/scenes.json','keyframes':f'phase1_transnet/{vid}/keyframes.json','embeddings':f'phase2_embeddings/{vid}/embeddings.npy','mapping':'embeddings[i] corresponds to keyframes[i]'})
    for name in ('run_summary.json','performance_summary.json'):
        p=out/name
        if p.exists(): zf.write(p,name)
    zf.writestr('manifest.json',json.dumps({'format_version':3,'num_videos':len(videos),'videos':videos},indent=2,ensure_ascii=False))
print(f'Created {archive} with {len(videos)} complete videos')
PY
"$PYTHON_BIN" - "$ARCHIVE" <<'PYZIPCHECK'
import sys, zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    bad_member = archive.testzip()
if bad_member is not None:
    raise SystemExit(f"ERROR: corrupt ZIP member: {bad_member}")
PYZIPCHECK
ls -lh "$ARCHIVE"
