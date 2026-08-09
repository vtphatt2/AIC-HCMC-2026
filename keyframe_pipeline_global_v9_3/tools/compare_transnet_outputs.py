#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description='Compare legacy/global TransNet outputs for correctness.')
p.add_argument('reference', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('--atol', type=float, default=1e-6)
a=p.parse_args()

ref_v={p.parent.name:p for p in a.reference.glob('*/transnet_predictions.npy')}
can_v={p.parent.name:p for p in a.candidate.glob('*/transnet_predictions.npy')}
ids=sorted(set(ref_v)&set(can_v))
if not ids:
    raise SystemExit('No matching transnet_predictions.npy files. Run both modes with --save-transnet-predictions.')
worst=0.0; exact_scenes=0; exact_keyframes=0
for vid in ids:
    x=np.load(ref_v[vid]); y=np.load(can_v[vid])
    if x.shape != y.shape:
        print(f'{vid}: SHAPE {x.shape} != {y.shape}'); continue
    err=float(np.max(np.abs(x-y))) if x.size else 0.0; worst=max(worst,err)
    rs=json.loads((a.reference/vid/'scenes.json').read_text())
    cs=json.loads((a.candidate/vid/'scenes.json').read_text())
    rk=json.loads((a.reference/vid/'keyframes.json').read_text())
    ck=json.loads((a.candidate/vid/'keyframes.json').read_text())
    scene_ok=rs.get('scenes')==cs.get('scenes'); key_ok=rk.get('keyframes')==ck.get('keyframes')
    exact_scenes += scene_ok; exact_keyframes += key_ok
    print(f'{vid}: max_abs={err:.3g} scenes={"OK" if scene_ok else "DIFF"} keyframes={"OK" if key_ok else "DIFF"}')
print(f'videos={len(ids)} worst_max_abs={worst:.6g} scenes_exact={exact_scenes}/{len(ids)} keyframes_exact={exact_keyframes}/{len(ids)}')
raise SystemExit(0 if worst <= a.atol and exact_scenes==len(ids) and exact_keyframes==len(ids) else 1)
