"""Stage N selections and verified PE-Core vectors without changing serving data.

Resumable: settings/source changes choose a new per-video generation. A complete
marker is written only after source checksums, vector rows and normalization pass.
Original scenes are reused; TransNet and L/M/S artifacts are never rewritten.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
sys.path.insert(0, str(ROOT / 'keyframe_pipeline_global_v9_3/src'))
os.environ.setdefault('HF_HOME', str(ROOT / 'challenge_resources/data/zip_embeddings/aic_runtime/hf_home'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from app.services import local_zip_media as media
from app.services.source_timeline import load_timeline, source_fingerprint, select_pictures, generation_signature
from app.services.readiness_policy import (SelectionPolicy, selection_policy,
    decode_provenance, verified_embed_decoder_threads)
from app.services.video_quarantine import release_blocked_video_ids
from app.services.staged_artifacts import (metadata_generation, reusable_vectors,
                                            adopt_source_verified_vectors, artifact_digests)
from pipeline.io_utils import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=ROOT / 'challenge_resources/data/zip_embeddings/readiness/n')
    parser.add_argument('--interval', type=float, default=selection_policy().interval_seconds)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--embed', action='store_true')
    parser.add_argument('--audit-release-blocked', action='store_true',
                        help='Stage an explicitly requested blocked video for offline verification only')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    args.stage = args.stage.resolve()
    os.chdir(ROOT / 'remote-server')
    if args.batch_size < 1: raise ValueError('batch-size must be positive')
    policy = SelectionPolicy(interval_seconds=args.interval)
    entries = media._scan_archives()
    requested = set(args.videos or [])
    blocked = release_blocked_video_ids()
    if requested - entries.keys():
        raise ValueError(f'Missing requested sources: {sorted(requested - entries.keys())}')
    if requested & blocked and not args.audit_release_blocked:
        raise ValueError(f'Release-blocked sources cannot be staged: {sorted(requested & blocked)}')
    if args.audit_release_blocked and not requested:
        raise ValueError('--audit-release-blocked requires explicit --videos')
    if not any(video.startswith('N') for video in entries):
        raise ValueError('No N sources found; check RAW_ZIP_DIR')
    root = ROOT / 'challenge_resources/data/zip_embeddings'
    jobs, failures = [], []
    for video, entry in sorted(entries.items()):
        if (not video.startswith('N') or (video in blocked and not args.audit_release_blocked) or
                args.videos and video not in args.videos): continue
        try:
            source_dir, = list(root.glob(f'output_*/videos__{video}'))
            old = json.loads((source_dir / 'keyframes.json').read_text())
            scenes = json.loads((source_dir / 'scenes.json').read_text())
            index = media._build_index(video, entry)
            table = load_timeline(video, index)
            if (index.duration_ticks is None or index.duration_ticks <= 0 or
                    index.timescale <= 0 or
                    index.duration_ticks < int(table[-1, 1]) - int(table[0, 1]) or
                    index.duration_ticks - (int(table[-1, 1]) - int(table[0, 1])) > index.timescale):
                raise ValueError(f'{video}: MP4 duration disagrees with decoded presentation timeline')
            rows, omitted = select_pictures(table, index.timescale, scenes['scenes'], old['keyframes'], policy)
            identity = source_fingerprint(index)
            decoder = decode_provenance(video, index)
            signature = generation_signature(identity, policy, scenes, old['keyframes'], decoder)
            payload = {**old, 'version': 2, 'selection': {'strategy': 'presentation_interval', **asdict(policy)},
                       'source_identity': identity, 'generation': signature,
                       'source_time_base': {'num': 1, 'den': index.timescale},
                       'source_duration_ms': (index.duration_ticks * 1000 + index.timescale // 2) // index.timescale,
                       'playback_origin_pts': int(table[0, 1]), 'keyframes': rows,
                       'num_keyframes': len(rows), 'omitted_entries': omitted,
                       'source_frame_count': len(table), 'submission_unit': 'milliseconds'}
            payload['decode_provenance'] = decoder
            folder, payload = metadata_generation(args.stage, video, payload, scenes)
            jobs.append((video, index, folder, payload))
        except Exception as exc:
            failures.append({'video_id': video, 'error': str(exc)})
    args.stage.mkdir(parents=True, exist_ok=True)
    report = {'version': 1, 'policy': asdict(policy),
              'release_blocked': sorted(v for v in blocked if v in entries), 'videos': [
        {'video_id': v, 'generation': p['generation'], 'path': str(f), 'rows': p['num_keyframes'],
         'omitted': len(p['omitted_entries'])} for v, i, f, p in jobs], 'failures': failures}
    atomic_json(args.stage / 'manifest.json', report)
    print(f"Staged {len(jobs)} videos / {sum(p['num_keyframes'] for _, _, _, p in jobs)} pictures; failures={len(failures)}", flush=True)
    if not args.embed:
        if failures: raise SystemExit(1)
        return
    import torch
    from pipeline.phase2_embed.model import load_image_encoder, DEFAULT_MODEL_ID
    from pipeline.phase2_embed.decode import _decode_selected_frames
    torch.set_num_threads(2)
    options = SimpleNamespace(model_source='huggingface', local_files_only=True, model_id=DEFAULT_MODEL_ID,
                              precision=None, amp='bf16', tf32=True, device='cuda', compile=False,
                              embed_ffmpeg_preprocess=True, ffmpeg_bin='/usr/bin/ffmpeg', ffprobe_bin='/usr/bin/ffprobe')
    model, preprocess, plan, amp = load_image_encoder(options)
    mean = torch.tensor(plan.mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(plan.std, device='cuda').view(1, 3, 1, 1)
    for number, (video, index, folder, payload) in enumerate(jobs, 1):
        if not args.audit_release_blocked and video in release_blocked_video_ids():
            failures.append({'video_id': video, 'phase': 'embedding',
                             'error': 'Source was release-blocked after this job started'})
            atomic_json(args.stage / 'failures.json', failures)
            print(f'SKIP newly release-blocked {video}', flush=True)
            continue
        marker = folder / 'verified.json'
        provenance = {'model': DEFAULT_MODEL_ID, 'preprocess': plan.ffmpeg_filter,
                      'amp': amp, 'weights_dtype': 'float32', 'tf32': True,
                      'decode_provenance': decode_provenance(video, index)}
        if reusable_vectors(folder, payload, provenance):
            print(f'{number}/{len(jobs)} reused {video}', flush=True)
            continue
        if adopt_source_verified_vectors(args.stage, video, folder, payload,
                                         json.loads((folder / 'scenes.json').read_text()), provenance):
            print(f'{number}/{len(jobs)} adopted source-verified {video}', flush=True)
            continue
        try:
            options.verified_decoder_threads = verified_embed_decoder_threads(video, index)
            rows = payload['keyframes']
            targets = [r['frame_number'] for r in rows]
            exact = {r['frame_number']: (r['source_pts'], r['source_checksum']) for r in rows}
            source = f'subfile,,start,{index.data_offset},end,{index.data_offset + index.video_size},,:{index.zip_path}'
            vectors, batch = [], []
            def flush():
                if not batch: return
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    encoded = model(torch.stack(batch).to('cuda').float().div_(255).sub_(mean).div_(std))
                vectors.append(torch.nn.functional.normalize(encoded.float(), dim=-1).cpu().numpy())
                batch.clear()
            for frame, tensor in _decode_selected_frames(options, source, index.fps, targets, preprocess, plan,
                                                          exact_pts=exact, exact_timebase=index.timescale):
                batch.append(tensor)
                if len(batch) >= args.batch_size: flush()
            flush()
            result = np.concatenate(vectors)
            if result.shape != (len(rows), 1280) or not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid embedding output')
            temporary = folder / 'embeddings.partial.npy'
            np.save(temporary, result)
            os.replace(temporary, folder / 'embeddings.npy')
            atomic_json(marker, {'version': 1, 'generation': payload['generation'], 'rows': len(rows),
                                 'source_checksums': 'exhaustive', 'source_time_base_verified': True,
                                 **provenance, **artifact_digests(folder), 'published': False})
            print(f'{number}/{len(jobs)} verified {video}: {len(rows)} vectors', flush=True)
        except Exception as exc:
            failures.append({'video_id': video, 'phase': 'embedding', 'error': str(exc)})
            atomic_json(args.stage / 'failures.json', failures)
            print(f'FAIL {video}: {exc}', flush=True)
    atomic_json(args.stage / 'failures.json', failures)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
