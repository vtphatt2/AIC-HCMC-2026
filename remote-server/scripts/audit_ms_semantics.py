"""Deterministic source-picture and PE-Core spot checks for every M/S video.

This audit leaves the existing vectors untouched. It samples first/middle/last
selections and selections adjacent to fixed scene-boundary quantiles, decodes
them from the original ZIP with independently recorded PTS/checksums, and
compares fresh PE-Core vectors with the packaged processing output. The report
labels these checks sampled; a mismatch requires an expanded audit.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
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
from app.services.exact_frame_pts import build_exact_pts, index_path, showinfo_frames
from app.services.source_timeline import source_fingerprint

DATA = ROOT / 'challenge_resources/data/zip_embeddings'
PTS_DIR = DATA / 'verification/ms_sample_pts'
DEFAULT_REPORT = DATA / 'verification/ms_semantic_audit.json'
COSINE_REVIEW_THRESHOLD = 0.995


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def selected_positions(rows: list[dict], scenes: list[dict], exhaustive: bool = False) -> list[int]:
    """Choose stable row positions and source pictures near scene boundaries."""
    if not rows:
        raise ValueError('No selected pictures')
    frames = [int(row['frame_number']) for row in rows]
    if frames != sorted(set(frames)):
        raise ValueError('Selected frame numbers are not unique and ordered')
    if exhaustive:
        return list(range(len(rows)))
    positions = {0, len(rows) // 2, len(rows) - 1}
    if not scenes:
        raise ValueError('No scene boundaries')
    for scene_index in {0, len(scenes) // 4, len(scenes) // 2,
                        (3 * len(scenes)) // 4, len(scenes) - 1}:
        scene = scenes[scene_index]
        for boundary in (int(scene['start_frame']), int(scene['end_frame'])):
            insertion = bisect_left(frames, boundary)
            choices = [position for position in (insertion - 1, insertion)
                       if 0 <= position < len(frames)]
            if choices:
                positions.add(min(choices, key=lambda pos: (abs(frames[pos] - boundary), pos)))
    return sorted(positions)


def source_folder(video: str) -> Path:
    lot = 'S01' if video.startswith('S') else video.split('_', 1)[0]
    return DATA / f'output_{lot}' / f'videos__{video}'


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, path)


def sampled_pts(video: str, index, targets: list[int], frame_count: int):
    """Reuse S's source-fingerprinted selected map; build M samples separately."""
    shared = index_path(video, index)
    candidates = [shared] if video.startswith('S') and shared.is_file() else []
    for path in candidates:
        indexed = np.load(path, mmap_mode='r', allow_pickle=False)
        if (indexed.ndim != 2 or indexed.shape[1] != 3 or indexed.dtype != np.int64 or not len(indexed) or
                np.any(indexed[1:, 0] <= indexed[:-1, 0]) or
                int(indexed[-1, 0]) >= frame_count):
            continue
        positions = np.searchsorted(indexed[:, 0], targets)
        if np.any(positions >= len(indexed)) or not np.array_equal(indexed[positions, 0], targets):
            continue
        return indexed[positions], path
    path = build_exact_pts(video, index, targets, frame_count, '/usr/bin/ffmpeg',
                           directory=PTS_DIR, force_dense=True)
    indexed = np.load(path, allow_pickle=False)
    if (indexed.ndim != 2 or indexed.shape[1] != 3 or indexed.dtype != np.int64 or
            not len(indexed) or np.any(indexed[1:, 0] <= indexed[:-1, 0])):
        raise ValueError('Invalid independent source PTS/checksum map')
    positions = np.searchsorted(indexed[:, 0], targets)
    if (np.any(positions >= len(indexed)) or
            not np.array_equal(indexed[positions, 0], targets)):
        raise ValueError('Independent source PTS/checksum map does not cover sample')
    return indexed[positions], path


def decoded_sample_tensors(source, targets, exact, index, plan, options, preprocess):
    """Seek to each audited PTS, accepting it only on exact checksum agreement.

    A source with unreliable random access falls back to one full sequential
    decode. No tensor is returned until the whole seek set has verified.
    """
    from pipeline.phase2_embed.decode import _decode_selected_frames
    from pipeline.phase2_embed.preprocess import fast_preprocess_rgb
    frame_bytes = plan.output_w * plan.output_h * 3
    sampled = []
    seek_failure = None
    for frame in targets:
        pts, checksum = exact[frame]
        seek_seconds = max(0.0, pts / index.timescale - 5.0)
        filters = f"select='eq(pts\\,{pts})',showinfo,{plan.ffmpeg_filter}"
        command = [options.ffmpeg_bin, '-hide_banner', '-nostdin', '-nostats',
                   '-loglevel', 'info', '-copyts', '-threads', '4',
                   '-ss', f'{seek_seconds:.6f}', '-i', source,
                   '-map', '0:v:0', '-an', '-vf', filters, '-filter_threads', '1',
                   '-vsync', '0', '-threads:v', '1', '-pix_fmt', 'rgb24',
                   '-frames:v', '1', '-f', 'rawvideo', 'pipe:1']
        try:
            result = subprocess.run(command, capture_output=True, timeout=120)
            observed = showinfo_frames(result.stderr)
            bases = {(int(a), int(b)) for a, b in re.findall(
                rb'config in time_base:\s*(\d+)/(\d+)', result.stderr)}
            if (result.returncode or len(result.stdout) != frame_bytes or
                    len(observed) != 1 or observed[0][1:] != (pts, checksum) or
                    bases != {(1, index.timescale)}):
                raise ValueError(f'{frame}: seek result exit={result.returncode} '
                                 f'frames={len(observed)} bytes={len(result.stdout)} '
                                 f'pts_checksums={[row[1:] for row in observed]} '
                                 f'time_bases={sorted(bases)}')
            sampled.append((frame, fast_preprocess_rgb(result.stdout, plan)))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            seek_failure = str(exc)
            break
    if seek_failure is None:
        return sampled, 'checksum_verified_input_seek', None
    replay = list(_decode_selected_frames(options, source, index.fps, targets,
                                          preprocess, plan, exact_pts=exact,
                                          exact_timebase=index.timescale))
    return replay, 'full_sequential_fallback', seek_failure


def audit_video(video: str, index, model, preprocess, plan, options, amp,
                exhaustive: bool = False) -> dict:
    import torch
    folder = source_folder(video)
    selected = json.loads((folder / 'keyframes.json').read_text())['keyframes']
    scenes_data = json.loads((folder / 'scenes.json').read_text())
    scenes = scenes_data['scenes']
    stored = np.load(folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    if stored.shape != (len(selected), 1280) or not np.isfinite(stored).all():
        raise ValueError('Stored vectors have invalid shape or nonfinite values')
    positions = selected_positions(selected, scenes, exhaustive)
    targets = [int(selected[position]['frame_number']) for position in positions]
    pts, pts_path = sampled_pts(video, index, targets, int(scenes_data['num_frames']))
    exact = {int(frame): (int(time), int(checksum)) for frame, time, checksum in pts}
    source = (f'subfile,,start,{index.data_offset},end,'
              f'{index.data_offset + index.video_size},,:{index.zip_path}')
    mean = torch.tensor(plan.mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(plan.std, device='cuda').view(1, 3, 1, 1)
    actual_frames = []
    vectors = []
    batch = []
    def flush():
        if not batch:
            return
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoded = model(torch.stack(batch).to('cuda').float().div_(255).sub_(mean).div_(std))
        vectors.append(torch.nn.functional.normalize(encoded.float(), dim=-1).cpu().numpy())
        batch.clear()
    decoded, decode_method, seek_failure = decoded_sample_tensors(
        source, targets, exact, index, plan, options, preprocess)
    for frame, tensor in decoded:
        actual_frames.append(frame)
        batch.append(tensor)
        if len(batch) >= 8:
            flush()
    flush()
    if actual_frames != targets or not vectors:
        raise ValueError('Decoded source frames differ from deterministic sample')
    fresh = np.concatenate(vectors)
    expected = np.asarray(stored[positions], dtype=np.float32)
    cosines = np.sum(fresh * expected, axis=1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(expected, axis=1))
    if not np.isfinite(cosines).all():
        raise ValueError('Nonfinite vector similarity')
    worst = np.argsort(cosines)[:min(5, len(cosines))]
    return {'status': 'pass' if float(np.min(cosines)) >= COSINE_REVIEW_THRESHOLD else 'review',
            'assessment_depth': 'all_selected' if exhaustive else 'deterministic_sample',
            'sampled_rows': len(positions), 'total_rows': len(selected),
            'sampled_frame_ids': targets, 'source_pts': pts[:, 1].tolist(),
            'source_checksums': pts[:, 2].tolist(),
            'decode_method': decode_method, 'seek_fallback_reason': seek_failure,
            'min_cosine': float(np.min(cosines)), 'median_cosine': float(np.median(cosines)),
            'worst_samples': [{'frame_number': targets[int(pos)], 'cosine': float(cosines[pos])}
                              for pos in worst],
            'model': options.model_id, 'preprocess': plan.ffmpeg_filter,
            'amp': amp, 'cosine_review_threshold': COSINE_REVIEW_THRESHOLD,
            'source': source_fingerprint(index), 'source_pts_map': str(pts_path),
            'keyframes_sha256': digest(folder / 'keyframes.json'),
            'scenes_sha256': digest(folder / 'scenes.json'),
            'embeddings_sha256': digest(folder / 'embeddings.npy')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--prepare-maps', action='store_true',
                        help='Prepare sampled source identities on CPU; do not load a model')
    parser.add_argument('--all-selected', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT / 'remote-server')
    entries = media._scan_archives()
    all_videos = sorted(v for v in entries if v.startswith(('M', 'S')))
    requested = set(args.videos or all_videos)
    if requested - set(all_videos):
        raise ValueError(f'Missing M/S source videos: {sorted(requested - set(all_videos))}')
    videos = [v for v in all_videos if v in requested]
    if args.inventory_only:
        summary = {}
        for video in videos:
            folder = source_folder(video)
            rows = json.loads((folder / 'keyframes.json').read_text())['keyframes']
            scenes = json.loads((folder / 'scenes.json').read_text())['scenes']
            summary[video] = {'selected_rows': len(rows),
                              'sampled_rows': len(selected_positions(rows, scenes, args.all_selected))}
        print(json.dumps({'videos': len(summary), 'total_sampled_rows': sum(
            x['sampled_rows'] for x in summary.values()), 'by_video': summary}, indent=2))
        return
    if args.prepare_maps:
        report_path = (DATA / 'verification/ms_sample_map_audit.json'
                       if args.report == DEFAULT_REPORT else args.report)
        report = {'version': 1, 'assessment_depth': 'sampled_source_maps_only_no_embedding_check',
                  'videos': {}}
        for number, video in enumerate(videos, 1):
            try:
                folder = source_folder(video)
                index = media._build_index(video, entries[video])
                selected = json.loads((folder / 'keyframes.json').read_text())['keyframes']
                scenes = json.loads((folder / 'scenes.json').read_text())
                positions = selected_positions(selected, scenes['scenes'], args.all_selected)
                targets = [int(selected[position]['frame_number']) for position in positions]
                table, path = sampled_pts(video, index, targets, int(scenes['num_frames']))
                result = {'status': 'pass', 'source': source_fingerprint(index),
                          'sampled_rows': len(targets), 'source_pts_map': str(path),
                          'source_frame_pts_checksum': table.tolist(),
                          'keyframes_sha256': digest(folder / 'keyframes.json'),
                          'scenes_sha256': digest(folder / 'scenes.json')}
            except Exception as exc:
                result = {'status': 'error', 'error': str(exc)}
            report['videos'][video] = result
            write_report(report_path, report)
            print(f'{number}/{len(videos)} source map {video}: {result["status"]} '
                  f'{result.get("error", "")}', flush=True)
        report['summary'] = {status: sum(row['status'] == status for row in report['videos'].values())
                             for status in ('pass', 'error')}
        write_report(report_path, report)
        if report['summary']['error']:
            raise SystemExit(1)
        return
    import torch
    from pipeline.phase2_embed.model import load_image_encoder, DEFAULT_MODEL_ID
    torch.set_num_threads(2)
    options = SimpleNamespace(model_source='huggingface', local_files_only=True,
                              model_id=DEFAULT_MODEL_ID, precision=None, amp='bf16', tf32=True,
                              device='cuda', compile=False, embed_ffmpeg_preprocess=True,
                              ffmpeg_bin='/usr/bin/ffmpeg', ffprobe_bin='/usr/bin/ffprobe',
                              verified_decoder_threads=4)
    model, preprocess, plan, amp = load_image_encoder(options)
    if plan is None:
        raise ValueError('Expected reproducible FFmpeg PE-Core preprocessing')
    try:
        report = json.loads(args.report.read_text())
        if report.get('version') != 1: report = None
    except (FileNotFoundError, ValueError):
        report = None
    if report is None:
        report = {'version': 1, 'sampling': 'first/middle/last plus nearest selected to fixed scene-boundary quantiles',
                  'cosine_review_threshold': COSINE_REVIEW_THRESHOLD, 'videos': {}}
    for number, video in enumerate(videos, 1):
        folder = source_folder(video)
        index = media._build_index(video, entries[video])
        identity = {'source': source_fingerprint(index),
                    'keyframes_sha256': digest(folder / 'keyframes.json'),
                    'scenes_sha256': digest(folder / 'scenes.json'),
                    'embeddings_sha256': digest(folder / 'embeddings.npy')}
        previous = report['videos'].get(video)
        depth = 'all_selected' if args.all_selected else 'deterministic_sample'
        if not args.force and previous and previous.get('status') in ('pass', 'review') and \
                previous.get('assessment_depth') == depth and all(
                    previous.get(key) == value for key, value in identity.items()):
            print(f'{number}/{len(videos)} reused {video}: {previous["status"]}', flush=True)
            continue
        try:
            result = audit_video(video, index, model, preprocess, plan, options, amp,
                                 exhaustive=args.all_selected)
        except Exception as exc:
            result = {**identity, 'status': 'error', 'error': str(exc)}
        report['videos'][video] = result
        write_report(args.report, report)
        print(f'{number}/{len(videos)} {video}: {result["status"]} '
              f'{result.get("sampled_rows", 0)} rows min_cos={result.get("min_cosine", "?")} '
              f'{result.get("error", "")}', flush=True)
    report['summary'] = {status: sum(row['status'] == status for row in report['videos'].values())
                         for status in ('pass', 'review', 'error')}
    write_report(args.report, report)
    if report['summary']['review'] or report['summary']['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
