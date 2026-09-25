"""Package verified N generations in isolated result ZIPs; never replace live ZIPs.

Each archive is built only when every non-blocked video in its source lot has
verified vectors. An explicitly audited recovery may be staged while its live
release block remains in force. Publication is a separate maintenance step.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from app.services import local_zip_media as media
from app.services.source_timeline import load_timeline, source_fingerprint, monotonic_entries
from app.services.staged_artifacts import artifact_digests, atomic_json, file_digest
from app.services.video_quarantine import release_blocked_video_ids
from scripts.ingest_zip_pipeline_results import video_directories


def stage_entries(*roots: Path):
    found = {}
    for root in roots:
        if not root:
            continue
        path = root / 'manifest.json'
        manifest = json.loads(path.read_text())
        for row in manifest['videos']:
            video = row['video_id']
            if video in found:
                raise ValueError(f'Duplicate staged generation for {video}')
            found[video] = (Path(row['path']), row['generation'])
    return found


def validate_video(video_id, index, folder: Path, generation: str):
    keyframes = json.loads((folder / 'keyframes.json').read_text())
    scenes = json.loads((folder / 'scenes.json').read_text())
    verified = json.loads((folder / 'verified.json').read_text())
    rows = keyframes['keyframes']
    if (keyframes.get('version') != 2 or keyframes.get('generation') != generation or
            verified.get('generation') != generation or verified.get('rows') != len(rows) or
            verified.get('source_checksums') != 'exhaustive' or
            verified.get('source_time_base_verified') is not True or
            verified.get('published') is not False or
            keyframes.get('source_identity') != source_fingerprint(index) or
            keyframes.get('num_keyframes') != len(rows)):
        raise ValueError(f'{video_id}: incomplete or stale verified generation')
    if any(verified.get(name) != digest for name, digest in artifact_digests(folder).items()):
        raise ValueError(f'{video_id}: verified artifact digest missing or changed')
    table = load_timeline(video_id, index)
    kept, omitted = monotonic_entries(table)
    frame_ids = [int(row['frame_number']) for row in rows]
    if (not frame_ids or len(frame_ids) != len(set(frame_ids)) or
            frame_ids != sorted(frame_ids) or frame_ids[0] < 0 or
            frame_ids[-1] >= len(table)):
        raise ValueError(f'{video_id}: invalid selected frame identities')
    allowed = set(map(int, kept[:, 0]))
    if any(frame not in allowed for frame in frame_ids):
        raise ValueError(f'{video_id}: selected omitted source frame')
    for row in rows:
        frame = int(row['frame_number'])
        if (int(row['source_pts']) != int(table[frame, 1]) or
                int(row['source_checksum']) != int(table[frame, 2]) or
                int(row['source_timebase']) != int(index.timescale)):
            raise ValueError(f'{video_id}/{frame}: source picture identity mismatch')
        timestamp_ms = (int(row['source_pts']) * 1000 + index.timescale // 2) // index.timescale
        if int(row['timestamp_ms']) != timestamp_ms:
            raise ValueError(f'{video_id}/{frame}: presentation timestamp mismatch')
    if len(set(int(row['source_pts']) for row in rows)) != len(rows):
        raise ValueError(f'{video_id}: repeated selected PTS')
    if set(int(row['frame_number']) for row in keyframes['omitted_entries']) != set(map(int, omitted[:, 0])):
        raise ValueError(f'{video_id}: omitted source identities changed')
    vectors = np.load(folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    if (vectors.shape != (len(rows), 1280) or vectors.dtype != np.float32 or
            not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError(f'{video_id}: invalid vector rows')
    if scenes.get('num_frames') != len(table):
        raise ValueError(f'{video_id}: scene frame count differs from decoded source')
    return keyframes, scenes, verified, vectors


def same_archive_members(first: Path, second: Path) -> bool:
    """ZIP timestamps/compression may differ; compare every logical member."""
    with zipfile.ZipFile(first) as left, zipfile.ZipFile(second) as right:
        names = left.namelist()
        if names != right.namelist() or len(names) != len(set(names)):
            return False
        return all(left.read(name) == right.read(name) for name in names)


def build_archive(source_archive: Path, entries: dict, stages: dict,
                  blocked: frozenset[str], recover: frozenset[str], output_dir: Path):
    lot = source_archive.stem.removesuffix('_results')
    with zipfile.ZipFile(source_archive) as original:
        expected = [v for v, _ in video_directories(set(original.namelist()))]
    source_ids = sorted(v for v, entry in entries.items()
                        if Path(entry['zip_path']).name == f'Video_{lot}.zip')
    if expected != source_ids:
        raise ValueError(f'{lot}: result/source inventory differs: {expected} vs {source_ids}')
    selected = [v for v in expected if v not in blocked or v in recover]
    excluded = [v for v in expected if v in blocked and v not in recover]
    validated = []
    for video in selected:
        if video not in stages:
            raise ValueError(f'{lot}: no staged generation for {video}')
        folder, generation = stages[video]
        index = media._build_index(video, entries[video])
        keyframes, scenes, verified, _ = validate_video(video, index, folder, generation)
        validated.append((video, folder, generation, keyframes, scenes, verified))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / source_archive.name
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f'.{path.stem}.',
                                     suffix='.partial', delete=False) as handle:
        temporary = Path(handle.name)
    manifest = {'version': 2, 'source_result_archive': str(source_archive.resolve()),
                'source_result_archive_sha256': file_digest(source_archive),
                'source_archive_lot': lot, 'videos': [], 'release_blocked_excluded': excluded,
                'audited_recoveries_staged_but_not_released': sorted(set(selected) & recover)}
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as zf:
            for video, folder, generation, keyframes, scenes, verified in validated:
                prefix = f'videos__{video}'
                members = {
                    f'phase1_transnet/{prefix}/scenes.json': (folder / 'scenes.json').read_bytes(),
                    f'phase1_transnet/{prefix}/keyframes.json': (folder / 'keyframes.json').read_bytes(),
                    f'phase2_embeddings/{prefix}/embeddings.npy': (folder / 'embeddings.npy').read_bytes(),
                    f'phase2_embeddings/{prefix}/verified.json': (folder / 'verified.json').read_bytes(),
                }
                for name, data in members.items():
                    zf.writestr(name, data)
                manifest['videos'].append({'video_id': video, 'generation': generation,
                                           'rows': len(keyframes['keyframes']),
                                           'source_fingerprint': keyframes['source_identity'],
                                           'embedding_sha256': hashlib.sha256(
                                               members[f'phase2_embeddings/{prefix}/embeddings.npy']).hexdigest(),
                                           'model': verified['model'],
                                           'preprocess': verified['preprocess']})
            zf.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(temporary) as zf:
            if zf.testzip() is not None:
                raise ValueError(f'{lot}: staged ZIP CRC failure')
            actual = [v for v, _ in video_directories(set(zf.namelist()))]
            if actual != selected:
                raise ValueError(f'{lot}: staged video inventory mismatch')
            for item in manifest['videos']:
                video = item['video_id']
                prefix = f'videos__{video}'
                payload = zf.read(f'phase2_embeddings/{prefix}/embeddings.npy')
                if hashlib.sha256(payload).hexdigest() != item['embedding_sha256']:
                    raise ValueError(f'{video}: staged vector member changed')
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if array.shape != (item['rows'], 1280):
                    raise ValueError(f'{video}: staged vector count changed')
        with path.with_suffix(path.suffix + '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if not same_archive_members(path, temporary):
                    raise FileExistsError(f'{path}: a different staged candidate already exists; '
                                          'choose a new --output-dir to preserve the previous generation')
            else:
                os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest['staged_archive'] = str(path)
    manifest['staged_sha256'] = file_digest(path)
    atomic_json(output_dir / f'{lot}_validation.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / 'challenge_resources/data/zip_embeddings'
    parser.add_argument('--stage', type=Path, default=root / 'readiness/n')
    parser.add_argument('--audit-stage', type=Path, default=root / 'readiness/n031_audit')
    parser.add_argument('--source-dir', type=Path, default=root)
    parser.add_argument('--output-dir', type=Path, default=root / 'readiness/publication')
    parser.add_argument('--archive', help='Stage only this result ZIP filename')
    parser.add_argument('--recover', nargs='*', default=[],
                        help='Include independently verified blocked videos in staged archive only')
    args = parser.parse_args()
    stages = stage_entries(args.stage, args.audit_stage)
    blocked = release_blocked_video_ids()
    recover = frozenset(args.recover)
    if recover - blocked:
        raise ValueError(f'Recovery is not release-blocked: {sorted(recover - blocked)}')
    entries = media._scan_archives()
    archives = sorted(args.source_dir.glob('N*_results.zip'))
    if args.archive:
        archives = [p for p in archives if p.name == args.archive]
    if not archives:
        raise ValueError('No requested N result archive found')
    for number, archive in enumerate(archives, 1):
        manifest = build_archive(archive, entries, stages, blocked, recover, args.output_dir)
        print(f'{number}/{len(archives)} staged {archive.name}: '
              f'{len(manifest["videos"])} videos, '
              f'{sum(x["rows"] for x in manifest["videos"])} vectors, '
              f'excluded={manifest["release_blocked_excluded"]}', flush=True)


if __name__ == '__main__':
    main()
