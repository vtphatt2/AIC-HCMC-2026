"""Independently replay complete N videos and compare every source PTS/pixel map row.

This is an audit, not an automatic repair: a decoder disagreement blocks a
release decision until the source picture can be identified without guessing.
The report is checkpointed after each video so archive-wide audits can resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from app.services import local_zip_media as media
from app.services.exact_frame_pts import index_path, showinfo_frames
from app.services.source_timeline import source_fingerprint
from app.services.video_quarantine import block_release


def audit_video(video_id, index, threads: int, timeout: int = 1200, *, map_path=None):
    map_path = map_path or index_path(video_id, index)
    stat = map_path.stat()
    identity = {'source': source_fingerprint(index), 'map_path': str(map_path),
                'map_size': stat.st_size, 'map_mtime_ns': stat.st_mtime_ns}
    table = np.load(map_path, mmap_mode='r', allow_pickle=False)
    if table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64:
        raise ValueError(f'{video_id}: invalid full source map')
    source = (f'subfile,,start,{index.data_offset},end,'
              f'{index.data_offset + index.video_size},,:{index.zip_path}')
    command = ['/usr/bin/ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel',
               'info', '-copyts', '-threads', str(threads), '-i', source,
               '-map', '0:v:0', '-an', '-vf', 'showinfo', '-filter_threads', '1',
               '-vsync', '0', '-f', 'null', '-']
    with tempfile.TemporaryFile() as diagnostics:
        result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                stderr=diagnostics, timeout=timeout)
        diagnostics.seek(0)
        log = diagnostics.read()
    observed = showinfo_frames(log)
    length = min(len(table), len(observed))
    pts_mismatches = []
    checksum_mismatches = []
    first = []
    for position in range(length):
        _, pts, checksum = observed[position]
        expected_pts, expected_checksum = map(int, table[position, 1:3])
        if pts != expected_pts:
            pts_mismatches.append(position)
        if checksum != expected_checksum:
            checksum_mismatches.append(position)
        if (pts != expected_pts or checksum != expected_checksum) and len(first) < 12:
            first.append({'frame_id': int(table[position, 0]),
                          'expected_pts': expected_pts, 'observed_pts': pts,
                          'expected_checksum': f'{expected_checksum:08X}',
                          'observed_checksum': f'{checksum:08X}'})
    bases = set((int(a), int(b)) for a, b in re.findall(
        rb'config in time_base:\s*(\d+)/(\d+)', log))
    timebase_ok = bases == {(1, int(index.timescale))}
    complete = (result.returncode == 0 and len(observed) == len(table)
                and np.array_equal(table[:, 0], np.arange(len(table))))
    status = ('exact' if complete and timebase_ok and not pts_mismatches
              and not checksum_mismatches else 'mismatch')
    return {**identity, 'video_id': video_id, 'status': status,
            'decoder_threads': threads, 'ffmpeg_exit': result.returncode,
            'source_map_rows': len(table), 'decoded_rows': len(observed),
            'time_bases': sorted([list(x) for x in bases]),
            'expected_time_base': [1, int(index.timescale)],
            'pts_mismatch_count': len(pts_mismatches),
            'checksum_mismatch_count': len(checksum_mismatches),
            'first_mismatches': first,
            'first_pts_mismatch_ids': pts_mismatches[:12],
            'first_checksum_mismatch_ids': checksum_mismatches[:12]}


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    report['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, path)


def block_mismatch(video: str, result: dict, threads: int, report_path: Path) -> None:
    try:
        evidence = str(report_path.relative_to(ROOT / 'challenge_resources/data'))
    except ValueError:
        evidence = str(report_path)
    block_release(
        video,
        reason='decoder_sensitive_source_picture',
        details=(f"Complete {threads}-thread replay differs from the current source map: "
                 f"PTS mismatches={result['pts_mismatch_count']}, pixel mismatches="
                 f"{result['checksum_mismatch_count']}. Organizer corruption is not "
                 "established; verified decoder-profile recovery and derivative "
                 "validation are required."),
        evidence=evidence,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, help='Archive filename, e.g. Video_N001-N010.zip')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--block-mismatches', action='store_true',
                        help='Atomically release-block complete decoder disagreements')
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    report_path = (args.report or ROOT / 'challenge_resources/data/zip_embeddings/verification'
                   / f'decoder_agreement_{Path(args.archive).stem}.json').resolve()
    os.chdir(ROOT / 'remote-server')
    entries = media._scan_archives()
    videos = sorted(v for v, entry in entries.items()
                    if v.startswith('N') and Path(entry['zip_path']).name == args.archive)
    if args.videos:
        requested = set(args.videos)
        if requested - set(videos):
            raise ValueError(f'Videos outside archive: {sorted(requested - set(videos))}')
        videos = [v for v in videos if v in requested]
    if not videos:
        raise ValueError(f'No N videos in {args.archive}')
    settings = {'archive': args.archive, 'decoder_threads': args.threads,
                'selected_videos': videos}
    try:
        previous = json.loads(report_path.read_text())
        report = previous if previous.get('settings') == settings else None
    except (FileNotFoundError, ValueError):
        report = None
    if report is None:
        report = {'version': 1, 'settings': settings, 'videos': {}}
    for number, video in enumerate(videos, 1):
        index = media._build_index(video, entries[video])
        map_path = index_path(video, index)
        stat = map_path.stat()
        identity = {'source': source_fingerprint(index), 'map_path': str(map_path),
                    'map_size': stat.st_size, 'map_mtime_ns': stat.st_mtime_ns}
        saved = report['videos'].get(video)
        if not args.force and saved and all(saved.get(k) == v for k, v in identity.items()) and \
                saved.get('status') in ('exact', 'mismatch'):
            if args.block_mismatches and saved['status'] == 'mismatch':
                block_mismatch(video, saved, args.threads, report_path)
            print(f'{number}/{len(videos)} reused {video}: {saved["status"]}', flush=True)
            continue
        try:
            result = audit_video(video, index, args.threads)
        except Exception as exc:
            result = {**identity, 'video_id': video, 'status': 'error', 'error': str(exc)}
        report['videos'][video] = result
        write_report(report_path, report)
        if args.block_mismatches and result['status'] == 'mismatch':
            block_mismatch(video, result, args.threads, report_path)
        print(f'{number}/{len(videos)} {video}: {result["status"]} '
              f'PTS={result.get("pts_mismatch_count", "?")} '
              f'pixels={result.get("checksum_mismatch_count", "?")}', flush=True)
    report['summary'] = {status: sum(row['status'] == status for row in report['videos'].values())
                         for status in ('exact', 'mismatch', 'error')}
    write_report(report_path, report)
    print(report['summary'], flush=True)
    if report['summary']['mismatch'] or report['summary']['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
