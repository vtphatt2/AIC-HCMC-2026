"""Classify cross-setting pixel differences by repeating the current map setting.

A one-thread/four-thread difference is harmless to the unified pipeline when a
fresh four-thread replay reproduces the authoritative four-thread map exactly.
Only a same-setting disagreement establishes map instability and warrants source
profile recovery. Reports are checkpointed per video.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
from app.services import local_zip_media as media
from app.services.exact_frame_pts import index_path
from app.services.readiness_policy import source_map_decoder_threads
from app.services.source_timeline import source_fingerprint, source_map_sha256
from app.services.video_quarantine import clear_release_block
from scripts.audit_decoder_agreement import audit_video, write_report


def replay_identity(video: str, index, map_path: Path, threads: int) -> dict:
    """Require a repeat of this source's current authoritative map setting."""
    if (map_path.resolve() != index_path(video, index).resolve() or
            threads != source_map_decoder_threads(video, index)):
        raise ValueError(f'{video}: stability replay must use the current source map and decoder')
    stat = map_path.stat()
    return {'source': source_fingerprint(index), 'map_path': str(map_path.resolve()),
            'map_size': stat.st_size, 'map_mtime_ns': stat.st_mtime_ns,
            'map_sha256': source_map_sha256(map_path), 'decoder_threads': threads}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-reports', nargs='+', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--clear-stable-blocks', action='store_true')
    args = parser.parse_args()
    args.report = args.report.resolve()
    os.chdir(ROOT / 'remote-server')
    entries = media._scan_archives()
    candidates = {}
    for path in args.archive_reports:
        source = json.loads(path.resolve().read_text())
        for video, row in source.get('videos', {}).items():
            if row.get('status') == 'mismatch':
                candidates[video] = row
    try:
        report = json.loads(args.report.read_text())
        if report.get('version') != 1 or report.get('threads') != args.threads:
            report = None
    except (FileNotFoundError, ValueError):
        report = None
    if report is None:
        report = {'version': 1, 'threads': args.threads, 'videos': {}}
    for number, (video, source_row) in enumerate(sorted(candidates.items()), 1):
        index = media._build_index(video, entries[video])
        map_path = Path(source_row['map_path'])
        identity = replay_identity(video, index, map_path, args.threads)
        if any(source_row.get(key) != identity[key] for key in
               ('source', 'map_path', 'map_size', 'map_mtime_ns')):
            raise ValueError(f'{video}: cross-setting report belongs to an old source map')
        saved = report['videos'].get(video)
        if saved and saved.get('identity') == identity and \
                saved.get('cross_setting_mismatch') == source_row and \
                saved.get('classification') in ('stable_current_map', 'unstable_current_map'):
            result = saved
        else:
            replay = audit_video(video, index, args.threads, map_path=map_path)
            classification = ('stable_current_map' if replay['status'] == 'exact'
                              else 'unstable_current_map')
            result = {'classification': classification, 'identity': identity,
                      'cross_setting_mismatch': source_row, 'same_setting_replay': replay}
            report['videos'][video] = result
            write_report(args.report, report)
        if args.clear_stable_blocks and result['classification'] == 'stable_current_map':
            clear_release_block(
                video, expected_reason='decoder_sensitive_source_picture',
                reason='verified_decoder_setting_variation',
                details=(f"A fresh {args.threads}-thread replay exactly reproduces the current "
                         "authoritative map. Cross-setting pixel variation is not a release defect."),
                evidence=str(args.report),
            )
        print(f'{number}/{len(candidates)} {video}: {result["classification"]}', flush=True)
    report['summary'] = {kind: sum(row['classification'] == kind
                                   for row in report['videos'].values())
                         for kind in ('stable_current_map', 'unstable_current_map')}
    write_report(args.report, report)
    print(report['summary'], flush=True)


if __name__ == '__main__':
    main()
