#!/usr/bin/env python3
"""Release N search visibility only from a current, complete live audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))

from app.services.video_quarantine import manifest_path, release_videos
from scripts.audit_readiness import file_identity

DATA = ROOT / 'challenge_resources/data'
RESULTS = DATA / 'zip_embeddings'


def validate_release_report(report: dict, *, image_reports: list[dict],
                            result_dir: Path, export_dir: Path,
                            quarantine_entries: dict) -> dict[str, str]:
    result_dir = result_dir.resolve()
    export_dir = export_dir.resolve()
    if (report.get('version') != 1 or report.get('status') != 'pass' or
            report.get('issues') or report.get('checks') != {
                'source_crc': True, 'live_indexes': True,
                'source_picture_semantics': 'not assessed here'} or
            Path(report.get('result_dir', '')).resolve() != result_dir or
            Path(report.get('export_dir', '')).resolve() != export_dir):
        raise ValueError('Release requires a passing CRC and live-index audit of the live paths')

    counts = report.get('counts', {})
    if (counts.get('source_archives') != 21 or counts.get('source_videos') != 614 or
            counts.get('result_archives') != 21 or counts.get('result_videos') != 614 or
            counts.get('source_by_lot', {}).get('N') != 298 or
            counts.get('release_blocked_missing_results') or
            counts.get('unexpected_numpy_rows') != 0):
        raise ValueError('Release audit inventory is incomplete')

    archives = report.get('result_archives', [])
    actual_names = {path.name for path in result_dir.glob('*_results.zip')
                    if path.name.startswith(('M', 'N', 'S'))}
    if len(archives) != 21 or {row.get('name') for row in archives} != actual_names:
        raise ValueError('Release result archive inventory changed')
    for row in archives:
        path = result_dir / row['name']
        if row.get('crc_ok') is not True or any(
                row.get(key) != value for key, value in file_identity(path).items()):
            raise ValueError(f'{path.name}: result archive changed after audit')

    numpy_report = report.get('numpy', {})
    if (numpy_report.get('mns_rows') != counts.get('result_rows') or
            numpy_report.get('vectors_file') != file_identity(export_dir / 'vectors.f32.npy') or
            numpy_report.get('metadata_file') != file_identity(export_dir / 'vectors.meta.npz')):
        raise ValueError('NumPy export changed after audit')

    videos = report.get('videos', {})
    n_videos = {video: row for video, row in videos.items() if video.startswith('N')}
    if len(n_videos) != 298:
        raise ValueError('Release audit does not contain all 298 N videos')
    for video, row in n_videos.items():
        exported = row.get('numpy_export', {})
        if (not row.get('verified_generation') or row.get('playback_copy') is not True or
                row.get('source_pts_rows') != row.get('selected') or
                not row.get('pts_map', {}).get('sha256') or
                exported.get('missing') != 0 or exported.get('differing') != 0):
            raise ValueError(f'{video}: release evidence is incomplete')

    image_rows = {}
    for image_report in image_reports:
        rows = image_report.get('videos', {})
        summary = image_report.get('summary', {})
        if (image_report.get('version') != 1 or summary.get('error') != 0 or
                summary.get('pass') != len(rows)):
            raise ValueError('Selected-card audit is incomplete')
        for video, row in rows.items():
            if video in image_rows:
                raise ValueError(f'{video}: duplicate selected-card audit')
            image_rows[video] = row
    if set(image_rows) != set(n_videos):
        raise ValueError('Selected-card audits do not cover all 298 N videos')
    for video, row in image_rows.items():
        expected = n_videos[video]
        map_path = Path(row.get('map_path', ''))
        try:
            map_stat = file_identity(map_path)
        except OSError as exc:
            raise ValueError(f'{video}: selected-card source map is unavailable') from exc
        if (row.get('status') != 'pass' or
                row.get('source_picture_identity') != 'all_selected_pts_checksums' or
                row.get('generation') != expected.get('verified_generation') or
                row.get('source') != expected.get('pts_map', {}).get('source') or
                row.get('cards') != expected.get('selected') or
                row.get('webp_cards') != expected.get('selected') or
                row.get('map_size') != map_stat['size'] or
                row.get('map_mtime_ns') != map_stat['mtime_ns']):
            raise ValueError(f'{video}: selected-card evidence is stale or incomplete')

    postgres = report.get('postgres', {})
    if postgres.get('videos') != 614 or postgres.get('missing') or postgres.get('extra'):
        raise ValueError('PostgreSQL does not match the audited generation')
    for algorithm in ('hnsw', 'flat'):
        index = report.get('indexes', {}).get(algorithm, {})
        if index.get('videos_checked') != 614 or index.get('mismatches') != 0:
            raise ValueError(f'{algorithm}: live vector index is incomplete')

    expected_reasons = {}
    for video, entry in quarantine_entries.items():
        if not video.startswith('N') or not isinstance(entry, dict):
            raise ValueError(f'{video}: invalid N quarantine entry')
        if entry.get('released') is True:
            continue
        if entry.get('release_blocked') is True:
            raise ValueError(f'{video}: source-identity release block is still active')
        if video not in n_videos or not entry.get('reason'):
            raise ValueError(f'{video}: missing audited release evidence')
        expected_reasons[video] = entry['reason']
    return expected_reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--image-report', type=Path, action='append', required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    quarantine = json.loads(manifest_path().read_text())
    if quarantine.get('version') != 1 or not isinstance(quarantine.get('videos'), dict):
        raise ValueError('Invalid quarantine manifest')
    image_report_paths = [path.resolve() for path in args.image_report]
    image_reports = [json.loads(path.read_text()) for path in image_report_paths]
    expected = validate_release_report(
        report, image_reports=image_reports, result_dir=RESULTS, export_dir=DATA,
        quarantine_entries=quarantine['videos'])
    if not expected:
        print('All audited N videos are already released')
        return
    image_digests = ','.join(
        f'{path}:sha256={hashlib.sha256(path.read_bytes()).hexdigest()}'
        for path in image_report_paths)
    evidence = (f'{report_path}:sha256={hashlib.sha256(report_bytes).hexdigest()};'
                f'images={image_digests}')
    release_videos(expected, evidence=evidence)
    print(f'Released {len(expected)} N videos from verified quarantine')


if __name__ == '__main__':
    main()
