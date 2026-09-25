"""Check every N presentation map and versioned timeline without decoding pixels.

This is a structural and timing audit. Decoder replay and selected-picture
audits provide separate source-picture evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from app.services import local_zip_media as media
from app.services.exact_frame_pts import index_path
from app.services.source_timeline import source_fingerprint, timeline_response

DATA = ROOT / 'challenge_resources/data/zip_embeddings'
REPORT = DATA / 'verification/n_timeline_audit.json'


def audit_video(video_id: str, entry: dict) -> dict:
    index = media._build_index(video_id, entry)
    source_lot = index.zip_path.stem.removeprefix('Video_')
    scenes = json.loads((DATA / f'output_{source_lot}' /
                         f'videos__{video_id}' / 'scenes.json').read_text())
    frame_count = int(scenes['num_frames'])
    path = index_path(video_id, index)
    table = np.load(path, mmap_mode='r', allow_pickle=False)
    if (table.shape != (frame_count, 3) or table.dtype != np.int64 or
            not np.array_equal(table[:, 0], np.arange(frame_count))):
        raise ValueError(f'Incomplete source frame map: {table.shape}/{frame_count}')
    timeline = timeline_response(video_id, index, table)
    ids = timeline['frame_ids']
    pts = timeline['source_pts']
    times = timeline['presentation_us']
    omitted = timeline['omitted_frame_ids']
    if (timeline['version'] != 2 or timeline['submission_unit'] != 'milliseconds' or
            timeline['verified_timing'] is not True or
            len(ids) + len(omitted) != frame_count or ids[0] != 0 or
            times[0] != 0 or ids != sorted(set(ids)) or
            sorted(ids + omitted) != list(range(frame_count)) or
            any(b <= a for a, b in zip(pts, pts[1:])) or
            any(b <= a for a, b in zip(times, times[1:])) or
            any(p != int(table[f, 1]) for f, p in zip(ids, pts)) or
            any(t != (p - pts[0]) * 1_000_000 // index.timescale
                for p, t in zip(pts, times))):
        raise ValueError('Versioned timeline lost source identity or presentation order')
    if any(int(table[f, 1]) > int(table[f - 1, 1]) and
           int(table[f, 1]) > max(map(int, table[:f, 1])) for f in omitted):
        raise ValueError('Omitted a strictly increasing source timestamp')
    positions = sorted({0, len(ids) // 2, len(ids) - 1})
    samples = [{'frame_id': ids[i], 'source_pts': pts[i],
                'presentation_us': times[i]} for i in positions]
    neighbors = []
    for frame in omitted:
        neighbors.append({'omitted_frame_id': frame,
                          'previous_kept_id': max((f for f in ids if f < frame), default=None),
                          'next_kept_id': next((f for f in ids if f > frame), None)})
    return {'status': 'pass', 'source': source_fingerprint(index),
            'map_path': str(path), 'source_frames': frame_count,
            'timeline_frames': len(ids), 'omitted_frame_ids': omitted,
            'time_base': timeline['time_base'], 'samples': samples,
            'omission_neighbors': neighbors,
            'assessment_depth': 'all_numeric_entries_no_pixel_decode'}


def main() -> None:
    os.chdir(ROOT / 'remote-server')
    entries = {video: entry for video, entry in media._scan_archives().items()
               if video.startswith('N')}
    report = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'expected_videos': 298, 'videos': {}, 'issues': []}
    if len(entries) != 298:
        report['issues'].append(f'N source inventory {len(entries)}/298')
    for video_id, entry in sorted(entries.items()):
        try:
            report['videos'][video_id] = audit_video(video_id, entry)
        except Exception as exc:
            report['videos'][video_id] = {'status': 'error', 'error': str(exc)}
            report['issues'].append(f'{video_id}: {exc}')
        print(f'{len(report["videos"])}/{len(entries)} {video_id}: '
              f'{report["videos"][video_id]["status"]}', flush=True)
    report['summary'] = {'pass': sum(row['status'] == 'pass' for row in report['videos'].values()),
                         'error': len(report['issues']),
                         'omitted_entries': sum(len(row.get('omitted_frame_ids', []))
                                                for row in report['videos'].values())}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix('.json.partial')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, REPORT)
    print(f'Timeline audit: {report["summary"]}; {REPORT}', flush=True)
    if report['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
