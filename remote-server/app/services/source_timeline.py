"""Source-picture identity and presentation selection, independent of encodings."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .exact_frame_pts import index_path
from .readiness_policy import SelectionPolicy


def source_fingerprint(index):
    stat = index.zip_path.stat()
    return {'path': str(index.zip_path.resolve()), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'offset': index.data_offset,
            'video_size': index.video_size}


def monotonic_entries(table):
    if (table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64 or
            not len(table) or not np.array_equal(table[:, 0], np.arange(len(table)))):
        raise ValueError('Full source frame/PTS/checksum map required')
    keep = np.ones(len(table), dtype=bool)
    if len(table) > 1:
        keep[1:] = table[1:, 1] > np.maximum.accumulate(table[:, 1])[:-1]
    return table[keep], table[~keep]


def load_timeline(video_id, index):
    return np.load(index_path(video_id, index), mmap_mode='r', allow_pickle=False)


def timeline_response(video_id, index, table=None):
    kept, omitted = monotonic_entries(load_timeline(video_id, index) if table is None else table)
    scale = int(index.timescale)
    if scale <= 0:
        raise ValueError('Invalid source time base')
    origin = int(kept[0, 1])
    return {'version': 2, 'video_id': video_id, 'submission_unit': 'milliseconds',
            'verified_timing': True, 'time_base': {'num': 1, 'den': scale},
            'playback_origin_pts': origin, 'source_fingerprint': source_fingerprint(index),
            'frame_ids': kept[:, 0].tolist(), 'source_pts': kept[:, 1].tolist(),
            'presentation_us': ((kept[:, 1] - origin) * 1_000_000 // scale).tolist(),
            'omitted_frame_ids': omitted[:, 0].tolist(),
            'omission_reason': 'non-increasing source presentation timestamp'}


def select_pictures(table, timescale, scenes, existing, policy=SelectionPolicy()):
    kept, omitted = monotonic_entries(table)
    if timescale <= 0:
        raise ValueError('Invalid source time base')
    pts = kept[:, 1]
    # Integer ticks avoid cumulative floating-point drift on long timelines.
    step = max(1, round(policy.interval_seconds * timescale))
    grid = np.arange(int(pts[0]), int(pts[-1]) + 1, step, dtype=np.int64)
    positions = np.searchsorted(pts, grid)
    left = np.maximum(positions - 1, 0)
    positions = np.minimum(positions, len(pts) - 1)
    positions = np.where(grid - pts[left] <= pts[positions] - grid, left, positions)
    selected = set(map(int, kept[positions, 0]))
    allowed = set(map(int, kept[:, 0]))
    if policy.retain_existing:
        selected.update(int(row['frame_number']) for row in existing if int(row['frame_number']) in allowed)
    ends = [int(scene['end_frame']) for scene in scenes]
    starts = [int(scene['start_frame']) for scene in scenes]
    if not ends or starts[0] < 0 or any(b < a for a, b in zip(starts, ends)) or any(
            starts[i] <= ends[i-1] for i in range(1, len(starts))):
        raise ValueError('Scenes must be ordered and non-overlapping')
    rows = []
    for frame in sorted(selected):
        scene = int(np.searchsorted(ends, frame))
        if scene >= len(scenes):
            raise ValueError(f'Frame {frame} is outside scene coverage')
        # TransNet may omit a transition picture between scene ranges. Keep
        # its verified source identity without inventing a scene boundary.
        if frame < starts[scene]:
            scene = -1
        _, timestamp, checksum = map(int, table[frame])
        rows.append({'frame_number': frame, 'scene_index': scene,
                     'source_pts': timestamp, 'source_checksum': checksum,
                     'source_timebase': int(timescale),
                     'timestamp_ms': (timestamp * 1000 + timescale // 2) // timescale})
        if scene == -1:
            rows[-1]['scene_association'] = 'transition gap in original TransNet output'
    return rows, [{'frame_number': int(f), 'source_pts': int(p), 'source_checksum': int(c),
                   'reason': 'non-increasing presentation timestamp'} for f, p, c in omitted]


def generation_signature(identity, policy, scenes, existing, decode_provenance=None):
    payload = {'source_identity': identity, 'selection': asdict(policy),
               'scenes': scenes, 'retained_frames': sorted(int(x['frame_number']) for x in existing)}
    if decode_provenance is not None:
        payload['decode_provenance'] = decode_provenance
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
