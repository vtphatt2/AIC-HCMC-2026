"""Offline playback derivatives; serving only reads already validated copies."""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .readiness_policy import playback_policy, exceptional_decode_provenance, playback_decoder_threads
from .source_timeline import source_fingerprint, load_timeline, monotonic_entries
from .exact_frame_pts import showinfo_frames

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIRECTORY = ROOT / 'challenge_resources/data/zip_embeddings/playback'


def directory():
    return Path(os.getenv('PLAYBACK_COPY_DIR', str(DEFAULT_DIRECTORY)))


def source_map_sha256(video_id, index):
    """Bind playback to decoded PTS and pixels, including map-only repairs."""
    table = load_timeline(video_id, index)
    if table.ndim != 2 or table.shape[1] != 3 or table.dtype != np.int64:
        raise ValueError(f'{video_id}: invalid source map for playback')
    digest = hashlib.sha256()
    digest.update(json.dumps({'shape': table.shape, 'dtype': str(table.dtype)}).encode())
    digest.update(table.tobytes(order='C'))
    return digest.hexdigest()


def copy_paths(video_id, index, policy=None, root=None):
    policy = policy or playback_policy()
    fingerprint = {'source': source_fingerprint(index), 'policy': asdict(policy),
                   'source_map_sha256': source_map_sha256(video_id, index)}
    exceptional = exceptional_decode_provenance(video_id, index)
    if exceptional is not None:
        fingerprint['decode_provenance'] = exceptional
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    base = (root or directory()) / f'{video_id}-{signature}'
    return base.with_suffix('.mp4'), base.with_suffix('.json')


def validated_copy(video_id, index, policy=None, root=None):
    try:
        path, marker = copy_paths(video_id, index, policy, root)
        meta = json.loads(marker.read_text())
        stat = path.stat()
        if (meta.get('version') != 1 or not meta.get('validated') or
                meta.get('source') != source_fingerprint(index) or
                meta.get('source_map_sha256') != source_map_sha256(video_id, index) or
                meta.get('size') != stat.st_size or meta.get('mtime_ns') != stat.st_mtime_ns):
            return None
    except (OSError, ValueError):
        return None
    return path, meta


def serving_copy(video_id, index, policy=None, root=None):
    """Require picture alignment evidence before exposing a playback derivative."""
    copy = validated_copy(video_id, index, policy, root)
    return copy if copy and copy[1].get('picture_alignment_verified') is True else None


def probe_copy(path, expected_us):
    result = subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_streams', '-show_frames', '-show_entries',
                             'stream=codec_name,pix_fmt,width,height:frame=best_effort_timestamp_time',
                             '-of', 'json', str(path)], capture_output=True, check=True, timeout=1800)
    report = json.loads(result.stdout)
    stream, = report['streams']
    times = np.array([round(float(f['best_effort_timestamp_time']) * 1_000_000)
                      for f in report['frames']], dtype=np.int64)
    if stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p':
        raise ValueError('Playback codec/pixel format is not browser compatible')
    if len(times) != len(expected_us) or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError(f'Playback frame coverage differs: {len(times)}/{len(expected_us)}')
    drift = int(np.max(np.abs(times - expected_us)))
    if drift > 2000:
        raise ValueError(f'Playback clock drift: {drift} microseconds')
    return {'frames': len(times), 'max_timing_error_us': drift, **stream}


def prepare_copy(video_id, index, policy=None, root=None, timeout=7200):
    policy = policy or playback_policy()
    reusable = validated_copy(video_id, index, policy, root)
    if reusable and reusable[1].get('picture_alignment_verified') is True:
        return reusable
    table, omitted = monotonic_entries(load_timeline(video_id, index))
    origin = int(table[0, 1])
    expected_us = (table[:, 1] - origin) * 1_000_000 // index.timescale
    path, marker = copy_paths(video_id, index, policy, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = f'subfile,,start,{index.data_offset},end,{index.data_offset + index.video_size},,:{index.zip_path}'
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.playback-') as scratch:
        temporary = Path(scratch) / 'video.mp4'
        filters = ("select='isnan(prev_selected_pts)+gt(pts,prev_selected_pts)',showinfo,"
                   f"setpts=PTS-{origin},scale=-2:min({policy.height}\\,ih),format={policy.pixel_format}")
        command = ['/usr/bin/ffmpeg', '-hide_banner', '-nostdin', '-v', 'info', '-copyts',
                   '-threads', str(playback_decoder_threads(video_id, policy, index)),
                   '-i', source, '-map', '0:v:0', '-map', '0:a:0?',
                   '-vf', filters, '-af', f'asetpts=PTS-{origin / index.timescale:.12f}/TB',
                   '-c:v', policy.codec, '-preset', policy.preset, '-crf', str(policy.crf),
                   '-pix_fmt', policy.pixel_format, '-threads:v', str(policy.threads),
                   '-filter_threads', '1', '-fps_mode', 'vfr', '-enc_time_base', f'1:{index.timescale}',
                   '-video_track_timescale', str(index.timescale), '-c:a', 'aac', '-b:a', '128k',
                   '-movflags', '+faststart', '-y', str(temporary)]
        with tempfile.TemporaryFile() as diagnostics:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=diagnostics,
                           check=True, timeout=timeout)
            diagnostics.seek(0)
            observed = showinfo_frames(diagnostics.read())
        if len(observed) != len(table) or any(
                pts != int(expected[1]) or checksum != int(expected[2])
                for (_, pts, checksum), expected in zip(observed, table)):
            mismatch = next(((i, (pts, checksum), (int(expected[1]), int(expected[2])))
                             for i, ((_, pts, checksum), expected) in enumerate(zip(observed, table))
                             if pts != int(expected[1]) or checksum != int(expected[2])), None)
            raise ValueError(f'{video_id}: playback source-picture alignment failed: '
                             f'{len(observed)}/{len(table)} frames; first mismatch={mismatch}')
        validation = probe_copy(temporary, expected_us)
        if validation['height'] > policy.height:
            raise ValueError('Playback resolution exceeds configured bound')
        os.replace(temporary, path)
        stat = path.stat()
        meta = {'version': 1, 'validated': True, 'picture_alignment_verified': True,
                'picture_alignment_method': 'all_source_pts_checksums_before_encode_and_all_output_pts',
                'source': source_fingerprint(index), 'policy': asdict(policy),
                'source_map_sha256': source_map_sha256(video_id, index),
                'source_time_base': {'num': 1, 'den': index.timescale}, 'source_origin_pts': origin,
                'source_to_playback_offset_seconds': -origin / index.timescale, 'playback_start_seconds': 0,
                'omitted_frame_ids': omitted[:, 0].tolist(), 'validation': validation,
                'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if exceptional_decode_provenance(video_id, index) is not None:
            meta['decode_provenance'] = exceptional_decode_provenance(video_id, index)
        temp_marker = Path(scratch) / 'metadata.json'
        temp_marker.write_text(json.dumps(meta, indent=2))
        os.replace(temp_marker, marker)
    return path, meta


async def open_range(video_id, index, range_header):
    from .local_zip_media import _parse_range, LocalZipUnavailable, STREAM_CHUNK
    copy = await asyncio.to_thread(serving_copy, video_id, index)
    if copy is None: raise LocalZipUnavailable(f'Validated playback copy unavailable for {video_id}')
    path, meta = copy
    size = meta['size']
    start, end = _parse_range(range_header, size)
    async def body():
        with path.open('rb') as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = await asyncio.to_thread(file.read, min(remaining, STREAM_CHUNK))
                if not chunk: raise OSError('Playback file was truncated')
                remaining -= len(chunk)
                yield chunk
    return {'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
            'Content-Range': f'bytes {start}-{end}/{size}', 'Content-Length': str(end-start+1)}, body()
