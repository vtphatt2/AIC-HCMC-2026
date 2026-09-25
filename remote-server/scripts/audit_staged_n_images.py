"""Verify staged N source pictures and the exact 640px cards exposed by HTTP.

The per-video report is checkpointed only after every selected full-size JPEG
and served card passes. A stopped run resumes without republishing a partial
video. Source JPEG generation validates PTS and pre-encoding pixel checksums.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from PIL import Image
from app.services import local_zip_media as media
from app.services.exact_frame_images import build_exact_images
from app.services.exact_frame_pts import index_path
from app.services.source_timeline import source_fingerprint


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, path)


async def audit_cards(video_id: str, rows: list[dict], paths: list[Path]) -> dict:
    max_mae = 0.0
    max_frame = None
    sizes = set()
    for row, path in zip(rows, paths):
        with Image.open(path) as file:
            source = file.convert('RGB')
        width = min(640, source.width)
        expected_size = (width, max(1, round(source.height * width / source.width)))
        original = await media.get_frame_jpeg(video_id, int(row['timestamp_ms']),
                                               frame_number=int(row['frame_number']))
        if original != path.read_bytes():
            raise ValueError(f'{video_id}/{row["frame_number"]}: exposed original differs from verified JPEG')
        card = await media.get_frame_jpeg(video_id, int(row['timestamp_ms']),
                                          width=640, frame_number=int(row['frame_number']))
        with Image.open(io.BytesIO(card)) as file:
            card_rgb = file.convert('RGB')
        if card_rgb.size != expected_size:
            raise ValueError(f'{video_id}/{row["frame_number"]}: card size {card_rgb.size}/{expected_size}')
        expected = source.resize(expected_size, Image.Resampling.LANCZOS)
        mae = float(np.mean(np.abs(np.asarray(expected, dtype=np.int16) -
                                   np.asarray(card_rgb, dtype=np.int16))))
        if mae > 10.0:
            raise ValueError(f'{video_id}/{row["frame_number"]}: card/source RGB MAE {mae:.2f}')
        if mae > max_mae:
            max_mae, max_frame = mae, int(row['frame_number'])
        sizes.add(card_rgb.size)
    return {'cards': len(rows), 'max_card_rgb_mae': round(max_mae, 4),
            'max_mae_frame': max_frame, 'card_sizes': sorted([list(size) for size in sizes])}


async def run(args) -> None:
    os.chdir(ROOT / 'remote-server')
    stage = args.stage.resolve()
    manifest = json.loads((stage / 'manifest.json').read_text())
    entries = media._scan_archives()
    jobs = {row['video_id']: row for row in manifest['videos']}
    requested = set(args.videos or jobs)
    if requested - jobs.keys():
        raise ValueError(f'Videos absent from stage: {sorted(requested - jobs.keys())}')
    report_path = (args.report or stage / 'image_audit.json').resolve()
    try:
        report = json.loads(report_path.read_text())
        if report.get('stage') != str(stage): report = None
    except (FileNotFoundError, ValueError):
        report = None
    if report is None:
        report = {'version': 1, 'stage': str(stage), 'videos': {}}
    for number, video in enumerate(sorted(requested), 1):
        job = jobs[video]
        index = media._build_index(video, entries[video])
        map_path = index_path(video, index)
        stat = map_path.stat()
        identity = {'source': source_fingerprint(index), 'generation': job['generation'],
                    'map_path': str(map_path), 'map_size': stat.st_size,
                    'map_mtime_ns': stat.st_mtime_ns}
        previous = report['videos'].get(video)
        if not args.force and previous and previous.get('status') == 'pass' and all(
                previous.get(k) == v for k, v in identity.items()):
            print(f'{number}/{len(requested)} reused {video}: {previous["cards"]} cards', flush=True)
            continue
        try:
            metadata = json.loads((Path(job['path']) / 'keyframes.json').read_text())
            rows = metadata['keyframes']
            if metadata['generation'] != job['generation'] or len(rows) != job['rows']:
                raise ValueError('Stage manifest and metadata differ')
            if (metadata.get('source_identity') != identity['source'] or
                    metadata.get('source_time_base') != {'num': 1, 'den': index.timescale}):
                raise ValueError('Stage source fingerprint or time base differs')
            table = np.load(map_path, mmap_mode='r', allow_pickle=False)
            frames = [int(row['frame_number']) for row in rows]
            if len(frames) != len(set(frames)) or frames != sorted(frames):
                raise ValueError('Selected source frames are not unique and ordered')
            if any(frame >= len(table) or frame < 0 or
                   int(table[frame, 1]) != int(row['source_pts']) or
                   int(table[frame, 2]) != int(row['source_checksum']) or
                   int(row['source_timebase']) != index.timescale or
                   int(row['timestamp_ms']) !=
                   (int(row['source_pts']) * 1000 + index.timescale // 2) // index.timescale
                   for frame, row in zip(frames, rows)):
                raise ValueError('Selected source IDs/PTS/checksums/units differ from complete map')
            paths = await asyncio.to_thread(build_exact_images, video, index, frames)
            if len(paths) != len(rows):
                raise ValueError('Incomplete verified source JPEG set')
            cards = await audit_cards(video, rows, paths)
            result = {**identity, 'video_id': video, 'status': 'pass',
                      'source_jpegs': len(paths), 'source_picture_identity': 'all_selected_pts_checksums',
                      **cards}
        except Exception as exc:
            result = {**identity, 'video_id': video, 'status': 'error', 'error': str(exc)}
        report['videos'][video] = result
        write_report(report_path, report)
        print(f'{number}/{len(requested)} {video}: {result["status"]} '
              f'cards={result.get("cards", 0)} {result.get("error", "")}', flush=True)
    report['summary'] = {status: sum(row['status'] == status for row in report['videos'].values())
                         for status in ('pass', 'error')}
    write_report(report_path, report)
    if report['summary']['error']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--force', action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
