"""Verify a candidate decoder for blocked N sources and register reusable profiles.

Source ZIPs and previous maps remain intact. Registration requires every decoded
frame ID, PTS and checksum to match a complete independent replay. This prepares
the source decoder policy only; staged vectors/images/playback and publication
are still required before a video can be released.
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
from app.services.decoder_profiles import register_profile
from app.services.exact_frame_pts import build_exact_pts
from app.services.video_quarantine import release_blocked_video_ids
from scripts.audit_decoder_agreement import audit_video, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report = args.report.resolve()
    os.chdir(ROOT / 'remote-server')
    if not 1 <= args.threads <= 16:
        raise ValueError('Decoder threads must be in 1..16')
    videos = sorted(set(args.videos))
    blocked = release_blocked_video_ids()
    if any(not video.startswith('N') or video not in blocked for video in videos):
        raise ValueError('Offline decoder recovery requires explicitly release-blocked N sources')
    entries = media._scan_archives()
    if set(videos) - entries.keys():
        raise ValueError(f'Missing sources: {sorted(set(videos) - entries.keys())}')
    root = ROOT / 'challenge_resources/data/zip_embeddings'
    report = {'version': 1, 'decoder_threads': args.threads, 'videos': {}}
    for video in videos:
        try:
            index = media._build_index(video, entries[video])
            folder, = root.glob(f'output_*/videos__{video}')
            scenes = json.loads((folder / 'scenes.json').read_text())
            rows = json.loads((folder / 'keyframes.json').read_text())['keyframes']
            candidate = build_exact_pts(video, index, [int(row['frame_number']) for row in rows],
                                        int(scenes['num_frames']), '/usr/bin/ffmpeg',
                                        store_full_timeline=True, decoder_threads=args.threads)
            evidence = audit_video(video, index, args.threads, map_path=candidate)
            report['videos'][video] = evidence
            write_report(args.report, report)
            if evidence['status'] != 'exact':
                raise ValueError('Candidate decoder does not reproduce the complete source map')
            register_profile(video, index, evidence, args.report)
            print(f'{video}: registered verified {args.threads}-thread source profile; release block retained', flush=True)
        except Exception as exc:
            report['videos'].setdefault(video, {}).update(status='error', error=str(exc))
            print(f'{video}: failed; release block retained: {exc}', flush=True)
        write_report(args.report, report)
    if any(row['status'] != 'exact' for row in report['videos'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
