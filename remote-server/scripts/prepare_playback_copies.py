"""Prepare N playback offline with bounded, resumable conversion workers."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
from app.services import local_zip_media as media
from app.services.playback_copies import prepare_copy, directory
from app.services.readiness_policy import BROWSER_COMPATIBLE_N_SOURCES
from app.services.video_quarantine import release_blocked_video_ids


def playback_job_settings(workers, nice):
    """Resolve resource controls without changing per-video artifact policy."""
    try:
        workers = int(os.getenv('PLAYBACK_PREPARE_WORKERS', '1')) if workers is None else workers
        nice = int(os.getenv('PLAYBACK_PREPARE_NICE', '0')) if nice is None else nice
    except ValueError as exc:
        raise ValueError('Playback worker and nice settings must be integers') from exc
    if not 1 <= workers <= 4:
        raise ValueError('workers must be 1..4')
    if not 0 <= nice <= 19:
        raise ValueError('nice must be 0..19')
    return workers, nice


def main():
    os.chdir(ROOT / 'remote-server')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='*')
    parser.add_argument('--workers', type=int,
                        help='Concurrent conversions (default: PLAYBACK_PREPARE_WORKERS or 1)')
    parser.add_argument('--nice', type=int,
                        help='Linux background priority increment (default: PLAYBACK_PREPARE_NICE or 0)')
    parser.add_argument('--audit-release-blocked', action='store_true',
                        help='Prepare explicitly named blocked videos offline; keep release blocks unchanged')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.workers, nice = playback_job_settings(args.workers, args.nice)
    if nice:
        os.nice(nice)
    print(f'Playback preparation: workers={args.workers} nice={nice}', flush=True)
    entries = media._scan_archives()
    blocked = release_blocked_video_ids()
    if args.audit_release_blocked and not args.videos:
        raise ValueError('--audit-release-blocked requires explicit --videos')
    if args.videos and set(args.videos) - entries.keys():
        raise ValueError(f'Missing requested sources: {sorted(set(args.videos) - entries.keys())}')
    if args.videos and set(args.videos) & blocked and not args.audit_release_blocked:
        raise ValueError(f'Release-blocked sources cannot be converted: {sorted(set(args.videos) & blocked)}')
    videos = args.videos or sorted(v for v in entries if v.startswith('N') and
                                  v not in blocked and v not in BROWSER_COMPATIBLE_N_SOURCES)
    failures = []
    def prepare(video):
        if not args.audit_release_blocked and video in release_blocked_video_ids():
            raise ValueError('Source was release-blocked after this job started')
        return prepare_copy(video, media._build_index(video, entries[video]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(prepare, video): video for video in videos}
        for number, future in enumerate(concurrent.futures.as_completed(pending), 1):
            video = pending[future]
            try:
                _, metadata = future.result()
                print(f'{number}/{len(videos)} verified {video}: {metadata["validation"]}', flush=True)
            except Exception as exc:
                failures.append({'video_id': video, 'error': str(exc)})
                print(f'FAIL {video}: {exc}', flush=True)
    directory().mkdir(parents=True, exist_ok=True)
    report = args.report or directory() / 'failures.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + '.partial')
    temporary.write_text(json.dumps(failures, indent=2))
    os.replace(temporary, report)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
