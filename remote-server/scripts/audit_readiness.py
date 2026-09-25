"""Read-only M/N/S readiness audit. Writes evidence even when checks fail.

Default checks every source/result/video/map/vector/export structurally. --crc
reads every source ZIP byte; --live compares PostgreSQL and all populated
Milvus raw indexes. Picture semantics and browser playback are separate checks.
"""
from __future__ import annotations
import argparse
import asyncio
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'remote-server'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'remote-server/.env')
import numpy as np
from app.services import local_zip_media as media
from app.services.exact_frame_pts import index_path
from app.services.source_timeline import monotonic_entries, source_fingerprint
from app.services.playback_copies import validated_copy
from app.services.video_quarantine import release_blocked_video_ids
from scripts.ingest_zip_pipeline_results import video_directories, selected_timestamp_ms

DATA = ROOT / 'challenge_resources/data'
RESULTS = DATA / 'zip_embeddings'
EXPECTED_ARCHIVES = {f'Videos_M{i:02d}.zip' for i in range(1,11)} | {f'Video_N{i:03d}-N{i+9:03d}.zip' for i in range(1,100,10)} | {'Video_S01.zip'}
# The NumPy export normalizes float32 rows once more. Packaging, in contrast,
# copies embedding bytes exactly and is checked with array_equal below.
MAX_EXPORT_VECTOR_ABS_ERROR = 1e-6


def issue(report, scope, detail):
    report['issues'].append({'scope': scope, 'detail': str(detail)})


def export_vector_error(packaged, exported):
    """Maximum component drift, or infinity for an invalid export row."""
    if (packaged.shape != exported.shape or not np.isfinite(packaged).all()
            or not np.isfinite(exported).all()):
        return float('inf')
    return float(np.max(np.abs(exported - packaged)))


def unexpected_export_ids(export_lookup, packaged_ids):
    return set(export_lookup) - set(packaged_ids)


def verify_packaged_generation(video_id, keyframes, marker, payloads):
    """A v2 ZIP carries its own verified generation; no scratch output is needed."""
    rows = keyframes['keyframes']
    if (keyframes.get('version') != 2 or
            marker.get('version') != 1 or
            marker.get('generation') != keyframes.get('generation') or
            marker.get('rows') != len(rows) or
            keyframes.get('num_keyframes') != len(rows) or
            marker.get('source_checksums') != 'exhaustive' or
            marker.get('source_time_base_verified') is not True or
            marker.get('published') is not False or
            not marker.get('model') or not marker.get('preprocess')):
        raise ValueError(f'{video_id}: incomplete packaged verification marker')
    for name, data in payloads.items():
        if marker.get(f'{name}_sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError(f'{video_id}: packaged {name} digest differs from verification')


def verify_n_source_identity(video_id, selected, table, timescale, identity):
    """Check every selected v2 N row against the decoded source map."""
    kept, omitted = monotonic_entries(table)
    if (selected.get('source_identity') != identity or
            selected.get('source_time_base') != {'num': 1, 'den': timescale} or
            selected.get('source_frame_count') != len(table)):
        raise ValueError(f'{video_id}: staged source identity/time base changed')
    expected_omissions = [
        {'frame_number': int(frame), 'source_pts': int(pts),
         'source_checksum': int(checksum),
         'reason': 'non-increasing presentation timestamp'}
        for frame, pts, checksum in omitted]
    if selected.get('omitted_entries', []) != expected_omissions:
        raise ValueError(f'{video_id}: omitted presentation identities changed')
    allowed = set(map(int, kept[:, 0]))
    for item in selected['keyframes']:
        frame = int(item['frame_number'])
        if frame not in allowed or frame >= len(table):
            raise ValueError(f'{video_id}/{frame}: selected omitted/out-of-range frame')
        if int(item['source_pts']) != int(table[frame, 1]):
            raise ValueError(f'{video_id}/{frame}: source PTS differs from map')
        if int(item['source_checksum']) != int(table[frame, 2]):
            raise ValueError(f'{video_id}/{frame}: source checksum differs from map')
        if int(item['source_timebase']) != timescale:
            raise ValueError(f'{video_id}/{frame}: source time base differs from map')
        selected_timestamp_ms(video_id, item, float('nan'), 2)


def verify_scene_association(video_id, selected, scenes):
    """N v2 may retain source-verified frames in omitted TransNet transitions."""
    versioned_n = video_id.startswith('N') and selected.get('version') == 2
    ends = [int(scene['end_frame']) for scene in scenes]
    for item in selected['keyframes']:
        frame = int(item['frame_number'])
        position = bisect_left(ends, frame)
        in_scene = (position < len(scenes) and
                    int(scenes[position]['start_frame']) <= frame)
        if not in_scene:
            if not versioned_n:
                raise ValueError(f'{video_id}/{frame}: selected frame lies in a TransNet transition gap')
            if (item.get('scene_index') != -1 or
                    item.get('scene_association') != 'transition gap in original TransNet output'):
                raise ValueError(f'{video_id}/{frame}: invalid transition gap scene association')
        elif versioned_n and item.get('scene_index') != position:
            raise ValueError(f'{video_id}/{frame}: selected scene association differs from TransNet')


def verify_result_inventory(archive_name, source_by_id, packaged_ids, manifest, blocked):
    """Only a documented, still-blocked source may be absent from an N ZIP."""
    lot = archive_name.removesuffix('_results.zip')
    expected = {video for video, source in source_by_id.items()
                if source['archive'] == f'Video_{lot}.zip'}
    if packaged_ids - expected:
        raise ValueError(f'{archive_name}: packaged videos are not in the source archive')
    missing = expected - packaged_ids
    if manifest is None:
        if missing:
            raise ValueError(f'{archive_name}: undocumented missing source videos {sorted(missing)}')
        return set()
    if manifest.get('version') != 2:
        raise ValueError(f'{archive_name}: invalid result manifest version')
    included = [row['video_id'] for row in manifest['videos']]
    excluded = manifest['release_blocked_excluded']
    if (len(included) != len(set(included)) or set(included) != packaged_ids or
            len(excluded) != len(set(excluded)) or set(excluded) != missing):
        raise ValueError(f'{archive_name}: undocumented or inconsistent result inventory')
    if missing - blocked:
        raise ValueError(f'{archive_name}: excluded source videos are not release-blocked')
    return missing


def audit(crc=False, live=False, output=None):
    os.chdir(ROOT / 'remote-server')
    started = time.time()
    report = {'version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'checks': {'source_crc': crc, 'live_indexes': live, 'source_picture_semantics': 'not assessed here'},
              'expected': {'source_archives': 21, 'videos': 614}, 'source_archives': [],
              'result_archives': [], 'videos': {}, 'indexes': {}, 'issues': []}
    source_by_id = {}
    archive_index = media._scan_archives()
    source_paths = sorted(p for p in media.raw_zip_dir().glob('*.zip') if p.name.startswith(('Videos_M','Video_N','Video_S')))
    names = {p.name for p in source_paths}
    for missing in sorted(EXPECTED_ARCHIVES - names): issue(report, 'source', f'missing archive {missing}')
    for extra in sorted(names - EXPECTED_ARCHIVES): issue(report, 'source', f'extra archive {extra}')
    for path in source_paths:
        stat = path.stat()
        row = {'name': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
               'videos': 0, 'crc_checked': crc, 'crc_ok': None}
        try:
            with zipfile.ZipFile(path) as zf:
                seen_names = set()
                for info in zf.infolist():
                    if info.filename in seen_names: issue(report, path.name, f'duplicate entry {info.filename}')
                    seen_names.add(info.filename)
                    if not info.filename.lower().endswith(('.mov','.mp4')): continue
                    video = Path(info.filename).stem
                    if video in source_by_id: issue(report, video, 'duplicate source video')
                    offset = media._data_offset(path, info.header_offset)
                    if offset + info.compress_size > stat.st_size or info.file_size <= 0 or info.compress_type != zipfile.ZIP_STORED:
                        issue(report, video, 'invalid stored media entry size/offset')
                    source_by_id[video] = {'archive': path.name, 'entry': info.filename,
                                           'offset': offset, 'size': info.file_size, 'crc32': info.CRC}
                    row['videos'] += 1
                if crc:
                    bad = zf.testzip()
                    row['crc_ok'] = bad is None
                    if bad: issue(report, path.name, f'CRC failed: {bad}')
        except Exception as exc: issue(report, path.name, f'source ZIP failed: {exc}')
        report['source_archives'].append(row)
        print(f'Source {path.name}: {row["videos"]} videos, CRC={row["crc_ok"]}', flush=True)
    if len(source_paths) != 21: issue(report, 'source', f'{len(source_paths)}/21 source archives')
    if len(source_by_id) != 614: issue(report, 'source', f'{len(source_by_id)}/614 unique videos')
    report['counts'] = {'source_archives': len(source_paths), 'source_videos': len(source_by_id),
                        'source_by_lot': dict(Counter(video[0] for video in source_by_id))}

    export = None
    timestamps = None
    export_lookup = {}
    try:
        export = np.load(DATA / 'vectors.f32.npy', mmap_mode='r', allow_pickle=False)
        with np.load(DATA / 'vectors.meta.npz', allow_pickle=False) as meta:
            ids = meta['frame_id']; videos = meta['video_id']; frames = meta['frame_number']; timestamps = meta['timestamp_ms']
        if export.shape != (len(ids), 1280): issue(report, 'numpy', f'vector/metadata shape {export.shape}/{len(ids)}')
        for i, video in enumerate(videos):
            if str(video).startswith(('M','N','S')):
                frame_id = str(ids[i])
                if frame_id in export_lookup: issue(report, 'numpy', f'duplicate frame {frame_id}')
                export_lookup[frame_id] = i
        report['numpy'] = {'total_rows': len(ids), 'mns_rows': len(export_lookup), 'shape': list(export.shape)}
    except Exception as exc:
        export = None
        timestamps = None
        export_lookup.clear()
        issue(report, 'numpy', exc)

    result_by_id = {}
    packaged_frame_ids = set()
    release_blocked = release_blocked_video_ids()
    archives = sorted(p for p in RESULTS.glob('*_results.zip') if p.name.startswith(('M','N','S')))
    for archive_path in archives:
        archive_report = {'name': archive_path.name, 'videos': 0, 'crc_ok': None}
        try:
            with zipfile.ZipFile(archive_path) as zf:
                members = zf.namelist()
                if len(members) != len(set(members)): issue(report, archive_path.name, 'duplicate ZIP member')
                bad = zf.testzip()
                archive_report['crc_ok'] = bad is None
                if bad: issue(report, archive_path.name, f'CRC failed: {bad}')
                video_folders = video_directories(set(members))
                for video, folder in video_folders:
                    archive_report['videos'] += 1
                    if video in result_by_id: issue(report, video, 'duplicate result video')
                    result_by_id[video] = archive_path.name
                    video_report = {'source': source_by_id.get(video), 'result': archive_path.name}
                    report['videos'][video] = video_report
                    try:
                        scenes_bytes = zf.read(f'phase1_transnet/{folder}/scenes.json')
                        keyframes_bytes = zf.read(f'phase1_transnet/{folder}/keyframes.json')
                        embedding_bytes = zf.read(f'phase2_embeddings/{folder}/embeddings.npy')
                        scenes = json.loads(scenes_bytes)
                        selected = json.loads(keyframes_bytes)
                        vectors = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
                        frames_selected = [int(r['frame_number']) for r in selected['keyframes']]
                        packaged_frame_ids.update(f'{video}_{frame:06d}' for frame in frames_selected)
                        num_frames = int(scenes['num_frames'])
                        ranges = scenes['scenes']
                        if (not ranges or int(ranges[0]['start_frame']) != 0 or int(ranges[-1]['end_frame']) != num_frames-1 or
                                any(int(s['start_frame']) > int(s['end_frame']) for s in ranges) or
                                any(int(ranges[i]['start_frame']) <= int(ranges[i-1]['end_frame']) for i in range(1,len(ranges)))):
                            issue(report, video, 'invalid scene order/coverage')
                        gaps = [(int(ranges[i-1]['end_frame'])+1, int(ranges[i]['start_frame'])-1)
                                for i in range(1,len(ranges)) if int(ranges[i]['start_frame']) > int(ranges[i-1]['end_frame'])+1]
                        # TransNet's predictions_to_scenes() omits pictures in
                        # transition runs between two scene ranges. Record them
                        # explicitly; they are not missing decoded source frames.
                        try:
                            verify_scene_association(video, selected, ranges)
                        except ValueError as exc:
                            issue(report, video, exc)
                        if (len(frames_selected) != len(set(frames_selected)) or frames_selected != sorted(frames_selected)
                                or any(frame < 0 or frame >= num_frames for frame in frames_selected)):
                            issue(report, video, 'invalid selected frame identities/order')
                        if vectors.shape != (len(frames_selected),1280) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                            issue(report, video, f'invalid vector shape/dtype/finiteness {vectors.shape}/{vectors.dtype}')
                        elif not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
                            issue(report, video, 'unnormalized vectors')
                        if selected.get('version') == 2:
                            marker = json.loads(zf.read(f'phase2_embeddings/{folder}/verified.json'))
                            verify_packaged_generation(video, selected, marker, {
                                'scenes': scenes_bytes, 'keyframes': keyframes_bytes,
                                'embeddings': embedding_bytes})
                            video_report['verified_generation'] = selected['generation']
                        else:
                            source_folder = RESULTS / f'output_{archive_path.name.removesuffix("_results.zip")}' / folder
                            if source_folder.exists():
                                original = np.load(source_folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
                                if original.shape != vectors.shape or not np.array_equal(original,vectors):
                                    issue(report,video,'result ZIP differs from processing output')
                            else: issue(report, video, 'processing output folder missing')
                        if export is not None:
                            missing_export = 0; differing = 0; max_export_error = 0.0
                            for i, frame in enumerate(frames_selected):
                                frame_id = f'{video}_{frame:06d}'
                                pos = export_lookup.get(frame_id)
                                if pos is None: missing_export += 1; continue
                                error = export_vector_error(vectors[i], export[pos])
                                max_export_error = max(max_export_error, error)
                                if (int(frames[pos]) != frame or str(videos[pos]) != video or
                                        error > MAX_EXPORT_VECTOR_ABS_ERROR):
                                    differing += 1
                                expected_ms = selected_timestamp_ms(
                                    video, selected['keyframes'][i], float(scenes['fps']),
                                    selected.get('version', 1))
                                if int(timestamps[pos]) != expected_ms: differing += 1
                            if missing_export or differing:
                                issue(report, video, f'NumPy export missing={missing_export} differing={differing}')
                            video_report['numpy_export'] = {'missing': missing_export,
                                                            'differing': differing,
                                                            'max_vector_abs_error': max_export_error}
                        if video.startswith('N'):
                            try:
                                entry = archive_index[video]
                                index = media._build_index(video,entry)
                                map_path = index_path(video,index)
                                table = np.load(map_path, mmap_mode='r', allow_pickle=False)
                                kept, omitted = monotonic_entries(table)
                                if len(table) != num_frames: issue(report,video,f'PTS map rows {len(table)}/{num_frames}')
                                if any(table[frame,0] != frame for frame in frames_selected): issue(report,video,'PTS map missing selection')
                                if selected.get('version') == 2:
                                    verify_n_source_identity(video, selected, table, index.timescale,
                                                             source_fingerprint(index))
                                video_report['pts_map'] = {'path':str(map_path), 'rows':len(table), 'time_base':[1,index.timescale],
                                                           'non_increasing':omitted[:,0].tolist()}
                                video_report['playback_copy'] = validated_copy(video,index) is not None
                            except Exception as exc: issue(report,video,f'PTS map failed: {exc}')
                        video_report.update({'scenes':len(ranges),'scene_gaps':gaps,
                                             'scene_transition_omitted_frames':sum(hi-lo+1 for lo,hi in gaps),
                                             'frames':num_frames,
                                             'selected':len(frames_selected),'vector_dim':vectors.shape[1] if vectors.ndim==2 else None,
                                             'selection':selected.get('selection'),'source_pts_rows':sum('source_pts' in r for r in selected['keyframes'])})
                    except Exception as exc: issue(report,video,f'result artifact failed: {exc}')
                if archive_path.name.startswith('N'):
                    try:
                        manifest = json.loads(zf.read('manifest.json')) if 'manifest.json' in members else None
                        excluded = verify_result_inventory(
                            archive_path.name, source_by_id,
                            {video for video, _ in video_folders}, manifest, release_blocked)
                        archive_report['documented_exclusions'] = sorted(excluded)
                    except Exception as exc:
                        issue(report, archive_path.name, f'result inventory failed: {exc}')
        except Exception as exc: issue(report,archive_path.name,f'result ZIP failed: {exc}')
        report['result_archives'].append(archive_report)
        print(f'Result {archive_path.name}: {archive_report["videos"]} videos, CRC={archive_report["crc_ok"]}', flush=True)
    if len(archives) != 21: issue(report,'results',f'{len(archives)}/21 result archives')
    missing_results = set(source_by_id) - set(result_by_id)
    extra_results = set(result_by_id) - set(source_by_id)
    if missing_results - release_blocked or extra_results:
        issue(report,'inventory',f'missing results={sorted(missing_results-release_blocked)[:20]}, extra={sorted(extra_results)[:20]}')
    report['counts']['release_blocked_missing_results'] = sorted(missing_results & release_blocked)
    report['counts']['result_archives'] = len(archives)
    report['counts']['result_videos'] = len(result_by_id)
    report['counts']['result_rows'] = sum(v.get('selected',0) for v in report['videos'].values())
    extra_export = unexpected_export_ids(export_lookup, packaged_frame_ids)
    report['counts']['unexpected_numpy_rows'] = len(extra_export)
    if extra_export:
        issue(report, 'numpy', f'{len(extra_export)} rows absent from result ZIPs: '
              f'{sorted(extra_export)[:20]}')
    report['counts']['scene_transition_omitted_frames'] = sum(
        v.get('scene_transition_omitted_frames', 0) for v in report['videos'].values())
    if live:
        audit_live(report, result_by_id, export_lookup, export, timestamps,
                   missing_results)
    report['elapsed_seconds'] = round(time.time()-started,2)
    report['status'] = 'pass' if not report['issues'] else 'exceptions'
    output = output or RESULTS / 'verification/readiness_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2))
    os.replace(temporary,output)
    print(f"Audit {report['status']}: {len(report['issues'])} exceptions; {output}",flush=True)
    return report


def audit_live(report,result_by_id, export_lookup, export, timestamps, absent_videos=()):
    from app.db import milvus_client, postgres_client
    from pymilvus import Collection, utility
    async def postgres():
        pool = await postgres_client.get_pool()
        rows = await pool.fetch("SELECT video_id, title, youtube_id, fps, duration_ms, frame_count FROM videos WHERE video_id LIKE 'M%' OR video_id LIKE 'N%' OR video_id LIKE 'S%'")
        await postgres_client.close_pool()
        return {r['video_id']:dict(r) for r in rows}
    try:
        postgres_rows = asyncio.run(postgres())
        missing = set(result_by_id)-set(postgres_rows)
        extra = set(postgres_rows)-set(result_by_id)
        report['postgres']={'videos':len(postgres_rows),'missing':sorted(missing),'extra':sorted(extra)}
        if missing or extra: issue(report,'postgres',f'missing={len(missing)} extra={len(extra)}')
        for video, data in postgres_rows.items():
            expected = report['videos'].get(video,{}).get('selected')
            if expected is not None and data['frame_count'] != expected:
                issue(report,video,f'PostgreSQL frame_count {data["frame_count"]}/{expected}')
    except Exception as exc: issue(report,'postgres',f'connection/query failed: {exc}')
    if export is None or timestamps is None:
        issue(report, 'milvus', 'live vector comparison requires a complete NumPy export')
        return
    try:
        milvus_client.connect()
        expected_ids = defaultdict(set)
        for frame_id in export_lookup:
            expected_ids[frame_id.rsplit('_',1)[0]].add(frame_id)
        for algorithm in ('hnsw','flat','scann'):
            name = milvus_client.collection_name_for_algorithm(algorithm)
            if not utility.has_collection(name):
                report['indexes'][algorithm]={'status':'unpopulated'}
                continue
            col = Collection(name); col.load()
            index_report={'collection':name,'entities':col.num_entities,'videos_checked':0,'mismatches':0}
            report['indexes'][algorithm]=index_report
            for number, video in enumerate(sorted(result_by_id),1):
                rows = col.query(expr=f'video_id == "{video}"', output_fields=['frame_id','video_id','frame_number','timestamp_ms','vector'],limit=16384)
                expected = report['videos'].get(video,{}).get('selected')
                if expected is None: continue
                if len(rows)!=expected:
                    issue(report,video,f'{algorithm} rows {len(rows)}/{expected}')
                    index_report['mismatches']+=1
                ids=set()
                for row in rows:
                    frame_id=row['frame_id']
                    if frame_id in ids: issue(report,video,f'{algorithm} duplicate {frame_id}')
                    ids.add(frame_id)
                    position=export_lookup.get(frame_id)
                    vector = np.asarray(row['vector'],dtype=np.float32)
                    if position is None:
                        issue(report,video,f'{algorithm} extra frame {frame_id}')
                        continue
                    if (int(row['frame_number'])!=int(frame_id.rsplit('_',1)[1]) or
                            int(row['timestamp_ms'])!=int(timestamps[position]) or
                            vector.shape != (1280,) or not np.isfinite(vector).all() or
                            not np.isclose(np.linalg.norm(vector),1,atol=1e-4) or
                            not np.allclose(vector,export[position],atol=2e-6,rtol=0)):
                        issue(report,video,f'{algorithm} invalid row {frame_id}')
                missing_ids=expected_ids[video]-ids
                if missing_ids: issue(report,video,f'{algorithm} missing {len(missing_ids)} frames')
                index_report['videos_checked']+=1
                if number%50==0: print(f'{algorithm}: {number}/{len(result_by_id)}',flush=True)
            for video in sorted(absent_videos):
                rows = col.query(expr=f'video_id == "{video}"',
                                 output_fields=['frame_id'], limit=16384)
                if rows:
                    issue(report, video, f'{algorithm} retains {len(rows)} rows from excluded result')
                    index_report['mismatches'] += 1
    except Exception as exc: issue(report,'milvus',f'connection/query failed: {exc}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crc',action='store_true')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.crc,args.live,args.output)
    raise SystemExit(0 if result['status']=='pass' else 2)
