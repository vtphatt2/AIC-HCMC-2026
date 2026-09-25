"""Immutable metadata generations and verified embedding resume checks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def metadata_generation(stage: Path, video_id: str, payload: dict, scenes: dict):
    """Preserve legacy generation names; a changed map starts a separate one.

    Source bytes may be unchanged while a repaired PTS/checksum map changes
    selected pictures. Never rewrite that old generation's metadata in place.
    """
    payload = dict(payload)
    base_generation = payload['generation']
    for attempt in range(2):
        folder = stage / video_id / payload['generation'][:16]
        if folder.exists() and any(folder.iterdir()):
            try:
                keyframes_path = folder / 'keyframes.json'
                scenes_path = folder / 'scenes.json'
                present = [path for path in (keyframes_path, scenes_path) if path.exists()]
                matches = bool(present) and all(
                    json.loads(path.read_text()) == expected
                    for path, expected in ((keyframes_path, payload), (scenes_path, scenes))
                    if path.exists()
                )
                # An interrupted metadata write can leave one final file or
                # only atomic-write scratch. Never reconstruct metadata around
                # existing vectors or a verification marker.
                if matches and len(present) < 2:
                    matches = not any((folder / name).exists() for name in
                                      ('embeddings.npy', 'verified.json'))
                if not present:
                    matches = all(path.name in ('keyframes.json.partial', 'scenes.json.partial')
                                  for path in folder.iterdir())
            except (OSError, ValueError):
                matches = False
            if matches:
                if not keyframes_path.exists():
                    atomic_json(keyframes_path, payload)
                if not scenes_path.exists():
                    atomic_json(scenes_path, scenes)
                return folder, payload
            if attempt:
                raise ValueError(f'{video_id}: conflicting staged generation; preserve it for review')
            identity = {'base_generation': base_generation, 'keyframes': payload, 'scenes': scenes}
            payload['generation'] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()).hexdigest()
            continue
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'keyframes.json', payload)
        atomic_json(folder / 'scenes.json', scenes)
        return folder, payload
    raise AssertionError('Unreachable generation selection')


def artifact_digests(folder: Path) -> dict:
    return {f'{name}_sha256': file_digest(folder / filename) for name, filename in (
        ('keyframes', 'keyframes.json'), ('scenes', 'scenes.json'), ('embeddings', 'embeddings.npy'))}


def reusable_vectors(folder: Path, payload: dict, provenance: dict, *, adopt_legacy=True) -> bool:
    """Accept a complete matching marker, with hashes for subsequent resumes.

    Old markers already attest exhaustive source checks. They are adopted only
    after metadata, generation, encoder settings and all vector rows validate.
    Hashes detect subsequent alteration; they do not replace picture evidence.
    """
    try:
        marker = json.loads((folder / 'verified.json').read_text())
        if (marker.get('version') != 1 or marker.get('generation') != payload['generation'] or
                marker.get('rows') != payload['num_keyframes'] or
                marker.get('source_checksums') != 'exhaustive' or
                marker.get('source_time_base_verified') is not True or
                json.loads((folder / 'keyframes.json').read_text()) != payload or
                any(marker.get(key) != value for key, value in provenance.items())):
            return False
        vectors = np.load(folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        if (vectors.shape != (payload['num_keyframes'], 1280) or vectors.dtype != np.float32 or
                not np.isfinite(vectors).all() or
                not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            return False
        digests = artifact_digests(folder)
        if any(key in marker and marker[key] != value for key, value in digests.items()):
            return False
        if not all(key in marker for key in digests):
            if not adopt_legacy:
                return False
            atomic_json(folder / 'verified.json', {**marker, **digests})
        return True
    except (OSError, ValueError, KeyError):
        return False


def adopt_source_verified_vectors(stage: Path, video_id: str, folder: Path,
                                  payload: dict, scenes: dict,
                                  provenance: dict) -> bool:
    """Reuse vectors whose every selected picture matched the same source map.

    A decoder policy change can alter an unselected frame while leaving every
    PE-Core input picture identical. The previous exhaustive checksum marker,
    unchanged selected rows, metadata and encoder settings establish reuse.
    Keep the old generation intact and write a complete marker only after the
    copied vectors pass the normal verification gate.
    """
    if (folder / 'verified.json').exists():
        return False
    # Duration describes the same video track but does not enter the encoder.
    comparable = lambda value: {key: item for key, item in value.items()
                                if key not in ('generation', 'decode_provenance',
                                               'source_duration_ms')}
    for candidate in sorted((stage / video_id).iterdir()):
        if candidate == folder or not candidate.is_dir():
            continue
        try:
            old_payload = json.loads((candidate / 'keyframes.json').read_text())
            if (comparable(old_payload) != comparable(payload) or
                    json.loads((candidate / 'scenes.json').read_text()) != scenes):
                continue
            old_provenance = {**provenance,
                              'decode_provenance': old_payload.get('decode_provenance')}
            if not reusable_vectors(candidate, old_payload, old_provenance):
                continue
            temporary = folder / 'embeddings.adopting.npy'
            shutil.copyfile(candidate / 'embeddings.npy', temporary)
            os.replace(temporary, folder / 'embeddings.npy')
            marker = {'version': 1, 'generation': payload['generation'],
                      'rows': payload['num_keyframes'],
                      'source_checksums': 'exhaustive',
                      'source_time_base_verified': True,
                      **provenance, **artifact_digests(folder), 'published': False,
                      'adopted_from_generation': old_payload['generation']}
            atomic_json(folder / 'verified.json', marker)
            return reusable_vectors(folder, payload, provenance, adopt_legacy=False)
        except (OSError, ValueError, KeyError):
            continue
    return False
