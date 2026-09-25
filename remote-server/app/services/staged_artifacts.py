"""Immutable metadata generations and verified embedding resume checks."""
from __future__ import annotations

import hashlib
import json
import os
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
                matches = (json.loads((folder / 'keyframes.json').read_text()) == payload and
                           json.loads((folder / 'scenes.json').read_text()) == scenes)
            except (OSError, ValueError):
                matches = False
            if matches:
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
