"""Versioned defaults for independent source, selection and playback derivatives."""
from dataclasses import asdict, dataclass
import math
import os
from .decoder_profiles import decoder_profile


@dataclass(frozen=True)
class SelectionPolicy:
    version: int = 1
    interval_seconds: float = 2.0
    retain_existing: bool = True

    def __post_init__(self):
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError('Sampling interval must be finite and positive')


@dataclass(frozen=True)
class PlaybackPolicy:
    version: int = 1
    height: int = 720
    codec: str = 'libx264'
    pixel_format: str = 'yuv420p'
    preset: str = 'veryfast'
    crf: int = 23
    threads: int = 2

    def __post_init__(self):
        if self.height < 2 or self.height % 2 or not 0 <= self.crf <= 51 or self.threads < 1:
            raise ValueError('Invalid playback settings')


def selection_policy():
    return SelectionPolicy(interval_seconds=float(os.getenv('N_SAMPLE_INTERVAL_SECONDS', '2')))


def playback_policy():
    return PlaybackPolicy(height=int(os.getenv('PLAYBACK_HEIGHT', '720')),
                          preset=os.getenv('PLAYBACK_PRESET', 'veryfast'),
                          crf=int(os.getenv('PLAYBACK_CRF', '23')),
                          threads=int(os.getenv('PLAYBACK_THREADS', '2')))


def submission_unit(video_id):
    return 'milliseconds' if video_id.startswith('N') else 'frames'


def source_map_decoder_threads(video_id, index):
    profile = decoder_profile(video_id, index)
    return profile['threads'] if profile else 4


def verified_embed_decoder_threads(video_id, index):
    profile = decoder_profile(video_id, index)
    return profile['threads'] if profile else 2


def playback_decoder_threads(video_id, policy, index):
    profile = decoder_profile(video_id, index)
    return profile['threads'] if profile else policy.threads


def exceptional_decode_provenance(video_id, index):
    """Only exceptional decoder settings change existing generation signatures."""
    profile = decoder_profile(video_id, index)
    if profile is None:
        return None
    return {'source_map_threads': profile['threads'], 'verified_embed_threads': profile['threads']}


# Browser audit before conversion: these original N010 sources played correctly.
BROWSER_COMPATIBLE_N_SOURCES = frozenset({'N010-V001', 'N010-V002', 'N010-V003'})
