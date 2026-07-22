# YouTube Storyboard Thumbnails — Workaround

## What this is

If `data/videos/<video_id>.mp4` exists, prefer `FRAME_IMAGE_SOURCE=local_video`.
It uses ffmpeg locally and returns the requested timestamp without YouTube access.
The storyboard mode below is only a fallback when neither video nor keyframe JPG is local.

`local-backend` (and, if wired up the same way, `remote-server`) can serve
`/static/frames/{video_id}/{frame}.jpg` from a source other than local files
on disk: the **storyboard sprite sheet** YouTube generates for its own
scrubber-hover preview. `app/services/youtube_thumbnail.py` fetches that
sprite via `yt-dlp` and crops out the tile nearest the frame's `timestamp_ms`.

This exists for one situation: **SAMPLE-mode local testing where you don't
have the real dataset keyframes** (e.g. only `metadata/` + `PECore-features/`
were copied, no `keyframes/` images — see `docs/setup.md` Option A/SAMPLE).
Search/ranking still runs on the real PE-Core vectors; this workaround only
fills in *something visual* to look at instead of a blank/placeholder image.

**It is a workaround, not a substitute for real keyframes.** Do not enable it
on `remote-server` for an actual demo/competition run — use the real
extracted keyframes there (`docs/setup.md` Option C).

## Why it's only approximate

YouTube's storyboard sprite has its own fixed tile interval (commonly a few
seconds, coarser on longer videos) — independent of the dataset's actual
frame rate. `get_thumbnail_jpeg()` picks the storyboard tile whose time
range contains `timestamp_ms`, so the image shown can be off by up to that
interval. It is a real frame from the real video, just not necessarily *the*
frame the vector search matched.

## How it works

1. `yt_dlp.YoutubeDL().extract_info(..., download=False)` returns format
   entries with `format_note == "storyboard"`. Each has `width`, `height`,
   `rows`, `columns`, `fps` (tiles per second), and a `fragments` list —
   each fragment URL returns one JPEG sprite sheet containing `rows ×
   columns` tiles.
2. We pick the storyboard level with the highest `fps` (finest granularity),
   compute which fragment + row/col contains `timestamp_ms`, download that
   one sprite sheet (cached on disk, see below), and crop out the tile with
   Pillow.
3. Sprite sheets are cached under `cache/youtube_storyboards/` (sha1 of the
   fragment URL → `.jpg`), so repeated requests for frames in the same time
   range of the same video don't re-download anything. Storyboard *metadata*
   (rows/cols/fragment URLs) is cached in-process per `youtube_id` for the
   life of the server.

## Configuration

`local-client/local-backend/.env`:

```env
# local              — serve real/placeholder files from AIC_SAMPLE_ROOT/keyframes (default)
# youtube_storyboard — this workaround
FRAME_IMAGE_SOURCE=youtube_storyboard

# Optional override (default: <repo-root>/cache/youtube_storyboards)
# YOUTUBE_STORYBOARD_CACHE_DIR=D:\path\to\cache
```

Setting `FRAME_IMAGE_SOURCE` to anything other than `youtube_storyboard`
(or leaving it unset) restores the normal static-file/proxy behavior in
`main.py` — no code changes needed to switch back.

## Dependencies

Not part of the base `requirements.txt` (keeps the default install light —
same reasoning as `PECORE_BACKEND=onnx` not requiring torch). Install
alongside it when you want this workaround:

```bash
pip install -r requirements.txt -r requirements-youtube-thumbnail.txt
```

This adds `yt-dlp` and `pillow`.

## Failure modes

- **`502` from `/static/frames/...`**: storyboard info could not be fetched
  for that `youtube_id` (video private/deleted/region-locked, or YouTube
  changed the storyboard response format). Falls back to an explicit error,
  not a silent blank image — check the backend log for the underlying
  `StoryboardUnavailable` message.
- **`404`**: the frame_id isn't in the loaded SAMPLE dataset, or the video
  has no `youtube_id` in its metadata JSON.
- Storyboard fragment URLs are signed and can expire; if thumbnails that
  worked earlier start failing, restart the backend to force re-resolution
  (metadata is cached in-process, not persisted to disk).

## Reliability note

This relies on an undocumented part of YouTube's player response (the same
data yt-dlp's own preview/thumbnail features use). It can break if YouTube
changes that response shape — `yt-dlp` is actively maintained against such
changes, so keeping it updated (`pip install -U yt-dlp`) is the main
maintenance task if this stops working.
