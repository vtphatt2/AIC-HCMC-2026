# Transcript Extraction & Processing (AIC2026) — Draft

**Pipeline.** Metadata `video_link` → YouTube auto-captions (`youtube_transcript_api`, prefer `vi`/`en`) → `*_Transcript.txt` in `[HH:MM:SS] text` format. Videos with no captions are logged to `failed_transcripts.txt` and re-processed with Whisper `large-v3` (openai-whisper locally, `faster-whisper` on Kaggle T4x2) → `.jsonl` `{start_time_ms, end_time_ms, text}`.

**Processing.** `convert_transcripts_to_jsonl.py` normalizes TXT→JSONL (`end_ms` = next start, last +5000 ms). `transcript_jsonl_reader.py` reads JSONL with TXT fallback. `preprocess/transcript/sentences.py` merges fragments and splits into non-overlapping sentences by terminal punctuation; when ASR has no punctuation it falls back to source-fragment boundaries (≤15 s / 320 chars) and interpolates timestamps. `build_keyframe_index.py` anchors each sentence to the nearest keyframe inside its interval (real PTS from `manifest.json`, else `frame_id/fps`). `add_transcript_banner.py` renders the sentence onto the keyframe for visual encoding. `index_transcripts.py` chunks, topic-labels, and embeds (`multilingual-e5-small`) into Milvus + PostgreSQL.

**Problems with missing auto-transcripts.**

- YouTube blocks/missing captions (`TranscriptsDisabled`, `NoTranscriptFound`, `RequestBlocked`): **60/835 videos** (28 L24, 6 L26, 24 L28, 2 L30).
- 28 L24 lion-dance videos have **no speech** → Whisper would hallucinate, so skipped and marked "no transcript".
- First Whisper pass partially failed → **retried 18 then 32 videos** (`transcript_retry/`).
- ASR cuts phrases at silences → mitigated with a 2-chunk sliding window.
- No punctuation/casing → fallback splitting, interpolated timestamps.
- Timestamp drift → prefer rendered-manifest PTS over `fps`.
- ~40% silent/B-roll videos penalized by fixed fusion → dynamic weights shift to OCR when transcript is empty.
- Frames outside any sentence get a blank banner / `anchor=None` (never a wrong frame).

**Result.** 835 transcript files (822 `.jsonl`); 60 videos lacked captions, 32 recovered via Whisper, 28 skipped as speechless. No video is dropped — each is either ASR-recovered or explicitly marked empty for OCR-weighted fusion.
