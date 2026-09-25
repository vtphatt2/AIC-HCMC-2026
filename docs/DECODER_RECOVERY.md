# Verified source decoder recovery

Two different failure patterns have been reproduced during the N audit:

| Pattern | Evidence | Shared handling |
|---|---|---|
| Decoder-dependent pixels | The same source bytes, frame IDs and PTS produce different pixel checksums with different decoder thread counts. | Prepare a separate candidate map, independently replay every source frame, and record one verified decoder profile used by all derivatives. |
| Repeated or backward PTS | Decoded source IDs remain valid, but a timestamp does not advance beyond the last retained timestamp. | Keep original IDs, record omitted entries, and omit those entries from selection and playback. Resolve image selection by source frame ID where PTS is ambiguous. |

These findings do not establish organizer corruption. An identical official
redownload rules out a differing local download; it does not prove that any
decoder output is the intended picture. Complete replay proves reproducibility
under the tested configuration. Source/derived picture checks and playback
validation remain required before release.

## Recovery workflow

1. Record the mismatch and release-block the affected video. Preserve the source,
   previous maps and derivatives. Expand verification to its archive/processing
   group; do not infer that other videos are faulty. Archive sweeps may use
   `audit_decoder_agreement.py --block-mismatches`; it atomically blocks only a
   completed replay disagreement and applies the same action when resuming a
   checkpointed mismatch.
2. From `remote-server`, run the offline recovery with explicitly named videos:

   ```bash
   .venv/bin/python scripts/recover_decoder_profiles.py \
     --videos N039-V001 --threads 1 \
     --report ../challenge_resources/data/zip_embeddings/verification/decoder_recovery.json
   ```

   The tool requires an existing release block. It builds a separately
   fingerprinted map, then compares a second complete decode against every
   source ID, PTS and checksum, including omitted presentation entries. Exact
   coverage, successful decoding and the source time base must agree.
3. Only a successful replay registers a source-bound profile in
   `challenge_resources/data/source_decoder_profiles.json`, configurable with
   `SOURCE_DECODER_PROFILES_PATH`. It records settings, verification evidence,
   source identity and the map digest. Registration preserves release blocks.
   Concurrent registrations are serialized and the manifest is replaced
   atomically. Changed sources or maps require renewed verification.
4. Stage the selected vectors, full-resolution source images, cards and playback
   using the common profile. Validate those artifacts and exports before
   publishing a consistent generation and removing the release block.

Source maps, exact image extraction, PE-Core input decoding, and offline playback
all resolve decoder settings through `app/services/readiness_policy.py`.
Every derivative uses the decoder thread setting that created its authoritative
source map. The normal map setting is currently four threads; the two-thread
playback policy applies to the H.264 encoder, not to source decoding. A verified
profile changes derivative generation identity while preserving canonical
video/frame IDs. Existing verified one-thread vectors and exact images remain
reusable. Playback copies are revalidated under the strengthened source-map and
decoder fingerprint. Normal generations made with the earlier two-thread
derivative decoder are invalidated and rebuilt because their provenance no
longer matches.

The complete one-thread replay runs across each N archive, rather than only the
video that exposed a mismatch. A pixel disagreement at identical frame ID and PTS
creates a source-bound recovery candidate. Videos whose one-thread replay matches
the existing four-thread map use the shared four-thread derivative policy and do
not need a per-video exception.

Changing the decoder implementation or FFmpeg build still requires a new replay;
the current registry records thread settings and source/map identity, not the
complete executable/library environment. Do not carry verification across such
an upgrade without rerunning it.

## Verification scope

`tests/test_decoder_profiles.py` checks arbitrary video IDs, shared settings,
incomplete and mismatched evidence, invalid time bases, changed maps and changed
sources. The timing/image/playback tests cover omitted identities, ambiguous PTS
and checksum rejection. Real-source evidence is kept under
`challenge_resources/data/zip_embeddings/verification/`.

The audit is not a general assertion that all N videos require one decoder
thread, nor that every source error can be repaired this way. If a source remains
unusable after decoder/process checks, the affected archive audit, and the final
official redownload comparison, exclude that video as specified in the readiness
plan and retain the evidence.
