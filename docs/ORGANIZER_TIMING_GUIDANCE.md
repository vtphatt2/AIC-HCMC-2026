# Organizer timing guidance and readiness decisions

## N001–N100 clarification

Source: organizer statement carried in the prior handoff and reiterated verbatim
by the user on 2026-09-25. This is existing task guidance, not a newly discovered
requirement. No announcement URL is recorded here.

> Các video camera giao thông (N001–N100) được encode ở tốc độ khung hình thay đổi (VFR), nên giá trị frame_idx trong map-keyframes chỉ mang tính tương đối, không phải giá trị chính xác tuyệt đối.
> Đối với dữ liệu này, các đội thi có thể dùng giá trị milisecond (pts_time × 1000) để nộp bài cho task KIS và QA thay cho frame_idx.
> Task TRAKE sẽ không sử dụng dữ liệu này.

Consequences for the system:

- N KIS/QA positions are milliseconds derived from source PTS and time base.
  The current integer export rounds `source_pts × time_base × 1000`.
  The announcement specifies the unit/formula, but does not specify integer
  rounding or establish acceptance of a particular exported file.
- N is unavailable for TRAKE.
- Organizer `frame_idx` values are approximate. A disagreement with nominal
  frame/FPS timing is expected for VFR and is not evidence of source corruption.
- Internal canonical frame IDs still identify our sequentially decoded source
  pictures. They are retained for thumbnails, checksums and vector association;
  they must not be presented as exact organizer submission positions.
- Source PTS and playback-relative time remain separate. A derived playback copy
  may start at zero; its offset must not silently alter the source submission PTS.
- The statement does not specify decoder threads, explain pixel differences at
  the same PTS, or guarantee behavior around non-increasing timestamps. Those
  cases retain their independent source/derivative verification requirements.

The supplied guidance confirms the N policy in the readiness plan. It does not
justify excluding a video merely because its frame index is approximate.

## Finals guide versus preliminary CSV rules

The local organizer document [HD-ChungKet-2026.pdf](../HD-ChungKet-2026.pdf),
pages 5–6, describes DRES submission with original-video milliseconds for KIS,
`QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>` for QA, and frame IDs in TRAKE text.
It does not prescribe a video decoder or establish integer rounding/timestamp
origin details for unusual source streams.

[SUBMISSION_RULES.md](SUBMISSION_RULES.md) records the earlier preliminary-round
CSV/ZIP format. Its L/M/S frame positions remain unchanged. The readiness work
adds the organizer's stated N millisecond exception and versioned local sidecars;
it does not equate a CSV export with an accepted DRES submission.

No live competition submission is part of readiness verification. Acceptance
remains unverified until organizer reference evidence or an acceptance result
is available.
