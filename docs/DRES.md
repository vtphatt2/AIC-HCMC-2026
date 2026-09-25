# Final-round DRES submission

Source: `HD-ChungKet-2026.pdf` on branch `zip-decode/fix`. This is separate from
the earlier CSV/ZIP upload workflow in [SUBMISSION.md](SUBMISSION.md). DRES
submission is triggered only by an operator clicking **Submit to DRES** on a
candidate in `/submissions`; Search and Verify Agents never submit.

## Connect DRES

1. Sign in at the address BTC gives (the PDF uses `https://eventretrieval.one/login`).
   Copy your team's `sessionId` from `/user`. Do not put it in Git, the tracked
   `local-client/frontend/.env.local`, or any `NEXT_PUBLIC_*` variable.
2. On the machine running the Next.js frontend, make the ignored file
   `local-client/frontend/.env.dres.local` with shell assignments:

   ```bash
   DRES_SESSION_ID='your-team-session-id'
   DRES_SUBMIT_PIN='a-private-pin-shared-with-the-two-operators'
   # Optional if BTC announces another DRES host:
   # DRES_BASE_URL='https://btc-dres-host'
   ```

3. From the repository root, load it and start the system:

   ```bash
   set -a
   source local-client/frontend/.env.dres.local
   set +a
   scripts/start-local.sh
   ```

   The frontend's server process needs these values. `DRES_SUBMIT_PIN` is
   entered by each operator in the dashboard before sending; it stays in the
   browser tab's memory. The session ID is sent only from the server to DRES.
   If the VORTA backend is on another host, also set `VORTA_BACKEND_URL` so
   KIS/Q&A frame numbers can be converted with that backend's video FPS.

## Submit one candidate

1. Create or open a dashboard session for the current query. Select **KIS**
   for Textual/Video KIS, **QA** for Q&A, or **TRAKE** for event chains.
2. Review the video/frame(s). For Q&A, fill in the answer on the candidate row.
   For TRAKE, keep one frame number per event in the required event order.
3. In **DRES live submission**, select an ACTIVE evaluation if more than one
   appears, then check that the shown task ID and `RUNNING` status match the
   current official query. Press **Refresh DRES** when the query changes.
4. Enter the submit PIN, click **Submit to DRES** on the chosen row, review the
   confirmation, and confirm. The row status displays DRES's verdict/response.

Payloads follow the PDF:

| Query type | DRES answer |
| --- | --- |
| KIS | `mediaItemName=VIDEO_ID`, `start=end=round(frame/fps*1000)` ms for L/M/S; verified source PTS ms for N |
| Q&A | `QA-ANSWER-VIDEO_ID-TIME_MS` |
| TRAKE | `TR-VIDEO_ID-FRAME_NUMBER1,FRAME_NUMBER2,...` |

The selected task ID is included in the DRES answer set so a task change cannot
silently redirect an answer. The API checks the task is still RUNNING and that
the row has not changed since the operator saw it. It blocks a duplicate exact
answer in the current Next.js process, including two near-simultaneous clicks.
For `N001–N100`, KIS/Q&A submission requires a row marked with verified source
PTS milliseconds by the merged timeline flow. Unresolved N rows are rejected;
N remains unavailable for TRAKE. See [organizer timing guidance](ORGANIZER_TIMING_GUIDANCE.md).
After a timeout, the result may have reached DRES; check the DRES website before
trying again. Restarting Next.js clears this local duplicate memory.

## Checks and fallback

From `local-client/frontend`, after `npm run build`:

```bash
npm test
node scripts/test-dres-smoke.cjs
```

The smoke uses fake local DRES and VORTA servers; it cannot submit to BTC.
Without `DRES_SESSION_ID` or `DRES_SUBMIT_PIN`, live submission stays disabled,
while manual search and CSV/ZIP export keep working. The official DRES endpoint
and credentials require a rehearsal with BTC before relying on live submission.
