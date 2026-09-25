import assert from 'node:assert/strict';
import test from 'node:test';
import { parsePosition, parseSubmissionCsv, serializeSubmissionRow, validateSubmissionRow } from './format.ts';
import { resolveNRow } from './timing.ts';
import { parseVersionedTimeline } from '../playback.ts';
import { fillKisRows } from './fillFrames.ts';

test('strict numeric parsing', () => {
  for (const value of ['1x', '1.5', '1e2', '-1', '', '9007199254740992']) assert.throws(() => parsePosition(value));
  assert.equal(parsePosition(' 0015 '), 15);
});
test('QA CSV preserves commas, quotes and embedded newlines with exactly three fields', () => {
  const row = {videoId: 'L01_V001', frames: [15], answer: 'a,"b"\nnext'};
  const csv = serializeSubmissionRow('qa', row);
  assert.equal(csv, 'L01_V001,15,"a,""b""\nnext"');
  assert.deepEqual(parseSubmissionCsv('qa', csv), [{...row, unit: 'frames'}]);
  assert.throws(() => parseSubmissionCsv('qa', 'L01_V001,15,a,b'));
});
test('N verified migration, TRAKE rejection and bounded neighbor pictures', () => {
  const timeline = parseVersionedTimeline({version: 2, verified_timing: true, submission_unit: 'milliseconds',
    frame_ids: [0, 1, 4, 5], source_pts: [200, 700, 1200, 1700], presentation_us: [0, 500000, 1000000, 1500000],
    time_base: {num: 1, den: 1000}, playback_origin_pts: 200});
  const legacy = {videoId: 'N031-V001', frames: [4], unit: 'frames'};
  assert.throws(() => validateSubmissionRow('kis', legacy, true));
  const row = resolveNRow(legacy, timeline);
  assert.deepEqual(row.frames, [1200]); assert.deepEqual(row.sourceFrames, [4]);
  validateSubmissionRow('kis', row, true);
  assert.throws(() => validateSubmissionRow('trake', row));
  const filled = fillKisRows([row], 100, 15, {'N031-V001': timeline});
  assert.equal(filled.length, 4);
  assert.equal(new Set(filled.map(r => r.frames[0])).size, 4);
  assert.deepEqual(filled.map(r => r.frames[0]), [1200, 700, 1700, 200]);
});
