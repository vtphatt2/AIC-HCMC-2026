import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';
import { parseSubmissionCsv, serializeSubmissionRow } from './format.ts';

const workspace = process.cwd();
const temporary = await mkdtemp(path.join(os.tmpdir(), 'aic-submission-test-'));
process.chdir(temporary);
const { default: handler } = await import('../../pages/api/submission.ts');
process.chdir(workspace);
const originalFetch = globalThis.fetch;
after(async () => { globalThis.fetch = originalFetch; await rm(temporary, {recursive: true, force: true}); });

function timeline(videoId = 'N001-V001') {
  return {version: 2, video_id: videoId, verified_timing: true, submission_unit: 'milliseconds',
    frame_ids: [0, 1, 4, 5], source_pts: [2001, 7002, 12004, 17006],
    presentation_us: [0, 500100, 1000300, 1500500],
    time_base: {num: 1, den: 10000}, playback_origin_pts: 2001};
}
function upstream(payload = timeline()) {
  globalThis.fetch = async () => ({ok: true, json: async () => payload});
}
async function call(method, session, body) {
  const response = {code: 200, payload: undefined, setHeader() {},
    status(code) { this.code = code; return this; }, json(payload) { this.payload = JSON.parse(JSON.stringify(payload)); return this; }};
  await handler({method, query: {session}, body}, response);
  return response;
}
async function create(session, queryType = 'kis') {
  assert.equal((await call('POST', '', {action: 'create', session, queryType})).code, 200);
}
const metadata = session => path.join(temporary, '.runtime/submissions', `${session}.meta.json`);

test('N modal selection, manual edit, sidecar and QA CSV preserve source units', async () => {
  upstream();
  await create('qa-roundtrip', 'qa');
  let response = await call('POST', 'qa-roundtrip', {action: 'add', videoId: 'N001-V001', frame: 4, sourceFrameInput: true});
  assert.equal(response.code, 200);
  assert.deepEqual(response.payload.rows[0].frames, [1200]);
  assert.deepEqual(response.payload.rows[0].sourceFrames, [4]);
  response = await call('POST', 'qa-roundtrip', {action: 'setAnswer', rowIndex: 0, answer: 'a,"b"\nnext'});
  assert.equal(response.code, 200);
  response = await call('POST', 'qa-roundtrip', {action: 'editFrame', rowIndex: 0, frameIndex: 0, frame: 700});
  assert.equal(response.code, 200);
  assert.deepEqual(response.payload.rows[0].sourceFrames, [1]);
  const saved = JSON.parse(await readFile(metadata('qa-roundtrip'), 'utf8'));
  assert.equal(saved.version, 2);
  assert.equal(saved.rows[0].unit, 'milliseconds');
  const csv = serializeSubmissionRow('qa', saved.rows[0]);
  assert.deepEqual(parseSubmissionCsv('qa', csv)[0].frames, [700]);
  response = await call('POST', 'qa-roundtrip', {action: 'replaceRaw', content: csv});
  assert.equal(response.code, 200);
  assert.deepEqual(response.payload.rows, saved.rows);
});

test('rounded N endpoint positions survive saving and CSV round trips', async () => {
  upstream();
  await create('endpoints');
  for (const frame of [0, 5]) {
    const response = await call('POST', 'endpoints', {action: 'add', videoId: 'N001-V001', frame, sourceFrameInput: true});
    assert.equal(response.code, 200, JSON.stringify(response.payload));
  }
  const saved = JSON.parse(await readFile(metadata('endpoints'), 'utf8'));
  const csv = saved.rows.map(row => serializeSubmissionRow('kis', row)).join('\n');
  const response = await call('POST', 'endpoints', {action: 'replaceRaw', content: csv});
  assert.equal(response.code, 200, JSON.stringify(response.payload));
  assert.deepEqual(response.payload.rows.map(row => row.sourceFrames[0]), [0, 5]);
});

test('two clients adding N candidates retain both updates', async () => {
  await create('concurrent');
  globalThis.fetch = async () => {
    await new Promise(resolve => setTimeout(resolve, 15));
    return {ok: true, json: async () => timeline()};
  };
  const results = await Promise.all([1, 4].map(frame => call('POST', 'concurrent',
    {action: 'add', videoId: 'N001-V001', frame, sourceFrameInput: true})));
  assert.ok(results.every(response => response.code === 200));
  const response = await call('GET', 'concurrent');
  assert.deepEqual(response.payload.rows.map(row => row.sourceFrames[0]).sort((a,b) => a-b), [1, 4]);
});

test('rename reserves its destination while source timing is checked', async () => {
  upstream();
  await create('rename-source');
  await call('POST', 'rename-source', {action: 'add', videoId: 'N001-V001', frame: 4, sourceFrameInput: true});
  globalThis.fetch = async () => {
    await new Promise(resolve => setTimeout(resolve, 15));
    return {ok: true, json: async () => timeline()};
  };
  const [renamed, created] = await Promise.all([
    call('POST', 'rename-source', {action: 'rename', newSession: 'rename-target'}),
    call('POST', '', {action: 'create', session: 'rename-target', queryType: 'kis'}),
  ]);
  assert.equal(renamed.code, 200);
  assert.equal(created.code, 409);
  assert.equal((await call('GET', 'rename-source')).code, 404);
  assert.deepEqual((await call('GET', 'rename-target')).payload.rows[0].sourceFrames, [4]);
});

test('delete waits for a pending save without resurrecting the session', async () => {
  await create('delete-pending');
  globalThis.fetch = async () => {
    await new Promise(resolve => setTimeout(resolve, 15));
    return {ok: true, json: async () => timeline()};
  };
  const results = await Promise.all([
    call('POST', 'delete-pending', {action: 'add', videoId: 'N001-V001', frame: 4, sourceFrameInput: true}),
    call('DELETE', 'delete-pending'),
  ]);
  assert.ok(results.every(response => response.code === 200));
  assert.equal((await call('GET', 'delete-pending')).code, 404);
});

test('legacy sessions migrate only mapped N frames and preserve L rows', async () => {
  upstream();
  await mkdir(path.dirname(metadata('legacy')), {recursive: true});
  await writeFile(metadata('legacy'), JSON.stringify({queryType: 'kis', draftRowIndex: 0, createdAt: 1, updatedAt: 1}));
  await writeFile(metadata('legacy').replace('.meta.json', '.csv'), 'L01_V001,15\nN001-V001,4\nN001-V001,2\n');
  const response = await call('GET', 'legacy');
  assert.equal(response.code, 200);
  assert.deepEqual(response.payload.rows[0].frames, [15]);
  assert.equal(response.payload.rows[0].unit, 'frames');
  assert.deepEqual(response.payload.rows[1].frames, [1200]);
  assert.equal(response.payload.rows[1].timingStatus, 'verified');
  assert.equal(response.payload.rows[2].timingStatus, 'unresolved');
  assert.deepEqual(response.payload.rows[2].frames, [2]);
});

test('failed verification and N TRAKE requests cannot alter a stored session', async () => {
  await create('reject', 'trake');
  const before = await readFile(metadata('reject'), 'utf8');
  upstream();
  assert.equal((await call('POST', 'reject', {action: 'add', videoId: 'N001-V001', frame: 4})).code, 422);
  assert.equal(await readFile(metadata('reject'), 'utf8'), before);
  await create('unavailable');
  const beforeMissing = await readFile(metadata('unavailable'), 'utf8');
  globalThis.fetch = async () => ({ok: false});
  assert.equal((await call('POST', 'unavailable', {action: 'addRow', videoId: 'N001-V001', frames: [1200]})).code, 422);
  assert.equal(await readFile(metadata('unavailable'), 'utf8'), beforeMissing);
});
