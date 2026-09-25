#!/usr/bin/env node
// Local-only integration smoke: fake DRES + fake VORTA, never the BTC server.
const assert = require("node:assert/strict");
const http = require("node:http");
const { spawn } = require("node:child_process");
const { mkdirSync, writeFileSync } = require("node:fs");
const path = require("node:path");

function serve(handler) {
  const server = http.createServer(handler);
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}
function port(server) { return server.address().port; }
function close(server) { return new Promise((resolve) => server.close(resolve)); }
function json(res, status, data) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}
async function request(base, path, method = "GET", body, pin) {
  const response = await fetch(`${base}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...(pin ? { "x-dres-submit-pin": pin } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  return { status: response.status, data: await response.json() };
}

(async () => {
  let taskId = "task-a";
  let submissions = [];
  let videoAvailable = true;
  const dres = await serve(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    assert.equal(url.searchParams.get("session"), "fake-session");
    if (url.pathname === "/api/v2/client/evaluation/list") return json(res, 200, [{ id: "eval-a", name: "Mock final", status: "ACTIVE" }]);
    if (url.pathname === "/api/v2/evaluation/eval-a/state") return json(res, 200, {
      evaluationStatus: "ACTIVE", taskStatus: "RUNNING", taskId,
    });
    if (url.pathname === "/api/v2/submit/eval-a") {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      submissions.push(JSON.parse(raw));
      await new Promise((resolve) => setTimeout(resolve, 50));
      return json(res, 200, { status: true, submission: "INDETERMINATE", description: "Mock accepted" });
    }
    return json(res, 404, { status: false, description: "Unknown mock route" });
  });
  const backend = await serve((req, res) => {
    if (!videoAvailable) return json(res, 503, { error: "Video unavailable" });
    return json(res, 200, { video_id: "L01_V001", fps: 25 });
  });
  const probe = await serve((_req, res) => res.end());
  const nextPort = port(probe);
  await close(probe);
  const next = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "-H", "127.0.0.1", "-p", String(nextPort)], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      DRES_BASE_URL: `http://127.0.0.1:${port(dres)}`,
      DRES_SESSION_ID: "fake-session",
      DRES_SUBMIT_PIN: "fake-pin",
      VORTA_BACKEND_URL: `http://127.0.0.1:${port(backend)}`,
    },
    stdio: "ignore",
  });
  const base = `http://127.0.0.1:${nextPort}`;
  const session = `dres-smoke-${process.pid}`;
  const nSession = `${session}-n`;
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) {
      try { ready = (await request(base, "/api/dres-submit")).status === 200; } catch {}
      if (ready) break;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    assert.equal(ready, true, "Next server did not start");
    assert.equal((await request(base, "/api/submission", "POST", { action: "create", session, queryType: "kis" })).status, 200);
    const saved = await request(base, `/api/submission?session=${session}`, "POST", {
      action: "addRow", videoId: "L01_V001", frames: [42],
    });
    assert.equal(saved.status, 200);
    const row = saved.data.rows[0];
    const body = { evaluationId: "eval-a", taskId, session, rowIndex: 0, expectedRow: JSON.stringify(row) };
    const info = await request(base, "/api/dres-submit");
    assert.equal(info.data.state.taskId, "task-a");
    assert.equal((await request(base, "/api/dres-submit", "POST", body)).status, 403);
    assert.equal((await request(base, "/api/dres-submit", "POST", { ...body, expectedRow: "{}" }, "fake-pin")).status, 409);
    assert.equal((await request(base, "/api/dres-submit", "POST", { ...body, taskId: "old-task" }, "fake-pin")).status, 409);
    videoAvailable = false;
    assert.equal((await request(base, "/api/dres-submit", "POST", body, "fake-pin")).status, 502);
    assert.equal(submissions.length, 0);
    videoAvailable = true;
    const accepted = await request(base, "/api/dres-submit", "POST", body, "fake-pin");
    assert.equal(accepted.status, 200);
    assert.deepEqual(submissions[0], { answerSets: [{ taskId: "task-a", answers: [{ mediaItemName: "L01_V001", start: 1680, end: 1680 }] }] });
    assert.equal((await request(base, "/api/dres-submit", "POST", body, "fake-pin")).status, 409);
    assert.equal(submissions.length, 1);
    const second = await request(base, `/api/submission?session=${session}`, "POST", {
      action: "addRow", videoId: "L01_V001", frames: [43],
    });
    const concurrentBody = { ...body, rowIndex: 1, expectedRow: JSON.stringify(second.data.rows[1]) };
    const parallel = await Promise.all([
      request(base, "/api/dres-submit", "POST", concurrentBody, "fake-pin"),
      request(base, "/api/dres-submit", "POST", concurrentBody, "fake-pin"),
    ]);
    assert.deepEqual(parallel.map((result) => result.status).sort(), [200, 409]);
    assert.equal(submissions.length, 2);
    const nRow = { videoId: "N001-V001", unit: "milliseconds", frames: [1743], sourceFrames: [42], timingStatus: "verified" };
    const submissionDir = path.join(process.cwd(), ".runtime", "submissions");
    mkdirSync(submissionDir, { recursive: true });
    writeFileSync(path.join(submissionDir, `${nSession}.csv`), "N001-V001,1743\r\n");
    writeFileSync(path.join(submissionDir, `${nSession}.meta.json`), JSON.stringify({
      version: 2, rows: [nRow], queryType: "kis", draftRowIndex: 0, createdAt: Date.now(), updatedAt: Date.now(),
    }));
    videoAvailable = false;
    writeFileSync(path.join(submissionDir, `${nSession}.csv`), "N001-V001,1744\r\n");
    assert.equal((await request(base, "/api/dres-submit", "POST", {
      evaluationId: "eval-a", taskId, session: nSession, rowIndex: 0, expectedRow: JSON.stringify(nRow),
    }, "fake-pin")).status, 404);
    writeFileSync(path.join(submissionDir, `${nSession}.csv`), "N001-V001,1743\r\n");
    const nAccepted = await request(base, "/api/dres-submit", "POST", {
      evaluationId: "eval-a", taskId, session: nSession, rowIndex: 0, expectedRow: JSON.stringify(nRow),
    }, "fake-pin");
    assert.equal(nAccepted.status, 200, JSON.stringify(nAccepted.data));
    assert.deepEqual(submissions[2], {
      answerSets: [{ taskId: "task-a", answers: [{ mediaItemName: "N001-V001", start: 1743, end: 1743 }] }],
    });
    taskId = "task-b";
    assert.equal((await request(base, "/api/dres-submit", "POST", body, "fake-pin")).status, 409);
    console.log("PASS: DRES status, PIN, stale candidate/task, missing media, CFR and N-PTS payloads, concurrent duplicate guard");
  } finally {
    await request(base, `/api/submission?session=${session}`, "DELETE").catch(() => {});
    await request(base, `/api/submission?session=${nSession}`, "DELETE").catch(() => {});
    next.kill("SIGTERM");
    await close(dres);
    await close(backend);
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
