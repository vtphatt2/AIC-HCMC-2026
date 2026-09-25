import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const plain = value => JSON.parse(JSON.stringify(value));
const response = score => ({ fps: 25, before: [], middle: [], after: [], scores: { F1: score } });
const args = () => ["V1", 100, 200, 20, [{ frame_id: "F1", frame_number: 5 }], ["teacher", "map"], [1, 2], 0.98];

function fixture() {
  let now = 0;
  const calls = [];
  let responder = async () => ({ ok: true, json: async () => response(0.8) });
  const modules = {};
  function load(name) {
    const exports = {};
    const source = readFileSync(new URL(`./${name}.ts`, import.meta.url), "utf8");
    const { outputText } = ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    });
    vm.runInNewContext(outputText, {
      exports, process: { env: { NEXT_PUBLIC_API_URL: "/" } }, URLSearchParams,
      Date: { now: () => now },
      require: path => modules[path] ?? {},
      fetch: async (url, init) => { calls.push({ url, init }); return responder(url, init); },
    });
    modules[`@/lib/${name}`] = exports;
    return exports;
  }
  const cache = load("recentContext");
  return { ...cache, ...load("api"), calls, tick: ms => { now += ms; }, respond: fn => { responder = fn; } };
}

test("returning context displays a cached preview but always fetches fresh scores", async () => {
  const f = fixture();
  await f.fetchScoredContext(...args());
  let release;
  f.respond(() => new Promise(resolve => { release = resolve; }));
  const seen = [];
  const refreshed = f.fetchScoredContext(...args(), undefined, value => seen.push(plain(value)));
  assert.deepEqual(seen, [response(0.8)]);
  assert.equal(f.calls.length, 2);
  release({ ok: true, json: async () => response(0.9) });
  assert.deepEqual(plain(await refreshed), response(0.9));
  f.respond(async () => ({ ok: true, json: async () => response(1) }));
  await f.fetchScoredContext(...args(), undefined, value => seen.push(plain(value)));
  assert.deepEqual(seen[1], response(0.9));
});

test("N frame URLs carry decoded frame identity and the new cache revision", () => {
  const f = fixture();
  assert.equal(f.zipFrameImageUrl("N001-V001", 1234, 42),
    "/api/zip-frame/N001-V001/1234?frame_number=42&v=7");
  assert.equal(f.zipFrameImageUrl("M01_V001", 1234, 42),
    "/api/zip-frame/M01_V001/1234");
  assert.throws(() => f.zipFrameImageUrl("N001-V001", 1234), /source frame identity/);
});

test("cache cannot cross videos, ranges, frame identities, query order, weights or thresholds", async () => {
  const changes = [
    a => { a[0] = "V2"; }, a => { a[1] = 101; }, a => { a[2] = 201; },
    a => { a[3] = 21; }, a => { a[4][0].frame_id = "F2"; },
    a => { a[4][0].frame_number = 6; }, a => { a[5][0] = "different query"; },
    a => { a[5].reverse(); }, a => { a[6] = [2, 1]; }, a => { a[7] = 0.95; },
  ];
  for (const change of changes) {
    const f = fixture();
    await f.fetchScoredContext(...args());
    const changed = args(); change(changed);
    let hits = 0;
    await f.fetchScoredContext(...changed, undefined, () => hits++);
    assert.equal(hits, 0);
  }
});

test("aborted requests cannot publish a preview or replace a valid cached response", async () => {
  const f = fixture();
  await f.fetchScoredContext(...args());
  const controller = new AbortController();
  controller.abort();
  f.respond(async () => ({ ok: true, json: async () => response(0.1) }));
  let hits = 0;
  await f.fetchScoredContext(...args(), controller.signal, () => hits++);
  assert.equal(hits, 0);
  let preview;
  await f.fetchScoredContext(...args(), undefined, value => { preview = value; });
  assert.deepEqual(plain(preview), response(0.8));

  let release;
  const pendingController = new AbortController();
  f.respond(() => new Promise(resolve => { release = resolve; }));
  const pending = f.fetchScoredContext(...args(), pendingController.signal);
  pendingController.abort();
  release({ ok: true, json: async () => response(0.2) });
  await pending;
  f.respond(async () => ({ ok: true, json: async () => response(0.3) }));
  await f.fetchScoredContext(...args(), undefined, value => { preview = value; });
  assert.deepEqual(plain(preview), response(0.1));
});

test("failed or partially scored responses do not survive in the preview cache", async () => {
  for (const failure of ["http", "network", "scores"]) {
    const f = fixture();
    await f.fetchScoredContext(...args());
    f.respond(async () => {
      if (failure === "network") throw new Error("offline");
      if (failure === "http") return { ok: false, status: 500 };
      return { ok: true, json: async () => ({ ...response(0), scores: null }) };
    });
    const pending = f.fetchScoredContext(...args());
    if (failure === "scores") await pending;
    else await assert.rejects(pending);
    f.respond(async () => ({ ok: true, json: async () => response(0.9) }));
    let hits = 0;
    await f.fetchScoredContext(...args(), undefined, () => hits++);
    assert.equal(hits, 0);
  }
});

test("older backend fallback keeps its existing neighbor and score requests", async () => {
  const f = fixture();
  f.respond(async url => {
    if (url.endsWith("context-scores")) return { ok: false, status: 404 };
    if (url.includes("context-frames")) return { ok: true, json: async () => ({ fps: 25, before: [{ frame_id: "F0", frame_number: 1 }], middle: [], after: [] }) };
    return { ok: true, json: async () => ({ scores: { F0: 0.5, F1: 0.8 } }) };
  });
  const result = await f.fetchScoredContext(...args());
  assert.deepEqual(plain(result.scores), { F0: 0.5, F1: 0.8 });
  assert.equal(f.calls.length, 3);
  assert.deepEqual(JSON.parse(f.calls[2].init.body).frame_ids, ["F0", "F1"]);
});

test("preview cache expires without extending age on reads, bounds storage and isolates consumers", () => {
  const f = fixture();
  const cache = new f.RecentContextCache(2, 120, 100);
  cache.set("a", { score: 1 });
  cache.set("b", { score: 2 });
  const value = cache.get("a"); value.score = 9;
  assert.equal(cache.get("a").score, 1);
  cache.set("c", { score: 3 });
  assert.equal(cache.get("b"), undefined);
  cache.set("oversized", "x".repeat(100));
  assert.equal(cache.get("oversized"), undefined);
  f.tick(90); assert.ok(cache.get("a"));
  f.tick(11); assert.equal(cache.get("a"), undefined);
  const small = new f.RecentContextCache(10, 60, 100);
  small.set("a", "1234567890"); small.set("b", "1234567890"); small.set("c", "1234567890");
  assert.equal(small.get("a"), undefined);
  assert.equal(small.get("c"), "1234567890");
});

test("transcript previews are video-specific and fresh text replaces cached text", async () => {
  const f = fixture();
  const transcript = text => ({ video_id: "V1", segments: [{ start_ms: 0, end_ms: 100, text }] });
  f.respond(async () => ({ ok: true, json: async () => transcript("previous") }));
  await f.fetchTranscript("V1");
  let release;
  f.respond(() => new Promise(resolve => { release = resolve; }));
  let preview;
  const pending = f.fetchTranscript("V1", value => { preview = plain(value); });
  assert.deepEqual(preview, transcript("previous"));
  assert.equal(f.calls.length, 2);
  release({ ok: true, json: async () => transcript("updated") });
  assert.deepEqual(plain(await pending), transcript("updated"));
  f.respond(async () => ({ ok: true, json: async () => transcript("updated") }));
  let hits = 0;
  await f.fetchTranscript("V2", () => hits++);
  assert.equal(hits, 0);
  // An unexpected video ID in a response must not become V2's preview.
  await f.fetchTranscript("V2", () => hits++);
  assert.equal(hits, 0);
  await f.fetchTranscript("V1", value => { preview = plain(value); });
  assert.deepEqual(preview, transcript("updated"));
});

test("transcript errors and expiry discard old previews without skipping the network", async () => {
  const f = fixture();
  const success = async () => ({ ok: true, json: async () => ({ video_id: "V1", segments: [] }) });
  f.respond(success);
  await f.fetchTranscript("V1");
  f.respond(async () => ({ ok: false, json: async () => ({ detail: "unavailable" }) }));
  await assert.rejects(f.fetchTranscript("V1"), /unavailable/);
  f.respond(success);
  let hits = 0;
  await f.fetchTranscript("V1", () => hits++);
  assert.equal(hits, 0);
  f.tick(120001);
  await f.fetchTranscript("V1", () => hits++);
  assert.equal(hits, 0);
  assert.equal(f.calls.length, 4);
});
