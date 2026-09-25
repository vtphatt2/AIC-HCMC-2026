import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

function fixture(mode = "server", connection) {
  const images = [];
  class Image {
    constructor() { images.push(this); }
  }
  const exports = {};
  const source = readFileSync(new URL("./frameImages.ts", import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  vm.runInNewContext(outputText, {
    exports, Image, URL, navigator: { connection }, document: { baseURI: "https://example.test/" },
    process: { env: { NEXT_PUBLIC_FRAME_DECODE: mode } },
  });
  return { ...exports, images };
}

test("card variants apply only to server-decoded archive frames", () => {
  const f = fixture();
  assert.equal(f.cardImageUrl("/api/zip-frame/V1/100"), "/api/zip-frame/V1/100?width=640");
  assert.equal(f.cardImageUrl("/api/zip-frame/N001-V001/100?frame_number=3"),
    "/api/zip-frame/N001-V001/100?frame_number=3&width=640");
  assert.equal(f.cardImageUrl("/api/zip-frame/N001-V001/100?frame_number=3&v=4"),
    "/api/zip-frame/N001-V001/100?frame_number=3&v=4&width=640");
  assert.equal(f.cardImageUrl("/api/zip-frame/N001-V001/100?frame_number=3&v=5"),
    "/api/zip-frame/N001-V001/100?frame_number=3&v=5&width=640");
  assert.equal(f.cardImageUrl("/api/zip-frame/N001-V001/100?frame_number=3&v=6"),
    "/api/zip-frame/N001-V001/100?frame_number=3&v=6&width=640");
  for (const url of ["/static/frame.jpg", "blob:frame", "/api/zip-video/V1", "/api/zip-frame/V1/100?width=640"])
    assert.equal(f.cardImageUrl(url), null);
  assert.equal(fixture("client").cardImageUrl("/api/zip-frame/V1/100"), null);
  assert.equal(f.needsOriginalFrame(250, 2), false);
  assert.equal(f.needsOriginalFrame(320, 2), false);
  assert.equal(f.needsOriginalFrame(321, 2), true);
});

test("prefetch is bounded, reuses successful requests and retries failures", () => {
  const f = fixture();
  f.prefetchOriginalFrame("/a");
  f.prefetchOriginalFrame("/a");
  f.prefetchOriginalFrame("/b");
  f.prefetchOriginalFrame("/c");
  assert.equal(f.images.length, 2);
  assert.equal(f.images[0].fetchPriority, "low");
  f.images[0].onload();
  f.prefetchOriginalFrame("/a");
  assert.equal(f.images.length, 2);
  f.images[1].onerror();
  f.prefetchOriginalFrame("/b");
  assert.equal(f.images.length, 3);
});

test("detail preview uses the displayed frame and never a different frame's URL", () => {
  const f = fixture();
  f.rememberCardImage("/a", "https://example.test/a?width=640");
  assert.equal(f.loadedCardImage("/a"), "https://example.test/a?width=640");
  assert.equal(f.loadedCardImage("/b"), undefined);
  f.rememberCardImage("/a", "https://example.test/a");
  f.prefetchOriginalFrame("/a");
  assert.equal(f.images.length, 0);
  f.rememberCardImage("/b", "blob:temporary");
  assert.equal(f.loadedCardImage("/b"), undefined);
});


test("weak connections suppress speculative originals without affecting explicit display URLs", () => {
  for (const connection of [{ saveData: true }, { downlink: 1.6, effectiveType: "4g" }, { effectiveType: "3g" }]) {
    const f = fixture("server", connection);
    assert.equal(f.isConstrainedConnection(), true);
    assert.equal(f.cardImageUrl("/api/zip-frame/V1/100", "webp"), "/api/zip-frame/V1/100?width=640&format=webp");
    f.prefetchOriginalFrame("/api/zip-frame/V1/100");
    assert.equal(f.images.length, 0);
  }
  for (const connection of [undefined, {}, { downlink: 0 }, { downlink: 10, effectiveType: "4g" }]) {
    const f = fixture("server", connection);
    assert.equal(f.isConstrainedConnection(), false);
    f.prefetchOriginalFrame("/api/zip-frame/V1/100");
    assert.equal(f.images.length, 1);
  }
});
