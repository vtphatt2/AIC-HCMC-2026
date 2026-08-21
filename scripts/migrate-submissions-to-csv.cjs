#!/usr/bin/env node
// One-time migration: submission sessions used to be stored as one
// {session}.json per session (entries[] + a separate rowOrder list +
// per-entry groupIndex). As of 2026-08-21 they're {session}.csv (the
// literal export file) + a small {session}.meta.json sidecar — see
// docs/SUBMISSION.md. .runtime/ is gitignored, so `git pull` never touches
// it; run this once on any machine that already has old-format sessions.
// Safe to run repeatedly — .json files are only touched (renamed away) if
// found, never re-read once converted.
//
// Node stdlib only — no dependency, nothing to install.

const fs = require("node:fs");
const path = require("node:path");

const DIR = path.join(__dirname, "..", "local-client", "frontend", ".runtime", "submissions");

function csvLine(queryType, row) {
  if (queryType === "qa") return [row.videoId, row.frames[0], row.answer ?? ""].join(",");
  return [row.videoId, ...row.frames].join(",");
}

function migrate(file) {
  const old = JSON.parse(fs.readFileSync(path.join(DIR, file), "utf8"));
  const { session, queryType } = old;
  const rowOrder = old.rowOrder || [];
  const pos = new Map(rowOrder.map((k, i) => [k, i]));

  let rows;
  if (queryType === "trake") {
    const groups = new Map();
    for (const e of old.entries) {
      if (!groups.has(e.groupIndex)) groups.set(e.groupIndex, []);
      groups.get(e.groupIndex).push(e);
    }
    rows = [...groups.entries()]
      .sort((a, b) => (pos.get(`g${a[0]}`) ?? Infinity) - (pos.get(`g${b[0]}`) ?? Infinity))
      .map(([, es]) => ({ videoId: es[0].videoId, frames: es.map((e) => e.frame).sort((a, b) => a - b) }));
  } else {
    rows = [...old.entries]
      .sort((a, b) => (pos.get(a.id) ?? Infinity) - (pos.get(b.id) ?? Infinity))
      .map((e) => ({ videoId: e.videoId, frames: [e.frame], answer: queryType === "qa" ? old.answer || "" : undefined }));
  }

  const meta = { queryType, draftRowIndex: 0, createdAt: old.createdAt, updatedAt: old.updatedAt };
  fs.writeFileSync(path.join(DIR, `${session}.meta.json`), `${JSON.stringify(meta, null, 2)}\n`, "utf8");
  const lines = rows.map((r) => csvLine(queryType, r));
  fs.writeFileSync(path.join(DIR, `${session}.csv`), lines.length ? `${lines.join("\r\n")}\r\n` : "", "utf8");
  fs.unlinkSync(path.join(DIR, file));
  console.log(`migrated ${session} (${rows.length} row${rows.length === 1 ? "" : "s"})`);
}

if (!fs.existsSync(DIR)) {
  console.log(`No ${DIR} — nothing to migrate.`);
  process.exit(0);
}

const oldFiles = fs.readdirSync(DIR).filter((f) => f.endsWith(".json") && !f.endsWith(".meta.json"));
if (oldFiles.length === 0) {
  console.log("No old-format sessions found — nothing to do.");
} else {
  oldFiles.forEach(migrate);
}
