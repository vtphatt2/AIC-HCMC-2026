import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "fs";
import path from "path";
import type { NextApiRequest, NextApiResponse } from "next";

import type { SubmissionQueryType, SubmissionRow, SubmissionSessionSummary, SubmissionState } from "@/types";
import { fillKisRows } from "@/lib/submission/fillFrames";
import { parseSubmissionCsv, serializeSubmissionRow, submissionUnit, validateSubmissionRow } from "@/lib/submission/format";
import { resolveNRow } from "@/lib/submission/timing";
import { parseVersionedTimeline, type FrameTimeline } from "@/lib/playback";

const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];
const ROOT = path.join(process.cwd(), ".runtime", "submissions");

interface Meta {
  version?: 2;
  rows?: SubmissionRow[];
  queryType: SubmissionQueryType;
  draftRowIndex: number;
  createdAt: number;
  updatedAt: number;
}

function csvFile(session: string): string {
  return path.join(ROOT, `${session}.csv`);
}
function metaFile(session: string): string {
  return path.join(ROOT, `${session}.meta.json`);
}

function isNonNegativeInt(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function serializeRow(queryType: SubmissionQueryType, row: SubmissionRow): string {
  return serializeSubmissionRow(queryType, row);
}

async function timelineFor(videoId: string): Promise<FrameTimeline> {
  const configured = process.env.SUBMISSION_TIMING_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const base = configured.startsWith("http") ? configured.replace(/\/$/, "") : "http://127.0.0.1:8000";
  const response = await fetch(`${base}/api/video/${encodeURIComponent(videoId)}/frame-timeline?version=2`, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error(`Verified timeline unavailable for ${videoId}`);
  const payload = await response.json();
  if (payload.video_id !== videoId) throw new Error("Timeline identity mismatch");
  return parseVersionedTimeline(payload);
}

function readState(session: string): SubmissionState | null {
  let meta: Meta;
  try {
    meta = JSON.parse(readFileSync(metaFile(session), "utf8"));
  } catch (error: any) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
  const csvText = existsSync(csvFile(session)) ? readFileSync(csvFile(session), "utf8") : "";
  // Version 2 metadata is the atomic session snapshot. CSV is its organizer
  // export, so a crash between the two renames cannot mix row units/order.
  const rows: SubmissionRow[] = meta.version === 2 && Array.isArray(meta.rows)
    ? meta.rows : parseSubmissionCsv(meta.queryType, csvText).map(row => ({
        ...row, unit: "frames", ...(row.videoId.startsWith("N") ? {
          timingStatus: "unresolved" as const, timingError: "Legacy N frame needs verified migration",
        } : {}),
      }));
  return { version: 2, session, queryType: meta.queryType, draftRowIndex: meta.draftRowIndex, rows, createdAt: meta.createdAt, updatedAt: meta.updatedAt };
}

function writeState(state: SubmissionState): void {
  mkdirSync(ROOT, { recursive: true });
  const meta: Meta = {
    version: 2,
    rows: state.rows,
    queryType: state.queryType,
    draftRowIndex: state.draftRowIndex,
    createdAt: state.createdAt,
    updatedAt: state.updatedAt,
  };
  const metaTmp = `${metaFile(state.session)}.${process.pid}.tmp`;
  writeFileSync(metaTmp, `${JSON.stringify(meta, null, 2)}\n`, "utf8");


  const csvContent = state.rows.length
    ? `${state.rows.map((r) => serializeRow(state.queryType, r)).join("\r\n")}\r\n`
    : "";
  const csvTmp = `${csvFile(state.session)}.${process.pid}.tmp`;
  writeFileSync(csvTmp, csvContent, "utf8");
  renameSync(csvTmp, csvFile(state.session));
  renameSync(metaTmp, metaFile(state.session));
}

function removeSessionFiles(session: string): void {
  if (existsSync(metaFile(session))) unlinkSync(metaFile(session));
  if (existsSync(csvFile(session))) unlinkSync(csvFile(session));
}

function listSessions(): SubmissionSessionSummary[] {
  let names: string[] = [];
  try {
    names = readdirSync(ROOT).filter((f) => f.endsWith(".meta.json"));
  } catch (error: any) {
    if (error?.code !== "ENOENT") throw error;
  }
  return names
    .map((name) => readState(name.slice(0, -".meta.json".length)))
    .filter((s): s is SubmissionState => s !== null)
    .map((s) => ({
      session: s.session,
      queryType: s.queryType,
      rowCount: s.rows.length,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    }))
    // Creation order, stable regardless of which session was edited most
    // recently — an activity-based sort reshuffled the list on every edit,
    // which was disorienting while actively working across sessions.
    .sort((a, b) => a.createdAt - b.createdAt);
}

// The frontend server owns this file store. Reserve all affected session names
// before awaiting timing lookups, so concurrent clients cannot overwrite each
// other's rows or race a rename/delete against a pending save.
const sessionTails = new Map<string, Promise<void>>();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const keys = [req.query.session, req.body?.session, req.body?.action === 'rename' ? req.body?.newSession : undefined]
    .filter((key): key is string => typeof key === 'string')
    .map(key => req.body?.action === 'rename' ? key.trim() : key)
    .filter(key => ID.test(key));
  const names = Array.from(new Set(keys));
  const previous = names.map(name => sessionTails.get(name));
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  names.forEach(name => sessionTails.set(name, pending));
  await Promise.all(previous);
  try {
    return await handleSessionRequest(req, res);
  } finally {
    release();
    names.forEach(name => { if (sessionTails.get(name) === pending) sessionTails.delete(name); });
  }
}

async function handleSessionRequest(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("Cache-Control", "no-store");
  const session = typeof req.query.session === "string" ? req.query.session : "";

  if (req.method === "GET") {
    if (!session) return res.status(200).json({ sessions: listSessions() });
    if (!ID.test(session)) return res.status(400).json({ error: "Invalid session name" });
    const state = readState(session);
    if (!state) return res.status(404).json({ error: "Session not found" });
    for (let i = 0; i < state.rows.length; i++) {
      const row = state.rows[i];
      if (row.videoId.startsWith("N") && row.unit !== "milliseconds") {
        try { state.rows[i] = resolveNRow(row, await timelineFor(row.videoId)); }
        catch (error: any) { row.timingStatus = "unresolved"; row.timingError = error.message; }
      }
    }
    return res.status(200).json(state);
  }

  if (req.method === "DELETE") {
    if (!ID.test(session)) return res.status(400).json({ error: "Invalid session name" });
    if (!existsSync(metaFile(session))) return res.status(404).json({ error: "Session not found" });
    removeSessionFiles(session);
    return res.status(200).json({ ok: true });
  }

  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const action = req.body?.action;

  if (action === "create") {
    const newSession = typeof req.body?.session === "string" ? req.body.session : "";
    if (!ID.test(newSession)) return res.status(400).json({ error: "Invalid session name" });
    if (existsSync(metaFile(newSession))) return res.status(409).json({ error: "Session already exists" });
    const queryType: SubmissionQueryType = QUERY_TYPES.includes(req.body?.queryType) ? req.body.queryType : "kis";
    const now = Date.now();
    const state: SubmissionState = {
      session: newSession,
      queryType,
      draftRowIndex: 0,
      rows: [],
      createdAt: now,
      updatedAt: now,
    };
    writeState(state);
    return res.status(200).json(state);
  }

  if (!ID.test(session)) return res.status(400).json({ error: "Invalid session name" });
  const current = readState(session);
  if (!current) return res.status(404).json({ error: "Session not found" });

  function rowAt(index: unknown): SubmissionRow | null {
    return isNonNegativeInt(index) && index < current!.rows.length ? current!.rows[index] : null;
  }

  if (action === "add") {
    // Quick-add from VideoModal — kis/qa always start a fresh row; trake
    // appends the frame to the in-progress candidate (draftRowIndex, or an
    // explicit rowIndex when reviewing an existing candidate from the
    // dashboard), starting a new row once that candidate has been closed out
    // by "+ New candidate" (draftRowIndex === rows.length).
    const videoId = typeof req.body?.videoId === "string" ? req.body.videoId : "";
    const frame = req.body?.frame;
    if (!ID.test(videoId)) return res.status(400).json({ error: "Invalid videoId" });
    if (!isNonNegativeInt(frame)) return res.status(400).json({ error: "Invalid frame" });

    if (current.queryType !== "trake") {
      current.rows.push({ videoId, unit: submissionUnit(videoId), frames: [frame], answer: current.queryType === "qa" ? "" : undefined });
    } else {
      const targetIndex = isNonNegativeInt(req.body?.rowIndex) ? req.body.rowIndex : current.draftRowIndex;
      const target = current.rows[targetIndex];
      if (target) {
        if (target.videoId !== videoId) {
          return res.status(409).json({
            error: `Candidate ${targetIndex + 1} already has frames from ${target.videoId} — every frame in a TRAKE candidate must be the same video. Start a new candidate first.`,
          });
        }
        if (!target.frames.includes(frame)) target.frames.push(frame);
        target.frames.sort((a, b) => a - b);
      } else {
        current.rows.splice(targetIndex, 0, { videoId, frames: [frame] });
      }
    }
  } else if (action === "addRow") {
    // Manual add: type a video_id + frame(s) directly, no search/VideoModal.
    const videoId = typeof req.body?.videoId === "string" ? req.body.videoId : "";
    const frames = req.body?.frames;
    if (!ID.test(videoId)) return res.status(400).json({ error: "Invalid videoId" });
    if (!Array.isArray(frames) || frames.length === 0 || frames.some((f) => !isNonNegativeInt(f))) {
      return res.status(400).json({ error: "Invalid frames" });
    }
    if (current.queryType !== "trake" && frames.length !== 1) {
      return res.status(400).json({ error: "kis/qa rows take exactly one frame" });
    }
    const answer = typeof req.body?.answer === "string" ? req.body.answer : "";
    if (answer.length > 100) return res.status(400).json({ error: "Answer must be 100 characters or fewer" });
    current.rows.push({
      videoId,
      unit: submissionUnit(videoId),
      frames: [...frames].sort((a, b) => a - b),
      answer: current.queryType === "qa" ? answer : undefined,
    });
  } else if (action === "editVideoId") {
    const row = rowAt(req.body?.rowIndex);
    const videoId = typeof req.body?.videoId === "string" ? req.body.videoId : "";
    if (!row) return res.status(404).json({ error: "Row not found" });
    if (!ID.test(videoId)) return res.status(400).json({ error: "Invalid videoId" });
    if (submissionUnit(row.videoId) !== submissionUnit(videoId)) return res.status(400).json({ error: "Changing position units requires a new row" });
    row.videoId = videoId;
    row.sourceFrames = undefined;
    row.timingStatus = undefined;
  } else if (action === "editFrame") {
    const row = rowAt(req.body?.rowIndex);
    const frameIndex = req.body?.frameIndex;
    const frame = req.body?.frame;
    if (!row || !isNonNegativeInt(frameIndex) || frameIndex >= row.frames.length) {
      return res.status(404).json({ error: "Row/frame not found" });
    }
    if (!isNonNegativeInt(frame)) return res.status(400).json({ error: "Invalid frame" });
    row.frames[frameIndex] = frame;
    row.frames.sort((a, b) => a - b);
  } else if (action === "removeFrame") {
    const row = rowAt(req.body?.rowIndex);
    const frameIndex = req.body?.frameIndex;
    if (!row || !isNonNegativeInt(frameIndex) || frameIndex >= row.frames.length) {
      return res.status(404).json({ error: "Row/frame not found" });
    }
    row.frames.splice(frameIndex, 1);
    if (row.frames.length === 0) current.rows.splice(current.rows.indexOf(row), 1);
  } else if (action === "removeRow") {
    if (!rowAt(req.body?.rowIndex)) return res.status(404).json({ error: "Row not found" });
    current.rows.splice(req.body.rowIndex, 1);
  } else if (action === "setAnswer") {
    const row = rowAt(req.body?.rowIndex);
    if (!row) return res.status(404).json({ error: "Row not found" });
    if (current.queryType !== "qa") return res.status(400).json({ error: "Only qa rows have an answer" });
    const answer = req.body?.answer;
    // Organizer cap: Q&A answers are compared as an exact string, max 100 chars.
    if (typeof answer !== "string" || answer.length > 100) {
      return res.status(400).json({ error: "Answer must be 100 characters or fewer" });
    }
    row.answer = answer;
  } else if (action === "setQueryType") {
    if (!QUERY_TYPES.includes(req.body?.queryType)) return res.status(400).json({ error: "Invalid queryType" });
    current.queryType = req.body.queryType;
  } else if (action === "rename") {
    const newSession = typeof req.body?.newSession === "string" ? req.body.newSession.trim() : "";
    if (!ID.test(newSession)) return res.status(400).json({ error: "Invalid session name" });
    if (newSession !== session && existsSync(metaFile(newSession))) {
      return res.status(409).json({ error: "A session with that name already exists" });
    }
    current.session = newSession;
  } else if (action === "newCandidate") {
    current.draftRowIndex = current.rows.length;
  } else if (action === "reset") {
    current.rows = [];
    current.draftRowIndex = 0;
  } else if (action === "reorderRows") {
    const order = req.body?.order;
    const n = current.rows.length;
    if (
      !Array.isArray(order) ||
      order.length !== n ||
      new Set(order).size !== n ||
      order.some((i: unknown) => !isNonNegativeInt(i) || (i as number) >= n)
    ) {
      return res.status(400).json({ error: "Invalid row order" });
    }
    current.rows = order.map((i: number) => current!.rows[i]);
  } else if (action === "replaceRaw") {
    // Raw CSV textarea save — replaces every row at once.
    const content = req.body?.content;
    if (typeof content !== "string") return res.status(400).json({ error: "Invalid content" });
    try { current.rows = parseSubmissionCsv(current.queryType, content); }
    catch (error: any) { return res.status(400).json({ error: error.message }); }
    const rows = current.rows;
    current.draftRowIndex = Math.min(current.draftRowIndex, rows.length);
  } else if (action === "fillNeighbors") {
    if (current.queryType !== "kis") {
      return res.status(400).json({ error: "Neighbor filler is only valid for KIS sessions" });
    }
    if (current.rows.length === 0) {
      return res.status(400).json({ error: "Add at least one KIS candidate before using filler" });
    }
    try {
      const timelines: Record<string, FrameTimeline> = {};
      for (const row of current.rows) if (row.videoId.startsWith("N") && !timelines[row.videoId]) timelines[row.videoId] = await timelineFor(row.videoId);
      current.rows = current.rows.map(row => timelines[row.videoId] ? resolveNRow(row, timelines[row.videoId]) : row);
      current.rows = fillKisRows(current.rows, 100, 15, timelines);
    } catch (error: any) { return res.status(422).json({ error: error.message }); }
  } else {
    return res.status(400).json({ error: "Unknown action" });
  }

  try {
    const timelines = new Map<string, FrameTimeline>();
    for (let i = 0; i < current.rows.length; i++) {
      const row = current.rows[i];
      validateSubmissionRow(current.queryType, row);
      if (row.videoId.startsWith("N")) {
        let timeline = timelines.get(row.videoId);
        if (!timeline) { timeline = await timelineFor(row.videoId); timelines.set(row.videoId, timeline); }
        current.rows[i] = resolveNRow(row, timeline, action === "add" && i === current.rows.length - 1 && req.body?.sourceFrameInput === true);
      } else { row.unit = "frames"; }
    }
  } catch (error: any) { return res.status(422).json({ error: error.message }); }
  current.updatedAt = Date.now();
  writeState(current);
  if (current.session !== session) removeSessionFiles(session);
  return res.status(200).json(current);
}
