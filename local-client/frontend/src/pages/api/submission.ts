import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "fs";
import path from "path";
import { randomUUID } from "crypto";
import type { NextApiRequest, NextApiResponse } from "next";

import type { SubmissionEntry, SubmissionQueryType, SubmissionSessionSummary, SubmissionState } from "@/types";

const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];
const ROOT = path.join(process.cwd(), ".runtime", "submissions");

function fileFor(session: string): string {
  return path.join(ROOT, `${session}.json`);
}

function readState(session: string): SubmissionState | null {
  try {
    const parsed = JSON.parse(readFileSync(fileFor(session), "utf8"));
    // Backfill for sessions created before rowOrder existed — every reader
    // (GET responses included, not just the mutating actions that run
    // syncRowOrder) needs a real array here, never undefined.
    return { ...parsed, rowOrder: parsed.rowOrder || [] };
  } catch (error: any) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function writeState(state: SubmissionState): void {
  mkdirSync(ROOT, { recursive: true });
  const file = fileFor(state.session);
  const temporary = `${file}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  renameSync(temporary, file);
}

function listSessions(): SubmissionSessionSummary[] {
  let names: string[] = [];
  try {
    names = readdirSync(ROOT).filter((f) => f.endsWith(".json"));
  } catch (error: any) {
    if (error?.code !== "ENOENT") throw error;
  }
  return names
    .map((name) => readState(name.slice(0, -".json".length)))
    .filter((s): s is SubmissionState => s !== null)
    .map((s) => ({
      session: s.session,
      queryType: s.queryType,
      queryNumber: s.queryNumber,
      entryCount: s.entries.length,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    }))
    // Creation order, stable regardless of which session was edited most
    // recently — an activity-based sort reshuffled the list on every edit,
    // which was disorienting while actively working across sessions.
    .sort((a, b) => a.createdAt - b.createdAt);
}

function isNonNegativeInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

// Keeps rowOrder (the ranked row list the CSV export/UI use) valid after any
// mutation: stale keys (removed entries, emptied-out candidates) drop out,
// newly-appeared keys append at the end in natural order, everything the
// user has manually ranked stays exactly where they put it. Called once,
// unconditionally, right before every write — covers every action without
// needing bespoke sync logic in each branch. A queryType switch naturally
// rebuilds rowOrder too, since validKeys is recomputed from the current type.
function syncRowOrder(state: SubmissionState): void {
  const validKeys = new Set(
    state.queryType === "trake"
      ? state.entries.map((e) => `g${e.groupIndex}`)
      : state.entries.map((e) => e.id),
  );
  const kept = state.rowOrder.filter((k) => validKeys.has(k));
  const keptSet = new Set(kept);
  const fresh = Array.from(validKeys).filter((k) => !keptSet.has(k));
  state.rowOrder = [...kept, ...fresh];
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("Cache-Control", "no-store");
  const session = typeof req.query.session === "string" ? req.query.session : "";

  if (req.method === "GET") {
    if (!session) return res.status(200).json({ sessions: listSessions() });
    if (!ID.test(session)) return res.status(400).json({ error: "Invalid session name" });
    const state = readState(session);
    if (!state) return res.status(404).json({ error: "Session not found" });
    return res.status(200).json(state);
  }

  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const action = req.body?.action;

  if (action === "create") {
    const newSession = typeof req.body?.session === "string" ? req.body.session : "";
    if (!ID.test(newSession)) return res.status(400).json({ error: "Invalid session name" });
    if (existsSync(fileFor(newSession))) {
      return res.status(409).json({ error: "Session already exists" });
    }
    const queryType: SubmissionQueryType = QUERY_TYPES.includes(req.body?.queryType) ? req.body.queryType : "kis";
    const queryNumber = isNonNegativeInt(req.body?.queryNumber) && req.body.queryNumber > 0 ? req.body.queryNumber : 1;
    const now = Date.now();
    const state: SubmissionState = {
      session: newSession,
      revision: 0,
      queryType,
      queryNumber,
      answer: "",
      nextGroupIndex: 0,
      rowOrder: [],
      entries: [],
      createdAt: now,
      updatedAt: now,
    };
    writeState(state);
    return res.status(200).json(state);
  }

  if (!ID.test(session)) return res.status(400).json({ error: "Invalid session name" });
  const current = readState(session);
  if (!current) return res.status(404).json({ error: "Session not found" });

  if (action === "add") {
    const videoId = typeof req.body?.videoId === "string" ? req.body.videoId : "";
    const frame = req.body?.frame;
    const fps = req.body?.fps;
    if (!ID.test(videoId)) return res.status(400).json({ error: "Invalid videoId" });
    if (!isNonNegativeInt(frame)) return res.status(400).json({ error: "Invalid frame" });
    if (typeof fps !== "number" || !Number.isFinite(fps) || fps <= 0) {
      return res.status(400).json({ error: "Invalid fps" });
    }
    const imageUrl = typeof req.body?.imageUrl === "string" ? req.body.imageUrl : undefined;
    const youtubeId =
      typeof req.body?.youtubeId === "string" && req.body.youtubeId.length <= 32
        ? req.body.youtubeId
        : undefined;
    // Reviewing an existing candidate from the dashboard and adding another
    // frame to it should land in *that* candidate, not whatever this
    // session's nextGroupIndex happens to be — callers with a specific
    // target (VideoModal opened for review) pass groupIndex explicitly. A
    // plain add (no override) just consumes the current pointer as-is and
    // must NOT advance it — nextGroupIndex only moves forward via
    // "+ New candidate" or an explicit override, otherwise every kis/qa add
    // (which never overrides) would scatter across ever-increasing
    // candidate numbers instead of everything staying ungrouped.
    const hasOverride = isNonNegativeInt(req.body?.groupIndex);
    const groupIndex = hasOverride ? req.body.groupIndex : current.nextGroupIndex;
    const entry: SubmissionEntry = {
      id: randomUUID(),
      videoId,
      frame,
      imageUrl,
      fps,
      youtubeId,
      groupIndex,
      addedAt: Date.now(),
    };
    current.entries.push(entry);
    if (hasOverride) current.nextGroupIndex = Math.max(current.nextGroupIndex, groupIndex + 1);
  } else if (action === "editFrame") {
    const id = typeof req.body?.id === "string" ? req.body.id : "";
    const frame = req.body?.frame;
    if (!isNonNegativeInt(frame)) return res.status(400).json({ error: "Invalid frame" });
    const entry = current.entries.find((e) => e.id === id);
    if (!entry) return res.status(404).json({ error: "Entry not found" });
    entry.frame = frame;
  } else if (action === "editGroup") {
    const id = typeof req.body?.id === "string" ? req.body.id : "";
    const groupIndex = req.body?.groupIndex;
    if (!isNonNegativeInt(groupIndex)) return res.status(400).json({ error: "Invalid groupIndex" });
    const entry = current.entries.find((e) => e.id === id);
    if (!entry) return res.status(404).json({ error: "Entry not found" });
    entry.groupIndex = groupIndex;
    current.nextGroupIndex = Math.max(current.nextGroupIndex, groupIndex + 1);
  } else if (action === "remove") {
    const id = typeof req.body?.id === "string" ? req.body.id : "";
    current.entries = current.entries.filter((e) => e.id !== id);
  } else if (action === "setMeta") {
    if (req.body?.queryType !== undefined) {
      if (!QUERY_TYPES.includes(req.body.queryType)) return res.status(400).json({ error: "Invalid queryType" });
      current.queryType = req.body.queryType;
    }
    if (req.body?.queryNumber !== undefined) {
      if (!isNonNegativeInt(req.body.queryNumber) || req.body.queryNumber < 1) {
        return res.status(400).json({ error: "Invalid queryNumber" });
      }
      current.queryNumber = req.body.queryNumber;
    }
    if (req.body?.answer !== undefined) {
      // Organizer cap: Q&A answers are compared as an exact string, max 100 chars.
      if (typeof req.body.answer !== "string" || req.body.answer.length > 100) {
        return res.status(400).json({ error: "Answer must be 100 characters or fewer" });
      }
      current.answer = req.body.answer;
    }
  } else if (action === "newCandidate") {
    current.nextGroupIndex += 1;
  } else if (action === "reset") {
    current.entries = [];
    current.nextGroupIndex = 0;
    current.answer = "";
    current.rowOrder = [];
  } else if (action === "reorderRows") {
    const rowOrder = req.body?.rowOrder;
    if (!Array.isArray(rowOrder) || rowOrder.length > 200 || rowOrder.some((k) => typeof k !== "string" || k.length > 128)) {
      return res.status(400).json({ error: "Invalid rowOrder" });
    }
    current.rowOrder = rowOrder;
  } else {
    return res.status(400).json({ error: "Unknown action" });
  }

  syncRowOrder(current);
  current.revision += 1;
  current.updatedAt = Date.now();
  writeState(current);
  return res.status(200).json(current);
}
