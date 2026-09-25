import type { NextApiRequest, NextApiResponse } from "next";
import { timingSafeEqual } from "crypto";
import { readCandidateForDres } from "./submission";
import { buildDresSubmission } from "@/lib/dres";

const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
const DRES_BASE = (process.env.DRES_BASE_URL || "https://eventretrieval.one").replace(/\/$/, "");
const BACKEND = (process.env.VORTA_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const attempts = new Map<string, "sending" | "accepted" | "uncertain">();

interface Evaluation { id: string; name: string; status: string }
interface EvaluationState { evaluationStatus: string; taskId?: string; taskStatus: string; timeLeft?: number }

async function dres(path: string, sessionToken: string, method: "GET" | "POST" = "GET", body?: unknown) {
  const url = new URL(`${DRES_BASE}${path}`);
  url.searchParams.set("session", sessionToken);
  const response = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(8000),
  });
  const data = await response.json().catch(() => null);
  return { response, data };
}

async function currentState(evaluationId: string, sessionToken: string): Promise<EvaluationState> {
  const { response, data } = await dres(`/api/v2/evaluation/${encodeURIComponent(evaluationId)}/state`, sessionToken);
  if (!response.ok || !data || typeof data !== "object") {
    throw new Error(`DRES task lookup failed (HTTP ${response.status})`);
  }
  return data as EvaluationState;
}

function publicError(error: unknown): string {
  if (error instanceof Error && error.name === "TimeoutError") return "DRES timed out; check its website before retrying.";
  return error instanceof Error ? error.message : "DRES is unavailable";
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "GET" && req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });
  const sessionToken = process.env.DRES_SESSION_ID;
  if (!sessionToken) return res.status(503).json({ error: "Set DRES_SESSION_ID on the frontend server to enable DRES submission" });

  if (req.method === "GET") {
    try {
      const { response, data } = await dres("/api/v2/client/evaluation/list", sessionToken);
      if (!response.ok || !Array.isArray(data)) {
        return res.status(502).json({ error: `DRES evaluation lookup failed (HTTP ${response.status})` });
      }
      const evaluations = (data as Evaluation[])
        .filter((item) => item && ID.test(item.id) && item.status === "ACTIVE")
        .map(({ id, name, status }) => ({ id, name, status }));
      const requested = typeof req.query.evaluationId === "string" ? req.query.evaluationId : "";
      if (requested && !evaluations.some((item) => item.id === requested)) {
        return res.status(400).json({ error: "Evaluation is not active" });
      }
      const selected = requested || (evaluations.length === 1 ? evaluations[0].id : "");
      const state = selected ? await currentState(selected, sessionToken) : null;
      return res.status(200).json({ evaluations, selectedEvaluationId: selected, state });
    } catch (error) {
      return res.status(502).json({ error: publicError(error) });
    }
  }

  const submitPin = process.env.DRES_SUBMIT_PIN;
  if (!submitPin) return res.status(503).json({ error: "Set DRES_SUBMIT_PIN on the frontend server to enable live submission" });
  const suppliedPin = req.headers["x-dres-submit-pin"];
  const supplied = Buffer.from(typeof suppliedPin === "string" ? suppliedPin : "");
  const expected = Buffer.from(submitPin);
  if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) {
    return res.status(403).json({ error: "Invalid DRES submit PIN" });
  }

  const { evaluationId, session, rowIndex, taskId, expectedRow } = req.body || {};
  if (![evaluationId, session, taskId].every((value) => typeof value === "string" && ID.test(value)) ||
      !Number.isInteger(rowIndex) || rowIndex < 0 || typeof expectedRow !== "string") {
    return res.status(400).json({ error: "Invalid submission request" });
  }
  const saved = readCandidateForDres(session, rowIndex);
  const row = saved?.row;
  if (!saved || !row) return res.status(404).json({ error: "Candidate no longer exists or its CSV line is malformed" });
  if (JSON.stringify(row) !== expectedRow) {
    return res.status(409).json({ error: "Candidate changed; review it again before submitting" });
  }

  let state: EvaluationState;
  try {
    state = await currentState(evaluationId, sessionToken);
  } catch (error) {
    return res.status(502).json({ error: publicError(error) });
  }
  if (state.evaluationStatus !== "ACTIVE" || state.taskStatus !== "RUNNING" || state.taskId !== taskId) {
    return res.status(409).json({ error: "DRES task changed or is not running; refresh the DRES status" });
  }

  let fps: number | undefined;
  if (saved.queryType !== "trake") {
    try {
      const response = await fetch(`${BACKEND}/api/video/${encodeURIComponent(row.videoId)}`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) return res.status(502).json({ error: "VORTA video lookup failed; cannot convert frame to milliseconds" });
      const video = await response.json();
      if (video.video_id !== row.videoId) return res.status(409).json({ error: "Video ID resolved to a different video" });
      fps = video.fps;
    } catch {
      return res.status(502).json({ error: "VORTA video metadata is unavailable; submission was not sent" });
    }
  }

  let payload;
  try {
    payload = buildDresSubmission(taskId, saved.queryType, row, fps);
  } catch (error) {
    return res.status(400).json({ error: publicError(error) });
  }
  const key = JSON.stringify([evaluationId, taskId, payload]);
  const previous = attempts.get(key);
  if (previous) return res.status(409).json({ error: `This exact answer was already ${previous}; check DRES before retrying` });
  attempts.set(key, "sending");

  try {
    const { response, data } = await dres(`/api/v2/submit/${encodeURIComponent(evaluationId)}`, sessionToken, "POST", payload);
    if (!response.ok || !data || data.status !== true) {
      attempts.delete(key);
      return res.status(502).json({
        error: `DRES rejected submission (HTTP ${response.status}): ${data?.description || "unknown response"}`,
      });
    }
    attempts.set(key, "accepted");
    return res.status(200).json({
      ok: true,
      evaluationId,
      taskId,
      answer: payload.answerSets[0].answers[0],
      verdict: data.submission || "PENDING",
      description: data.description || "Accepted by DRES",
    });
  } catch (error) {
    attempts.set(key, "uncertain");
    return res.status(502).json({ error: `${publicError(error)} Submission status is uncertain; check DRES before retrying.` });
  }
}
