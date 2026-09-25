import type { NextApiRequest, NextApiResponse } from "next";

const AGENT_SERVER_URL = (process.env.AGENT_SERVER_URL || "http://127.0.0.1:8012").replace(/\/$/, "");

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("Cache-Control", "no-store");
  let method: "GET" | "POST" | "DELETE";
  let path = "/api/agent/verify";
  let body: string | undefined;

  if (req.method === "GET" || req.method === "DELETE") {
    method = req.method;
  } else if (req.method === "POST" && req.body?.action === "cancel") {
    method = "POST";
    path += "/cancel";
  } else if (req.method === "POST") {
    method = "POST";
    body = JSON.stringify({ query: req.body?.query, candidate: req.body?.candidate });
  } else {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const upstream = await fetch(`${AGENT_SERVER_URL}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body,
      signal: AbortSignal.timeout(5000),
    });
    const payload = await upstream.json();
    return res.status(upstream.status).json(payload);
  } catch {
    return res.status(503).json({ error: "Verify Agent is unavailable. Manual inspection is still available." });
  }
}
