import { mkdirSync, readFileSync, renameSync, writeFileSync } from "fs";
import path from "path";
import type { NextApiRequest, NextApiResponse } from "next";

import type { StrategyConfigDraft, StrategyConfigValue } from "@/types";


const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const ROOT = path.join(process.cwd(), ".runtime", "strategy-config-drafts");

function readDraft(file: string): StrategyConfigDraft {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch (error: any) {
    if (error?.code === "ENOENT") return { revision: 0, overrides: {} };
    throw error;
  }
}

function validOverrides(value: unknown): value is Record<string, StrategyConfigValue> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const entries = Object.entries(value);
  return entries.length <= 100 && entries.every(([key, item]) =>
    /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(key)
    && ((typeof item === "number" && Number.isFinite(item))
      || (Array.isArray(item) && item.length <= 100 && item.every((number) => typeof number === "number" && Number.isFinite(number))))
  );
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const strategyId = typeof req.query.strategy_id === "string" ? req.query.strategy_id : "";
  const configId = typeof req.query.config_id === "string" ? req.query.config_id : "";
  if (!ID.test(strategyId) || !ID.test(configId)) {
    return res.status(400).json({ error: "Invalid strategy_id or config_id" });
  }

  const file = path.join(ROOT, strategyId, `${configId}.json`);
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "GET") return res.status(200).json(readDraft(file));
  if (req.method !== "PUT") return res.status(405).json({ error: "Method not allowed" });
  if (!validOverrides(req.body?.overrides)) return res.status(400).json({ error: "Invalid overrides" });

  const current = readDraft(file);
  const draft = {
    revision: current.revision + 1,
    overrides: { ...current.overrides, ...req.body.overrides },
  };
  mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(draft, null, 2)}\n`, "utf8");
  renameSync(temporary, file);
  return res.status(200).json(draft);
}
