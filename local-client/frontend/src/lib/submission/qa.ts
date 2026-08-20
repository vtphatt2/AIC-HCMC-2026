import type { SubmissionEntry } from "@/types";
import type { CandidateGroup } from "./types";

// QA: <video>,<frame>,<answer> — same independent-row shape as KIS, plus
// the session's shared answer text on every row.
export function toGroups(entries: SubmissionEntry[]): CandidateGroup[] {
  return entries.map((e) => ({ videoId: e.videoId, frameIds: [e.frame] }));
}

export function buildRows(groups: CandidateGroup[], answer: string): (string | number)[][] {
  return groups.map((g) => [g.videoId, g.frameIds[0], answer]);
}
