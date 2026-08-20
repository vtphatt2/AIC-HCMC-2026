import type { SubmissionEntry } from "@/types";
import { sortByRowOrder, type CandidateGroup } from "./types";

// QA: <video>,<frame>,<answer> — same independent-row shape as KIS, plus
// the session's shared answer text on every row. rowOrder ranks the rows
// (row key = entry.id).
export function toGroups(entries: SubmissionEntry[], rowOrder: string[]): CandidateGroup[] {
  return sortByRowOrder(entries, rowOrder, (e) => e.id).map((e) => ({ videoId: e.videoId, frameIds: [e.frame] }));
}

export function buildRows(groups: CandidateGroup[], answer: string): (string | number)[][] {
  return groups.map((g) => [g.videoId, g.frameIds[0], answer]);
}
