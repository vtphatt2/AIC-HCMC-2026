import type { SubmissionEntry } from "@/types";
import { sortByRowOrder, type CandidateGroup } from "./types";

// KIS: <video>,<frame> — every entry is its own independent row, never
// clustered into candidates. rowOrder ranks the rows (row key = entry.id).
export function toGroups(entries: SubmissionEntry[], rowOrder: string[]): CandidateGroup[] {
  return sortByRowOrder(entries, rowOrder, (e) => e.id).map((e) => ({ videoId: e.videoId, frameIds: [e.frame] }));
}

export function buildRows(groups: CandidateGroup[]): (string | number)[][] {
  return groups.map((g) => [g.videoId, g.frameIds[0]]);
}
