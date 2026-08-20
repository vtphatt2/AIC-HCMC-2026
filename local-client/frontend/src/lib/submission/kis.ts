import type { SubmissionEntry } from "@/types";
import type { CandidateGroup } from "./types";

// KIS: <video>,<frame> — every entry is its own independent row, never
// clustered into candidates.
export function toGroups(entries: SubmissionEntry[]): CandidateGroup[] {
  return entries.map((e) => ({ videoId: e.videoId, frameIds: [e.frame] }));
}

export function buildRows(groups: CandidateGroup[]): (string | number)[][] {
  return groups.map((g) => [g.videoId, g.frameIds[0]]);
}
