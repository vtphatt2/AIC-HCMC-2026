import type { SubmissionEntry } from "@/types";
import { sortByRowOrder, type CandidateGroup } from "./types";

export function groupKey(groupIndex: number): string {
  return `g${groupIndex}`;
}

// Groups by groupIndex (arbitrary insertion order — callers sort by rowOrder
// afterward), frames ascending within each group.
function byGroupIndex(entries: SubmissionEntry[]): SubmissionEntry[][] {
  const groups = new Map<number, SubmissionEntry[]>();
  for (const e of entries) {
    if (!groups.has(e.groupIndex)) groups.set(e.groupIndex, []);
    groups.get(e.groupIndex)!.push(e);
  }
  return Array.from(groups.values()).map((es) => [...es].sort((a, b) => a.frame - b.frame));
}

// TRAKE: <video>,<frame_1>,...,<frame_N> — one row per candidate. Frame
// order within a candidate is guaranteed ascending by frame id, since one
// video's events happen in a single forward timeline; that's the correct
// event order regardless of the order frames were added in. rowOrder ranks
// the candidates (row key = groupKey(groupIndex)).
export function toGroups(entries: SubmissionEntry[], rowOrder: string[]): CandidateGroup[] {
  const groups = sortByRowOrder(byGroupIndex(entries), rowOrder, (es) => groupKey(es[0].groupIndex));
  return groups.map((es) => ({ videoId: es[0].videoId, frameIds: es.map((e) => e.frame) }));
}

export function buildRows(groups: CandidateGroup[]): (string | number)[][] {
  return groups.map((g) => [g.videoId, ...g.frameIds]);
}

// Full entries, grouped and ordered exactly as the export sees them
// (candidates by rowOrder, frames ascending within each) — for UI display,
// where the thumbnail/imageUrl/id fields toGroups() strips are still needed.
export function sortedGroups(entries: SubmissionEntry[], rowOrder: string[]): SubmissionEntry[][] {
  return sortByRowOrder(byGroupIndex(entries), rowOrder, (es) => groupKey(es[0].groupIndex));
}

// Organizer rules `toGroups`/`buildRows` can't catch on their own:
// - every candidate's frames must come from the same video (toGroups
//   silently keeps only the first video id per candidate, so this has to be
//   checked against the raw entries before that collapse happens)
// - every candidate must have the same frame count (the query's event count)
// Both are surfaced as warnings, not hard errors — a half-built candidate
// mid-edit shouldn't be treated as broken. Candidate numbers use the
// entries' own groupIndex, not array position — those diverge once a
// candidate has been emptied out by moving all its frames elsewhere.
export function validate(entries: SubmissionEntry[]): { videoMismatch: number[]; sizeMismatch: number[] } {
  const groups = byGroupIndex(entries);

  const videoMismatch = groups
    .map((es) => ({ candidate: es[0].groupIndex + 1, distinctVideos: new Set(es.map((e) => e.videoId)).size }))
    .filter((g) => g.distinctVideos > 1)
    .map((g) => g.candidate);

  const sizes = new Set(groups.map((es) => es.length));
  const sizeMismatch = sizes.size > 1 ? Array.from(sizes).sort((a, b) => a - b) : [];

  return { videoMismatch, sizeMismatch };
}
