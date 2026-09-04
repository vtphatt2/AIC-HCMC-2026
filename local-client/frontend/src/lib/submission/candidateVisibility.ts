export const COLLAPSED_CANDIDATE_LIMIT = 5;

export function visibleCandidateRows<T>(
  rows: readonly T[],
  expanded: boolean,
  limit: number = COLLAPSED_CANDIDATE_LIMIT,
): readonly T[] {
  if (expanded || rows.length <= limit) return rows;
  return rows.slice(0, limit);
}
