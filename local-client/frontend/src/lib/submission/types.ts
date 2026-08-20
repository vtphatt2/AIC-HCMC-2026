// The minimal shape every query-type module's row/validation logic works
// with — deliberately stripped of SubmissionEntry's storage/UI fields (id,
// imageUrl, fps, youtubeId, groupIndex) so each type's rules are easy to
// reason about and test in isolation.
export interface CandidateGroup {
  videoId: string;
  frameIds: number[];
}

// Generic row-ranking sort — not a query-type rule, just shared infra: sort
// `items` by their position in `rowOrder` (a list of row keys), unknown keys
// (not yet synced server-side) fall to the end in their original order.
export function sortByRowOrder<T>(items: T[], rowOrder: string[], keyOf: (item: T) => string): T[] {
  const pos = new Map(rowOrder.map((k, i) => [k, i]));
  return [...items].sort((a, b) => (pos.get(keyOf(a)) ?? Infinity) - (pos.get(keyOf(b)) ?? Infinity));
}

// Drag-and-drop reorder: move draggedKey to where targetKey currently sits.
// Direction matters — dragging downward (dragged started above target) must
// land *after* the target, dragging upward must land *before* it, or a
// downward drag keeps snapping back above whatever it's dropped on.
export function reorderKeys(current: string[], draggedKey: string, targetKey: string): string[] {
  if (draggedKey === targetKey) return current;
  const draggingDown = current.indexOf(draggedKey) < current.indexOf(targetKey);
  const filtered = current.filter((k) => k !== draggedKey);
  const targetIdx = filtered.indexOf(targetKey);
  filtered.splice(draggingDown ? targetIdx + 1 : targetIdx, 0, draggedKey);
  return filtered;
}
