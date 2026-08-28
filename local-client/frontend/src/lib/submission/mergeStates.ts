export function mergeFetchedSubmissionStates<T>(
  sessions: ReadonlyArray<{ session: string }>,
  fetched: ReadonlyArray<T | null | undefined>,
  previous: Readonly<Record<string, T>>,
): Record<string, T> {
  const next: Record<string, T> = {};
  sessions.forEach((summary, index) => {
    const state = fetched[index] ?? previous[summary.session];
    if (state !== undefined && state !== null) {
      next[summary.session] = state;
    }
  });
  return next;
}
