export interface FillableSubmissionRow {
  videoId: string;
  frames: number[];
  answer?: string;
}

interface Anchor {
  videoId: string;
  frame: number;
  attempt: number;
}

function neighborFrame(anchor: Anchor, step: number): number {
  const distance = (Math.floor(anchor.attempt / 2) + 1) * step;
  const direction = anchor.attempt % 2 === 0 ? -1 : 1;
  anchor.attempt += 1;
  return anchor.frame + direction * distance;
}

/**
 * Preserve the existing KIS rank order and append nearby candidates until
 * `targetRows` is reached. Each distinct seed gets one result per round, so
 * a session with several strong candidates spends the remaining row budget
 * evenly instead of exhausting the neighborhood around rank 1 first.
 */
export function fillKisRows<T extends FillableSubmissionRow>(
  rows: readonly T[],
  targetRows: number = 100,
  step: number = 15,
): T[] {
  if (!Number.isInteger(targetRows) || targetRows < 0) {
    throw new Error("Filler target must be a non-negative integer.");
  }
  if (!Number.isInteger(step) || step <= 0) {
    throw new Error("Filler step must be a positive integer.");
  }

  const result = rows.map((row) => ({ ...row, frames: [...row.frames] })) as T[];
  if (result.length === 0 || result.length >= targetRows) return result;

  const seen = new Set<string>();
  const anchors: Anchor[] = [];
  for (const row of rows) {
    const frame = row.frames[0];
    if (!Number.isInteger(frame) || frame < 0 || !row.videoId) continue;
    const key = `${row.videoId}:${frame}`;
    seen.add(key);
    if (!anchors.some((anchor) => anchor.videoId === row.videoId && anchor.frame === frame)) {
      anchors.push({ videoId: row.videoId, frame, attempt: 0 });
    }
  }
  if (anchors.length === 0) return result;

  while (result.length < targetRows) {
    for (const anchor of anchors) {
      let frame: number;
      let key: string;
      do {
        frame = neighborFrame(anchor, step);
        key = `${anchor.videoId}:${frame}`;
      } while (frame < 0 || seen.has(key));

      seen.add(key);
      result.push({ videoId: anchor.videoId, frames: [frame] } as T);
      if (result.length === targetRows) break;
    }
  }

  return result;
}
