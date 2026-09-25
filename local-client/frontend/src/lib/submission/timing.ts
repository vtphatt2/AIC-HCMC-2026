import type { SubmissionRow } from '../../types';
import { frameAtSubmissionMilliseconds, submissionMilliseconds, type FrameTimeline } from '../playback';

export function resolveNRow(row: SubmissionRow, timeline: FrameTimeline, sourceFrameInput = false): SubmissionRow {
  if (!row.videoId.startsWith('N')) return { ...row, unit: 'frames' };
  const sourceFrames = row.frames.map(position => sourceFrameInput || row.unit === 'frames'
    ? position : frameAtSubmissionMilliseconds(timeline, position));
  const frames = sourceFrames.map(frame => submissionMilliseconds(timeline, frame));
  return { ...row, unit: 'milliseconds', frames, sourceFrames, timingStatus: 'verified', timingError: undefined };
}
