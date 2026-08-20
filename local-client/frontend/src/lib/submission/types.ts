// The minimal shape every query-type module's row/validation logic works
// with — deliberately stripped of SubmissionEntry's storage/UI fields (id,
// imageUrl, fps, youtubeId, groupIndex) so each type's rules are easy to
// reason about and test in isolation.
export interface CandidateGroup {
  videoId: string;
  frameIds: number[];
}
