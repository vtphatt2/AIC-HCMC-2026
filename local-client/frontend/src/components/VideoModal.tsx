import { useCallback, useEffect, useRef, useState } from "react";
import type { SearchResult, SubmissionState, TranscriptSegment } from "@/types";
import { apiUrl, fetchTranscript } from "@/lib/api";
import { addSubmissionRowFrame, fetchSubmission } from "@/lib/submission";
import { SUBMISSION_SESSION_KEY } from "@/components/SubmissionPanel";

interface Props {
  result: SearchResult;
  onClose: () => void;
  showTranscript: boolean;
  onToggleTranscript: () => void;
  onOpenSubmissionPanel: () => void;
  // Set when this modal was opened to review one specific existing entry
  // (e.g. from the /submissions dashboard) — "Add to submission" then
  // targets that entry's own session/candidate instead of whatever this
  // browser has picked in SubmissionPanel (localStorage), which may be a
  // different session entirely.
  overrideSession?: string;
  overrideRowIndex?: number;
}

declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

const PLAYER_DOM_ID = "yt-player-container";

export default function VideoModal({
  result,
  onClose,
  showTranscript,
  onToggleTranscript,
  onOpenSubmissionPanel,
  overrideSession,
  overrideRowIndex,
}: Props) {
  const playerRef = useRef<any>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const youtubeId = result.youtube_id || "";
  const startSeconds = result.timestamp_ms / 1000;
  const fps = result.fps;

  // YouTube is tried first whenever a video has a known youtube_id — but
  // "known youtube_id" doesn't guarantee a *working* embed: the uploader may
  // have disabled embedding, or the video may have gone private/been taken
  // down since ingestion. onError below catches that and falls back to
  // streaming straight from the organizer's remote ZIP via local-backend's
  // Range proxy (see app/services/remote_zip_proxy.py) — the same fallback
  // used outright for videos with no youtube_id at all (e.g. ones ingested
  // straight from the organizer zips, which carry no YouTube mapping).
  const zipVideoUrl = result.video_id ? apiUrl(`/api/zip-video/${encodeURIComponent(result.video_id)}`) : "";
  const [zipVideoFailed, setZipVideoFailed] = useState(false);
  const [youtubeFailed, setYoutubeFailed] = useState(false);
  const useYoutube = Boolean(youtubeId) && !youtubeFailed;
  const useZipVideo = !useYoutube && Boolean(zipVideoUrl) && !zipVideoFailed;
  // Show the SHARP frame (same one the results grid shows), with the fast
  // low-res preview as a blurred stand-in underneath until it loads — a
  // blur-up, same as ResultCard. The grid already loaded the sharp image
  // for this frame, so it's usually browser-cached and appears instantly;
  // the blurred preview only covers the rare cache-miss so there's never a
  // black screen while it (or the player) loads.
  const resolve = (u?: string) => (u ? (u.startsWith("http") ? u : apiUrl(u)) : "");
  const sharpUrl = resolve(result.frame_image_url);
  const previewUrl = resolve(result.frame_preview_url);
  const [sharpLoaded, setSharpLoaded] = useState(false);

  // Live playback position — updated by the polling interval below
  const [currentTimeSec, setCurrentTimeSec] = useState(startSeconds);
  // True once playback has actually started (by click or keyboard) — not
  // once the player is merely ready. Until then the frame image stays the
  // visible layer: opening the modal shows the frame, paused, on purpose.
  const [hasStartedPlaying, setHasStartedPlaying] = useState(false);
  const wantsPlayRef = useRef(false);
  const currentFrame = Math.floor(currentTimeSec * fps);

  // "Add to submission" — targets whichever session this browser last picked
  // in the SubmissionPanel (localStorage.SUBMISSION_SESSION_KEY). No session
  // yet -> open the panel to pick/create one instead of adding blindly.
  const [addStatus, setAddStatus] = useState<"idle" | "added" | "error">("idle");
  const [sessionInfo, setSessionInfo] = useState<SubmissionState | null>(null);

  const refreshSessionInfo = useCallback(() => {
    const session = overrideSession || window.localStorage.getItem(SUBMISSION_SESSION_KEY);
    if (!session) { setSessionInfo(null); return; }
    fetchSubmission(session).then(setSessionInfo).catch(() => setSessionInfo(null));
  }, [overrideSession]);

  // Shows which session/candidate "Add to submission" targets *before* the
  // click, not just after — matters most for TRAKE, where the target
  // candidate changes as the session is worked on by any teammate.
  useEffect(() => {
    refreshSessionInfo();
  }, [refreshSessionInfo]);

  const handleAddToSubmission = useCallback(async () => {
    const session = overrideSession || window.localStorage.getItem(SUBMISSION_SESSION_KEY);
    if (!session) {
      onOpenSubmissionPanel();
      return;
    }
    try {
      // The add response already IS the updated state — no need for a
      // second round trip just to re-fetch what we already have.
      const updated = await addSubmissionRowFrame(session, result.video_id, currentFrame, overrideRowIndex);
      setSessionInfo(updated);
      setAddStatus("added");
      setTimeout(() => setAddStatus("idle"), 1500);
    } catch (err: any) {
      setAddStatus("error");
      window.alert(err.message || "Failed to add to submission");
      setTimeout(() => setAddStatus("idle"), 1500);
    }
  }, [result.video_id, currentFrame, onOpenSubmissionPanel, overrideSession, overrideRowIndex]);

  // New result (possibly a different video) — give the zip source a fresh
  // try and reset to the paused-on-frame-image state.
  useEffect(() => {
    setZipVideoFailed(false);
    setYoutubeFailed(false);
    setHasStartedPlaying(false);
    wantsPlayRef.current = false;
  }, [result.video_id]);

  const toggleNativeVideoPlayback = useCallback(() => {
    const el = videoElRef.current;
    if (!el) return;
    if (el.paused) {
      if (!hasStartedPlaying) el.currentTime = startSeconds;
      el.play();
    } else {
      el.pause();
    }
  }, [hasStartedPlaying, startSeconds]);

  const togglePlayback = useCallback(() => {
    const player = playerRef.current;
    if (!player || typeof player.playVideo !== "function") {
      wantsPlayRef.current = true;
      return;
    }
    const YT = window.YT;
    const state = typeof player.getPlayerState === "function" ? player.getPlayerState() : -1;
    const neverStarted = !YT || state === YT.PlayerState.UNSTARTED || state === YT.PlayerState.CUED;
    if (neverStarted) {
      if (typeof player.seekTo === "function") player.seekTo(startSeconds, true);
      player.playVideo();
    } else if (state === YT.PlayerState.PLAYING) {
      player.pauseVideo();
    } else {
      player.playVideo();
    }
  }, [startSeconds]);

  const handleTogglePlayback = useCallback(() => {
    if (useZipVideo) {
      toggleNativeVideoPlayback();
    } else {
      togglePlayback();
    }
  }, [useZipVideo, toggleNativeVideoPlayback, togglePlayback]);

  // Transcript panel state — fetched once per video, no search involved
  // (see app/services/transcript_index.py): just "what's being said now."
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const activeSegmentRef = useRef<HTMLDivElement | null>(null);
  const transcriptPanelRef = useRef<HTMLDivElement | null>(null);

  // Full keyboard control at the window level, so the user never has to
  // click into the iframe to seek — which matters because once the
  // cross-origin YouTube iframe has focus, keydowns go into it and our
  // parent listener stops receiving them (Escape would stop working, the
  // exact bug reported). Handling everything here means clicking the iframe
  // is never necessary:
  //   Enter / Space — first press starts playback seeking to the target
  //                   frame; after that, toggles play/pause. Never re-seeks
  //                   back once the user has played past the start point.
  //   ← / →          — seek back / forward by SEEK_STEP_SECONDS.
  //   Escape         — close.
  // Reads live player state (not React state) so it never runs stale.
  useEffect(() => {
    const SEEK_STEP_SECONDS = 5;

    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }

      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleTogglePlayback();
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const delta = e.key === "ArrowLeft" ? -SEEK_STEP_SECONDS : SEEK_STEP_SECONDS;

        if (useZipVideo) {
          const el = videoElRef.current;
          if (el) el.currentTime = Math.max(0, el.currentTime + delta);
          return;
        }

        const player = playerRef.current;
        const ready = player && typeof player.playVideo === "function";
        if (!ready || typeof player.seekTo !== "function") return;
        const now = typeof player.getCurrentTime === "function" ? player.getCurrentTime() : startSeconds;
        player.seekTo(Math.max(0, now + delta), true);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, startSeconds, useZipVideo, handleTogglePlayback]);

  // Fetch the full transcript once per video (not per frame) — only when
  // the panel is actually shown, and only once (result.video_id is stable
  // for the lifetime of one modal instance).
  useEffect(() => {
    if (!showTranscript || !result.video_id) return;
    let cancelled = false;
    setSegments([]);
    setTranscriptError(null);
    fetchTranscript(result.video_id)
      .then((res) => { if (!cancelled) setSegments(res.segments); })
      .catch((err) => { if (!cancelled) setTranscriptError(err.message || "No transcript available"); });
    return () => { cancelled = true; };
  }, [showTranscript, result.video_id]);

  const activeSegmentIndex = segments.findIndex(
    (s) => currentTimeSec * 1000 >= s.start_ms && currentTimeSec * 1000 < s.end_ms,
  );

  // Auto-reveal the active line only when it's actually scrolled out of the
  // panel's view — otherwise the 100ms playback poll would keep yanking the
  // panel back and fight the user scrolling it by hand.
  useEffect(() => {
    const panel = transcriptPanelRef.current;
    const seg = activeSegmentRef.current;
    if (!panel || !seg) return;
    const p = panel.getBoundingClientRect();
    const s = seg.getBoundingClientRect();
    if (s.top < p.top || s.bottom > p.bottom) {
      seg.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [activeSegmentIndex]);

  // Mount the YouTube player — only once the zip source isn't in play (either
  // never available for this video, or it just failed) so the common case
  // never touches the YouTube iframe API at all.
  useEffect(() => {
    if (!useYoutube) return;
    let player: any = null;
    let cancelled = false;
    setHasStartedPlaying(false);
    wantsPlayRef.current = false;

    function createPlayer() {
      if (cancelled || !youtubeId || !document.getElementById(PLAYER_DOM_ID)) return;

      player = new window.YT.Player(PLAYER_DOM_ID, {
        videoId: youtubeId,
        playerVars: {
          // No autoplay — opens paused on the frame image on purpose; only
          // The Play button or keyboard starts playback.
          // No `start` either — it only accepts whole seconds (rounds off
          // the exact frame); the real position comes from seekTo() below,
          // which takes fractional seconds.
          autoplay: 0,
          rel: 0,
          playsinline: 1,
          origin: typeof window !== "undefined" ? window.location.origin : undefined
        },
        events: {
          onReady: (event: any) => {
            if (cancelled) {
              event.target.destroy();
              return;
            }
            playerRef.current = event.target;
            event.target.seekTo(startSeconds, true);
            event.target.pauseVideo();
            if (wantsPlayRef.current) {
              // Re-seek in case the first seekTo landed before the video
              // was buffered enough for it to stick.
              event.target.seekTo(startSeconds, true);
              event.target.playVideo();
            }
          },
          onStateChange: (event: any) => {
            if (!cancelled && event.data === window.YT.PlayerState.PLAYING) {
              setHasStartedPlaying(true);
            }
          },
          // error codes: 2 invalid param, 5 HTML5 player error, 100 video
          // removed/private, 101/150 embedding disabled by the uploader —
          // all mean this embed can never play, not a transient hiccup.
          onError: () => {
            if (!cancelled) setYoutubeFailed(true);
          },
        },
      });
    }

    if (window.YT && window.YT.Player) {
      createPlayer();
    } else {
      if (!document.getElementById("yt-iframe-api")) {
        const tag = document.createElement("script");
        tag.id = "yt-iframe-api";
        tag.src = "https://www.youtube.com/iframe_api";
        document.head.appendChild(tag);
      }
      window.onYouTubeIframeAPIReady = createPlayer;
    }

    return () => {
      cancelled = true;
      if (player) {
        player.destroy();
      }
      playerRef.current = null;
    };
  }, [youtubeId, useYoutube]);

  // Seek without rebuilding the player when only the target timestamp
  // changes — stays paused (a newly-shown frame should also start paused).
  useEffect(() => {
    if (useZipVideo) {
      if (videoElRef.current) videoElRef.current.currentTime = startSeconds;
    } else if (playerRef.current && typeof playerRef.current.seekTo === "function") {
      playerRef.current.seekTo(startSeconds, true);
      playerRef.current.pauseVideo();
    }
    setCurrentTimeSec(startSeconds);
    setHasStartedPlaying(false);
  }, [startSeconds, useZipVideo]);

  // Poll the player every 100ms to get the live playback position.
  // This updates frame number and timestamp whenever the video plays or the user scrubs.
  useEffect(() => {
    const interval = setInterval(() => {
      if (useZipVideo) {
        if (videoElRef.current) setCurrentTimeSec(videoElRef.current.currentTime);
      } else if (playerRef.current && typeof playerRef.current.getCurrentTime === "function") {
        setCurrentTimeSec(playerRef.current.getCurrentTime());
      }
    }, 100);
    return () => clearInterval(interval);
  }, [useZipVideo]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className={`relative w-full ${showTranscript ? "max-w-6xl" : "max-w-4xl"} flex flex-col lg:flex-row gap-3 transition-[max-width] duration-150`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="relative flex-1 min-w-0">
          {/* Close + transcript toggle */}
          <div className="absolute -top-8 right-0 flex items-center gap-3">
            <button
              onClick={onToggleTranscript}
              className="font-retro text-stone-300 hover:text-orange-400 text-xs uppercase tracking-wide transition"
              title="Toggle transcript panel (or type /transcript on|off in the command panel)"
            >
              {showTranscript ? "▾ Hide transcript" : "▸ Show transcript"}
            </button>
            <button
              onClick={onClose}
              className="font-retro text-stone-300 hover:text-orange-400 text-xs uppercase tracking-wide transition"
            >
              ✕ Close (Esc)
            </button>
          </div>

          {/* 16:9 aspect ratio wrapper */}
          <div className="relative w-full bg-black border-2 border-stone-700" style={{ paddingTop: "56.25%" }}>
            {/* Frame placeholder — explicit z-10, stays the visible layer
                until playback actually starts (opens paused, on purpose).
                Blur-up: blurred low-res preview underneath, sharp frame on
                top fading in on load — same treatment as the grid cards. */}
            {!hasStartedPlaying && (
              <div className="absolute inset-0 z-10 bg-black">
                {previewUrl && !sharpLoaded && (
                  <img
                    src={previewUrl}
                    alt=""
                    aria-hidden="true"
                    className="absolute inset-0 w-full h-full object-contain blur-sm scale-105"
                  />
                )}
                {sharpUrl && (
                  <img
                    src={sharpUrl}
                    alt=""
                    onLoad={() => setSharpLoaded(true)}
                    className={`relative w-full h-full object-contain transition-opacity duration-200 ${
                      sharpLoaded ? "opacity-100" : "opacity-0"
                    }`}
                  />
                )}
                {(useZipVideo || useYoutube) && (
                  <button
                    type="button"
                    onClick={handleTogglePlayback}
                    aria-label="Play video from selected frame"
                    title="Play video"
                    className="absolute inset-0 z-20 m-auto h-16 w-16 rounded-full border-2 border-white/80 bg-black/65 text-3xl text-white transition hover:scale-105 hover:bg-orange-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-orange-400"
                  >
                    <span aria-hidden="true" className="ml-1">▶</span>
                  </button>
                )}
                {(useZipVideo || useYoutube) && (
                  <div className="font-retro absolute bottom-2 right-2 bg-stone-900/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wide">
                    ▸ Click/Enter play · ←/→ seek · Esc close
                  </div>
                )}
              </div>
            )}
            {useZipVideo && (
              <div className="absolute inset-0 z-0">
                <video
                  ref={videoElRef}
                  src={zipVideoUrl}
                  controls
                  playsInline
                  preload="metadata"
                  className="w-full h-full"
                  onLoadedMetadata={() => {
                    if (videoElRef.current) videoElRef.current.currentTime = startSeconds;
                  }}
                  onPlay={() => setHasStartedPlaying(true)}
                  onError={() => setZipVideoFailed(true)}
                />
              </div>
            )}
            {useYoutube && (
              <div className="absolute inset-0 z-0">
                <div id={PLAYER_DOM_ID} className="w-full h-full" />
              </div>
            )}
          </div>

          {/* Live info bar — updates as the video plays */}
          <div className="mt-2 flex items-center gap-4 text-sm text-stone-400 font-mono">
            <span>{result.video_id}</span>
            <span>·</span>
            <span>
              Frame <span className="text-white font-bold">{currentFrame}</span>
            </span>
            <span>·</span>
            <span>
              <span className="text-white font-bold">{currentTimeSec.toFixed(2)}</span>s
            </span>
            <span>·</span>
            <span>{fps} fps</span>
            <div className="ml-auto flex items-center gap-2">
              {sessionInfo && (
                <span className="text-xs text-stone-500 normal-case">
                  → {sessionInfo.session}
                  {sessionInfo.queryType === "trake" && ` · candidate ${sessionInfo.draftRowIndex + 1}`}
                </span>
              )}
              <button
                type="button"
                onClick={handleAddToSubmission}
                className="font-retro text-xs uppercase tracking-wide border-2 border-stone-500 rounded px-2 py-1 text-stone-300 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition"
              >
                {addStatus === "added" ? "✓ Added" : addStatus === "error" ? "✕ Failed" : "＋ Add to submission"}
              </button>
            </div>
          </div>
        </div>

        {/* Transcript panel — plain "what's being said now" from timing, no search/highlighting yet */}
        {showTranscript && (
          <div
            ref={transcriptPanelRef}
            className="w-full lg:w-80 shrink-0 max-h-[70vh] lg:max-h-[calc(56.25vw*0.5625+3rem)] overflow-y-auto bg-stone-900 border-2 border-stone-700 rounded p-3 space-y-1"
          >
            {transcriptError && (
              <p className="text-xs text-stone-500 italic">{transcriptError}</p>
            )}
            {!transcriptError && segments.length === 0 && (
              <p className="text-xs text-stone-500 italic">Loading transcript…</p>
            )}
            {segments.map((seg, i) => (
              <div
                key={i}
                ref={i === activeSegmentIndex ? activeSegmentRef : undefined}
                className={`text-sm rounded px-2 py-1 transition-colors ${
                  i === activeSegmentIndex
                    ? "bg-orange-700/30 text-orange-200 font-medium"
                    : "text-stone-400"
                }`}
              >
                {seg.text}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
