import { useEffect, useRef, useState } from "react";
import type { SearchResult, TranscriptSegment } from "@/types";
import { apiUrl, fetchTranscript } from "@/lib/api";

interface Props {
  result: SearchResult;
  onClose: () => void;
  showTranscript: boolean;
  onToggleTranscript: () => void;
}

declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

const PLAYER_DOM_ID = "yt-player-container";

export default function VideoModal({ result, onClose, showTranscript, onToggleTranscript }: Props) {
  const playerRef = useRef<any>(null);
  const youtubeId = result.youtube_id || "";
  const startSeconds = result.timestamp_ms / 1000;
  const fps = result.fps;
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
  // True once playback has actually started (by Enter, see below) — not
  // once the player is merely ready. Until then the frame image stays the
  // visible layer: opening the modal shows the frame, paused, on purpose.
  const [hasStartedPlaying, setHasStartedPlaying] = useState(false);
  const wantsPlayRef = useRef(false);
  const currentFrame = Math.floor(currentTimeSec * fps);

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

      const player = playerRef.current;
      const ready = player && typeof player.playVideo === "function";

      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (!ready) {
          wantsPlayRef.current = true; // player not ready yet — play as soon as it is
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
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        if (!ready || typeof player.seekTo !== "function") return;
        const now = typeof player.getCurrentTime === "function" ? player.getCurrentTime() : startSeconds;
        const delta = e.key === "ArrowLeft" ? -SEEK_STEP_SECONDS : SEEK_STEP_SECONDS;
        player.seekTo(Math.max(0, now + delta), true);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, startSeconds]);

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

  // Mount the YouTube player (only when youtubeId changes to prevent redundant iframe rebuilds)
  useEffect(() => {
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
          // Enter (or the state-change fallback below) starts playback.
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
  }, [youtubeId]);

  // Seek without rebuilding the iframe when only the target timestamp
  // changes — stays paused (a newly-shown frame should also start paused).
  useEffect(() => {
    if (playerRef.current && typeof playerRef.current.seekTo === "function") {
      playerRef.current.seekTo(startSeconds, true);
      playerRef.current.pauseVideo();
    }
    setCurrentTimeSec(startSeconds);
    setHasStartedPlaying(false);
  }, [startSeconds]);

  // Poll the player every 100ms to get the live playback position.
  // This updates frame number and timestamp whenever the video plays or the user scrubs.
  useEffect(() => {
    const interval = setInterval(() => {
      if (playerRef.current && typeof playerRef.current.getCurrentTime === "function") {
        setCurrentTimeSec(playerRef.current.getCurrentTime());
      }
    }, 100);
    return () => clearInterval(interval);
  }, []);

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
                {youtubeId && (
                  <div className="font-retro absolute bottom-2 right-2 bg-stone-900/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wide">
                    ▸ Enter play · ←/→ seek · Esc close
                  </div>
                )}
              </div>
            )}
            {youtubeId && (
              <div className="absolute inset-0 z-0">
                <div id={PLAYER_DOM_ID} className="w-full h-full" />
              </div>
            )}
          </div>

          {/* Live info bar — updates as the video plays */}
          <div className="mt-2 flex gap-4 text-sm text-stone-400 font-mono">
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
