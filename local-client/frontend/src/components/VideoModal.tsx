import { useEffect, useRef, useState } from "react";
import type { SearchResult } from "@/types";
import { apiUrl } from "@/lib/api";

interface Props {
  result: SearchResult;
  onClose: () => void;
}

declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

const PLAYER_DOM_ID = "yt-player-container";

export default function VideoModal({ result, onClose }: Props) {
  const playerRef = useRef<any>(null);
  const youtubeId = result.youtube_id || "";
  const startSeconds = result.timestamp_ms / 1000;
  const fps = result.fps;
  const frameImageUrl = result.frame_image_url
    ? result.frame_image_url.startsWith("http")
      ? result.frame_image_url
      : apiUrl(result.frame_image_url)
    : "";

  // Live playback position — updated by the polling interval below
  const [currentTimeSec, setCurrentTimeSec] = useState(startSeconds);
  const [playerMounted, setPlayerMounted] = useState(false);
  const currentFrame = Math.floor(currentTimeSec * fps);

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Mount the YouTube player (only when youtubeId changes to prevent redundant iframe rebuilds)
  useEffect(() => {
    let player: any = null;
    let cancelled = false;
    setPlayerMounted(false);

    function createPlayer() {
      if (cancelled || !youtubeId || !document.getElementById(PLAYER_DOM_ID)) return;

      player = new window.YT.Player(PLAYER_DOM_ID, {
        videoId: youtubeId,
        playerVars: {
          autoplay: 1,
          rel: 0,
          start: Math.floor(startSeconds),
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
            event.target.playVideo();
            event.target.seekTo(startSeconds, true);
            setPlayerMounted(true);
          },
          onStateChange: (event: any) => {
            if (!cancelled && event.data === window.YT.PlayerState.PLAYING) {
              setPlayerMounted(true);
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

  // Seek without rebuilding the iframe when only the target timestamp changes.
  useEffect(() => {
    if (playerRef.current && typeof playerRef.current.seekTo === "function") {
      playerRef.current.seekTo(startSeconds, true);
      playerRef.current.playVideo();
    }
    setCurrentTimeSec(startSeconds);
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
        className="relative w-full max-w-4xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="font-retro absolute -top-8 right-0 text-stone-300 hover:text-orange-400 text-xs uppercase tracking-wide transition"
        >
          ✕ Close (Esc)
        </button>

        {/* 16:9 aspect ratio wrapper */}
        <div className="relative w-full bg-black border-2 border-stone-700" style={{ paddingTop: "56.25%" }}>
          {!playerMounted && (
            <div className="absolute inset-0">
              {frameImageUrl && (
                <img
                  src={frameImageUrl}
                  alt=""
                  className="w-full h-full object-contain"
                />
              )}
              {youtubeId && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20 text-sm text-white">
                  Loading video…
                </div>
              )}
            </div>
          )}
          {youtubeId && (
            <div
              className={`absolute inset-0 transition-opacity ${
                playerMounted ? "opacity-100" : "opacity-0"
              }`}
            >
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
    </div>
  );
}
