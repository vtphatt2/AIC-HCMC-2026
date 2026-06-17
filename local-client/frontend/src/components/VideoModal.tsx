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
  const frameImageUrl = result.frame_image_url.startsWith("http")
    ? result.frame_image_url
    : apiUrl(result.frame_image_url);

  // Live playback position — updated by the polling interval below
  const [currentTimeSec, setCurrentTimeSec] = useState(startSeconds);
  const [playbackStarted, setPlaybackStarted] = useState(false);
  const currentFrame = Math.floor(currentTimeSec * fps);

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Mount the YouTube player
  useEffect(() => {
    function createPlayer() {
      if (!youtubeId || !document.getElementById(PLAYER_DOM_ID)) return;
      playerRef.current = new window.YT.Player(PLAYER_DOM_ID, {
        videoId: youtubeId,
        playerVars: { autoplay: 1, rel: 0, start: Math.floor(startSeconds) },
        events: {
          onReady: (event: any) => {
            event.target.playVideo();
          },
          onStateChange: (event: any) => {
            if (event.data === window.YT.PlayerState.PLAYING) {
              setPlaybackStarted(true);
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
      if (playerRef.current) {
        playerRef.current.destroy();
        playerRef.current = null;
      }
    };
  }, [startSeconds, youtubeId]);

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
          className="absolute -top-8 right-0 text-slate-400 hover:text-white text-sm transition"
        >
          Close (Esc)
        </button>

        {/* 16:9 aspect ratio wrapper */}
        <div className="relative w-full bg-black" style={{ paddingTop: "56.25%" }}>
          {!playbackStarted && (
            <div className="absolute inset-0">
              <img
                src={frameImageUrl}
                alt=""
                className="w-full h-full object-contain"
              />
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
                playbackStarted ? "opacity-100" : "opacity-0"
              }`}
            >
              <div id={PLAYER_DOM_ID} className="w-full h-full" />
            </div>
          )}
        </div>

        {/* Live info bar — updates as the video plays */}
        <div className="mt-2 flex gap-4 text-sm text-slate-400 font-mono">
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
