import { useState } from "react";
import type { SearchResult } from "@/types";
import { fetchVideoById } from "@/lib/api";

interface VideoLookupProps {
  onOpen: (result: SearchResult) => void;
}

export default function VideoLookup({ onOpen }: VideoLookupProps) {
  const [jumpVideoId, setJumpVideoId] = useState("");
  const [jumpLoading, setJumpLoading] = useState(false);

  async function handleJumpToVideo() {
    const videoId = jumpVideoId.trim();
    if (!videoId) return;
    setJumpLoading(true);
    try {
      onOpen(await fetchVideoById(videoId));
      setJumpVideoId("");
    } catch (err: any) {
      window.alert(err.message || `Video "${videoId}" not found`);
    } finally {
      setJumpLoading(false);
    }
  }

  return (
    <>
      <input
        value={jumpVideoId}
        onChange={(e) => setJumpVideoId(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") handleJumpToVideo(); }}
        placeholder="video_id…"
        disabled={jumpLoading}
        title="Jump straight to a video by its exact video_id"
        className="h-7 w-28 rounded border-2 border-stone-800 dark:border-stone-500 px-2 text-xs font-mono bg-cream-card dark:bg-stone-800 text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600"
      />
      <button
        onClick={handleJumpToVideo}
        disabled={jumpLoading || !jumpVideoId.trim()}
        className="h-7 flex items-center justify-center rounded border-2 border-stone-800 dark:border-stone-500 px-2 text-stone-600 dark:text-stone-300 hover:text-orange-700 dark:hover:text-orange-400 hover:border-orange-700 dark:hover:border-orange-400 transition disabled:opacity-40"
        title="Jump to video_id"
      >
        <span className="text-xs font-bold">{jumpLoading ? "…" : "↗"}</span>
      </button>
    </>
  );
}
