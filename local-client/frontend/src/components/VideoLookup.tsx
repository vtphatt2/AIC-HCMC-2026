import { useState } from "react";
import type { SearchResult } from "@/types";
import { fetchVideoById } from "@/lib/api";

interface VideoLookupProps {
  onOpen: (result: SearchResult) => void;
}

export default function VideoLookup({ onOpen }: VideoLookupProps) {
  const [jumpVideoQuery, setJumpVideoQuery] = useState("");
  const [jumpLoading, setJumpLoading] = useState(false);

  async function handleJumpToVideo() {
    const lookup = jumpVideoQuery.trim();
    if (!lookup) return;
    setJumpLoading(true);
    try {
      onOpen(await fetchVideoById(lookup));
      setJumpVideoQuery("");
    } catch (err: any) {
      window.alert(err.message || `Video “${lookup}” not found`);
    } finally {
      setJumpLoading(false);
    }
  }

  return (
    <>
      <input
        value={jumpVideoQuery}
        onChange={(event) => setJumpVideoQuery(event.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") handleJumpToVideo(); }}
        placeholder="Title or video ID…"
        disabled={jumpLoading}
        title="Open the highest-ranked video by title, full ID, or ID prefix such as L0_"
        className="h-7 w-32 rounded border-2 border-stone-800 dark:border-stone-500 px-2 text-xs bg-cream-card dark:bg-stone-800 text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600"
      />
      <button
        onClick={handleJumpToVideo}
        disabled={jumpLoading || !jumpVideoQuery.trim()}
        className="h-7 flex items-center justify-center rounded border-2 border-stone-800 dark:border-stone-500 px-2 text-stone-600 dark:text-stone-300 hover:text-orange-700 dark:hover:text-orange-400 hover:border-orange-700 dark:hover:border-orange-400 transition disabled:opacity-40"
        title="Open video by title or ID"
      >
        <span className="text-xs font-bold">{jumpLoading ? "…" : "↗"}</span>
      </button>
    </>
  );
}
