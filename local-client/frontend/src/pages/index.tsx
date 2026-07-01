import { useState, useEffect } from "react";
import Head from "next/head";
import Script from "next/script";
import type {
  Strategy,
  QueryGroup,
  SearchResult,
  SearchResponse,
  TranscriptChunkResult,
  TranscriptChunkSearchResponse,
} from "@/types";
import {
  fetchStrategies,
  runSearch,
  translateTexts,
  warmupTextEncoder,
  searchTranscriptChunks,
} from "@/lib/api";
import QueryGroupComponent from "@/components/QueryGroup";
import ResultGrid from "@/components/ResultGrid";
import VideoGroupGrid from "@/components/VideoGroupGrid";
import TranscriptChunkCard from "@/components/TranscriptChunkCard";
import VideoModal from "@/components/VideoModal";

const DEFAULT_GROUP: QueryGroup = {
  semanticQuery: "",
  textQuery: "",
  temporalOffsetMs: 5000,
  translateSemantic: false,
  translatedQuery: "",
};

const ALL_GENRES = [
  "All",
  "Ẩm thực", "Công nghệ", "Du lịch", "Thể thao", "Giáo dục",
  "Kinh tế", "Sức khỏe", "Giải trí", "Thời sự", "Văn hóa",
  "Đời sống", "Môi trường", "Giao thông", "Pháp luật",
];

function normalizeTopK(value: string): number {
  return Math.min(1000, Math.max(1, Number.parseInt(value, 10) || 1));
}

export default function Home() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [queryGroups, setQueryGroups] = useState<QueryGroup[]>([
    { ...DEFAULT_GROUP, temporalOffsetMs: 0 },
  ]);
  const [topKInput, setTopKInput] = useState("100");
  const [collapsed, setCollapsed] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [totalTimeMs, setTotalTimeMs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Searching…");
  const [error, setError] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);
  const [viewMode, setViewMode] = useState<"score" | "video">("score");
  const [videoGenre, setVideoGenre] = useState("All");

  // ── Transcript search state ────────────────────────────────────────────────
  const [searchMode, setSearchMode] = useState<"frames" | "transcripts">("frames");
  const [transcriptQuery, setTranscriptQuery] = useState("");
  const [transcriptTopK, setTranscriptTopK] = useState("20");
  const [transcriptResponse, setTranscriptResponse] = useState<TranscriptChunkSearchResponse | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptTimeMs, setTranscriptTimeMs] = useState(0);
  const [transcriptGenre, setTranscriptGenre] = useState("");

  // ── Load strategies on mount ───────────────────────────────────────────────
  useEffect(() => {
    fetchStrategies()
      .then((list) => {
        setStrategies(list);
        if (list.length > 0) setSelectedStrategy(list[0].id);
      })
      .catch(() => setError("Cannot connect to backend. Is the local backend running?"));

    // Hide GPU wake-up work while the user prepares the first query.
    warmupTextEncoder().catch(() => undefined);
  }, []);

  // ── Query group helpers ────────────────────────────────────────────────────
  function updateGroup(index: number, updated: QueryGroup) {
    setQueryGroups((prev) => prev.map((g, i) => (i === index ? updated : g)));
  }

  function addGroup() {
    setQueryGroups((prev) => [...prev, { ...DEFAULT_GROUP }]);
  }

  function removeGroup(index: number) {
    setQueryGroups((prev) => prev.filter((_, i) => i !== index));
  }

  // ── Frame search ───────────────────────────────────────────────────────────
  async function handleSearch() {
    const hasInput = queryGroups.some((g) => g.semanticQuery.trim() || g.textQuery.trim());
    if (!hasInput) {
      setError("Enter at least one search query.");
      return;
    }
    if (!selectedStrategy) {
      setError("Select a strategy.");
      return;
    }

    setError(null);
    setLoading(true);
    setLoadingLabel("Searching…");
    const topK = normalizeTopK(topKInput);
    setTopKInput(String(topK));
    const started = performance.now();
    try {
      const groupsForSearch = queryGroups.map((group) => ({ ...group }));
      const pendingIndices = groupsForSearch
        .map((group, index) => ({ group, index }))
        .filter(({ group }) =>
          group.translateSemantic &&
          group.semanticQuery.trim() &&
          !group.translatedQuery
        )
        .map(({ index }) => index);

      if (pendingIndices.length > 0) {
        setLoadingLabel("Translating…");
        const translation = await translateTexts(
          pendingIndices.map((index) => groupsForSearch[index].semanticQuery),
        );
        pendingIndices.forEach((groupIndex, translationIndex) => {
          groupsForSearch[groupIndex].translatedQuery =
            translation.translations[translationIndex];
        });
        setQueryGroups(groupsForSearch);
      }

      setLoadingLabel("Searching…");
      const res = await runSearch(selectedStrategy, groupsForSearch, topK, videoGenre);
      setTotalTimeMs(Math.round(performance.now() - started));
      setResponse(res);
    } catch (err: any) {
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  // ── Transcript search ──────────────────────────────────────────────────────
  async function handleTranscriptSearch() {
    if (!transcriptQuery.trim()) {
      setError("Enter a transcript search query.");
      return;
    }
    setError(null);
    setTranscriptLoading(true);
    const topK = normalizeTopK(transcriptTopK);
    setTranscriptTopK(String(topK));
    const started = performance.now();
    try {
      const res = await searchTranscriptChunks(
        transcriptQuery.trim(), topK,
        transcriptGenre || undefined,
      );
      setTranscriptTimeMs(Math.round(performance.now() - started));
      setTranscriptResponse(res);
    } catch (err: any) {
      setError(err.message || "Transcript search failed.");
    } finally {
      setTranscriptLoading(false);
    }
  }

  function handleChunkCardClick(chunk: TranscriptChunkResult) {
    const searchResult: SearchResult = {
      video_id: chunk.video_id,
      youtube_id: undefined,
      frame_id: `${chunk.video_id}_000000`,
      frame_number: 0,
      timestamp_ms: chunk.start_time_ms,
      confidence: chunk.score,
      frame_image_url: "",
      fps: 25,
    };
    setActiveResult(searchResult);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") {
      if (searchMode === "transcripts") {
        handleTranscriptSearch();
      } else {
        handleSearch();
      }
    }
  }

  const currentStrategy = strategies.find((s) => s.id === selectedStrategy);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-900 text-slate-200" onKeyDown={handleKeyDown}>
      <Head>
        <link rel="preconnect" href="https://www.youtube.com" />
        <link rel="preconnect" href="https://www.google.com" />
        <link rel="preconnect" href="https://i.ytimg.com" />
        <link rel="preconnect" href="https://s.ytimg.com" />
      </Head>
      <Script
        id="yt-iframe-api"
        src="https://www.youtube.com/iframe_api"
        strategy="afterInteractive"
      />

      {/* ── Sticky top block: header + search panel ── */}
      <div className="sticky top-0 z-30 bg-slate-900 border-b border-slate-700 shadow-lg shadow-black/40">

        {/* Header row */}
        <header className="px-6 py-3 flex items-center justify-between gap-4 border-b border-slate-800">
          <div className="flex items-center gap-4 shrink-0">
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">AIC 2026</h1>
              <p className="text-xs text-slate-500">Video Retrieval Playground</p>
            </div>

            {/* Frames / Transcripts toggle */}
            <div className="flex rounded-lg border border-slate-600 overflow-hidden">
              <button
                onClick={() => { setSearchMode("frames"); setError(null); }}
                className={`px-3 py-1 text-xs font-medium transition ${
                  searchMode === "frames"
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:text-white"
                }`}
              >
                Frames
              </button>
              <button
                onClick={() => { setSearchMode("transcripts"); setError(null); }}
                className={`px-3 py-1 text-xs font-medium transition ${
                  searchMode === "transcripts"
                    ? "bg-emerald-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:text-white"
                }`}
              >
                Transcripts
              </button>
            </div>
          </div>

          {/* Strategy selector — only in frames mode */}
          {searchMode === "frames" && (
            <div className="flex items-center gap-3 min-w-0">
              <label className="text-sm text-slate-400 shrink-0">Strategy</label>
              <select
                value={selectedStrategy}
                onChange={(e) => setSelectedStrategy(e.target.value)}
                className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {strategies.length === 0 && <option value="">Loading…</option>}
                {strategies.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
              {currentStrategy && (
                <span className="hidden lg:inline text-xs text-slate-500 truncate max-w-xs">
                  {currentStrategy.description} — by {currentStrategy.author}
                </span>
              )}
            </div>
          )}
          {searchMode === "transcripts" && <div />}
        </header>

        {/* Frames search panel */}
        {searchMode === "frames" && (
          <div className="px-4 py-3 space-y-3">
            {/* Control row */}
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => setCollapsed((v) => !v)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-slate-400 hover:text-white text-sm transition"
                title={collapsed ? "Expand search panel" : "Collapse search panel"}
              >
                <span className="text-xs">{collapsed ? "▶" : "▼"}</span>
                {collapsed ? "Show Search" : "Hide Search"}
              </button>

              {!collapsed && (
                <button
                  onClick={addGroup}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-slate-600 text-slate-400 hover:border-slate-400 hover:text-white text-sm transition"
                >
                  <span className="text-base leading-none">+</span>
                  Add Temporal Step
                </button>
              )}

              <div className="flex-1" />

              {/* View mode toggle */}
              <div className="flex items-center bg-slate-800 border border-slate-600 rounded-lg overflow-hidden">
                <button
                  onClick={() => setViewMode("score")}
                  className={`px-3 py-1.5 text-sm transition ${
                    viewMode === "score"
                      ? "bg-blue-600 text-white"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Score
                </button>
                <button
                  onClick={() => setViewMode("video")}
                  className={`px-3 py-1.5 text-sm transition ${
                    viewMode === "video"
                      ? "bg-blue-600 text-white"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Video
                </button>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 shrink-0">Top K</label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={topKInput}
                  onChange={(e) => setTopKInput(e.target.value)}
                  onBlur={() => setTopKInput(String(normalizeTopK(topKInput)))}
                  className="w-20 bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 shrink-0">Genre</label>
                <select
                  value={videoGenre}
                  onChange={(e) => setVideoGenre(e.target.value)}
                  className="bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {ALL_GENRES.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleSearch}
                disabled={loading}
                className="px-5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
              >
                {loading ? loadingLabel : "Search"}
              </button>
            </div>

            {!collapsed && (
              <div className="space-y-3">
                {queryGroups.map((group, i) => (
                  <QueryGroupComponent
                    key={i}
                    group={group}
                    index={i}
                    isFirst={i === 0}
                    onChange={(updated) => updateGroup(i, updated)}
                    onRemove={() => removeGroup(i)}
                  />
                ))}
                <p className="text-xs text-slate-600 pl-1">Tip: Enter to search</p>
              </div>
            )}
          </div>
        )}

        {/* Transcripts search panel */}
        {searchMode === "transcripts" && (
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 flex-wrap">
              <input
                type="text"
                placeholder="Search transcript chunks (e.g. cách nấu phở, kẹt xe, AI chip)…"
                value={transcriptQuery}
                onChange={(e) => setTranscriptQuery(e.target.value)}
                className="flex-1 min-w-[200px] bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 shrink-0">Top K</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={transcriptTopK}
                  onChange={(e) => setTranscriptTopK(e.target.value)}
                  onBlur={() => setTranscriptTopK(String(normalizeTopK(transcriptTopK)))}
                  className="w-20 bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 shrink-0">Topic</label>
                <select
                  value={transcriptGenre}
                  onChange={(e) => setTranscriptGenre(e.target.value)}
                  className="bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                >
                  <option value="">Auto</option>
                  {ALL_GENRES.filter(g => g !== "All").map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleTranscriptSearch}
                disabled={transcriptLoading}
                className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
              >
                {transcriptLoading ? "Searching…" : "Search Transcripts"}
              </button>
            </div>
          </div>
        )}

      </div>

      {/* ── Main content ── */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {error && (
          <div className="bg-red-900/40 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {/* Frame search results */}
        {searchMode === "frames" && response && viewMode === "score" && (
          <ResultGrid
            results={response.results}
            total={response.total}
            executionTimeMs={totalTimeMs}
            onCardClick={(r) => setActiveResult(r)}
          />
        )}
        {searchMode === "frames" && response && viewMode === "video" && (
          <VideoGroupGrid
            results={response.results}
            total={response.total}
            executionTimeMs={totalTimeMs}
            onCardClick={(r) => setActiveResult(r)}
          />
        )}

        {/* Transcript chunk search results */}
        {searchMode === "transcripts" && transcriptResponse && (
          <div className="space-y-4">
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <span>
                <span className="text-white font-semibold">{transcriptResponse.total}</span> chunk matches
              </span>
              <span>·</span>
              <span title="Total time for transcript chunk search">
                <span className="text-white font-semibold">{transcriptTimeMs}</span> ms
              </span>
            </div>

            {transcriptResponse.results.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {transcriptResponse.results.map((chunk, i) => (
                  <TranscriptChunkCard
                    key={chunk.chunk_id}
                    result={chunk}
                    rank={i + 1}
                    onClick={handleChunkCardClick}
                  />
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-center py-16">
                No matching chunks. Try different keywords.
              </p>
            )}
          </div>
        )}
      </main>

      {/* ── Floating back-to-top button ── */}
      {(response || transcriptResponse) && (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          className="fixed bottom-6 right-6 z-40 w-11 h-11 flex items-center justify-center rounded-full bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/50 transition"
          title="Back to top"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
          </svg>
        </button>
      )}

      {/* ── Video modal ── */}
      {activeResult && (
        <VideoModal
          result={activeResult}
          onClose={() => setActiveResult(null)}
        />
      )}
    </div>
  );
}
