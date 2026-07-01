import { useState, useEffect, useRef } from "react";
import Head from "next/head";
import Script from "next/script";
import type {
  Strategy,
  QueryGroup,
  SearchResult,
  SearchResponse,
  TranscriptResult,
  TranscriptSearchResponse,
  VectorSearchAlgorithm,
} from "@/types";
import {
  fetchVectorSearchAlgorithms,
  fetchStrategies,
  runSearch,
  translateTexts,
  warmupTextEncoder,
  searchTranscript,
} from "@/lib/api";
import QueryGroupComponent from "@/components/QueryGroup";
import ResultGrid from "@/components/ResultGrid";
import VideoGroupGrid from "@/components/VideoGroupGrid";
import TranscriptResultList from "@/components/TranscriptResultList";
import VideoModal from "@/components/VideoModal";

const DEFAULT_GROUP: QueryGroup = {
  semanticQuery: "",
  textQuery: "",
  temporalOffsetMs: 5000,
  translateSemantic: false,
  translatedQuery: "",
};

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
  const [vectorAlgorithms, setVectorAlgorithms] = useState<VectorSearchAlgorithm[]>([]);
  const [selectedVectorAlgorithm, setSelectedVectorAlgorithm] = useState("");
  const [algorithmMenuOpen, setAlgorithmMenuOpen] = useState(false);

  // ── Transcript search state ────────────────────────────────────────────────
  const [searchMode, setSearchMode] = useState<"frames" | "transcripts">("frames");
  const [transcriptQuery, setTranscriptQuery] = useState("");
  const [transcriptTopK, setTranscriptTopK] = useState("20");
  const [transcriptResponse, setTranscriptResponse] = useState<TranscriptSearchResponse | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptTimeMs, setTranscriptTimeMs] = useState(0);

  // ── Load strategies on mount ───────────────────────────────────────────────
  useEffect(() => {
    fetchStrategies()
      .then((list) => {
        setStrategies(list);
        if (list.length > 0) setSelectedStrategy(list[0].id);
      })
      .catch(() => setError("Cannot connect to backend. Is the local backend running?"));

    fetchVectorSearchAlgorithms()
      .then((payload) => {
        setVectorAlgorithms(payload.algorithms);
        const fallback = payload.algorithms.find((item) => item.available)?.id || "";
        const defaultIsAvailable = payload.algorithms.some(
          (item) => item.id === payload.default && item.available
        );
        setSelectedVectorAlgorithm(defaultIsAvailable ? payload.default : fallback);
      })
      .catch(() => undefined);

    // Hide GPU wake-up work while the user prepares the first query.
    warmupTextEncoder().catch(() => undefined);
  }, []);

  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setAlgorithmMenuOpen(false);
      }
    }
    if (algorithmMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [algorithmMenuOpen]);

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
      const res = await runSearch(
        selectedStrategy,
        groupsForSearch,
        topK,
        selectedVectorAlgorithm || undefined,
      );
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
      const res = await searchTranscript(transcriptQuery.trim(), topK);
      setTranscriptTimeMs(Math.round(performance.now() - started));
      setTranscriptResponse(res);
    } catch (err: any) {
      setError(err.message || "Transcript search failed.");
    } finally {
      setTranscriptLoading(false);
    }
  }

  function handleTranscriptCardClick(transcriptResult: TranscriptResult) {
    const searchResult: SearchResult = {
      video_id: transcriptResult.video_id,
      youtube_id: transcriptResult.youtube_id,
      frame_id: transcriptResult.nearest_frame_id || `${transcriptResult.video_id}_000000`,
      frame_number: 0,
      timestamp_ms: transcriptResult.start_time_ms,
      confidence: transcriptResult.score,
      frame_image_url: transcriptResult.frame_image_url || "",
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
  const currentVectorAlgorithm = vectorAlgorithms.find((item) => item.id === selectedVectorAlgorithm);

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

              <button
                onClick={handleSearch}
                disabled={loading}
                className="px-5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
              >
                {loading ? loadingLabel : "Search"}
              </button>


              {vectorAlgorithms.length > 0 && (
                <div className="relative" ref={menuRef}>
                  <button
                    type="button"
                    onClick={() => setAlgorithmMenuOpen((open) => !open)}
                    className="h-8 min-w-20 px-2.5 rounded-lg bg-slate-800 border border-slate-600 text-slate-300 hover:text-white hover:border-slate-400 text-xs font-medium transition"
                    title="Vector search algorithm"
                  >
                    <span className="mr-1">⚙</span>
                    {selectedVectorAlgorithm === "flat" ? "FLAT - Linear search" : (currentVectorAlgorithm?.name || "Vector")}
                  </button>

                  {algorithmMenuOpen && (
                    <div className="absolute right-0 top-10 z-40 w-64 rounded-lg border border-slate-600 bg-slate-900 shadow-xl shadow-black/50 overflow-hidden divide-y divide-slate-800">
                      {/* EXACT Section */}
                      {vectorAlgorithms.some(a => a.id === "flat" || a.id === "linear") && (
                        <div className="py-1">
                          <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                            Exact
                          </div>
                          {vectorAlgorithms
                            .filter(a => a.id === "flat" || a.id === "linear")
                            .map((algorithm) => (
                              <button
                                key={algorithm.id}
                                type="button"
                                disabled={!algorithm.available}
                                onClick={() => {
                                  if (!algorithm.available) return;
                                  setSelectedVectorAlgorithm(algorithm.id);
                                  setAlgorithmMenuOpen(false);
                                }}
                                className={`w-full px-3 py-1.5 text-left transition ${
                                  selectedVectorAlgorithm === algorithm.id
                                    ? "bg-blue-600/30 text-white"
                                    : "text-slate-300 hover:bg-slate-800"
                                } ${!algorithm.available ? "opacity-45 cursor-not-allowed hover:bg-transparent" : ""}`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="text-xs font-semibold">
                                    {algorithm.id === "flat" ? "FLAT - Linear search" : algorithm.name}
                                  </span>
                                  {!algorithm.available && (
                                    <span className="text-[9px] font-bold uppercase tracking-wide text-slate-500">
                                      Later
                                    </span>
                                  )}
                                </div>
                                <p className="text-[10px] text-slate-500 leading-normal">
                                  {algorithm.description}
                                </p>
                              </button>
                            ))}
                        </div>
                      )}

                      {/* ANN Section */}
                      {vectorAlgorithms.some(a => a.id === "hnsw" || (a.id === "cagra" && a.available)) && (
                        <div className="py-1">
                          <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                            ANN
                          </div>
                          {vectorAlgorithms
                            .filter(a => a.id === "hnsw" || (a.id === "cagra" && a.available))
                            .map((algorithm) => (
                              <button
                                key={algorithm.id}
                                type="button"
                                disabled={!algorithm.available}
                                onClick={() => {
                                  if (!algorithm.available) return;
                                  setSelectedVectorAlgorithm(algorithm.id);
                                  setAlgorithmMenuOpen(false);
                                }}
                                className={`w-full px-3 py-1.5 text-left transition ${
                                  selectedVectorAlgorithm === algorithm.id
                                    ? "bg-blue-600/30 text-white"
                                    : "text-slate-300 hover:bg-slate-800"
                                } ${!algorithm.available ? "opacity-45 cursor-not-allowed hover:bg-transparent" : ""}`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="text-xs font-semibold">{algorithm.name}</span>
                                  {!algorithm.available && (
                                    <span className="text-[9px] font-bold uppercase tracking-wide text-slate-500">
                                      Later
                                    </span>
                                  )}
                                </div>
                                <p className="text-[10px] text-slate-500 leading-normal">
                                  {algorithm.description}
                                </p>
                              </button>
                            ))}
                        </div>
                      )}

                      {/* QUANTIZATION Section */}
                      {vectorAlgorithms.some(a => a.id === "scann") && (
                        <div className="py-1">
                          <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                            Quantization ANN
                          </div>
                          {vectorAlgorithms
                            .filter(a => a.id === "scann")
                            .map((algorithm) => (
                              <button
                                key={algorithm.id}
                                type="button"
                                disabled={!algorithm.available}
                                onClick={() => {
                                  if (!algorithm.available) return;
                                  setSelectedVectorAlgorithm(algorithm.id);
                                  setAlgorithmMenuOpen(false);
                                }}
                                className={`w-full px-3 py-1.5 text-left transition ${
                                  selectedVectorAlgorithm === algorithm.id
                                    ? "bg-blue-600/30 text-white"
                                    : "text-slate-300 hover:bg-slate-800"
                                } ${!algorithm.available ? "opacity-45 cursor-not-allowed hover:bg-transparent" : ""}`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="text-xs font-semibold">{algorithm.name}</span>
                                  {!algorithm.available && (
                                    <span className="text-[9px] font-bold uppercase tracking-wide text-slate-500">
                                      Later
                                    </span>
                                  )}
                                </div>
                                <p className="text-[10px] text-slate-500 leading-normal">
                                  {algorithm.description}
                                </p>
                              </button>
                            ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
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
                placeholder="Search transcript text (e.g. flood, goal, hospital)…"
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

        {/* Transcript search results */}
        {searchMode === "transcripts" && transcriptResponse && (
          <TranscriptResultList
            results={transcriptResponse.results}
            total={transcriptResponse.total}
            executionTimeMs={transcriptTimeMs}
            onCardClick={handleTranscriptCardClick}
          />
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
