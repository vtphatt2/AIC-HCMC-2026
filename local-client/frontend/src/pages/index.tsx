import { useState, useEffect, useRef, useMemo } from "react";
import Head from "next/head";
import Script from "next/script";
import type {
  Strategy,
  QueryGroup,
  SearchResult,
  SearchResponse,
  TranscriptChunkResult,
  TranscriptChunkSearchResponse,
  VectorSearchAlgorithm,
} from "@/types";
import {
  fetchVectorSearchAlgorithms,
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

const THEME_STORAGE_KEY = "aic2026-theme";
const SIDEBAR_WIDTH_KEY = "aic2026-sidebar-width";
const SIDEBAR_DEFAULT_WIDTH = 520;
const SIDEBAR_MIN_WIDTH = 300;
const SIDEBAR_MAX_WIDTH = 900;
const MAIN_MIN_WIDTH = 360; // keep the results pane usable even on narrow windows

function normalizeTopK(value: string): number {
  return Math.min(1000, Math.max(1, Number.parseInt(value, 10) || 1));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export default function Home() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [queryGroups, setQueryGroups] = useState<QueryGroup[]>([
    { ...DEFAULT_GROUP, temporalOffsetMs: 0 },
  ]);
  const [topKInput, setTopKInput] = useState("100");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT_WIDTH);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [totalTimeMs, setTotalTimeMs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Searching…");
  const [error, setError] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);
  const [viewMode, setViewMode] = useState<"score" | "video">("score");
  const [videoGenre, setVideoGenre] = useState("All");
  const [vectorAlgorithms, setVectorAlgorithms] = useState<VectorSearchAlgorithm[]>([]);
  const [selectedVectorAlgorithm, setSelectedVectorAlgorithm] = useState("");
  const [algorithmMenuOpen, setAlgorithmMenuOpen] = useState(false);

  // ── Transcript search state ────────────────────────────────────────────────
  const [searchMode, setSearchMode] = useState<"frames" | "transcripts">("frames");
  const [transcriptQuery, setTranscriptQuery] = useState("");
  const [transcriptTopK, setTranscriptTopK] = useState("20");
  const [transcriptResponse, setTranscriptResponse] = useState<TranscriptChunkSearchResponse | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptTimeMs, setTranscriptTimeMs] = useState(0);
  const [transcriptGenre, setTranscriptGenre] = useState("");
  const [transcriptViewMode, setTranscriptViewMode] = useState<"score" | "video">("score");

  const mainRef = useRef<HTMLDivElement>(null);

  // ── Theme: load saved preference, reflect onto <html class="dark"> ────────
  useEffect(() => {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
      return;
    }
    const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
    setTheme(prefersLight ? "light" : "dark");
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  // ── Sidebar width: load saved preference, persist on change ───────────────
  useEffect(() => {
    const saved = window.localStorage.getItem(SIDEBAR_WIDTH_KEY);
    const parsed = saved ? Number.parseInt(saved, 10) : NaN;
    if (!Number.isNaN(parsed)) {
      setSidebarWidth(clamp(parsed, SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH));
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
  }, [sidebarWidth]);

  // ── Sidebar resize drag (VSCode-style) ─────────────────────────────────────
  function startSidebarResize(e: React.MouseEvent) {
    e.preventDefault();
    setIsResizingSidebar(true);
  }

  useEffect(() => {
    if (!isResizingSidebar) return;

    function handleMouseMove(e: MouseEvent) {
      const maxWidth = Math.min(SIDEBAR_MAX_WIDTH, window.innerWidth - MAIN_MIN_WIDTH);
      setSidebarWidth(clamp(e.clientX, SIDEBAR_MIN_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, maxWidth)));
    }
    function handleMouseUp() {
      setIsResizingSidebar(false);
    }

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    const prevCursor = document.body.style.cursor;
    const prevUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevUserSelect;
    };
  }, [isResizingSidebar]);

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
        selectedStrategy, groupsForSearch, topK,
        videoGenre,
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
    const midMs = (chunk.start_time_ms + chunk.end_time_ms) / 2;
    const computedFrameNumber = Math.floor((midMs / 1000) * 25);
    const frameNumber = chunk.frame_number > 0 ? chunk.frame_number : computedFrameNumber;
    const timestampMs = chunk.nearest_timestamp_ms ?? chunk.start_time_ms;

    const searchResult: SearchResult = {
      video_id: chunk.video_id,
      youtube_id: chunk.youtube_id,
      frame_id: `${chunk.video_id}_${String(frameNumber).padStart(6, "0")}`,
      frame_number: frameNumber,
      timestamp_ms: timestampMs,
      confidence: chunk.score,
      frame_image_url: chunk.frame_image_url,
      fps: 25,
    };
    setActiveResult(searchResult);
  }

  function handleTranscriptFrameClick(
    videoId: string,
    youtubeId: string,
    frameNumber: number,
    timestampMs: number,
    frameImageUrl: string,
  ) {
    const searchResult: SearchResult = {
      video_id: videoId,
      youtube_id: youtubeId || undefined,
      frame_id: `${videoId}_${String(frameNumber).padStart(6, "0")}`,
      frame_number: frameNumber,
      timestamp_ms: timestampMs,
      confidence: 0,
      frame_image_url: frameImageUrl,
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

  const transcriptFrameResults = useMemo<SearchResult[]>(() => {
    if (!transcriptResponse) return [];
    return transcriptResponse.results.filter((chunk) => chunk.frame_image_url).map((chunk) => {
      const midMs = (chunk.start_time_ms + chunk.end_time_ms) / 2;
      const computedFrameNumber = Math.floor((midMs / 1000) * 25);
      const frameNumber = chunk.frame_number > 0 ? chunk.frame_number : computedFrameNumber;
      const timestampMs = chunk.nearest_timestamp_ms ?? Math.round((frameNumber / 25) * 1000);

      return {
        video_id: chunk.video_id,
        youtube_id: chunk.youtube_id,
        frame_id: `${chunk.video_id}_${String(frameNumber).padStart(6, "0")}`,
        frame_number: frameNumber,
        timestamp_ms: timestampMs,
        confidence: chunk.score,
        frame_image_url: chunk.frame_image_url,
        fps: 25,
      };
    });
  }, [transcriptResponse]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className="h-screen flex flex-col bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200 overflow-hidden"
      onKeyDown={handleKeyDown}
    >
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

      <div className="flex flex-1 min-h-0">
        {/* ── Left sidebar: search panel (VSCode-explorer-style collapse + resize) ── */}
        <aside
          className={`flex flex-col shrink-0 bg-white dark:bg-slate-900 ${
            isResizingSidebar ? "" : "transition-[width] duration-150"
          } ${sidebarCollapsed ? "w-12 border-r border-slate-200 dark:border-slate-700" : ""}`}
          style={sidebarCollapsed ? undefined : { width: sidebarWidth }}
        >
          {sidebarCollapsed ? (
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="h-full w-full flex flex-col items-center gap-2 pt-4 text-slate-400 hover:text-slate-900 dark:hover:text-white transition"
              title="Show search panel"
            >
              <span className="text-sm">▶</span>
              <span className="text-[11px] tracking-wide" style={{ writingMode: "vertical-rl" }}>
                Search
              </span>
            </button>
          ) : (
            <>
              {/* Sidebar header */}
              <header className="px-4 py-3 flex items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 shrink-0">
                <div className="flex items-center gap-3 min-w-0">
                  <button
                    onClick={() => setSidebarCollapsed(true)}
                    className="shrink-0 w-7 h-7 flex items-center justify-center rounded-lg border border-slate-300 dark:border-slate-600 text-slate-500 hover:text-slate-900 dark:hover:text-white transition"
                    title="Collapse search panel"
                  >
                    <span className="text-xs">◀</span>
                  </button>
                  <div className="min-w-0">
                    <h1 className="text-lg font-bold text-slate-900 dark:text-white leading-tight">AIC 2026</h1>
                    <p className="text-xs text-slate-500">Video Retrieval Playground</p>
                  </div>
                </div>

                <button
                  onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
                  className="shrink-0 w-7 h-7 flex items-center justify-center rounded-lg border border-slate-300 dark:border-slate-600 text-slate-500 hover:text-slate-900 dark:hover:text-white transition"
                  title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
                >
                  <span className="text-xs">{theme === "dark" ? "☀" : "☾"}</span>
                </button>
              </header>

              {/* Frames / Transcripts mode toggle */}
              <div className="px-4 py-2 border-b border-slate-200 dark:border-slate-800 shrink-0">
                <div className="flex rounded-lg border border-slate-300 dark:border-slate-600 overflow-hidden w-fit">
                  <button
                    onClick={() => { setSearchMode("frames"); setError(null); }}
                    className={`px-3 py-1 text-xs font-medium transition ${
                      searchMode === "frames"
                        ? "bg-blue-600 text-white"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Frames
                  </button>
                  <button
                    onClick={() => { setSearchMode("transcripts"); setError(null); }}
                    className={`px-3 py-1 text-xs font-medium transition ${
                      searchMode === "transcripts"
                        ? "bg-emerald-600 text-white"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Transcripts
                  </button>
                </div>
              </div>

              {/* Scrollable search panel body */}
              <div className="flex-1 overflow-y-auto">
                {/* Frames search panel */}
                {searchMode === "frames" && (
                  <div className="px-4 py-3 space-y-3">
                    {/* Strategy selector */}
                    <div className="flex items-center gap-2">
                      <label className="text-sm text-slate-500 dark:text-slate-400 shrink-0">Strategy</label>
                      <select
                        value={selectedStrategy}
                        onChange={(e) => setSelectedStrategy(e.target.value)}
                        className="flex-1 min-w-0 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        {strategies.length === 0 && <option value="">Loading…</option>}
                        {strategies.map((s) => (
                          <option key={s.id} value={s.id}>{s.name}</option>
                        ))}
                      </select>
                    </div>
                    {currentStrategy && (
                      <p className="text-xs text-slate-500 -mt-1">
                        {currentStrategy.description} — by {currentStrategy.author}
                      </p>
                    )}

                    {/* Control row */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <button
                        onClick={addGroup}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-slate-400 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:border-slate-600 dark:hover:border-slate-400 hover:text-slate-900 dark:hover:text-white text-sm transition"
                      >
                        <span className="text-base leading-none">+</span>
                        Add Temporal Step
                      </button>

                      {/* View mode toggle */}
                      <div className="flex items-center bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg overflow-hidden">
                        <button
                          onClick={() => setViewMode("score")}
                          className={`px-3 py-1.5 text-sm transition ${
                            viewMode === "score"
                              ? "bg-blue-600 text-white"
                              : "text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                          }`}
                        >
                          Score
                        </button>
                        <button
                          onClick={() => setViewMode("video")}
                          className={`px-3 py-1.5 text-sm transition ${
                            viewMode === "video"
                              ? "bg-blue-600 text-white"
                              : "text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                          }`}
                        >
                          Video
                        </button>
                      </div>

                      <div className="flex items-center gap-2">
                        <label className="text-sm text-slate-500 dark:text-slate-400 shrink-0">Top K</label>
                        <input
                          type="number"
                          min={1}
                          max={1000}
                          value={topKInput}
                          onChange={(e) => setTopKInput(e.target.value)}
                          onBlur={() => setTopKInput(String(normalizeTopK(topKInput)))}
                          className="w-20 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-900 dark:text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div className="flex items-center gap-2">
                        <label className="text-sm text-slate-500 dark:text-slate-400 shrink-0">Genre</label>
                        <select
                          value={videoGenre}
                          onChange={(e) => setVideoGenre(e.target.value)}
                          className="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {ALL_GENRES.map((g) => (
                            <option key={g} value={g}>{g}</option>
                          ))}
                        </select>
                      </div>

                      {vectorAlgorithms.length > 0 && (
                        <div className="relative" ref={menuRef}>
                          <button
                            type="button"
                            onClick={() => setAlgorithmMenuOpen((open) => !open)}
                            className="h-8 min-w-20 px-2.5 rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:border-slate-400 text-xs font-medium transition"
                            title="Vector search algorithm"
                          >
                            <span className="mr-1">⚙</span>
                            {selectedVectorAlgorithm === "flat" ? "FLAT - Linear search" : (currentVectorAlgorithm?.name || "Vector")}
                          </button>

                          {algorithmMenuOpen && (
                            <div className="absolute left-0 top-10 z-40 w-64 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 shadow-xl shadow-black/20 dark:shadow-black/50 overflow-hidden divide-y divide-slate-200 dark:divide-slate-800">
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
                                            ? "bg-blue-600/20 dark:bg-blue-600/30 text-slate-900 dark:text-white"
                                            : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
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

                              {/* Graph ANN Section */}
                              {vectorAlgorithms.some(a => a.id === "hnsw" || (a.id === "cagra" && a.available)) && (
                                <div className="py-1">
                                  <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                                    Graph ANN
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
                                            ? "bg-blue-600/20 dark:bg-blue-600/30 text-slate-900 dark:text-white"
                                            : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
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

                              {/* Quantized CPU ANN Section */}
                              {vectorAlgorithms.some(a => a.id === "scann") && (
                                <div className="py-1">
                                  <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                                    Quantized CPU ANN
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
                                            ? "bg-blue-600/20 dark:bg-blue-600/30 text-slate-900 dark:text-white"
                                            : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
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

                      <button
                        onClick={handleSearch}
                        disabled={loading}
                        className="px-5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
                      >
                        {loading ? loadingLabel : "Search"}
                      </button>
                    </div>

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
                      <p className="text-xs text-slate-500 dark:text-slate-600 pl-1">Tip: Enter to search</p>
                    </div>
                  </div>
                )}

                {/* Transcripts search panel */}
                {searchMode === "transcripts" && (
                  <div className="px-4 py-3 space-y-3">
                    <input
                      type="text"
                      placeholder="Search transcript chunks (e.g. cách nấu phở, kẹt xe, AI chip)…"
                      value={transcriptQuery}
                      onChange={(e) => setTranscriptQuery(e.target.value)}
                      className="w-full bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />

                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="flex items-center gap-2">
                        <label className="text-sm text-slate-500 dark:text-slate-400 shrink-0">Top K</label>
                        <input
                          type="number"
                          min={1}
                          max={200}
                          value={transcriptTopK}
                          onChange={(e) => setTranscriptTopK(e.target.value)}
                          onBlur={() => setTranscriptTopK(String(normalizeTopK(transcriptTopK)))}
                          className="w-20 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-900 dark:text-white text-center focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        />
                      </div>

                      {/* View mode toggle for transcripts */}
                      <div className="flex items-center bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg overflow-hidden">
                        <button
                          onClick={() => setTranscriptViewMode("score")}
                          className={`px-3 py-1.5 text-sm transition ${
                            transcriptViewMode === "score"
                              ? "bg-emerald-600 text-white"
                              : "text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                          }`}
                        >
                          Score
                        </button>
                        <button
                          onClick={() => setTranscriptViewMode("video")}
                          className={`px-3 py-1.5 text-sm transition ${
                            transcriptViewMode === "video"
                              ? "bg-emerald-600 text-white"
                              : "text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                          }`}
                        >
                          Video
                        </button>
                      </div>

                      <div className="flex items-center gap-2">
                        <label className="text-sm text-slate-500 dark:text-slate-400 shrink-0">Topic</label>
                        <select
                          value={transcriptGenre}
                          onChange={(e) => setTranscriptGenre(e.target.value)}
                          className="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
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
                        className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
                      >
                        {transcriptLoading ? "Searching…" : "Search Transcripts"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </aside>

        {/* ── Drag handle: resize sidebar (VSCode-style) ── */}
        {!sidebarCollapsed && (
          <div
            onMouseDown={startSidebarResize}
            className="w-1 shrink-0 cursor-col-resize bg-slate-200 dark:bg-slate-700 hover:bg-blue-400 active:bg-blue-500 transition-colors"
            title="Drag to resize sidebar"
          />
        )}

        {/* ── Right pane: results ── */}
        <main ref={mainRef} className="flex-1 min-w-0 overflow-y-auto">
          <div className="px-4 py-6 space-y-6">
            {error && (
              <div className="bg-red-100 dark:bg-red-900/40 border border-red-300 dark:border-red-700 text-red-700 dark:text-red-300 rounded-lg px-4 py-3 text-sm">
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

            {/* Transcript chunk search results — Score View */}
            {searchMode === "transcripts" && transcriptResponse && transcriptViewMode === "score" && (
              <div className="space-y-4">
                <div className="flex items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
                  <span>
                    <span className="text-slate-900 dark:text-white font-semibold">{transcriptResponse.total}</span> chunk matches
                  </span>
                  <span>·</span>
                  <span title="Total time for transcript chunk search">
                    <span className="text-slate-900 dark:text-white font-semibold">{transcriptTimeMs}</span> ms
                  </span>
                </div>

                {transcriptResponse.results.length > 0 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {transcriptResponse.results.map((chunk, i) => (
                      <TranscriptChunkCard
                        key={chunk.chunk_id}
                        result={chunk}
                        rank={i + 1}
                        onClick={handleChunkCardClick}
                        onFrameClick={handleTranscriptFrameClick}
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

            {/* Transcript chunk search results — Video View */}
            {searchMode === "transcripts" && transcriptResponse && transcriptViewMode === "video" && (
              <VideoGroupGrid
                results={transcriptFrameResults}
                total={transcriptResponse.total}
                executionTimeMs={transcriptTimeMs}
                onCardClick={(r) => setActiveResult(r)}
              />
            )}
          </div>

        </main>
      </div>

      {/* ── Floating back-to-top button (scrolls the results pane) ── */}
      {(response || transcriptResponse) && (
        <button
          onClick={() => mainRef.current?.scrollTo({ top: 0, behavior: "smooth" })}
          className="fixed bottom-6 right-6 z-40 w-11 h-11 flex items-center justify-center rounded-full bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/30 dark:shadow-blue-900/50 transition"
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
