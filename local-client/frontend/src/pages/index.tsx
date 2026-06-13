import { useState, useEffect } from "react";
import type {
  Strategy,
  QueryGroup,
  SearchResult,
  SearchResponse,
} from "@/types";
import { fetchStrategies, runSearch, translateTexts } from "@/lib/api";
import QueryGroupComponent from "@/components/QueryGroup";
import ResultGrid from "@/components/ResultGrid";
import VideoModal from "@/components/VideoModal";

const DEFAULT_GROUP: QueryGroup = {
  semanticQuery: "",
  textQuery: "",
  temporalOffsetMs: 5000,
  translateSemantic: false,
  translatedQuery: "",
};

export default function Home() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [queryGroups, setQueryGroups] = useState<QueryGroup[]>([
    { ...DEFAULT_GROUP, temporalOffsetMs: 0 },
  ]);
  const [topK, setTopK] = useState<number>(100);
  const [collapsed, setCollapsed] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Searching…");
  const [error, setError] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);

  // ── Load strategies on mount ───────────────────────────────────────────────
  useEffect(() => {
    fetchStrategies()
      .then((list) => {
        setStrategies(list);
        if (list.length > 0) setSelectedStrategy(list[0].id);
      })
      .catch(() => setError("Cannot connect to backend. Is the local backend running?"));
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

  // ── Search ─────────────────────────────────────────────────────────────────
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
      const res = await runSearch(selectedStrategy, groupsForSearch, topK);
      setResponse(res);
    } catch (err: any) {
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSearch();
  }

  const currentStrategy = strategies.find((s) => s.id === selectedStrategy);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-900 text-slate-200" onKeyDown={handleKeyDown}>

      {/* ── Sticky top block: header + search panel ── */}
      <div className="sticky top-0 z-30 bg-slate-900 border-b border-slate-700 shadow-lg shadow-black/40">

        {/* Header row */}
        <header className="px-6 py-3 flex items-center justify-between gap-4 border-b border-slate-800">
          <div className="shrink-0">
            <h1 className="text-lg font-bold text-white leading-tight">AIC 2026</h1>
            <p className="text-xs text-slate-500">Video Retrieval Playground</p>
          </div>

          {/* Strategy selector */}
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
        </header>

        {/* Search panel */}
        <div className="px-4 py-3 space-y-3">

          {/* Control row — always visible even when collapsed */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* Collapse toggle */}
            <button
              onClick={() => setCollapsed((v) => !v)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-slate-400 hover:text-white text-sm transition"
              title={collapsed ? "Expand search panel" : "Collapse search panel"}
            >
              <span className="text-xs">{collapsed ? "▶" : "▼"}</span>
              {collapsed ? "Show Search" : "Hide Search"}
            </button>

            {/* Add temporal step — only when expanded */}
            {!collapsed && (
              <button
                onClick={addGroup}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-slate-600 text-slate-400 hover:border-slate-400 hover:text-white text-sm transition"
              >
                <span className="text-base leading-none">+</span>
                Add Temporal Step
              </button>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Top K input */}
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-400 shrink-0">Top K</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-20 bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Search button */}
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-5 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg text-sm transition"
            >
              {loading ? loadingLabel : "Search"}
            </button>
          </div>

          {/* Collapsible query groups */}
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
              <p className="text-xs text-slate-600 pl-1">Tip: Ctrl+Enter to search</p>
            </div>
          )}

        </div>
      </div>

      {/* ── Main content ── */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {error && (
          <div className="bg-red-900/40 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {response && (
          <ResultGrid
            results={response.results}
            total={response.total}
            executionTimeMs={response.execution_time_ms}
            onCardClick={(r) => setActiveResult(r)}
          />
        )}
      </main>

      {/* ── Floating back-to-top button — visible whenever there are results ── */}
      {response && (
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
