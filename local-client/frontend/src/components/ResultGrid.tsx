import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { SearchResult } from "@/types";
import ResultCard from "./ResultCard";
import VerifyAction from "./VerifyAction";

export interface ResultGridHandle {
  focus: () => void;
}

interface Props {
  results: SearchResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: SearchResult) => void;
  onVerify?: (result: SearchResult) => void;
  scrollContainerRef: React.RefObject<HTMLElement | null>;
  onFocusQuery: () => void;
  // False while the video modal is open — the grid ignores all keys so the
  // modal is truly modal (no phantom nav/scroll behind it).
  active: boolean;
}

const ALIGN_TOP_PADDING = 12;

interface ClusterRowProps {
  result: SearchResult;
  index: number;
  focusedStep: number;
  isFocused: boolean;
  setCardRef: (itemIndex: number, stepIndex: number) => (node: HTMLButtonElement | null) => void;
  onCardClick: (r: SearchResult) => void;
  onVerify?: (r: SearchResult) => void;
  onSelect: (itemIndex: number, stepIndex: number) => void;
}

function ClusterRow({ result, index, focusedStep, isFocused, setCardRef, onCardClick, onVerify, onSelect }: ClusterRowProps) {
  const steps = result.steps && result.steps.length > 1 ? result.steps : [result];
  return (
    <div
      // No fixed/full width here — the box sizes to its own frame count so
      // small clusters (2-3 steps) can sit side by side; a big cluster just
      // ends up wide enough to take its row on its own. `overflow-x-auto`
      // is only a fallback for clusters wider than the results pane itself.
      className={`inline-flex flex-col max-w-full border-2 rounded overflow-hidden bg-cream dark:bg-stone-800/50 transition ${
        isFocused
          ? "border-orange-700 dark:border-orange-500"
          : "border-stone-800 dark:border-stone-600"
      }`}
    >
      <div className="px-3 py-1.5 bg-cream-card dark:bg-stone-800 border-b-2 border-stone-800 dark:border-stone-600 flex items-center justify-between gap-3">
        <span className="font-mono text-sm text-stone-900 dark:text-white font-semibold">{result.video_id}</span>
        <span className="text-xs text-teal-700 dark:text-teal-400 font-semibold">
          match: {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex overflow-x-auto gap-1 p-1.5 scrollbar-thin">
        {steps.map((stepFrame, stepIndex) => (
          <div key={stepFrame.frame_id} className="shrink-0 w-48">
            <ResultCard
              ref={setCardRef(index, stepIndex)}
              result={stepFrame}
              rank={stepIndex + 1}
              imageLoading={index === 0 && stepIndex < 6 ? "eager" : "lazy"}
              onClick={(r) => { onSelect(index, stepIndex); onCardClick(r); }}
              badgeLabel={`Step ${stepIndex + 1}`}
              focused={isFocused && stepIndex === focusedStep}
              compact
            />
          </div>
        ))}
      </div>
      {onVerify && <div className="px-2 pb-2"><VerifyAction onClick={() => onVerify(result)} /></div>}
    </div>
  );
}

const ResultGrid = forwardRef<ResultGridHandle, Props>(function ResultGrid(
  { results, total, executionTimeMs, onCardClick, onVerify, scrollContainerRef, onFocusQuery, active },
  ref,
) {
  const isClusterView = results.some((r) => r.steps && r.steps.length > 1);
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRefMap = useRef(new Map<string, HTMLButtonElement>());
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [focusedStep, setFocusedStep] = useState(0);

  useImperativeHandle(ref, () => ({
    focus: () => containerRef.current?.focus(),
  }));

  // Reset selection whenever a new result set arrives.
  useEffect(() => {
    setFocusedIndex(0);
    setFocusedStep(0);
  }, [results]);

  function setCardRef(itemIndex: number, stepIndex: number) {
    return (node: HTMLButtonElement | null) => {
      const key = `${itemIndex}:${stepIndex}`;
      if (node) cardRefMap.current.set(key, node);
      else cardRefMap.current.delete(key);
    };
  }

  function getColumnCount(): number {
    const first = cardRefMap.current.get("0:0");
    if (!first) return 1;
    const firstTop = first.offsetTop;
    for (let i = 1; i < results.length; i++) {
      const node = cardRefMap.current.get(`${i}:0`);
      if (node && node.offsetTop !== firstTop) return i;
    }
    return Math.max(results.length, 1);
  }

  // Reveal the focused card only when it's actually outside the visible
  // area — scroll-into-view-if-needed, not re-center-every-time. Selection
  // now only moves via click (lands on an already-visible card → no-op) or
  // arrow keys (may go off-screen → minimal reveal), so this never fights
  // the mouse wheel the way the old always-recenter behavior did.
  useEffect(() => {
    const container = scrollContainerRef.current;
    const key = `${focusedIndex}:${isClusterView ? focusedStep : 0}`;
    const card = cardRefMap.current.get(key);
    if (!container || !card) return;
    const c = container.getBoundingClientRect();
    const k = card.getBoundingClientRect();
    if (k.top < c.top + ALIGN_TOP_PADDING) {
      container.scrollBy({ top: k.top - c.top - ALIGN_TOP_PADDING, behavior: "smooth" });
    } else if (k.bottom > c.bottom - ALIGN_TOP_PADDING) {
      container.scrollBy({ top: k.bottom - c.bottom + ALIGN_TOP_PADDING, behavior: "smooth" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedIndex, focusedStep, isClusterView]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (!active) return; // video modal is open — it owns the keyboard
    if (e.key === "Escape" && e.shiftKey) {
      // Dedicated combo for switching panels — plain Escape stays free for
      // closing the video modal without ambiguity.
      e.preventDefault();
      onFocusQuery();
      return;
    }
    if (results.length === 0) return;

    // Both layouts can now pack multiple items per visual row (clusters wrap
    // like the dense grid when they're narrow), so Up/Down always move by
    // the actual measured row width in either mode.
    const cols = getColumnCount();
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusedIndex((i) => Math.max(0, i - cols));
      if (isClusterView) setFocusedStep(0);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusedIndex((i) => Math.min(results.length - 1, i + cols));
      if (isClusterView) setFocusedStep(0);
      return;
    }
    if (isClusterView) {
      // At the first/last step of a cluster, Left/Right spill over into the
      // previous/next cluster instead of stopping dead.
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (focusedStep > 0) {
          setFocusedStep(focusedStep - 1);
        } else if (focusedIndex > 0) {
          const prevStepCount = results[focusedIndex - 1]?.steps?.length ?? 1;
          setFocusedIndex(focusedIndex - 1);
          setFocusedStep(prevStepCount - 1);
        }
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const stepCount = results[focusedIndex]?.steps?.length ?? 1;
        if (focusedStep < stepCount - 1) {
          setFocusedStep(focusedStep + 1);
        } else if (focusedIndex < results.length - 1) {
          setFocusedIndex(focusedIndex + 1);
          setFocusedStep(0);
        }
        return;
      }
    } else {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setFocusedIndex((i) => Math.max(0, i - 1));
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setFocusedIndex((i) => Math.min(results.length - 1, i + 1));
        return;
      }
    }

    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation(); // don't let it bubble to any outer handler
      const item = results[focusedIndex];
      if (!item) return;
      const target = isClusterView ? (item.steps?.[focusedStep] ?? item) : item;
      onCardClick(target);
    }
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className="space-y-4 focus:outline-none"
    >
      {/* Stats bar */}
      <div className="flex items-center gap-4 text-sm text-stone-500 dark:text-stone-400 font-mono">
        <span>
          <span className="text-stone-900 dark:text-white font-semibold">{total}</span> results
        </span>
        <span>·</span>
        <span title="Total search time">
          <span className="text-stone-900 dark:text-white font-semibold">{executionTimeMs}</span> ms
        </span>
      </div>

      {isClusterView ? (
        <div className="flex flex-wrap items-start gap-3">
          {results.map((r, i) => (
            <ClusterRow
              key={r.frame_id}
              result={r}
              index={i}
              focusedStep={focusedStep}
              isFocused={i === focusedIndex}
              setCardRef={setCardRef}
              onCardClick={onCardClick}
              onVerify={onVerify}
              onSelect={(itemIndex, stepIndex) => { setFocusedIndex(itemIndex); setFocusedStep(stepIndex); }}
            />
          ))}
        </div>
      ) : (
        // Grid — sized for a half-width results pane, not the full viewport
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {results.map((r, i) => (
            <div key={r.frame_id}>
              <ResultCard
              ref={setCardRef(i, 0)}
              result={r}
              rank={i + 1}
              onClick={(res) => { setFocusedIndex(i); onCardClick(res); }}
              focused={i === focusedIndex}
              />
              {onVerify && <VerifyAction onClick={() => onVerify(r)} />}
            </div>
          ))}
        </div>
      )}

      {results.length === 0 && (
        <p className="text-stone-500 text-center py-16">No results. Try a different query.</p>
      )}
    </div>
  );
});

export default ResultGrid;
