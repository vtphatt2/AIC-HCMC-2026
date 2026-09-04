import Head from "next/head";
import { useRouter } from "next/router";
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchSubmission, rowThumbUrl, useVideoInfo } from "@/lib/submission";
import type { SubmissionState } from "@/types";

const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition";

export default function SubmissionGridPage() {
  const router = useRouter();
  const session = typeof router.query.session === "string" ? router.query.session : "";
  const [state, setState] = useState<SubmissionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!session) return;
    try {
      setState(await fetchSubmission(session));
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load submission");
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark",
      window.localStorage.getItem("aic2026-theme") !== "light",
    );
  }, []);

  useEffect(() => {
    if (!router.isReady) return;
    if (!session) {
      setError("Missing submission session.");
      setLoading(false);
      return;
    }
    load();
    const interval = window.setInterval(load, 5000);
    return () => window.clearInterval(interval);
  }, [load, router.isReady, session]);

  const videoIds = useMemo(
    () => state?.rows.map((row) => row.videoId) ?? [],
    [state],
  );
  const videoInfo = useVideoInfo(videoIds);
  const frames = useMemo(
    () => state?.rows.flatMap((row, rowIndex) =>
      row.frames.map((frame, frameIndex) => ({ row, rowIndex, frame, frameIndex })),
    ) ?? [],
    [state],
  );

  return (
    <>
      <Head><title>{state ? `${state.session} — Grid review` : "Submission grid review"}</title></Head>
      <main className="min-h-screen bg-cream dark:bg-stone-900 text-stone-900 dark:text-stone-50 p-4 md:p-7">
        <div className="max-w-7xl mx-auto space-y-5">
          <header className="flex flex-wrap items-start justify-between gap-3 sticky top-0 z-10 bg-cream/95 dark:bg-stone-900/95 py-2 backdrop-blur">
            <div>
              <h1 className="font-retro text-xl md:text-2xl font-bold uppercase">
                {state?.session ?? (session || "Submission grid")}
              </h1>
              {state && (
                <p className="text-sm text-stone-500 dark:text-stone-400">
                  {state.queryType.toUpperCase()} · {state.rows.length} candidate{state.rows.length === 1 ? "" : "s"} · {frames.length} frame{frames.length === 1 ? "" : "s"}
                  {" · refreshes every 5 seconds"}
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <button className={BTN} onClick={load}>↻ Refresh</button>
              <a className={BTN} href="/submissions">← Dashboard</a>
            </div>
          </header>

          {loading && <p className="text-sm text-stone-500 italic">Loading submission…</p>}
          {error && <p className="border-2 border-red-600 rounded p-3 text-sm text-red-700 dark:text-red-400">{error}</p>}
          {!loading && !error && frames.length === 0 && (
            <p className="text-sm text-stone-500 italic">This submission has no frames yet.</p>
          )}

          {frames.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {frames.map(({ row, rowIndex, frame, frameIndex }) => {
                const fps = videoInfo[row.videoId]?.fps ?? 25;
                return (
                  <article
                    key={`${rowIndex}:${frameIndex}:${row.videoId}:${frame}`}
                    className="border-2 border-stone-800 dark:border-stone-600 rounded overflow-hidden bg-cream-card dark:bg-stone-800"
                  >
                    <img
                      src={rowThumbUrl(row.videoId, frame, fps)}
                      alt={`${row.videoId}, frame ${frame}`}
                      loading="lazy"
                      className="w-full aspect-video object-cover bg-stone-200 dark:bg-stone-700"
                    />
                    <div className="p-2 leading-tight">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-xs font-bold truncate" title={row.videoId}>{row.videoId}</span>
                        <span className="font-retro text-[10px] text-stone-500 shrink-0">
                          #{rowIndex + 1}{row.frames.length > 1 ? `.${frameIndex + 1}` : ""}
                        </span>
                      </div>
                      <p className="font-mono text-sm text-orange-700 dark:text-orange-400">frame {frame}</p>
                      {state?.queryType === "qa" && row.answer && (
                        <p className="text-xs mt-1 line-clamp-2" title={row.answer}>{row.answer}</p>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
