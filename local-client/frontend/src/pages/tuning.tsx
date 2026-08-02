import Head from "next/head";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

import {
  fetchStrategies,
  fetchStrategyConfigs,
  fetchStrategyConfigDraft,
  saveStrategyConfigDraft,
} from "@/lib/api";
import type {
  Strategy,
  StrategyConfigField,
  StrategyConfigPreset,
  StrategyConfigValue,
} from "@/types";


const INPUT = "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-3 py-2 text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";


function WeightControl({ label, field, value, onChange }: {
  label: string;
  field: StrategyConfigField;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block space-y-2">
      <div className="flex justify-between gap-4 text-sm">
        <span>{label}</span>
        <input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className={`${INPUT} w-24 text-right py-1`}
        />
      </div>
      <input
        type="range"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-orange-700"
      />
    </label>
  );
}


export default function TuningPage() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [strategyId, setStrategyId] = useState("");
  const [configs, setConfigs] = useState<StrategyConfigPreset[]>([]);
  const [schema, setSchema] = useState<Record<string, StrategyConfigField>>({});
  const [configId, setConfigId] = useState("default");
  const [weights, setWeights] = useState<Record<string, StrategyConfigValue>>({});
  const [draftRevision, setDraftRevision] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState("Loading…");

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark",
      window.localStorage.getItem("aic2026-theme") !== "light",
    );
  }, []);

  useEffect(() => {
    if (!router.isReady) return;
    fetchStrategies()
      .then((list) => {
        const tunable = list.filter((strategy) => strategy.configurable);
        setStrategies(tunable);
        const requested = typeof router.query.strategy === "string" ? router.query.strategy : "";
        setStrategyId(tunable.some((strategy) => strategy.id === requested) ? requested : (tunable[0]?.id || ""));
      })
      .catch((error) => setStatus(error instanceof Error ? error.message : "Failed to load strategies"));
  }, [router.isReady, router.query.strategy]);

  useEffect(() => {
    if (!strategyId) return;
    setStatus("Loading…");
    fetchStrategyConfigs(strategyId)
      .then((payload) => {
        const requested = typeof router.query.config === "string" ? router.query.config : "default";
        const selected = payload.configs.find((config) => config.id === requested) || payload.configs[0];
        setSchema(payload.schema);
        setConfigs(payload.configs);
        setConfigId(selected.id);
        setWeights(selected.weights);
        setDraftRevision(0);
        setDirty(false);
        setStatus("Saved");
      })
      .catch((error) => setStatus(error instanceof Error ? error.message : "Failed to load configs"));
  }, [strategyId, router.query.config]);

  useEffect(() => {
    if (!dirty) return;
    setStatus("Saving…");
    const timer = window.setTimeout(() => {
      saveStrategyConfigDraft(strategyId, configId, weights)
        .then((draft) => {
          setDraftRevision(draft.revision);
          setDirty(false);
          setStatus(`Saved local draft ${draft.revision}`);
        })
        .catch((error) => setStatus(error instanceof Error ? error.message : "Save failed"));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [configId, dirty, strategyId, weights]);

  useEffect(() => {
    if (!strategyId || !configId || dirty) return;
    const preset = configs.find((config) => config.id === configId);
    if (!preset) return;
    let revision = draftRevision;
    let cancelled = false;

    async function refreshDraft() {
      try {
        const draft = await fetchStrategyConfigDraft(strategyId, configId);
        if (cancelled) return;
        if (draft.revision === revision) return;
        revision = draft.revision;
        setDraftRevision(draft.revision);
        setWeights({ ...preset.weights, ...draft.overrides });
        setStatus(`Synced local draft ${draft.revision}`);
      } catch {
        // Keep sliders usable through a temporary network interruption.
      }
    }

    void refreshDraft();
    const timer = window.setInterval(refreshDraft, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [configId, configs, dirty, draftRevision, strategyId]);

  const selectedConfig = configs.find((config) => config.id === configId);

  function chooseConfig(nextId: string) {
    const selected = configs.find((config) => config.id === nextId);
    if (!selected) return;
    setConfigId(selected.id);
    setWeights(selected.weights);
    setDraftRevision(0);
    setDirty(false);
    setStatus("Loading local draft...");
  }

  function updateWeight(key: string, value: StrategyConfigValue) {
    setWeights((current) => ({ ...current, [key]: value }));
    setDirty(true);
  }

  return (
    <>
      <Head><title>Strategy Weight Tuning</title></Head>
      <main className="min-h-screen bg-cream dark:bg-stone-900 text-stone-900 dark:text-stone-50 p-5 md:p-8">
        <div className="max-w-3xl mx-auto space-y-6">
          <header className="flex items-start justify-between gap-4">
            <div>
              <h1 className="font-retro text-2xl font-bold uppercase">Strategy Weight Tuning</h1>
              <p className="text-sm text-stone-500 dark:text-stone-400">Presets stay read-only on the backend; tuning is saved on this frontend machine.</p>
            </div>
            <a href="/" className="font-retro text-sm text-orange-700 dark:text-orange-400 hover:underline">Back to search</a>
          </header>

          <section className="border-2 border-stone-800 dark:border-stone-500 rounded p-4 bg-cream-card dark:bg-stone-800 space-y-4">
            <div className="grid sm:grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-xs font-bold uppercase text-stone-500">Strategy</span>
                <select value={strategyId} onChange={(event) => setStrategyId(event.target.value)} className={`${INPUT} w-full`}>
                  {strategies.map((strategy) => <option key={strategy.id} value={strategy.id}>{strategy.name}</option>)}
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-xs font-bold uppercase text-stone-500">Config</span>
                <select value={configId} onChange={(event) => chooseConfig(event.target.value)} className={`${INPUT} w-full`}>
                  {configs.map((config) => <option key={config.id} value={config.id}>{config.id}</option>)}
                </select>
              </label>
            </div>

            <div className="space-y-4">
              {Object.entries(schema).map(([key, field]) => {
                const current = weights[key] ?? field.default;
                if (field.type === "number_list") {
                  const values = Array.isArray(current) ? current : [];
                  return (
                    <fieldset key={key} className="space-y-4 border-t border-stone-300 dark:border-stone-600 pt-4">
                      <legend className="font-bold pr-3">{field.label}</legend>
                      {values.map((value, index) => (
                        <WeightControl
                          key={index}
                          label={(field.item_label || "Item {index}").replace("{index}", String(index + 1))}
                          field={field}
                          value={value}
                          onChange={(next) => updateWeight(key, values.map((item, itemIndex) => itemIndex === index ? next : item))}
                        />
                      ))}
                    </fieldset>
                  );
                }
                const value = typeof current === "number" ? current : 1;
                return <WeightControl key={key} label={field.label} field={field} value={value} onChange={(next) => updateWeight(key, next)} />;
              })}
            </div>

            <p className="text-xs text-stone-500 dark:text-stone-400">
              {status}{selectedConfig ? ` · preset ${selectedConfig.id}@${selectedConfig.revision} · draft@${draftRevision}` : ""}
            </p>
          </section>

        </div>
      </main>
    </>
  );
}
