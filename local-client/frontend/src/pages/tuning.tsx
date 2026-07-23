import Head from "next/head";
import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";

import {
  deleteStrategyConfig,
  fetchStrategies,
  fetchStrategyConfigs,
  saveStrategyConfig,
  strategyConfigUpdateKey,
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
  const [newId, setNewId] = useState("");
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
        setDirty(false);
        setStatus("Saved");
      })
      .catch((error) => setStatus(error instanceof Error ? error.message : "Failed to load configs"));
  }, [strategyId, router.query.config]);

  useEffect(() => {
    if (!dirty) return;
    setStatus("Saving…");
    const timer = window.setTimeout(() => {
      saveStrategyConfig(strategyId, configId, weights)
        .then((saved) => {
          setConfigs((current) => current.map((config) => config.id === saved.id ? saved : config));
          setDirty(false);
          setStatus(`Saved revision ${saved.revision}`);
          window.localStorage.setItem(
            strategyConfigUpdateKey(strategyId, saved.id),
            `${saved.revision}:${Date.now()}`,
          );
        })
        .catch((error) => setStatus(error instanceof Error ? error.message : "Save failed"));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [configId, dirty, strategyId, weights]);

  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === configId),
    [configId, configs],
  );

  function chooseConfig(nextId: string) {
    const selected = configs.find((config) => config.id === nextId);
    if (!selected) return;
    setConfigId(selected.id);
    setWeights(selected.weights);
    setDirty(false);
    setStatus("Saved");
  }

  function updateWeight(key: string, value: StrategyConfigValue) {
    setWeights((current) => ({ ...current, [key]: value }));
    setDirty(true);
  }

  async function saveAs() {
    const id = newId.trim();
    if (!id) return;
    if (configs.some((config) => config.id === id)) {
      setStatus(`Config '${id}' already exists`);
      return;
    }
    try {
      const saved = await saveStrategyConfig(strategyId, id, weights);
      setConfigs((current) => [...current.filter((config) => config.id !== saved.id), saved]);
      setConfigId(saved.id);
      setNewId("");
      setDirty(false);
      setStatus(`Saved revision ${saved.revision}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Save failed");
    }
  }

  async function removeCurrent() {
    if (configId === "default") return;
    if (!window.confirm(`Delete config '${configId}'?`)) return;
    try {
      await deleteStrategyConfig(strategyId, configId);
      const fallback = configs.find((config) => config.id === "default")!;
      setConfigs((current) => current.filter((config) => config.id !== configId));
      setConfigId("default");
      setWeights(fallback.weights);
      setDirty(false);
      setStatus("Preset deleted");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Delete failed");
    }
  }

  return (
    <>
      <Head><title>Strategy Weight Tuning</title></Head>
      <main className="min-h-screen bg-cream dark:bg-stone-900 text-stone-900 dark:text-stone-50 p-5 md:p-8">
        <div className="max-w-3xl mx-auto space-y-6">
          <header className="flex items-start justify-between gap-4">
            <div>
              <h1 className="font-retro text-2xl font-bold uppercase">Strategy Weight Tuning</h1>
              <p className="text-sm text-stone-500 dark:text-stone-400">Saved presets are available to the search UI and `/config` command.</p>
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
                      <div className="flex gap-2">
                        <button
                          type="button"
                          disabled={values.length >= (field.max_items ?? Number.POSITIVE_INFINITY)}
                          onClick={() => updateWeight(key, [...values, 1])}
                          className="font-retro text-xs px-3 py-1 border-2 border-stone-700 dark:border-stone-400 rounded disabled:opacity-40"
                        >+ Event</button>
                        <button
                          type="button"
                          disabled={values.length <= (field.min_items ?? 0)}
                          onClick={() => updateWeight(key, values.slice(0, -1))}
                          className="font-retro text-xs px-3 py-1 border-2 border-stone-400 rounded disabled:opacity-40"
                        >- Event</button>
                      </div>
                    </fieldset>
                  );
                }
                const value = typeof current === "number" ? current : 1;
                return <WeightControl key={key} label={field.label} field={field} value={value} onChange={(next) => updateWeight(key, next)} />;
              })}
            </div>

            <p className="text-xs text-stone-500 dark:text-stone-400">
              {status}{selectedConfig ? ` · ${selectedConfig.id}@${selectedConfig.revision}` : ""}
            </p>
          </section>

          <section className="flex flex-wrap items-end gap-2">
            <label className="flex-1 min-w-52 space-y-1">
              <span className="text-xs font-bold uppercase text-stone-500">New preset ID</span>
              <input value={newId} onChange={(event) => setNewId(event.target.value)} placeholder="transcript-heavy" className={`${INPUT} w-full`} />
            </label>
            <button onClick={saveAs} disabled={!newId.trim()} className="font-retro px-4 py-2 border-2 border-stone-900 dark:border-stone-100 rounded bg-orange-700 text-white disabled:opacity-40">Save as</button>
            <button onClick={removeCurrent} disabled={configId === "default"} className="font-retro px-4 py-2 border-2 border-rose-700 rounded text-rose-700 dark:text-rose-400 disabled:opacity-40">Delete</button>
          </section>
        </div>
      </main>
    </>
  );
}
