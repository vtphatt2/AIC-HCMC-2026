// Original result URLs remain authoritative for playback and submissions.
export function cardImageUrl(original: string, format: "jpeg" | "webp" = "jpeg"): string | null {
  if ((process.env.NEXT_PUBLIC_FRAME_DECODE || "server").toLowerCase() === "client") return null;
  return /\/api\/zip-frame\/[^/?#]+\/\d+(?:\?frame_number=\d+(?:&v=[4567])?)?$/.test(original)
    ? `${original}${original.includes("?") ? "&" : "?"}width=640${format === "webp" ? "&format=webp" : ""}`
    : null;
}

export function needsOriginalFrame(cssWidth: number, devicePixelRatio: number): boolean {
  return cssWidth > 0 && cssWidth * Math.max(1, devicePixelRatio) > 640;
}

export function isConstrainedConnection(): boolean {
  if (typeof navigator === "undefined") return false;
  const connection = (navigator as Navigator & { connection?: {
    saveData?: boolean; effectiveType?: string; downlink?: number;
  } }).connection;
  return Boolean(connection && (connection.saveData ||
    ["slow-2g", "2g", "3g"].includes(connection.effectiveType || "") ||
    (typeof connection.downlink === "number" && connection.downlink > 0 && connection.downlink <= 2)));
}

const loadedCards = new Map<string, string>();
const prefetched = new Set<string>();
const prefetching = new Map<string, HTMLImageElement>();

export function rememberCardImage(original: string, displayed: string): void {
  if (displayed.startsWith("blob:")) return;
  loadedCards.delete(original);
  loadedCards.set(original, displayed);
  if (loadedCards.size > 256) loadedCards.delete(loadedCards.keys().next().value!);
}

export function loadedCardImage(original: string): string | undefined {
  return loadedCards.get(original);
}

export function prefetchOriginalFrame(original: string): void {
  if (typeof Image === "undefined" || isConstrainedConnection() ||
      loadedCards.get(original) === new URL(original, document.baseURI).href ||
      prefetched.has(original) || prefetching.has(original) || prefetching.size >= 2) return;
  const image = new Image();
  prefetching.set(original, image);
  image.fetchPriority = "low";
  image.onload = () => {
    prefetching.delete(original);
    prefetched.add(original);
    if (prefetched.size > 128) prefetched.delete(prefetched.values().next().value!);
  };
  image.onerror = () => { prefetching.delete(original); };
  image.src = original;
}
