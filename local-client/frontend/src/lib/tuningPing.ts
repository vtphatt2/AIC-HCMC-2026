// Cross-tab "the tuning draft changed" signal, so the query page and the
// tuning page don't have to poll each other once a second.
//
// The browser's native `storage` event fires only in *other* tabs of the same
// origin on the same machine — which is exactly the scope we want now that the
// app is shared over a tunnel: a teammate moving a slider must not re-trigger
// your search. It also fires only on an actual write, so a page that nobody is
// tuning makes zero requests.

const KEY = "aic:tuning-draft-ping";

export function pingTuningDraft(revision: number): void {
  try {
    localStorage.setItem(KEY, `${Date.now()}:${revision}`);
  } catch {
    // Private mode / storage disabled — the tuning page just won't sync live.
  }
}

export function onTuningDraftPing(handler: () => void): () => void {
  const listener = (event: StorageEvent) => {
    if (event.key === KEY) handler();
  };
  window.addEventListener("storage", listener);
  return () => window.removeEventListener("storage", listener);
}
