import { useEffect } from "react";

interface Props {
  onClose: () => void;
}

// A key/command row: mono "key" chip on the left, description on the right.
function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-1">
      <code className="shrink-0 font-mono text-xs bg-stone-100 dark:bg-stone-800 border border-stone-300 dark:border-stone-600 rounded px-1.5 py-0.5 text-orange-800 dark:text-orange-300 whitespace-nowrap">
        {k}
      </code>
      <span className="text-sm text-stone-700 dark:text-stone-300 leading-snug">{children}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1">
      <h3 className="font-retro text-sm font-bold uppercase tracking-wide text-orange-700 dark:text-orange-400 border-b-2 border-stone-200 dark:border-stone-700 pb-1">
        {title}
      </h3>
      <div className="pt-1">{children}</div>
    </section>
  );
}

export default function HelpModal({ onClose }: Props) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl max-h-[85vh] flex flex-col bg-cream dark:bg-stone-900 border-2 border-stone-800 dark:border-stone-500 rounded shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b-2 border-stone-800 dark:border-stone-600 shrink-0">
          <h2 className="font-retro text-lg font-bold text-stone-900 dark:text-stone-50">
            How to use <span className="text-orange-700 dark:text-orange-400">AIC 2026</span>
          </h2>
          <button
            onClick={onClose}
            className="font-retro text-stone-500 hover:text-orange-700 dark:hover:text-orange-400 text-xs uppercase tracking-wide transition"
          >
            ✕ Close (Esc)
          </button>
        </div>

        {/* Scrollable body */}
        <div className="overflow-y-auto px-5 py-4 space-y-5">
          <p className="text-sm text-stone-600 dark:text-stone-400 leading-relaxed">
            Search indexed videos by <strong>image</strong> (semantic scene description),
            by <strong>text</strong> (OCR / transcript keywords), or both — then browse the
            matching frames and jump straight to the moment in the video.
          </p>

          <Section title="Two ways to build a query">
            <div className="space-y-2 text-sm text-stone-700 dark:text-stone-300 leading-snug">
              <p>
                <code className="font-mono text-xs">✎ Manual</code> — the classic form: type in
                the Semantic / Text boxes, add temporal steps, pick strategy / Top&nbsp;K / genre,
                then press <strong>Enter</strong> (in any input) or hit <strong>Search</strong>.
              </p>
              <p>
                <code className="font-mono text-xs">&gt;_ Chat</code> — a command bar (like a code
                agent). Type a query and press Enter to search instantly, or use slash-commands
                for everything else. Toggle between the two with the ✎ / &gt;_ buttons at the top
                of this panel.
              </p>
            </div>
          </Section>

          <Section title="Search modes">
            <Row k="Frames">Visual + text search over video keyframes (the main mode).</Row>
            <Row k="Transcripts">Search transcript chunks by topic/keywords.</Row>
            <Row k="Score / Video">
              Two ways to view results: a flat grid ranked by score, or frames grouped per video.
            </Row>
            <Row k="Temporal step">
              Chain multiple sub-queries with a time gap (e.g. “red car” then “person walking”
              N seconds later) to find a sequence, not just one frame.
            </Row>
          </Section>

          <Section title="Command bar (chat mode)">
            <Row k="<text> ⏎">Set the active step’s semantic query and search.</Row>
            <Row k="/step add [x]">Insert a step at position x (default: end).</Row>
            <Row k="/step del [x]">Remove step x (default: last).</Row>
            <Row k="/step clear [x]">Clear a step’s content.</Row>
            <Row k="/step <n>">Jump to step n.</Row>
            <Row k="/text <value>">Set the active step’s OCR/transcript text field.</Row>
            <Row k="/translate on|off">Toggle VI→EN translation for the active step.</Row>
            <Row k="/mode frames|transcripts">Switch search mode.</Row>
            <Row k="/view score|video">Switch results view.</Row>
            <Row k="/topk <n>">Set number of results.</Row>
            <Row k="/genre <name>">Filter by genre.</Row>
            <Row k="/strategy <id>">Choose the search strategy.</Row>
            <Row k="/transcript on|off">Show/hide the transcript panel in the video modal.</Row>
            <Row k="/search">Run the search now.</Row>
            <Row k="/clear">Reset to a single empty step.</Row>
            <Row k="/help">List all commands.</Row>
          </Section>

          <Section title="Command bar keys">
            <Row k="Tab">Autocomplete / cycle commands and argument values.</Row>
            <Row k="↑ / ↓">Cycle suggestions, or recall command history / move between steps.</Row>
            <Row k="Enter">Confirm the current field and auto-advance to the next.</Row>
            <Row k="F2">Edit the active step’s field.</Row>
            <Row k="Esc">Cancel the current field / return to idle.</Row>
            <Row k="Shift+Esc">Jump focus between the command bar and the results.</Row>
            <Row k="Ctrl+Z">Undo the last step edit.</Row>
            <Row k="Ctrl+Enter">Search immediately, from anywhere.</Row>
          </Section>

          <Section title="Browsing results">
            <Row k="← ↑ → ↓">Move the selection between result frames.</Row>
            <Row k="Enter / Space">Open the focused frame in the video player.</Row>
            <Row k="Click / Hover">Click a card to open it; hover just previews it.</Row>
            <Row k="Shift+Esc">Move focus back to the command bar.</Row>
          </Section>

          <Section title="Video player">
            <Row k="Enter / Space">Play / pause (first press starts at the exact frame).</Row>
            <Row k="← / →">Seek backward / forward 5 seconds.</Row>
            <Row k="Esc">Close the player.</Row>
            <Row k="Transcript">
              Toggle the side transcript panel with the button in the player, or
              <code className="font-mono text-xs"> /transcript on|off</code>. It follows along as
              the video plays.
            </Row>
          </Section>

          <p className="text-xs text-stone-500 dark:text-stone-500 pt-1">
            Full written guide: <code className="font-mono">docs/USAGE.md</code> in the repo.
          </p>
        </div>
      </div>
    </div>
  );
}
