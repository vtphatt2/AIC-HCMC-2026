# AIC 2026 — Usage Guide

The playground lets you search indexed videos by **image** (semantic scene
description), by **text** (OCR / transcript keywords), or both — then browse the
matching frames and jump straight to the moment in the video.

This guide is also available in-app: click the **?** icon at the top of the left
panel (next to the light/dark toggle).

---

## Two ways to build a query

The left panel has a **✎ Manual** / **>_ Chat** toggle at the top.

- **✎ Manual** — the classic form. Type in the Semantic / Text boxes, add
  temporal steps, pick strategy / Top K / genre, then press **Enter** in any
  input or click **Search**.
- **>_ Chat** — a command bar, like a code agent. Type a query and press Enter to
  search instantly, or use slash-commands for everything else. Selection and
  query state are shared with Manual mode, so you can switch freely.

## Search modes

| Mode | What it does |
|---|---|
| **Frames** | Visual + text search over video keyframes (the main mode). |
| **Transcripts** | Search transcript chunks by topic / keywords. |
| **Score / Video view** | Flat grid ranked by score, or frames grouped per video. |
| **Temporal step** | Chain sub-queries with a time gap (e.g. "red car" then "person walking" N seconds later) to find a sequence, not just one frame. |

---

## Command bar (Chat mode)

Type plain text and press **Enter** to set the active step's semantic query and
search. Everything else is a slash-command:

| Command | Effect |
|---|---|
| `/step add [x]` | Insert a step at position x (default: end). Steps at x.. shift +1. |
| `/step del [x]` | Remove step x (default: last). |
| `/step clear [x]` | Clear a step's semantic / text (default: active step). |
| `/step <n>` | Jump to step n. |
| `/text <value>` | Set the active step's OCR / transcript text field. |
| `/mode frames\|transcripts` | Switch search mode. |
| `/view score\|video` | Switch results view. |
| `/topk <n>` | Set number of results to fetch. |
| `/genre <name>` | Filter results by genre. |
| `/strategy <id\|name>` | Choose the search strategy. |
| `/transcript on\|off` | Show/hide the transcript panel in the video modal. |
| `/search` | Run the search immediately. |
| `/clear` | Reset to a single empty step. |
| `/help` | List all commands. |

### Command bar keys

| Key | Action |
|---|---|
| `Tab` | Autocomplete / cycle commands and argument values. |
| `↑` / `↓` | Cycle suggestions, or recall command history / move between steps. |
| `Enter` | Confirm the current field and auto-advance to the next. |
| `F2` | Edit the active step's field. |
| `Esc` | Cancel the current field / return to idle. |
| `Shift+Esc` | Jump focus between the command bar and the results. |
| `Ctrl+Z` | Undo the last step edit. |
| `Ctrl+Enter` | Search immediately, from anywhere. |

---

## Browsing results

| Key / action | Effect |
|---|---|
| `←` `↑` `→` `↓` | Move the selection between result frames. |
| `Enter` / `Space` | Open the focused frame in the video player. |
| Click | Open that frame. |
| Hover | Preview only — does not change the selection. |
| `Shift+Esc` | Move focus back to the command bar. |

In **Score view**, multi-step temporal matches are grouped into cluster boxes
(one frame per step). In **Video view**, frames are grouped per video with the
best match centered and, when the transcript panel is on, the transcript around
the best frame shown with the top 1–3 moments highlighted.

## Video player

Opens **paused** on the frame image (press a key to start — it never autoplays).

| Key | Action |
|---|---|
| `Enter` / `Space` | Play / pause (first press starts at the exact frame). |
| `←` / `→` | Seek backward / forward 5 seconds. |
| `Esc` | Close the player. |

All controls work from the keyboard, so you never need to click into the video
(clicking the YouTube iframe would steal keyboard focus and break `Esc`).

Toggle the side **transcript panel** with the button in the player or
`/transcript on|off`. It scrolls along with playback, highlighting the current
line.
