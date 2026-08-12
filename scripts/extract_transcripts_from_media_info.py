"""Extract transcripts from media-info JSONs → JSONL (multi-threaded, graceful backoff)."""
import json, re, time, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from http.cookiejar import MozillaCookieJar
import requests

MEDIA_DIR = Path("AIC2026_sample/media-info-aic25-b1/media-info")
OUT_DIR = Path("AIC2026_sample/transcripts")
COOKIES_FILE = Path("cookies.txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YT_ID_RE = re.compile(r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})")

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

print_lock = Lock()
rate_limiter = {"last_request": 0, "min_interval": 0.5, "backoff_until": 0}

def wait_rate_limit():
    """Centralized rate limiter with slot-based scheduling."""
    with print_lock:
        now = time.time()
        if now < rate_limiter["backoff_until"]:
            wait = rate_limiter["backoff_until"] - now
            time.sleep(wait)
            now = time.time()
        gap = now - rate_limiter["last_request"]
        if gap < rate_limiter["min_interval"]:
            time.sleep(rate_limiter["min_interval"] - gap)
        rate_limiter["last_request"] = time.time()

def apply_backoff(multiplier=1.0):
    """Graceful backoff: increase interval exponentially on rate limit."""
    with print_lock:
        rate_limiter["min_interval"] = min(rate_limiter["min_interval"] * 2, 10.0)
        rate_limiter["backoff_until"] = time.time() + 10 * multiplier

def reduce_backoff():
    """Gradually reduce interval back to normal on success."""
    with print_lock:
        rate_limiter["min_interval"] = max(rate_limiter["min_interval"] * 0.8, 0.5)

def create_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    if COOKIES_FILE.exists():
        cj = MozillaCookieJar(COOKIES_FILE)
        cj.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(cj)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        session.proxies = {"https": proxy, "http": proxy}
    return session

def process_one(fp, total, idx):
    vid = fp.stem
    out_path = OUT_DIR / f"{vid}.jsonl"
    if out_path.exists():
        with print_lock:
            print(f"  [{idx}/{total}] SKIP {vid}")
        return ("skip", vid)

    data = json.loads(fp.read_text())
    url = data.get("watch_url", "")
    m = YT_ID_RE.search(url)
    if not m:
        with print_lock:
            print(f"  [{idx}/{total}] FAIL {vid} — no YouTube ID")
        return ("fail", vid)

    yt_id = m.group(1)
    session = create_session()
    yta = YouTubeTranscriptApi(http_client=session)

    for attempt in range(10):
        wait_rate_limit()
        try:
            t = yta.fetch(yt_id, languages=["vi", "en"])
            break
        except (TranscriptsDisabled, NoTranscriptFound):
            return ("fail", f"{vid} — no transcript")
        except VideoUnavailable:
            return ("fail", f"{vid} — video unavailable")
        except Exception as e:
            err = str(e)
            if "IpBlocked" in err or "Too Many" in err or "429" in err:
                wait = min(30 * (attempt + 1), 180)
                with print_lock:
                    print(f"  [{idx}/{total}] BLOCK {vid} — backoff {wait}s (attempt {attempt+1}/10)")
                apply_backoff(attempt + 1)
                time.sleep(wait)
                continue
            return ("fail", f"{vid} — {type(e).__name__}: {err[:80]}")
    else:
        return ("fail", f"{vid} — blocked after 10 retries")

    snips = t.snippets
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(snips):
            start_ms = int(s.start * 1000)
            end_ms = int(snips[i + 1].start * 1000) if i + 1 < len(snips) else start_ms + 5000
            f.write(json.dumps({"start_time_ms": start_ms, "end_time_ms": end_ms, "text": s.text}, ensure_ascii=False) + "\n")

    reduce_backoff()
    with print_lock:
        print(f"  [{idx}/{total}] OK   {vid} — {len(snips)} segments")
    return ("ok", vid)

def main():
    files = sorted(MEDIA_DIR.glob("*.json"))
    total = len(files)

    existing = sum(1 for fp in files if (OUT_DIR / f"{fp.stem}.jsonl").exists())
    pending = total - existing

    if COOKIES_FILE.exists():
        print(f"Cookies: {COOKIES_FILE}")

    print(f"Total: {total} | Done: {existing} | Pending: {pending}\n")

    if pending == 0:
        print("All done.")
        return

    workers = min(max(4, os.cpu_count() or 4), 16)  # ponytail: 4-16 threads, enough
    print(f"Workers: {workers}\n")

    stats = {"ok": 0, "skip": 0, "fail": 0}
    futures_map = {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        idx = 0
        for fp in files:
            idx += 1
            if (OUT_DIR / f"{fp.stem}.jsonl").exists():
                stats["skip"] += 1
                continue
            f = ex.submit(process_one, fp, total, idx)
            futures_map[f] = fp

        for f in as_completed(futures_map):
            status, info = f.result()
            stats[status] += 1
            completed = stats["ok"] + stats["fail"]
            if completed % 50 == 0 and completed > 0:
                with print_lock:
                    print(f"  --- Progress: {completed}/{pending} (OK={stats['ok']} FAIL={stats['fail']}) ---")

    print(f"\nDone. OK={stats['ok']} SKIP={stats['skip']} FAIL={stats['fail']}")

if __name__ == "__main__":
    main()
