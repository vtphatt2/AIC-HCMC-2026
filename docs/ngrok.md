# Ngrok — Setup and Usage

Ngrok shows up in this repo in two unrelated places. Pick the one matching
what you're running:

| You're running | Ngrok exposes | Section |
|---|---|---|
| `local-backend` + frontend on your own laptop, want a teammate to try it | one tunnel, both dev servers behind `share-proxy.cjs` | [Sharing a local dev instance](#sharing-a-local-dev-instance) |
| `remote-server` on the GPU workstation, teammates run `local-backend` in `ENV_MODE=LOCAL` against it | port 8000 only | [Exposing the remote server](#exposing-the-remote-server) |

Both need ngrok installed once. Skip to whichever section matches your case.

---

## One-time ngrok setup

1. Create a free account at [ngrok.com](https://ngrok.com) and install the
   `ngrok` CLI so it's on `PATH` (`ngrok version` should print something).
2. Add your authtoken once — this writes to ngrok's own config file
   (`~/AppData/Local/ngrok/ngrok.yml` on Windows), **not** anything in this
   repo:
   ```bash
   ngrok config add-authtoken <your-authtoken>
   ```
3. (Recommended) Claim a free static domain from the ngrok dashboard
   (`https://dashboard.ngrok.com/domains`) — something like
   `your-name.ngrok-free.dev`. Without a static domain the tunnel gets a new
   random URL every time it restarts, which is annoying to redistribute.

Nothing in this repo reads `NGROK_AUTHTOKEN` — the `.env.example` entry for
it is only a note of where the value belongs. `NGROK_DOMAIN` **is** read, by
`scripts/start-local.ps1 -Ngrok`, from the repo-root `.env`.

---

## Sharing a local dev instance

Use this to let a teammate use your laptop's `local-backend` + frontend
(`ZIP` or `LOCAL` mode) from their own browser, without them setting up
anything.

Windows only — `scripts/start-local.ps1` has an `-Ngrok` switch;
`start-local.sh` (Mac/Linux/WSL) does not currently support it. On those
platforms, run `share-proxy.cjs` and `ngrok http 3001` manually (same idea,
see [How it works](#how-it-works) below).

### 1. Set your domain

Repo-root `.env` (copy from `.env.example` if you don't have one yet):

```env
NGROK_DOMAIN=your-name.ngrok-free.dev
```

Or pass `-NgrokDomain your-name.ngrok-free.dev` on the command line instead.

### 2. Start everything

```powershell
scripts\start-local.ps1 -Ngrok
```

This opens four terminal windows: backend, frontend, `share-proxy.cjs` (port
3001), and `ngrok http 3001 --domain <your-domain>`. Share
`https://your-name.ngrok-free.dev` with your teammate — that's it, one URL,
nothing else to configure on their end.

Combine with other flags as needed, e.g. `-Backend torch-cuda -Ngrok`.

### How it works

`share-proxy.cjs` is a ~50-line Node script (stdlib only, nothing to
install) that sits in front of both dev servers on one port:

| Path | Goes to |
|---|---|
| `/api/tuning-draft` | Next dev server (it's a page route reading a local file) |
| `/api/*`, `/static/*` | backend on port 8000 |
| everything else | Next dev server on port 3000 |

`-Ngrok` also sets `NEXT_PUBLIC_API_URL=/`, so the browser calls whatever
origin served the page (the tunnel) instead of hardcoding `localhost:8000`.
One tunnel, no CORS setup needed, nothing bound to a public port directly.

**Don't replace this with `next.config.js` rewrites.** That was tried first
and killed the Next dev server: a result grid loads up to 100
`/api/zip-frame` thumbnails, the browser aborts the in-flight ones on every
new search, and the resulting ECONNRESET storm took the process down. Next
is a compiler, not a proxy — keep the media traffic off it, which is the
whole reason `share-proxy.cjs` exists.

### What your teammate sees

- **First visit**: the free ngrok plan shows an interstitial "you are about
  to visit..." warning page. They click "Visit Site" once; a cookie
  suppresses it after that.
- **Performance**: everyone is sharing your one CPU-bound backend process —
  concurrent searches from multiple people queue rather than run in
  parallel.

---

## Exposing the remote server

Use this on the GPU workstation running `remote-server` (`ENV_MODE=SERVER`),
so teammates can point their own `local-backend` at it with
`ENV_MODE=LOCAL` (see [setup.md Option B](setup.md#option-b--local-backend-connected-to-gpu-server-local-mode)).

Unlike the local-sharing case above, this exposes the backend port directly
— no `share-proxy.cjs` involved, since there's no frontend dev server to
protect on this machine.

### 1. Start the server normally

```bash
bash scripts/start-remote.sh
```

or `scripts\start-remote.ps1` on Windows. See
[running.md → Remote server](running.md#remote-server-server-mode).

### 2. Allow the tunnel's origin in CORS

`remote-server/.env`:

```env
CORS_ORIGINS=http://localhost:3000,https://your-name.ngrok-free.dev
```

Restart the backend after editing `.env` (`--reload` only covers `.py`
edits).

### 3. Start the tunnel

```bash
ngrok http 8000
```

Or with a claimed static domain:

```bash
ngrok http 8000 --domain your-name.ngrok-free.dev
```

### 4. Share the URL

Give teammates the `https://...ngrok-free.dev` (or `.ngrok.io`) URL shown in
the ngrok terminal / at `http://localhost:4040`. They set it in their own
`local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=https://your-name.ngrok-free.dev
```

Verify from their machine:

```bash
curl https://your-name.ngrok-free.dev/api/health
```

If that 404s or hangs, check `docker compose ps` / the backend terminal on
the host machine before suspecting the tunnel.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ngrok not found on PATH` | CLI not installed, or not on `PATH` | Reinstall, confirm `ngrok version` works in a fresh terminal |
| `Set NGROK_DOMAIN in .env or pass -NgrokDomain` | `-Ngrok` used without a domain configured | Add `NGROK_DOMAIN=...` to repo-root `.env`, or pass `-NgrokDomain` |
| `Ngrok: agent already running (port 4040 in use)` | A previous `ngrok` process is still up | Fine to ignore if it's the tunnel you want; otherwise close the old terminal window first |
| Teammate sees an interstitial warning page | Free ngrok plan behavior, not a bug | They click "Visit Site" once; a cookie suppresses it afterward |
| CORS error in the browser console (remote-server case) | Tunnel origin missing from `CORS_ORIGINS` | Add it to `remote-server/.env` and restart the backend |
| Thumbnails/searches fail only when shared, work locally | `next.config.js` rewrites used instead of `share-proxy.cjs`, or `NEXT_PUBLIC_API_URL` not set to `/` | Use `-Ngrok` as documented above — don't hand-roll the proxy |
| Tunnel URL changes every restart | No static domain claimed | Claim one free static domain in the ngrok dashboard and reuse it |
