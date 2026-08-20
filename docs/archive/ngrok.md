# Ngrok — Share a Running Instance

Assumes the system is already running somewhere. This page only covers
turning on the tunnel — and, just as important, **verifying it actually
works before sending the link**, since the most common failure looks fine
(the page loads) while every search silently fails.

## Which scenario are you in?

| Where's the frontend? | Where's the backend? | Section |
|---|---|---|
| Same machine as the backend | `local-backend` or `remote-server`, same machine | [Case A](#case-a--frontend--backend-on-the-same-machine) |
| Not shared — each teammate runs their own | `remote-server`, only the backend is shared | [Case B](#case-b--only-a-backend-is-shared-remote-server) |

If you're not sure: **if you're sharing one link that shows the whole app**,
you're in Case A. **If you're sharing a URL that goes into someone else's
`.env` file** (`REMOTE_SERVER_URL`), you're in Case B.

---

## Case A — frontend + backend on the same machine

Covers both flavors of backend — `local-backend` and `remote-server`.
`share-proxy.cjs` forwards purely by port (3000 for the frontend, 8000 for
the backend); it has no idea which backend is actually listening there, so
the steps are identical either way.

### A1. Backend was started by `start-local.ps1`

Only true for `local-backend` — `start-local.ps1` cannot start
`remote-server`. Stop and restart with `-Ngrok` added, it wires up
everything for you:

```powershell
scripts\start-local.ps1 -Ngrok
```

Skip to [Verify before sharing](#verify-before-sharing).

### A2. Frontend and backend are already running in their own terminals

This is the only path when the backend is `remote-server` (since
`start-local.ps1` never manages it), and also works if you started
`local-backend` by hand instead of via the launch script.

**Do this in order — order matters, see the pitfall below:**

1. Confirm both are actually up:
   ```bash
   curl http://127.0.0.1:8000/api/health   # backend
   curl http://127.0.0.1:3000              # frontend
   ```

2. **Stop the frontend** if it's running, and restart it with
   `NEXT_PUBLIC_API_URL=/`:
   ```bash
   # in local-client/frontend
   set NEXT_PUBLIC_API_URL=/
   npm run dev
   ```
   ⚠️ **This is the step people skip, and it's the exact bug behind "the
   page loads but search does nothing."** `NEXT_PUBLIC_*` variables are
   baked in when Next.js *starts*, not read live — if the frontend was
   already running before you set this, it is still shipping the old
   value (typically `http://localhost:8000`) to every browser that loads
   the page, including your teammate's, where `localhost:8000` means
   *their own machine*, not yours. There is no fix except restarting the
   frontend process after the variable is set.

3. Start the proxy:
   ```bash
   # from the repo root
   set SHARE_PORT=3001
   set FRONTEND_PORT=3000
   set BACKEND_PORT=8000
   node scripts/share-proxy.cjs
   ```
   (The three `set` lines are only needed if you're using non-default
   ports; the values above are the defaults.)

4. Start the tunnel, pointed at **3001** (the proxy), never at 3000 or 8000
   directly:
   ```bash
   ngrok http 3001
   # or, with a claimed static domain:
   ngrok http 3001 --domain your-domain.ngrok-free.dev
   ```

---

## Case B — only a backend is shared (`remote-server`)

Teammates run their own frontend + `local-backend` locally
(`ENV_MODE=LOCAL`); your machine only shares the backend, no proxy needed.

```bash
ngrok http 8000
```

Give teammates the printed URL. They set it in their own
`local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=https://your-domain.ngrok-free.dev
```

If instead a teammate's **browser** is going to call this URL directly
(frontend-only access — see
[setup.md](setup.md#teammate-frontend-only-access)), add the tunnel URL to
`CORS_ORIGINS` in `remote-server/.env` and restart the backend — that request
comes straight from their browser, so CORS applies, unlike the
`local-backend` → `remote-server` request above which is server-to-server.

---

## Verify before sharing

Don't send the link on faith. From **your own machine**, hitting the public
URL exactly like a teammate would (not `127.0.0.1`):

```bash
curl https://your-domain.ngrok-free.dev/api/health
```

- **JSON back** (`{"status":"ok",...}`) → good, send the link.
- **HTML back**, or a 404 with a Next.js error page → the tunnel is
  pointed at the frontend port instead of the proxy (Case A2, ngrok must
  target 3001, not 3000).
- **Connection refused / 502** → nothing is listening where the tunnel or
  proxy expects; recheck `BACKEND_PORT` on `share-proxy.cjs` and that the
  backend is actually up.

Then open the tunnel URL itself in a **private/incognito browser window**
(so nothing is cached from your own `localhost` testing) and run one real
search. This is the only check that also catches the
`NEXT_PUBLIC_API_URL` mistake above — `curl`ing `/api/health` alone won't,
since that call doesn't go through the frontend's baked-in URL at all.

---

## One-time ngrok setup (skip if already done)

```bash
ngrok config add-authtoken <your-authtoken>   # from https://dashboard.ngrok.com
```

Optional: claim a free static domain at
[dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains) so the URL
doesn't change every time you restart the tunnel.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Page loads, but every search fails / spins forever | `NEXT_PUBLIC_API_URL` baked in before it was set to `/` | Restart the frontend process — see [A2 step 2](#a2-frontend-and-backend-are-already-running-in-their-own-terminals) |
| `curl <tunnel-url>/api/health` returns HTML, not JSON | ngrok tunnel points at port 3000 (frontend) instead of 3001 (proxy) | Restart the tunnel: `ngrok http 3001` |
| `curl <tunnel-url>/api/health` returns connection refused / 502 | `share-proxy.cjs` not running, or its `BACKEND_PORT` doesn't match where the backend actually listens | Confirm `curl http://127.0.0.1:8000/api/health` works locally first, then restart the proxy with matching `BACKEND_PORT` |
| `ngrok not found on PATH` | CLI not installed, or not on `PATH` | Reinstall, open a fresh terminal |
| Teammate sees an interstitial "you are about to visit..." page | Normal on the free ngrok plan | They click "Visit Site" once; a cookie skips it after |
| CORS error in browser console | Only relevant to Case B frontend-only access; tunnel URL missing from `CORS_ORIGINS` | Add it to `remote-server/.env` and restart the backend |
| Tunnel URL changes every time you restart | No static domain claimed | Claim one free static domain (see one-time setup above) |
| `agent already running (port 4040 in use)` | A previous `ngrok` process is still up | Fine if it's the tunnel you want; otherwise close that terminal first |
