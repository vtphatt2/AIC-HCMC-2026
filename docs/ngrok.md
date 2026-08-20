# Ngrok — Share a Running Instance

Assumes you already have the system running locally. This page only covers
turning on the tunnel, nothing else.

## One-time setup (skip if already done)

```bash
ngrok config add-authtoken <your-authtoken>   # from https://dashboard.ngrok.com
```

Optional: claim a free static domain at
[dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains) so the URL
doesn't change every time you restart the tunnel.

---

## Case A — frontend + backend both running on your machine

Applies whether the backend is `local-backend` **or** `remote-server` —
`share-proxy.cjs` just forwards by port (3000 for the frontend, 8000 for the
backend), it doesn't care which one is actually listening there.

**Started them with `start-local.ps1`?** (this only ever starts
`local-backend`, not `remote-server`) — stop and restart with `-Ngrok`
added:

```powershell
scripts\start-local.ps1 -Ngrok
```

This opens the tunnel automatically along with everything else. Share the
`https://your-domain.ngrok-free.dev` URL it prints — that's the only link
your teammate needs.

**Backend (8000, `local-backend` or `remote-server`) and frontend (3000)
already running in their own terminals and you don't want to restart
them?** — this is the case for a `remote-server` backend, since
`start-local.ps1` can't manage it. Just add the proxy + tunnel on top:

```bash
node scripts/share-proxy.cjs
ngrok http 3001
```

(With a claimed static domain: `ngrok http 3001 --domain your-domain.ngrok-free.dev`.)

---

## Case B — remote-server running on the GPU machine

```bash
ngrok http 8000
```

Give teammates the printed URL. They set it as `REMOTE_SERVER_URL` in their
own `local-client/local-backend/.env` (`ENV_MODE=LOCAL`).

If a teammate's **browser** calls this URL directly (frontend-only access,
no local-backend in between), add the tunnel URL to `CORS_ORIGINS` in
`remote-server/.env` and restart the backend.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ngrok not found on PATH` | Reinstall ngrok, open a fresh terminal |
| Teammate sees an interstitial "you are about to visit..." page | Normal on the free plan — they click "Visit Site" once, a cookie skips it after |
| CORS error in browser console | Add the tunnel URL to `CORS_ORIGINS` and restart the backend |
| Tunnel URL changes every time | Claim a free static domain (see one-time setup above) |
| `agent already running (port 4040 in use)` | A previous `ngrok` is still up — close that terminal first if you need a different tunnel |
