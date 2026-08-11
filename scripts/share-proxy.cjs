#!/usr/bin/env node
// One public port that fronts both dev servers, so the app can be shared
// through a single ngrok tunnel.
//
// Why this exists instead of next.config.js rewrites: a result grid loads up
// to 100 thumbnails, each a slow /api/zip-frame decode, and the browser
// aborts the in-flight ones every time you search again. Pushing that through
// the Next dev server's proxy killed it (ECONNRESET storms, then the dev
// server exited). Next is a compiler, not a proxy. This keeps the media
// firehose off it entirely.
//
// Node stdlib only — no dependency, nothing to install.

const http = require("node:http");

const PORT = Number(process.env.SHARE_PORT) || 3001;
const NEXT = { host: "127.0.0.1", port: Number(process.env.FRONTEND_PORT) || 3000 };
const API = { host: "127.0.0.1", port: Number(process.env.BACKEND_PORT) || 8000 };

// /api/tuning-draft is a Next page route (reads a local JSON file), not a
// backend route — it has to stay on the Next side.
function upstreamFor(url) {
  if (url.startsWith("/api/tuning-draft")) return NEXT;
  if (url.startsWith("/api/") || url.startsWith("/static/")) return API;
  return NEXT;
}

const server = http.createServer((req, res) => {
  const up = upstreamFor(req.url);
  const proxied = http.request(
    // agent: false — a fresh socket per request. Pooled keep-alive sockets
    // race with uvicorn reaping idle ones, which surfaces as "socket hang up".
    { ...up, method: req.method, path: req.url, headers: req.headers, agent: false },
    (upRes) => {
      res.writeHead(upRes.statusCode, upRes.headers);
      upRes.pipe(res);
    },
  );

  // An aborted thumbnail is normal (user searched again), not an error worth
  // logging or crashing over. Just tear down the upstream half.
  proxied.on("error", () => {
    if (!res.headersSent) res.writeHead(502, { "content-type": "text/plain" });
    res.end("upstream unavailable");
  });
  res.on("close", () => proxied.destroy());

  req.pipe(proxied);
});

// Next's HMR websocket. Without this the client reconnects forever.
server.on("upgrade", (req, socket, head) => {
  const up = upstreamFor(req.url);
  const proxied = http.request({ ...up, path: req.url, headers: req.headers, agent: false });
  proxied.on("upgrade", (upRes, upSocket, upHead) => {
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\n` +
        Object.entries(upRes.headers).map(([k, v]) => `${k}: ${v}\r\n`).join("") +
        "\r\n",
    );
    if (upHead && upHead.length) socket.unshift(upHead);
    upSocket.pipe(socket).pipe(upSocket);
    upSocket.on("error", () => socket.destroy());
  });
  proxied.on("error", () => socket.destroy());
  socket.on("error", () => proxied.destroy());
  if (head && head.length) proxied.write(head);
  proxied.end();
});

server.on("clientError", (_err, socket) => socket.destroy());

server.listen(PORT, "127.0.0.1", () => {
  console.log(`share-proxy on http://127.0.0.1:${PORT}`);
  console.log(`  /api/* + /static/*  -> ${API.host}:${API.port}   (backend)`);
  console.log(`  everything else     -> ${NEXT.host}:${NEXT.port}   (next dev)`);
});
