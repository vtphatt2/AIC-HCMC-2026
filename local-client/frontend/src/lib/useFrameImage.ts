// Picks where a thumbnail gets decoded, and hides the choice from callers.
//
// Server mode is the existing behaviour: the <img> points at /api/zip-frame and
// the backend runs ffmpeg. Client mode asks the backend only for a byte range
// and decodes with WebCodecs here, which matters once several people share one
// backend — a result grid is up to 100 thumbnails *per viewer*, and in server
// mode all of them queue behind the host's decode semaphore.
//
// Falls back to the server URL on anything: no WebCodecs, an unsupported
// codec, a decode failure. There is deliberately no error state — a thumbnail
// that took the slow path looks identical.

import { useEffect, useRef, useState } from "react";
import { canDecodeInBrowser, decodeFrameInBrowser } from "@/lib/zipFrameDecoder";

const CLIENT_DECODE =
  (process.env.NEXT_PUBLIC_FRAME_DECODE || "server").toLowerCase() === "client";

/** /api/zip-frame/{video_id}/{timestamp_ms} — the only shape we can decode here. */
const ZIP_FRAME = /^\/api\/zip-frame\/([^/]+)\/(\d+)$/;

export function useFrameImage(serverUrl: string, rawPath: string | undefined): string {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  // Kept in a ref as well so cleanup can revoke it without re-running on every
  // state change — a revoked URL on a still-mounted <img> shows a broken image.
  const objectUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const match = rawPath?.match(ZIP_FRAME);
    if (!CLIENT_DECODE || !match || !canDecodeInBrowser()) return;

    const controller = new AbortController();
    let cancelled = false;

    decodeFrameInBrowser(decodeURIComponent(match[1]), Number(match[2]), controller.signal)
      .then((url) => {
        if (cancelled || !url) {
          if (url) URL.revokeObjectURL(url);
          return;
        }
        objectUrlRef.current = url;
        setObjectUrl(url);
      })
      .catch(() => {
        /* server URL is already showing; nothing to report */
      });

    return () => {
      cancelled = true;
      controller.abort();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [rawPath]);

  return objectUrl || serverUrl;
}
