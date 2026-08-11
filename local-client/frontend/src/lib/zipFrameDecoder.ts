// Decode a keyframe thumbnail in the browser instead of on the backend.
//
// The server route (/api/zip-frame) runs one ffmpeg process per thumbnail, on
// one machine, for every viewer — a shared result grid asks for up to 100 of
// them each. This path asks the backend only where the bytes are, fetches that
// one range, and decodes with WebCodecs on the viewer's own CPU. Decode cost
// then scales with viewers instead of stacking on the host.
//
// Everything here degrades to the server route: no WebCodecs, an unsupported
// codec, a malformed GOP — all return null and the caller uses the <img> URL.

import { apiUrl } from "@/lib/api";

type FramePlan = {
  video_id: string;
  timestamp_ms: number;
  codec: string;
  nal_length_size: number;
  timescale: number;
  target_pts: number;
  parameter_sets: string[];
  bytes_url: string;
  samples: { o: number; s: number; p: number }[];
};

export function canDecodeInBrowser(): boolean {
  return typeof window !== "undefined" && typeof (window as any).VideoDecoder === "function";
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const START_CODE = new Uint8Array([0, 0, 0, 1]);

/** AVCC stores each NAL as <length><payload>; Annex-B wants a start code instead. */
function avccToAnnexB(sample: Uint8Array, nalLengthSize: number): Uint8Array {
  const parts: Uint8Array[] = [];
  let i = 0;
  while (i + nalLengthSize <= sample.length) {
    let len = 0;
    for (let k = 0; k < nalLengthSize; k++) len = (len << 8) | sample[i + k];
    i += nalLengthSize;
    if (len <= 0 || i + len > sample.length) break;
    parts.push(START_CODE, sample.subarray(i, i + len));
    i += len;
  }
  return concat(parts);
}

function concat(parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/**
 * Returns an object URL for the decoded frame, or null if this browser can't
 * do it — callers must treat null as "use the server URL", never as an error.
 *
 * Frames are matched by presentation timestamp rather than by counting decoder
 * output. With B-frames the decoder emits in presentation order, which is not
 * the order the samples went in, so an index would pick the wrong picture.
 */
export async function decodeFrameInBrowser(
  videoId: string,
  timestampMs: number,
  signal?: AbortSignal,
): Promise<string | null> {
  if (!canDecodeInBrowser()) return null;

  const planRes = await fetch(
    apiUrl(`/api/zip-frame-plan/${encodeURIComponent(videoId)}/${timestampMs}`),
    { signal },
  );
  if (!planRes.ok) return null;
  const plan: FramePlan = await planRes.json();
  if (!plan.samples?.length) return null;

  const support = await (window as any).VideoDecoder.isConfigSupported({ codec: plan.codec });
  if (!support?.supported) return null;

  const bytesRes = await fetch(apiUrl(plan.bytes_url), { signal });
  if (!bytesRes.ok) return null;
  const region = new Uint8Array(await bytesRes.arrayBuffer());

  const usToTicks = 1_000_000 / (plan.timescale || 1);
  const targetUs = Math.round(plan.target_pts * usToTicks);

  return await new Promise<string | null>((resolve) => {
    let settled = false;
    let best: { frame: any; delta: number } | null = null;

    const finish = async () => {
      if (settled) return;
      settled = true;
      try {
        decoder.close();
      } catch {
        /* already closed */
      }
      if (!best) return resolve(null);
      const { frame } = best;
      try {
        const canvas = document.createElement("canvas");
        canvas.width = frame.displayWidth || frame.codedWidth;
        canvas.height = frame.displayHeight || frame.codedHeight;
        canvas.getContext("2d")!.drawImage(frame, 0, 0);
        frame.close();
        canvas.toBlob(
          (blob) => resolve(blob ? URL.createObjectURL(blob) : null),
          "image/jpeg",
          0.85,
        );
      } catch {
        try {
          frame.close();
        } catch {
          /* ignore */
        }
        resolve(null);
      }
    };

    const decoder = new (window as any).VideoDecoder({
      output: (frame: any) => {
        const delta = Math.abs((frame.timestamp ?? 0) - targetUs);
        if (best && delta >= best.delta) {
          frame.close();
          return;
        }
        best?.frame.close();
        best = { frame, delta };
      },
      error: () => {
        if (!settled) {
          settled = true;
          resolve(null);
        }
      },
    });

    signal?.addEventListener("abort", () => {
      if (!settled) {
        settled = true;
        try {
          decoder.close();
        } catch {
          /* ignore */
        }
        best?.frame.close();
        resolve(null);
      }
    });

    try {
      // Annex-B mode: configure() must NOT carry a description, or WebCodecs
      // expects AVCC and rejects every chunk.
      decoder.configure({ codec: plan.codec, optimizeForLatency: true });

      const paramSets = plan.parameter_sets.map(base64ToBytes);
      const preamble = concat(paramSets.flatMap((nal) => [START_CODE, nal]));

      plan.samples.forEach((sample, i) => {
        const raw = region.subarray(sample.o, sample.o + sample.s);
        const annexb = avccToAnnexB(raw, plan.nal_length_size);
        // SPS/PPS ride in front of the keyframe, exactly as the server path
        // writes them ahead of the GOP.
        const data = i === 0 ? concat([preamble, annexb]) : annexb;
        decoder.decode(
          new (window as any).EncodedVideoChunk({
            type: i === 0 ? "key" : "delta",
            timestamp: Math.round(sample.p * usToTicks),
            data,
          }),
        );
      });

      decoder.flush().then(finish, finish);
    } catch {
      if (!settled) {
        settled = true;
        resolve(null);
      }
    }
  });
}
