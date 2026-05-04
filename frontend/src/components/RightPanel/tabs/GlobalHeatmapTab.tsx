import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGlobalClipDetail, videoStreamUrl } from "../../../lib/api";
import { hslToRgb } from "../../../lib/utils";
import { useStore } from "../../../stores/useStore";
import type { GlobalClipMetadata } from "../../../types/index";

type SubTab = "illumination-change" | "illumination-shift";

// ── Helpers ──────────────────────────────────────────────────────────────

async function decodeBase64ToImageData(b64: string): Promise<ImageData> {
  const blob = await fetch(`data:image/jpeg;base64,${b64}`).then((r) =>
    r.blob(),
  );
  const bmp = await createImageBitmap(blob);
  const offscreen = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = offscreen.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  return ctx.getImageData(0, 0, bmp.width, bmp.height);
}

async function decodeBase64ToBlurredImageData(b64: string): Promise<ImageData> {
  const blob = await fetch(`data:image/jpeg;base64,${b64}`).then((r) =>
    r.blob(),
  );
  const bmp = await createImageBitmap(blob);
  const offscreen = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = offscreen.getContext("2d")!;
  ctx.filter = "blur(4px)";
  ctx.drawImage(bmp, 0, 0);
  return ctx.getImageData(0, 0, bmp.width, bmp.height);
}

// ── Shared legend component ───────────────────────────────────────────────

function ColorBarLegend({
  gradient,
  leftLabel,
  rightLabel,
  title,
}: {
  gradient: string;
  leftLabel: string;
  rightLabel: string;
  title: string;
}) {
  return (
    <div className="shrink-0 px-4 pb-3 pt-1">
      <p className="text-[10px] text-gray-400 text-center mb-1 uppercase tracking-wide">
        {title}
      </p>
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-gray-400 shrink-0">{leftLabel}</span>
        <div
          className="flex-1 h-3 rounded"
          style={{ background: gradient, opacity: 0.85 }}
        />
        <span className="text-[10px] text-gray-400 shrink-0">{rightLabel}</span>
      </div>
    </div>
  );
}

// ── Clip video players ────────────────────────────────────────────────────

function ClipVideoPlayer({ clip }: { clip: GlobalClipMetadata }) {
  const { selectedVideoId } = useStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const clipRef = useRef(clip);
  const streamUrl = selectedVideoId ? videoStreamUrl(selectedVideoId) : "";

  useEffect(() => {
    clipRef.current = clip;
    const video = videoRef.current;
    if (!video || !streamUrl) return;
    video.currentTime = clip.start_time;
    video.play().catch(() => {});
  }, [clip, streamUrl]);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    const c = clipRef.current;
    if (!video) return;
    if (video.currentTime >= c.end_time) {
      video.currentTime = c.start_time;
    }
  }, []);

  return (
    <div className="flex-1 relative bg-black overflow-hidden rounded">
      {streamUrl && (
        <video
          ref={videoRef}
          src={streamUrl}
          className="w-full h-full object-contain"
          muted
          onTimeUpdate={handleTimeUpdate}
        />
      )}
      <div className="absolute bottom-1 left-1 bg-black/60 px-1.5 py-0.5 rounded text-[10px] text-gray-300">
        Clip {clip.clip_index} · {clip.start_time.toFixed(1)}s–
        {clip.end_time.toFixed(1)}s
      </div>
    </div>
  );
}

function ClipVideoRow({
  clip1,
  clip2,
}: {
  clip1: GlobalClipMetadata;
  clip2: GlobalClipMetadata;
}) {
  return (
    <div className="flex-1 min-h-0 flex gap-2 px-3 pb-2 pt-1 border-t border-gray-800">
      <ClipVideoPlayer clip={clip1} />
      <ClipVideoPlayer clip={clip2} />
    </div>
  );
}

// ── Sub-tab 1: Illumination Change ───────────────────────────────────────

function IlluminationChangeHeatmap({
  clip1,
  clip2,
}: {
  clip1: GlobalClipMetadata;
  clip2: GlobalClipMetadata;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const [d1, d2] = await Promise.all([
        clip1.median_frame_b64 ? clip1 : fetchGlobalClipDetail(clip1.clip_id),
        clip2.median_frame_b64 ? clip2 : fetchGlobalClipDetail(clip2.clip_id),
      ]);

      if (!d1.median_frame_b64 || !d2.median_frame_b64) {
        setError("Median frames not available for selected clips");
        setBusy(false);
        return;
      }

      const [img1, img2] = await Promise.all([
        decodeBase64ToImageData(d1.median_frame_b64),
        decodeBase64ToImageData(d2.median_frame_b64),
      ]);

      const valid = [img1, img2];
      const W = valid[0].width;
      const H = valid[0].height;
      const N = valid.length;

      // Compute per-pixel luminance variance across the 2 frames
      const sums = new Float32Array(W * H);
      const sumSqs = new Float32Array(W * H);
      for (const img of valid) {
        for (let i = 0; i < W * H; i++) {
          const r = img.data[i * 4];
          const g = img.data[i * 4 + 1];
          const b = img.data[i * 4 + 2];
          const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          sums[i] += lum;
          sumSqs[i] += lum * lum;
        }
      }
      const variance = new Float32Array(W * H);
      let maxVar = 0;
      for (let i = 0; i < W * H; i++) {
        const v = sumSqs[i] / N - (sums[i] / N) ** 2;
        variance[i] = v;
        if (v > maxVar) maxVar = v;
      }

      const canvas = canvasRef.current;
      if (!canvas) {
        setBusy(false);
        return;
      }
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext("2d")!;

      // Black background
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);

      // Build heatmap pixels with lightness baked in (black where stable)
      const heatData = ctx.createImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const t = maxVar > 0 ? variance[i] / maxVar : 0;
        const hue = 240 - t * 240; // blue (stable) → red (changing)
        const [r, g, b] = hslToRgb(hue, 1, t * 0.5); // lightness 0→0.5 as t→1
        heatData.data[i * 4] = r;
        heatData.data[i * 4 + 1] = g;
        heatData.data[i * 4 + 2] = b;
        heatData.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(heatData, 0, 0);
    } catch (e: unknown) {
      setError(String(e));
    }
    setBusy(false);
  }, [clip1, clip2]);

  useEffect(() => {
    run();
  }, [run]);

  return (
    <div className="h-full flex flex-col bg-black overflow-hidden">
      <div className="flex-1 flex items-center justify-center p-2 min-h-0 relative">
        <canvas
          ref={canvasRef}
          className="h-full w-auto max-w-full border border-white/30"
        />
        {busy && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <p className="text-gray-300 text-sm">Computing…</p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
      </div>
      <ColorBarLegend
        title="Change Intensity"
        leftLabel="Stable"
        rightLabel="Changing"
        gradient="linear-gradient(to right, #000000, hsl(240,100%,50%), hsl(120,100%,50%), hsl(0,100%,50%))"
      />
      <ClipVideoRow clip1={clip1} clip2={clip2} />
    </div>
  );
}

// ── Sub-tab 2: Illumination Shift ─────────────────────────────────────────

function IlluminationShiftHeatmap({
  clip1,
  clip2,
}: {
  clip1: GlobalClipMetadata;
  clip2: GlobalClipMetadata;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const illuminationGradient = useMemo(() => {
    const stops = 32;
    const parts: string[] = [];
    for (let i = 0; i <= stops; i++) {
      const t = (i / stops) * 2 - 1;
      const hue = 240 - ((t + 1) / 2) * 240;
      const lightness = Math.abs(t) * 0.5;
      const [r, g, b] = hslToRgb(hue, 1.0, lightness);
      parts.push(`rgb(${r},${g},${b}) ${Math.round((i / stops) * 100)}%`);
    }
    return `linear-gradient(to right, ${parts.join(", ")})`;
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const [d1, d2] = await Promise.all([
        clip1.median_frame_b64 ? clip1 : fetchGlobalClipDetail(clip1.clip_id),
        clip2.median_frame_b64 ? clip2 : fetchGlobalClipDetail(clip2.clip_id),
      ]);

      if (!d1.median_frame_b64 || !d2.median_frame_b64) {
        setError("Median frames not available");
        setBusy(false);
        return;
      }

      const [img1, img2] = await Promise.all([
        decodeBase64ToBlurredImageData(d1.median_frame_b64),
        decodeBase64ToBlurredImageData(d2.median_frame_b64),
      ]);

      const W = img1.width;
      const H = img1.height;
      const canvas = canvasRef.current;
      if (!canvas) {
        setBusy(false);
        return;
      }
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext("2d")!;

      // Black background
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);

      // First pass: compute luminance deltas and find max absolute delta
      const deltas = new Float32Array(W * H);
      let maxAbsDelta = 0;
      for (let i = 0; i < W * H; i++) {
        const lum1 = 0.2126 * img1.data[i * 4] + 0.7152 * img1.data[i * 4 + 1] + 0.0722 * img1.data[i * 4 + 2];
        const lum2 = 0.2126 * img2.data[i * 4] + 0.7152 * img2.data[i * 4 + 1] + 0.0722 * img2.data[i * 4 + 2];
        deltas[i] = lum2 - lum1;
        if (Math.abs(deltas[i]) > maxAbsDelta) maxAbsDelta = Math.abs(deltas[i]);
      }

      // Second pass: color mapping normalized to observed max (not absolute ±255)
      const overlay = ctx.createImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const t = maxAbsDelta > 0 ? deltas[i] / maxAbsDelta : 0; // −1…+1
        const hue = 240 - ((t + 1) / 2) * 240; // blue (darker) → red (brighter)
        const [r, g, b] = hslToRgb(hue, 1, Math.abs(t) * 0.5);
        overlay.data[i * 4] = r;
        overlay.data[i * 4 + 1] = g;
        overlay.data[i * 4 + 2] = b;
        overlay.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(overlay, 0, 0);
    } catch (e: unknown) {
      setError(String(e));
    }
    setBusy(false);
  }, [clip1, clip2]);

  useEffect(() => {
    run();
  }, [run]);

  return (
    <div className="h-full flex flex-col bg-black overflow-hidden">
      <div className="flex-1 flex items-center justify-center p-2 min-h-0 relative">
        <canvas
          ref={canvasRef}
          className="h-full w-auto max-w-full border border-white/30"
        />
        {busy && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <p className="text-gray-300 text-sm">
              Computing illumination diff…
            </p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
      </div>
      <ColorBarLegend
        title="Illumination Change"
        leftLabel="Darker"
        rightLabel="Brighter"
        gradient={illuminationGradient}
      />
      <ClipVideoRow clip1={clip1} clip2={clip2} />
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function GlobalHeatmapTab() {
  const { twoPointSelection } = useStore();
  const [subTab, setSubTab] = useState<SubTab>("illumination-change");

  const hasTwoPoints = twoPointSelection !== null;

  return (
    <div className="flex flex-col h-full">
      {/* Sub-tab bar */}
      <div className="flex border-b border-gray-700 bg-gray-900 shrink-0 text-xs">
        {(
          ["illumination-change", "illumination-shift"] as SubTab[]
        ).map((st) => {
          const label =
            st === "illumination-change"
              ? "Illumination Change"
              : "Illumination Shift";
          const disabled = !hasTwoPoints;
          return (
            <button
              key={st}
              onClick={() => !disabled && setSubTab(st)}
              disabled={disabled}
              title={disabled ? "Use 2-point selection to enable" : undefined}
              className={`px-3 py-2 border-b-2 transition-colors ${
                subTab === st
                  ? "border-blue-500 text-white"
                  : disabled
                    ? "border-transparent text-gray-600 cursor-not-allowed"
                    : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Sub-tab content */}
      <div className="flex-1 overflow-hidden">
        {!hasTwoPoints && (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm px-4 text-center">
            Use the 2-point selection tool to select 2 representative clips
          </div>
        )}
        {hasTwoPoints && subTab === "illumination-change" && (
          <IlluminationChangeHeatmap
            clip1={twoPointSelection!.clip1}
            clip2={twoPointSelection!.clip2}
          />
        )}
        {hasTwoPoints && subTab === "illumination-shift" && (
          <IlluminationShiftHeatmap
            clip1={twoPointSelection!.clip1}
            clip2={twoPointSelection!.clip2}
          />
        )}
      </div>
    </div>
  );
}
