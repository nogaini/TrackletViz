import { useEffect, useRef, useState } from 'react';
import { useStore } from '../../../stores/useStore';
import type { TrackletMetadata } from '../../../types/index';

interface Props {
  selectedTracklets: TrackletMetadata[];
}

const GRID_W = 128;
const GRID_H = 72;

export default function HeatmapTab({ selectedTracklets }: Props) {
  const { videoMetadata } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const [computing, setComputing] = useState(false);

  // Keep a stable worker across renders
  useEffect(() => {
    const worker = new Worker(new URL('../../../workers/heatmap.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const worker = workerRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!videoMetadata || selectedTracklets.length === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const vw = videoMetadata.width;
    const vh = videoMetadata.height;
    canvas.width = vw;
    canvas.height = vh;

    // Step 1: draw background image first (synchronous)
    const drawHeatmapOverlay = (grid: Float32Array, maxVal: number) => {
      const cellW = vw / GRID_W;
      const cellH = vh / GRID_H;

      ctx.save();
      ctx.globalAlpha = 0.6;
      for (let gy = 0; gy < GRID_H; gy++) {
        for (let gx = 0; gx < GRID_W; gx++) {
          const val = grid[gy * GRID_W + gx] / maxVal;
          if (val <= 0) continue;
          const hue = Math.round(240 - val * 240);
          ctx.fillStyle = `hsl(${hue},100%,50%)`;
          ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
        }
      }
      ctx.restore();
      setComputing(false);
    };

    // Collect all bounding boxes for the worker
    const bboxes: { x1: number; y1: number; x2: number; y2: number }[] = [];
    for (const t of selectedTracklets) {
      for (const box of t.bounding_boxes) {
        bboxes.push({ x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 });
      }
    }

    const runWithWorker = (drawBg: () => void) => {
      if (!worker) {
        // Fallback: compute synchronously if worker unavailable
        const grid = new Float32Array(GRID_W * GRID_H);
        for (const box of bboxes) {
          const gx1 = Math.max(0, Math.floor((box.x1 / vw) * GRID_W));
          const gy1 = Math.max(0, Math.floor((box.y1 / vh) * GRID_H));
          const gx2 = Math.min(GRID_W, Math.ceil((box.x2 / vw) * GRID_W));
          const gy2 = Math.min(GRID_H, Math.ceil((box.y2 / vh) * GRID_H));
          for (let gy = gy1; gy < gy2; gy++) {
            for (let gx = gx1; gx < gx2; gx++) {
              grid[gy * GRID_W + gx] += 1;
            }
          }
        }
        const maxVal = Math.max(...grid, 1);
        drawBg();
        drawHeatmapOverlay(grid, maxVal);
        return;
      }

      setComputing(true);

      // Replace handler each time so stale results are ignored
      worker.onmessage = (e: MessageEvent<{ grid: Float32Array; maxVal: number }>) => {
        drawBg();
        drawHeatmapOverlay(e.data.grid, e.data.maxVal);
      };

      worker.postMessage({ bboxes, vw, vh, gridW: GRID_W, gridH: GRID_H }, []);
    };

    if (videoMetadata.background_image_base64) {
      const img = new Image();
      img.onload = () => {
        runWithWorker(() => ctx.drawImage(img, 0, 0, vw, vh));
      };
      img.src = `data:image/jpeg;base64,${videoMetadata.background_image_base64}`;
    } else {
      runWithWorker(() => {
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, vw, vh);
      });
    }
  }, [selectedTracklets, videoMetadata]);

  if (selectedTracklets.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">Select tracklets in the scatter plot to begin.</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-black overflow-hidden">
      <div className="flex-1 flex items-center justify-center p-2 min-h-0 relative">
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full object-contain"
          style={{ imageRendering: 'pixelated' }}
        />
        {computing && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <div className="flex items-center gap-2 bg-gray-900/90 px-4 py-2 rounded-lg">
              <div className="w-3 h-3 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" />
              <span className="text-xs text-gray-300">Computing heatmap…</span>
            </div>
          </div>
        )}
      </div>
      <div className="shrink-0 px-4 pb-3 pt-1">
        <p className="text-[10px] text-gray-400 text-center mb-1 uppercase tracking-wide">Density</p>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400 shrink-0">Low</span>
          <div
            className="flex-1 h-3 rounded"
            style={{
              background:
                'linear-gradient(to right, hsl(240,100%,50%), hsl(180,100%,50%), hsl(120,100%,50%), hsl(60,100%,50%), hsl(0,100%,50%))',
              opacity: 0.85,
            }}
          />
          <span className="text-[10px] text-gray-400 shrink-0">High</span>
        </div>
      </div>
    </div>
  );
}
