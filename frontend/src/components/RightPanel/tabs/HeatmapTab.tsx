import { useEffect, useRef } from 'react';
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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!videoMetadata || selectedTracklets.length === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const vw = videoMetadata.width;
    const vh = videoMetadata.height;

    // Step 1: draw background
    if (videoMetadata.background_image_base64) {
      const img = new Image();
      img.onload = () => {
        canvas.width = vw;
        canvas.height = vh;
        ctx.drawImage(img, 0, 0, vw, vh);
        renderHeatmap(ctx, vw, vh);
      };
      img.src = `data:image/jpeg;base64,${videoMetadata.background_image_base64}`;
    } else {
      canvas.width = vw;
      canvas.height = vh;
      ctx.fillStyle = '#111827';
      ctx.fillRect(0, 0, vw, vh);
      renderHeatmap(ctx, vw, vh);
    }
  }, [selectedTracklets, videoMetadata]);

  function renderHeatmap(ctx: CanvasRenderingContext2D, vw: number, vh: number) {
    // Accumulate grid
    const grid = new Float32Array(GRID_W * GRID_H);
    for (const t of selectedTracklets) {
      for (const box of t.bounding_boxes) {
        const gx1 = Math.floor((box.x1 / vw) * GRID_W);
        const gy1 = Math.floor((box.y1 / vh) * GRID_H);
        const gx2 = Math.ceil((box.x2 / vw) * GRID_W);
        const gy2 = Math.ceil((box.y2 / vh) * GRID_H);
        for (let gy = Math.max(0, gy1); gy < Math.min(GRID_H, gy2); gy++) {
          for (let gx = Math.max(0, gx1); gx < Math.min(GRID_W, gx2); gx++) {
            grid[gy * GRID_W + gx] += 1;
          }
        }
      }
    }

    const maxVal = Math.max(...grid, 1);

    // Draw heat overlay
    const cellW = vw / GRID_W;
    const cellH = vh / GRID_H;

    ctx.save();
    ctx.globalAlpha = 0.6;
    for (let gy = 0; gy < GRID_H; gy++) {
      for (let gx = 0; gx < GRID_W; gx++) {
        const val = grid[gy * GRID_W + gx] / maxVal;
        if (val <= 0) continue;
        // cool (blue=240) to warm (red=0): hsl(240-val*240, 100%, 50%)
        const hue = Math.round(240 - val * 240);
        ctx.fillStyle = `hsl(${hue},100%,50%)`;
        ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
      }
    }
    ctx.restore();
  }

  if (selectedTracklets.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">Select tracklets in the scatter plot to begin.</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-black overflow-hidden">
      <div className="flex-1 flex items-center justify-center p-2 min-h-0">
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full object-contain"
          style={{ imageRendering: 'pixelated' }}
        />
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
