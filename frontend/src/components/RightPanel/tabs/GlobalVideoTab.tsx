import { useCallback, useEffect, useRef, useState } from 'react';
import { useStore } from '../../../stores/useStore';
import { videoStreamUrl } from '../../../lib/api';
import type { GlobalClipMetadata } from '../../../types/index';

interface Props {
  selectedClips: GlobalClipMetadata[];
}

export default function GlobalVideoTab({ selectedClips }: Props) {
  const { selectedVideoId, videoMetadata } = useStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineCanvasRef = useRef<HTMLCanvasElement>(null);
  const [loopClip, setLoopClip] = useState<GlobalClipMetadata | null>(null);
  const loopRef = useRef<GlobalClipMetadata | null>(null);

  const duration = videoMetadata?.duration ?? 0;
  const streamUrl = selectedVideoId ? videoStreamUrl(selectedVideoId) : '';

  // ── Loop enforcement ──────────────────────────────────────────────────────
  useEffect(() => {
    loopRef.current = loopClip;
    const video = videoRef.current;
    if (!video || !loopClip) return;
    video.currentTime = loopClip.start_time;
    video.play().catch(() => {});
  }, [loopClip]);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    const lc = loopRef.current;
    if (!video || !lc) return;
    if (video.currentTime >= lc.end_time) {
      video.currentTime = lc.start_time;
    }
  }, []);

  const stopLoop = useCallback(() => {
    loopRef.current = null;
    setLoopClip(null);
    videoRef.current?.pause();
  }, []);

  // ── Timeline rendering ────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = timelineCanvasRef.current;
    if (!canvas || duration <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    if (W === 0 || H === 0) return;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, W, H);
    for (const clip of selectedClips) {
      const x = (clip.start_time / duration) * W;
      const w = Math.max(4, ((clip.end_time - clip.start_time) / duration) * W);
      const isActive = loopClip?.clip_id === clip.clip_id;
      ctx.fillStyle = isActive ? 'rgba(96,165,250,0.9)' : 'rgba(59,130,246,0.7)';
      ctx.fillRect(x, 0, w, H);
    }
  }, [selectedClips, duration, loopClip]);

  // ── Timeline click ────────────────────────────────────────────────────────
  const handleTimelineClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!duration || selectedClips.length === 0) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const canvasW = rect.width;
      const clickX = e.clientX - rect.left;
      const clickedTime = (clickX / canvasW) * duration;

      // Exact hit
      let hit = selectedClips.find(
        c => c.start_time <= clickedTime && c.end_time >= clickedTime,
      ) ?? null;

      // Tolerance hit (±6px → time units)
      if (!hit && canvasW > 0) {
        const toleranceTime = (6 / canvasW) * duration;
        let minDist = Infinity;
        for (const c of selectedClips) {
          const dist =
            clickedTime < c.start_time ? c.start_time - clickedTime
            : clickedTime > c.end_time ? clickedTime - c.end_time
            : 0;
          if (dist <= toleranceTime && dist < minDist) {
            minDist = dist;
            hit = c;
          }
        }
      }

      if (hit) setLoopClip(hit);
    },
    [duration, selectedClips],
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Timeline — above video, only when clips selected */}
      {selectedClips.length > 0 && (
        <div className="px-3 pt-3 pb-2 shrink-0">
          <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">
            Timeline
          </p>
          <canvas
            ref={timelineCanvasRef}
            className="w-full h-5 rounded overflow-hidden cursor-pointer block"
            onClick={handleTimelineClick}
          />
        </div>
      )}

      {/* Video player */}
      <div className="flex-1 relative overflow-hidden bg-black">
        {selectedVideoId && (
          <video
            ref={videoRef}
            src={streamUrl}
            className="w-full h-full object-contain"
            onTimeUpdate={handleTimeUpdate}
            controls={loopClip === null}
            muted
          />
        )}

        {/* Empty state overlay */}
        {selectedClips.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <p className="text-gray-500 text-sm">
              Select clips in the scatter plot to view their timeline.
            </p>
          </div>
        )}

        {/* Loop mode controls */}
        {loopClip && (
          <div className="absolute top-2 right-2 z-10 flex items-center gap-2 bg-gray-900/80 backdrop-blur px-2 py-1 rounded">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-gray-300">Clip {loopClip.clip_index}</span>
            <button
              onClick={stopLoop}
              className="text-[10px] text-gray-400 hover:text-white ml-1"
            >
              Stop
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
