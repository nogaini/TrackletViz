import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../../stores/useStore';
import { fetchTrackletBatch, videoStreamUrl } from '../../../lib/api';
import { mergeIntervals } from '../../../lib/utils';
import { BBOX_COLOR, BBOX_TRACK_COLOR } from '../../../lib/colors';
import LazyThumbnail from '../../shared/LazyThumbnail';
import type { TrackletMetadata, BoundingBox } from '../../../types/index';

interface Props {
  selectedTracklets: TrackletMetadata[];
}

interface Segment {
  start: number;
  end: number;
}

interface PopoverState {
  segment: Segment;
  x: number;
  canvasW: number;
}

/**
 * Binary search for the bounding box whose timestamp is closest to `time`.
 * Assumes `boxes` is sorted ascending by timestamp (which it always is — boxes
 * are stored in frame order from the indexer).
 */
function findClosestBox(boxes: BoundingBox[] | undefined | null, time: number): BoundingBox | null {
  if (!boxes || boxes.length === 0) return null;
  let lo = 0;
  let hi = boxes.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (boxes[mid].timestamp < time) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // lo is the first index with timestamp >= time; check lo-1 too
  if (lo > 0 && Math.abs(boxes[lo - 1].timestamp - time) <= Math.abs(boxes[lo].timestamp - time)) {
    return boxes[lo - 1];
  }
  return boxes[lo];
}

export default function VideoPlayerTab({ selectedTracklets }: Props) {
  const { selectedVideoId, videoMetadata } = useStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const bboxCanvasRef = useRef<HTMLCanvasElement>(null);
  const timelineCanvasRef = useRef<HTMLCanvasElement>(null);
  const loopTrackletRef = useRef<TrackletMetadata | null>(null);
  const rafRef = useRef<number | null>(null);

  const [loopTracklet, setLoopTracklet] = useState<TrackletMetadata | null>(null);
  const [popover, setPopover] = useState<PopoverState | null>(null);

  const duration = videoMetadata?.duration ?? 0;

  // Compute merged timeline segments
  const mergedSegments = useMemo<Segment[]>(() => {
    const intervals: [number, number][] = selectedTracklets.map(t => [
      t.start_timestamp,
      t.end_timestamp,
    ]);
    return mergeIntervals(intervals).map(([s, e]) => ({ start: s, end: e }));
  }, [selectedTracklets]);

  // ── Canvas bbox drawing loop ────────────────────────────────────────────
  useEffect(() => {
    const video = videoRef.current;
    const canvas = bboxCanvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const tracklet = loopTrackletRef.current;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (tracklet) {
        const box = findClosestBox(tracklet.bounding_boxes ?? [], video.currentTime);
        if (box) {
          const containerW = canvas.clientWidth || canvas.width;
          const containerH = canvas.clientHeight || canvas.height;
          const videoW = videoMetadata?.width ?? canvas.width;
          const videoH = videoMetadata?.height ?? canvas.height;
          const videoAR = videoW / videoH;
          const containerAR = containerW / containerH;

          let renderedW: number, renderedH: number, offsetX: number, offsetY: number;
          if (containerAR > videoAR) {
            renderedH = containerH;
            renderedW = containerH * videoAR;
            offsetX = (containerW - renderedW) / 2;
            offsetY = 0;
          } else {
            renderedW = containerW;
            renderedH = containerW / videoAR;
            offsetX = 0;
            offsetY = (containerH - renderedH) / 2;
          }

          const scaleX = (renderedW / videoW) * (canvas.width / containerW);
          const scaleY = (renderedH / videoH) * (canvas.height / containerH);
          const originX = (offsetX / containerW) * canvas.width;
          const originY = (offsetY / containerH) * canvas.height;

          ctx.strokeStyle = BBOX_COLOR;
          ctx.lineWidth = 2;
          ctx.strokeRect(
            originX + box.x1 * scaleX,
            originY + box.y1 * scaleY,
            (box.x2 - box.x1) * scaleX,
            (box.y2 - box.y1) * scaleY,
          );
          ctx.strokeStyle = BBOX_TRACK_COLOR;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          (tracklet.bounding_boxes ?? []).forEach((b, i) => {
            const cx = originX + b.center_x * scaleX;
            const cy = originY + b.center_y * scaleY;
            if (i === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          });
          ctx.stroke();
        }
      }
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [videoMetadata]);

  // ── Canvas-based timeline ───────────────────────────────────────────────
  useEffect(() => {
    const canvas = timelineCanvasRef.current;
    if (!canvas || duration === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    if (W === 0 || H === 0) return;

    canvas.width = W * dpr;
    canvas.height = H * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#1f2937'; // gray-800
    ctx.fillRect(0, 0, W, H);

    // Segments
    for (const seg of mergedSegments) {
      const x = (seg.start / duration) * W;
      const w = Math.max(6, ((seg.end - seg.start) / duration) * W);
      const isActive =
        loopTracklet !== null &&
        loopTracklet.start_timestamp < seg.end &&
        loopTracklet.end_timestamp > seg.start;
      ctx.fillStyle = isActive ? 'rgba(96,165,250,0.9)' : 'rgba(59,130,246,0.7)';
      ctx.fillRect(x, 0, w, H);
    }
  }, [mergedSegments, duration, loopTracklet]);

  const handleTimelineClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (duration === 0) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const canvasW = rect.width;
      const clickX = e.clientX - rect.left;
      const time = (clickX / canvasW) * duration;

      // 1. Exact hit-test
      let seg = mergedSegments.find(s => s.start <= time && time <= s.end) ?? null;

      // 2. Tolerance hit-test: ±6 CSS pixels converted to time units
      if (!seg && canvasW > 0) {
        const toleranceTime = (6 / canvasW) * duration;
        let minDist = Infinity;
        for (const s of mergedSegments) {
          const dist = time < s.start ? s.start - time : time > s.end ? time - s.end : 0;
          if (dist <= toleranceTime && dist < minDist) {
            minDist = dist;
            seg = s;
          }
        }
      }

      if (seg) {
        setPopover({ segment: seg, x: clickX, canvasW });
      } else {
        setPopover(null);
      }
    },
    [duration, mergedSegments],
  );

  // ── Loop playback ───────────────────────────────────────────────────────
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    const tracklet = loopTrackletRef.current;
    if (!video || !tracklet) return;
    if (video.currentTime >= tracklet.end_timestamp) {
      video.currentTime = tracklet.start_timestamp;
    }
  }, []);

  const playTracklet = useCallback((tracklet: TrackletMetadata) => {
    loopTrackletRef.current = tracklet;
    setLoopTracklet(tracklet);
    setPopover(null);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = tracklet.start_timestamp;
    void video.play();
    // If bboxes not yet loaded, fetch just this one tracklet immediately
    if (!tracklet.bounding_boxes?.length) {
      fetchTrackletBatch([tracklet.tracklet_id]).then(([full]) => {
        if (full?.bounding_boxes?.length) {
          const enriched = { ...tracklet, bounding_boxes: full.bounding_boxes };
          // Only apply if user is still viewing this tracklet
          if (loopTrackletRef.current?.tracklet_id === tracklet.tracklet_id) {
            loopTrackletRef.current = enriched;
          }
        }
      });
    }
  }, []);

  const stopLoop = useCallback(() => {
    loopTrackletRef.current = null;
    setLoopTracklet(null);
    videoRef.current?.pause();
  }, []);

  // Tracklets in the currently hovered segment
  const popoverTracklets = popover
    ? selectedTracklets.filter(
        t =>
          t.start_timestamp <= popover.segment.end &&
          t.end_timestamp >= popover.segment.start,
      )
    : [];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Annotated timeline — canvas-based, only shown when tracklets are selected */}
      {selectedTracklets.length > 0 && (
        <div className="px-3 pt-3 pb-2 shrink-0">
          <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Timeline</p>
          <div className="relative">
            <canvas
              ref={timelineCanvasRef}
              className="w-full h-5 rounded overflow-hidden cursor-pointer block"
              onClick={handleTimelineClick}
            />

            {/* Popover tracklet list */}
            {popover && (
              <div
                className="absolute z-30 bg-gray-800 border border-gray-600 rounded-lg p-2 shadow-xl mt-1 max-h-48 overflow-y-auto"
                style={
                  popover.x <= popover.canvasW / 2
                    ? { left: popover.x }
                    : { right: popover.canvasW - popover.x }
                }
              >
                <p className="text-[10px] text-gray-400 mb-1.5 uppercase tracking-wide">
                  Tracklets in segment
                </p>
                <div className="flex flex-col gap-1">
                  {popoverTracklets.map(t => (
                    <button
                      key={t.tracklet_id}
                      onClick={() => playTracklet(t)}
                      className="flex items-center gap-2 text-left p-1.5 rounded hover:bg-gray-700 transition-colors"
                    >
                      <LazyThumbnail
                        trackletId={t.tracklet_id}
                        srcOverride={t.thumbnail_base64}
                        className="w-10 h-10 object-cover rounded shrink-0"
                        alt="thumb"
                      />
                      <span className="text-xs text-gray-200 capitalize">{t.class_name}</span>
                      <span className="text-[10px] text-gray-500 ml-auto">#{t.tracklet_id}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Video player — always mounted so refs stay valid */}
      <div className="flex-1 relative overflow-hidden bg-black" onClick={() => setPopover(null)}>
        {selectedVideoId && (
          <>
            <video
              ref={videoRef}
              src={videoStreamUrl(selectedVideoId)}
              className="w-full h-full object-contain"
              onTimeUpdate={handleTimeUpdate}
              controls={loopTracklet === null}
              muted
            />
            <canvas
              ref={bboxCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              width={videoMetadata?.width ?? 1280}
              height={videoMetadata?.height ?? 720}
            />
          </>
        )}

        {/* Empty state overlay */}
        {selectedTracklets.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <p className="text-gray-500 text-sm">Select tracklets in the scatter plot to begin.</p>
          </div>
        )}

        {/* Loop mode controls */}
        {loopTracklet && (
          <div className="absolute top-2 right-2 z-10 flex items-center gap-2 bg-gray-900/80 backdrop-blur px-2 py-1 rounded">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-gray-300 capitalize">{loopTracklet.class_name}</span>
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
