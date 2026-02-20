import { useCallback, useEffect, useRef, useState } from 'react';
import { useStore } from '../../../stores/useStore';
import { videoStreamUrl } from '../../../lib/api';
import { mergeIntervals } from '../../../lib/utils';
import { BBOX_COLOR, BBOX_TRACK_COLOR } from '../../../lib/colors';
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
}

function findClosestBox(boxes: BoundingBox[] | undefined | null, time: number): BoundingBox | null {
  if (!boxes || boxes.length === 0) return null;
  let best = boxes[0];
  let bestDiff = Math.abs(boxes[0].timestamp - time);
  for (const box of boxes) {
    const diff = Math.abs(box.timestamp - time);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = box;
    }
  }
  return best;
}

export default function VideoPlayerTab({ selectedTracklets }: Props) {
  const { selectedVideoId, videoMetadata } = useStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loopTrackletRef = useRef<TrackletMetadata | null>(null);
  const rafRef = useRef<number | null>(null);

  const [loopTracklet, setLoopTracklet] = useState<TrackletMetadata | null>(null);
  const [popover, setPopover] = useState<PopoverState | null>(null);

  const duration = videoMetadata?.duration ?? 0;

  // Compute merged timeline segments
  const mergedSegments: Segment[] = (() => {
    const intervals: [number, number][] = selectedTracklets.map(t => [
      t.start_timestamp,
      t.end_timestamp,
    ]);
    return mergeIntervals(intervals).map(([s, e]) => ({ start: s, end: e }));
  })();

  // Canvas bbox drawing loop
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const tracklet = loopTrackletRef.current;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (tracklet) {
        const box = findClosestBox(tracklet.bounding_boxes, video.currentTime);
        if (box) {
          // Letterbox-aware coordinate transform:
          // The <video> uses object-contain, centering the content with bars.
          // The <canvas> fills the same container but has internal px = video resolution.
          // We must offset by the bar size so coords map onto video content, not the bars.
          const containerW = canvas.clientWidth || canvas.width;
          const containerH = canvas.clientHeight || canvas.height;
          const videoW = videoMetadata?.width ?? canvas.width;
          const videoH = videoMetadata?.height ?? canvas.height;
          const videoAR = videoW / videoH;
          const containerAR = containerW / containerH;

          let renderedW: number, renderedH: number, offsetX: number, offsetY: number;
          if (containerAR > videoAR) {
            // Pillarbox: bars on left & right
            renderedH = containerH;
            renderedW = containerH * videoAR;
            offsetX = (containerW - renderedW) / 2;
            offsetY = 0;
          } else {
            // Letterbox: bars on top & bottom
            renderedW = containerW;
            renderedH = containerW / videoAR;
            offsetX = 0;
            offsetY = (containerH - renderedH) / 2;
          }

          // Map video pixel → canvas internal coordinate
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
          // Track line
          ctx.strokeStyle = BBOX_TRACK_COLOR;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          tracklet.bounding_boxes.forEach((b, i) => {
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

  // Loop playback
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
  }, []);

  const stopLoop = useCallback(() => {
    loopTrackletRef.current = null;
    setLoopTracklet(null);
    videoRef.current?.pause();
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Annotated timeline — only shown when tracklets are selected */}
      {selectedTracklets.length > 0 && <div className="px-3 pt-3 pb-2 shrink-0">
        <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Timeline</p>
        <div
          className="relative h-5 bg-gray-800 rounded overflow-hidden cursor-pointer"
          onClick={e => {
            if (duration === 0) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            const time = frac * duration;
            // Find segment at this time
            const seg = mergedSegments.find(s => s.start <= time && time <= s.end);
            if (seg) {
              setPopover({ segment: seg, x: e.clientX - rect.left });
            } else {
              setPopover(null);
            }
          }}
        >
          {duration > 0 &&
            mergedSegments.map((seg, i) => (
              <div
                key={i}
                className={`absolute h-full transition-colors ${
                  loopTracklet !== null
                  && loopTracklet.start_timestamp < seg.end
                  && loopTracklet.end_timestamp   > seg.start
                    ? 'bg-blue-400/90 ring-2 ring-white/90 ring-inset'
                    : 'bg-blue-500/70 hover:bg-blue-400/80'
                }`}
                style={{
                  left: `${(seg.start / duration) * 100}%`,
                  width: `${((seg.end - seg.start) / duration) * 100}%`,
                }}
              />
            ))}
        </div>

        {/* Popover tracklet list */}
        {popover && (
          <div
            className="absolute z-30 bg-gray-800 border border-gray-600 rounded-lg p-2 shadow-xl mt-1 max-h-48 overflow-y-auto"
            style={{ left: popover.x }}
          >
            <p className="text-[10px] text-gray-400 mb-1.5 uppercase tracking-wide">
              Tracklets in segment
            </p>
            <div className="flex flex-col gap-1">
              {selectedTracklets
                .filter(
                  t =>
                    t.start_timestamp <= popover.segment.end &&
                    t.end_timestamp >= popover.segment.start,
                )
                .map(t => (
                  <button
                    key={t.tracklet_id}
                    onClick={() => playTracklet(t)}
                    className="flex items-center gap-2 text-left p-1.5 rounded hover:bg-gray-700 transition-colors"
                  >
                    {t.thumbnail_base64 && (
                      <img
                        src={`data:image/jpeg;base64,${t.thumbnail_base64}`}
                        alt="thumb"
                        className="w-10 h-10 object-cover rounded shrink-0"
                      />
                    )}
                    <span className="text-xs text-gray-200 capitalize">{t.class_name}</span>
                    <span className="text-[10px] text-gray-500 ml-auto">#{t.tracklet_id}</span>
                  </button>
                ))}
            </div>
          </div>
        )}
      </div>}

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
              ref={canvasRef}
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
