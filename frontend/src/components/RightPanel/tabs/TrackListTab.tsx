import { useVirtualizer } from "@tanstack/react-virtual";
import Konva from "konva";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Image as KonvaImage, Layer, Shape, Rect, Stage } from "react-konva";
import { fetchTrackletBatch, videoStreamUrl } from "../../../lib/api";
import { getClassColorHex, speedToColorHex } from "../../../lib/colors";
import { useStore } from "../../../stores/useStore";
import type { BoundingBox, TrackletMetadata } from "../../../types/index";

interface Props {
  selectedTracklets: TrackletMetadata[];
}

const MIN_SCALE = 0.3;
const MAX_SCALE = 8;

function SpeedSparkline({
  points,
  maxSpeed,
  width = 200,
  height = 24,
}: {
  points: BoundingBox[];
  maxSpeed: number;
  width?: number;
  height?: number;
}) {
  if (!points || points.length < 2) return null;
  const len = points.length;
  const speeds = points.map((p) => p.speed || 0);
  const localMax = Math.max(...speeds, 0.01);

  const segments = [];
  for (let i = 0; i < len - 1; i++) {
    const x1 = (i / (len - 1)) * width;
    const x2 = ((i + 1) / (len - 1)) * width;
    const y1 = height - (speeds[i] / localMax) * (height - 2) - 1;
    const y2 = height - (speeds[i + 1] / localMax) * (height - 2) - 1;
    const avgSpeed = (speeds[i] + speeds[i + 1]) / 2;
    segments.push(
      <line
        key={i}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={speedToColorHex(avgSpeed, maxSpeed)}
        strokeWidth={1.5}
        strokeLinecap="round"
      />,
    );
  }

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="block"
    >
      {segments}
    </svg>
  );
}

function findBoxAtTime(boxes: BoundingBox[], time: number): BoundingBox | null {
  if (!boxes || boxes.length === 0) return null;
  let lo = 0, hi = boxes.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (boxes[mid].timestamp < time) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(boxes[lo - 1].timestamp - time) <= Math.abs(boxes[lo].timestamp - time))
    return boxes[lo - 1];
  return boxes[lo];
}

function findNearestTracklet(
  tracklets: TrackletMetadata[],
  vx: number,
  vy: number,
  radiusPx: number,
  bboxMap: Map<string, BoundingBox[]>,
): TrackletMetadata | null {
  let best: TrackletMetadata | null = null;
  let bestDist = radiusPx * radiusPx;
  for (const t of tracklets) {
    for (const box of (bboxMap.get(t.tracklet_id) ?? [])) {
      const dx = box.center_x - vx;
      const dy = box.center_y - vy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist) {
        bestDist = d2;
        best = t;
      }
    }
  }
  return best;
}

const CARD_HEIGHT = 84; // px — estimated fixed height per card

function TrackCardList({
  tracklets,
  selectedTrackletId,
  maxSpeed,
  onTrackClick,
  onMouseEnter,
  onMouseLeave,
  bboxMap,
}: {
  tracklets: TrackletMetadata[];
  selectedTrackletId: string | null;
  maxSpeed: number;
  onTrackClick: (t: TrackletMetadata) => void;
  onMouseEnter: (id: string) => void;
  onMouseLeave: () => void;
  bboxMap: Map<string, BoundingBox[]>;
}) {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: tracklets.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => CARD_HEIGHT,
    overscan: 5,
    measureElement: (el) => el.getBoundingClientRect().height,
  });

  if (tracklets.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-xs text-gray-500 text-center py-4">
          No tracks match current filters
        </p>
      </div>
    );
  }

  return (
    <div ref={parentRef} className="flex-1 overflow-y-auto">
      <div
        style={{ height: rowVirtualizer.getTotalSize(), position: "relative" }}
        className="px-2 py-1"
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const t = tracklets[virtualRow.index];
          const isSelected = t.tracklet_id === selectedTrackletId;
          return (
            <div
              key={t.tracklet_id}
              ref={rowVirtualizer.measureElement}
              data-index={virtualRow.index}
              style={{
                position: "absolute",
                top: virtualRow.start,
                left: 0,
                right: 0,
                padding: "0 4px 6px",
              }}
            >
              <button
                onClick={() => onTrackClick(t)}
                onMouseEnter={() => onMouseEnter(t.tracklet_id)}
                onMouseLeave={onMouseLeave}
                className={`w-full text-left rounded-lg p-2 transition-colors cursor-pointer ${
                  isSelected
                    ? "bg-blue-600/20 border border-blue-500"
                    : "bg-gray-800/60 border border-gray-700 hover:border-gray-500"
                }`}
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <div
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: getClassColorHex(t.class_name) }}
                  />
                  <span className="text-xs font-medium capitalize text-gray-200 truncate">
                    {t.class_name}
                  </span>
                  <span className="text-[10px] text-gray-500 ml-auto font-mono">
                    #{t.tracklet_id}
                  </span>
                </div>
                <div className="mb-1.5 rounded overflow-hidden bg-gray-900/50 px-1 py-0.5">
                  <SpeedSparkline points={bboxMap.get(t.tracklet_id) ?? []} maxSpeed={maxSpeed} />
                </div>
                <div className="flex gap-3 text-[10px] text-gray-400">
                  <span>
                    {t.avg_speed.toFixed(1)}{" "}
                    <span className="text-gray-500">px/s</span>
                  </span>
                  <span>
                    {t.duration.toFixed(1)}
                    <span className="text-gray-500">s</span>
                  </span>
                  <span>
                    {t.point_count} <span className="text-gray-500">pts</span>
                  </span>
                </div>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function TrackListTab({ selectedTracklets }: Props) {
  const { selectedVideoId, videoMetadata, setHighlightedTrackletId, selectedTrackletIds } =
    useStore();

  // Stable key derived from selected IDs — avoids effect loop from array identity changes
  const selectionKey = useMemo(
    () => [...selectedTrackletIds].sort().join(','),
    [selectedTrackletIds],
  );

  // Per-tracklet bbox data, loaded progressively in background chunks
  const [bboxMap, setBboxMap] = useState<Map<string, BoundingBox[]>>(new Map());

  useEffect(() => {
    setBboxMap(new Map());
    const ids = selectionKey ? selectionKey.split(',') : [];
    if (ids.length === 0) return;
    let cancelled = false;
    const CHUNK_SIZE = 20;

    async function loadChunks() {
      for (let i = 0; i < ids.length; i += CHUNK_SIZE) {
        if (cancelled) break;
        const chunk = ids.slice(i, i + CHUNK_SIZE);
        try {
          const results = await fetchTrackletBatch(chunk);
          if (!cancelled) {
            setBboxMap(prev => {
              const next = new Map(prev);
              for (const r of results) {
                if (r.bounding_boxes?.length) next.set(r.tracklet_id, r.bounding_boxes);
              }
              return next;
            });
          }
        } catch { /* skip failed chunk, continue */ }
      }
    }
    loadChunks();
    return () => { cancelled = true; };
  }, [selectionKey]);

  // Filters
  const [activeClasses, setActiveClasses] = useState<Set<string> | null>(null);
  const [minSpeed, setMinSpeed] = useState(0);
  const [selectedTrackletId, setSelectedTrackletId] = useState<string | null>(
    null,
  );

  // Konva canvas
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 1, h: 1 });
  const [stageScale, setStageScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
  // Video playback
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const animRef = useRef<Konva.Animation | null>(null);
  const bgAnimRef = useRef<Konva.Animation | null>(null);
  const bboxRectRef = useRef<Konva.Rect>(null);
  const bboxLayerRef = useRef<Konva.Layer>(null);
  const bgLayerRef = useRef<Konva.Layer>(null);
  const endTimeRef = useRef(0);
  const startTimeRef = useRef(0);
  const activeBoxesRef = useRef<BoundingBox[]>([]);

  const handleVideoRef = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
    setVideoEl(el);
  }, []);

  const vw = videoMetadata?.width ?? 1280;
  const vh = videoMetadata?.height ?? 720;

  // Resize observer for canvas container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ w: width || 1, h: height || 1 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Center video in canvas whenever canvas size or video dimensions change
  useEffect(() => {
    const sf = Math.min(canvasSize.w / vw, canvasSize.h / vh);
    setStagePos({
      x: (canvasSize.w - vw * sf) / 2,
      y: (canvasSize.h - vh * sf) / 2,
    });
  }, [canvasSize, vw, vh]);

  const scaleFactor = Math.min(canvasSize.w / vw, canvasSize.h / vh);

  // Continuously redraw bgLayer so KonvaImage shows live video frames
  useEffect(() => {
    const layer = bgLayerRef.current;
    if (!videoEl || !layer) return;
    const anim = new Konva.Animation(() => {
      // empty body — just drives layer redraws at 60fps
    }, [layer]);
    anim.start();
    bgAnimRef.current = anim;
    return () => {
      anim.stop();
      bgAnimRef.current = null;
    };
  }, [videoEl]);

  // Register timeupdate whenever videoEl changes — loops playback back to start
  useEffect(() => {
    const vid = videoEl;
    if (!vid) return;
    const onTimeUpdate = () => {
      if (endTimeRef.current > 0 && vid.currentTime >= endTimeRef.current) {
        vid.currentTime = startTimeRef.current;
        void vid.play();
      }
    };
    vid.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      vid.removeEventListener("timeupdate", onTimeUpdate);
      vid.pause();
      if (animRef.current) {
        animRef.current.stop();
        animRef.current = null;
      }
      if (bgAnimRef.current) {
        bgAnimRef.current.stop();
        bgAnimRef.current = null;
      }
    };
  }, [videoEl]);

  // Reset when video changes
  useEffect(() => {
    stopPlayback();
    endTimeRef.current = 0;
    startTimeRef.current = 0;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVideoId]);

  const stopPlayback = useCallback(() => {
    videoRef.current?.pause();
    if (animRef.current) {
      animRef.current.stop();
      animRef.current = null;
    }
    if (bboxRectRef.current) bboxRectRef.current.visible(false);
    bboxLayerRef.current?.batchDraw();
    activeBoxesRef.current = [];
    endTimeRef.current = 0;
    startTimeRef.current = 0;
  }, []);

  const startPlayback = useCallback(
    (startTime: number, endTime: number, boxes: BoundingBox[]) => {
      const vid = videoRef.current;
      if (!vid) return;
      startTimeRef.current = startTime;
      endTimeRef.current = endTime;
      activeBoxesRef.current = boxes;
      vid.currentTime = startTime;

      const onSeeked = () => {
        vid.removeEventListener("seeked", onSeeked);
        void vid.play();
        const layers = bboxLayerRef.current ? [bboxLayerRef.current] : [];
        if (layers.length > 0) {
          const anim = new Konva.Animation(() => {
            const rect = bboxRectRef.current;
            const boxes2 = activeBoxesRef.current;
            if (rect && boxes2.length > 0) {
              const box = findBoxAtTime(boxes2, vid.currentTime);
              if (box) {
                rect.setAttrs({
                  x: box.x1 * scaleFactor,
                  y: box.y1 * scaleFactor,
                  width: (box.x2 - box.x1) * scaleFactor,
                  height: (box.y2 - box.y1) * scaleFactor,
                  visible: true,
                });
              }
            }
          }, layers);
          anim.start();
          animRef.current = anim;
        }
      };
      vid.addEventListener("seeked", onSeeked);
    },
    [scaleFactor],
  );

  // Available classes from selection
  const availableClasses = useMemo(() => {
    const names = new Set<string>();
    for (const t of selectedTracklets) names.add(t.class_name);
    return Array.from(names).sort();
  }, [selectedTracklets]);

  // Max speed for sparkline scale
  const maxSpeed = useMemo(() => {
    return Math.max(...selectedTracklets.map((t) => t.avg_speed), 1);
  }, [selectedTracklets]);

  const defaultMinSpeed = 400;

  // Filtered tracklets
  const filteredTracklets = useMemo(() => {
    return selectedTracklets
      .filter((t) => {
        if (activeClasses !== null && !activeClasses.has(t.class_name))
          return false;
        if (t.avg_speed < minSpeed) return false;
        return true;
      })
      .sort((a, b) => b.avg_speed - a.avg_speed);
  }, [selectedTracklets, activeClasses, minSpeed]);

  // Reset filters when tracklets change
  useEffect(() => {
    setActiveClasses(null);
    setMinSpeed(defaultMinSpeed);
    setSelectedTrackletId(null);
  }, [selectedTracklets, defaultMinSpeed]);

  const toggleClass = useCallback(
    (cls: string) => {
      setActiveClasses((prev) => {
        if (prev === null) {
          if (availableClasses.length <= 1) return null;
          return new Set([cls]);
        }
        const next = new Set(prev);
        if (next.has(cls)) {
          next.delete(cls);
          if (next.size === 0) return null;
        } else {
          next.add(cls);
          if (next.size === availableClasses.length) return null;
        }
        return next;
      });
    },
    [availableClasses],
  );

  const handleTrackClick = useCallback(
    (tracklet: TrackletMetadata) => {
      if (selectedTrackletId === tracklet.tracklet_id) {
        setSelectedTrackletId(null);
        stopPlayback();
      } else {
        setSelectedTrackletId(tracklet.tracklet_id);
        stopPlayback();
        const cached = bboxMap.get(tracklet.tracklet_id);
        if (cached && cached.length > 0) {
          startPlayback(tracklet.start_timestamp, tracklet.end_timestamp, cached);
        } else {
          // Fetch just this one tracklet immediately so playback starts fast
          fetchTrackletBatch([tracklet.tracklet_id]).then(([full]) => {
            const boxes = full?.bounding_boxes ?? [];
            if (boxes.length > 0) {
              setBboxMap(prev => new Map(prev).set(tracklet.tracklet_id, boxes));
            }
            startPlayback(tracklet.start_timestamp, tracklet.end_timestamp, boxes);
          });
        }
      }
    },
    [selectedTrackletId, stopPlayback, startPlayback, bboxMap],
  );

  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    if (!stage) return;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const newScale = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, oldScale * (1 + direction * 0.1)),
    );
    setStageScale(newScale);
    setStagePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  }, []);

  // Draw all track lines imperatively into a single canvas — avoids creating
  // thousands of React/Konva elements which would freeze the reconciler.
  const drawTrackLines = useCallback(
    (ctx: Konva.Context) => {
      const native = (ctx as any)._context as CanvasRenderingContext2D;
      native.save();
      for (const t of filteredTracklets) {
        const isSelected = t.tracklet_id === selectedTrackletId;
        const opacity = selectedTrackletId === null ? 0.85 : isSelected ? 1 : 0.15;
        const strokeW = (isSelected ? 1 : 0.5) / scaleFactor;
        native.globalAlpha = opacity;
        native.lineWidth = strokeW;
        native.lineCap = "round";
        const tBoxes = bboxMap.get(t.tracklet_id);
        for (let i = 0; tBoxes && i < tBoxes.length - 1; i++) {
          const p = tBoxes[i];
          const q = tBoxes[i + 1];
          const speed = ((p.speed || 0) + (q.speed || 0)) / 2;
          native.strokeStyle = speedToColorHex(speed, maxSpeed);
          native.beginPath();
          native.moveTo(p.center_x * scaleFactor, p.center_y * scaleFactor);
          native.lineTo(q.center_x * scaleFactor, q.center_y * scaleFactor);
          native.stroke();
        }
      }
      native.globalAlpha = 1;
      native.restore();
    },
    [filteredTracklets, selectedTrackletId, scaleFactor, maxSpeed, bboxMap],
  );

  const selectedTrackMeta = selectedTrackletId
    ? selectedTracklets.find((t) => t.tracklet_id === selectedTrackletId)
    : null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top half: video + track line overlay */}
      <div className="h-1/2 flex shrink-0 overflow-hidden">

      {/* Stage container — takes remaining width */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden"
      >
        {selectedTracklets.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-gray-900/60">
            <p className="text-gray-400 text-sm">
              Select tracklets in the scatter plot to begin.
            </p>
          </div>
        )}
        {/* Hidden video element — drawn into Konva bgLayer via KonvaImage */}
        <video
          ref={handleVideoRef}
          muted
          playsInline
          preload="auto"
          src={selectedVideoId ? videoStreamUrl(selectedVideoId) : undefined}
          style={{ display: "none" }}
        />

        <Stage
          width={canvasSize.w}
          height={canvasSize.h}
          scaleX={stageScale}
          scaleY={stageScale}
          x={stagePos.x}
          y={stagePos.y}
          draggable
          onWheel={handleWheel}
          onDragEnd={(e: Konva.KonvaEventObject<DragEvent>) =>
            setStagePos({ x: e.target.x(), y: e.target.y() })
          }
          onClick={(e: Konva.KonvaEventObject<MouseEvent>) => {
            const stage = e.target.getStage();
            if (!stage) return;
            const pos = stage.getPointerPosition();
            if (!pos) return;
            // Convert from stage container coords to video-pixel coords
            const videoX = (pos.x - stagePos.x) / stageScale / scaleFactor;
            const videoY = (pos.y - stagePos.y) / stageScale / scaleFactor;
            const HIT_RADIUS = 20 / stageScale;
            const found = findNearestTracklet(filteredTracklets, videoX, videoY, HIT_RADIUS, bboxMap);
            if (found) {
              handleTrackClick(found);
            } else {
              setSelectedTrackletId(null);
              stopPlayback();
            }
          }}
        >
          {/* bgLayer: renders live video frames via KonvaImage */}
          <Layer ref={bgLayerRef} listening={false}>
            {videoEl && (
              <KonvaImage
                image={videoEl}
                width={vw * scaleFactor}
                height={vh * scaleFactor}
              />
            )}
          </Layer>
          <Layer ref={bboxLayerRef} listening={false}>
            <Rect
              ref={bboxRectRef}
              stroke={
                selectedTrackMeta
                  ? getClassColorHex(selectedTrackMeta.class_name)
                  : "#fff"
              }
              strokeWidth={0.25 / scaleFactor}
              fill="transparent"
              visible={false}
            />
          </Layer>
          <Layer listening={false}>
            <Shape listening={false} sceneFunc={drawTrackLines} />
          </Layer>
        </Stage>

      </div>

      {/* Vertical speed legend — right of frame, centered */}
      <div className="flex flex-col items-center justify-center gap-1.5 px-2 py-3 shrink-0 border-l border-gray-700">
        <span
          className="text-[10px] text-gray-400 mb-1"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          Speed
        </span>
        <span className="text-[10px] text-gray-400 whitespace-nowrap">{maxSpeed.toFixed(0)}</span>
        <div
          className="w-3 rounded"
          style={{
            flex: "1 1 0",
            maxHeight: "7rem",
            background:
              "linear-gradient(to bottom, rgb(239,68,68), rgb(234,179,8), rgb(34,197,94), rgb(6,182,212), rgb(59,130,246))",
          }}
        />
        <span className="text-[10px] text-gray-400">0</span>
        <span
          className="text-[10px] text-gray-500 mt-1"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          px/s
        </span>
      </div>

      </div>

      {/* Track list: bottom half */}
      <div className="h-1/2 flex flex-col overflow-hidden border-t border-gray-700">
        {/* Filters */}
        <div className="px-3 py-2 border-b border-gray-700 shrink-0 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">
              Tracks ({filteredTracklets.length} / {selectedTracklets.length})
            </span>
            {activeClasses !== null && (
              <button
                onClick={() => setActiveClasses(null)}
                className="text-[10px] text-blue-400 hover:text-blue-300"
              >
                Reset
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {availableClasses.map((cls) => {
              const active = activeClasses === null || activeClasses.has(cls);
              return (
                <button
                  key={cls}
                  onClick={() => toggleClass(cls)}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium capitalize transition-colors cursor-pointer ${
                    active
                      ? "bg-gray-700 text-gray-200"
                      : "bg-gray-800/40 text-gray-500 line-through"
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: getClassColorHex(cls) }}
                  />
                  {cls}
                </button>
              );
            })}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">
              Min speed
            </span>
            <span className="text-[10px] text-gray-400 font-mono">
              {minSpeed} px/s
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.ceil(maxSpeed)}
            step={1}
            value={minSpeed}
            onChange={(e) => setMinSpeed(Number(e.target.value))}
            className="w-full h-1 accent-blue-500 cursor-pointer"
          />
        </div>

        {/* Track cards — virtualised to stay smooth with thousands of items */}
        {selectedTracklets.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-xs text-gray-500">No tracklets selected.</p>
          </div>
        ) : (
          <TrackCardList
            tracklets={filteredTracklets}
            selectedTrackletId={selectedTrackletId}
            maxSpeed={maxSpeed}
            onTrackClick={handleTrackClick}
            onMouseEnter={(id) => setHighlightedTrackletId(id)}
            onMouseLeave={() => setHighlightedTrackletId(null)}
            bboxMap={bboxMap}
          />
        )}
      </div>
    </div>
  );
}
