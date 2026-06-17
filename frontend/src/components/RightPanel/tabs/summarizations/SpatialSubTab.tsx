import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchTrackletBatch } from '../../../../lib/api';
import { useStore } from '../../../../stores/useStore';
import type { BoundingBox } from '../../../../types/index';

const GRID_W = 128;
const GRID_H = 72;
const CHUNK_SIZE = 20;

type Mode = 'centroid' | 'bbox';

// ── Module-level caches (survive component unmounts / tab switches) ───────────

// videoId → (trackletId → bboxes)
const bboxCache = new Map<string, Map<string, BoundingBox[]>>();

type GridResult = { grids: Float32Array[]; maxVals: number[]; numBuckets: number };
const gridCache = new Map<string, GridResult>();

// Persists config/committedKey across tab switches so the tab re-opens unchanged
let savedSpatialState: {
  videoId: string;
  numBuckets: number;
  mode: Mode;
  selectedClasses: Set<string> | null;
  committedKey: string | null;
} = { videoId: '', numBuckets: 4, mode: 'centroid', selectedClasses: null, committedKey: null };

// ── Pure helpers ──────────────────────────────────────────────────────────────

function formatTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function toBucketIdx(ts: number, minTime: number, bucketDur: number, n: number): number {
  return Math.min(n - 1, Math.max(0, Math.floor((ts - minTime) / bucketDur)));
}

function accumCentroid(grid: Float32Array, box: BoundingBox, vw: number, vh: number) {
  const gx = Math.max(0, Math.min(GRID_W - 1, Math.floor((box.center_x / vw) * GRID_W)));
  const gy = Math.max(0, Math.min(GRID_H - 1, Math.floor((box.center_y / vh) * GRID_H)));
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = gx + dx, ny = gy + dy;
      if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) grid[ny * GRID_W + nx] += 1;
    }
  }
}

function accumBbox(grid: Float32Array, box: BoundingBox, vw: number, vh: number) {
  const gx1 = Math.max(0, Math.floor((box.x1 / vw) * GRID_W));
  const gy1 = Math.max(0, Math.floor((box.y1 / vh) * GRID_H));
  const gx2 = Math.min(GRID_W, Math.ceil((box.x2 / vw) * GRID_W));
  const gy2 = Math.min(GRID_H, Math.ceil((box.y2 / vh) * GRID_H));
  for (let gy = gy1; gy < gy2; gy++)
    for (let gx = gx1; gx < gx2; gx++)
      grid[gy * GRID_W + gx] += 1;
}

function drawResultOnCanvases(
  result: GridResult,
  canvases: (HTMLCanvasElement | null)[],
  bgImg: HTMLImageElement | null,
  vw: number,
  vh: number,
) {
  const cellW = vw / GRID_W;
  const cellH = vh / GRID_H;
  for (let bi = 0; bi < result.numBuckets; bi++) {
    const canvas = canvases[bi];
    if (!canvas) continue;
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');
    if (!ctx) continue;
    if (bgImg) ctx.drawImage(bgImg, 0, 0, vw, vh);
    else { ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, vw, vh); }
    ctx.save();
    ctx.globalAlpha = 0.6;
    for (let gy = 0; gy < GRID_H; gy++) {
      for (let gx = 0; gx < GRID_W; gx++) {
        const val = result.grids[bi][gy * GRID_W + gx] / result.maxVals[bi];
        if (val <= 0) continue;
        ctx.fillStyle = `hsl(${Math.round(240 - val * 240)},100%,50%)`;
        ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
      }
    }
    ctx.restore();
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SpatialSubTab() {
  const {
    tracklets,
    globalClips,
    videoMetadata,
    selectedVideoId,
    setGlobalColorMode,
    setHighlightedSpatialClipIds,
  } = useStore();

  const videoId = selectedVideoId ?? '';

  const allClasses = useMemo(
    () => [...new Set(tracklets.map(t => t.class_name))].sort(),
    [tracklets],
  );

  // ── Config state — initialised from savedSpatialState ────────────────────
  const sameVideo = savedSpatialState.videoId === videoId;

  const [numBuckets, setNumBuckets] = useState(() => sameVideo ? savedSpatialState.numBuckets : 4);
  const [inputVal, setInputVal] = useState(() => String(sameVideo ? savedSpatialState.numBuckets : 4));
  const [mode, setMode] = useState<Mode>(() => sameVideo ? savedSpatialState.mode : 'centroid');
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(() =>
    sameVideo && savedSpatialState.selectedClasses !== null
      ? savedSpatialState.selectedClasses
      : new Set(allClasses),
  );
  const [committedKey, setCommittedKey] = useState<string | null>(() =>
    sameVideo ? savedSpatialState.committedKey : null,
  );

  // If allClasses arrives after initial render (data not loaded yet), initialise once
  const classesInited = useRef(sameVideo && savedSpatialState.selectedClasses !== null);
  useEffect(() => {
    if (classesInited.current || allClasses.length === 0) return;
    setSelectedClasses(new Set(allClasses));
    classesInited.current = true;
  }, [allClasses]);

  // ── Other state ───────────────────────────────────────────────────────────
  const [fetchProgress, setFetchProgress] = useState<{ done: number; total: number } | null>(null);
  const [clickedBucket, setClickedBucket] = useState<number | null>(null);

  const bgImgRef = useRef<HTMLImageElement | null>(null);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  // Preload background image
  useEffect(() => {
    if (!videoMetadata?.background_image_base64) { bgImgRef.current = null; return; }
    const img = new Image();
    img.src = `data:image/jpeg;base64,${videoMetadata.background_image_base64}`;
    img.onload = () => { bgImgRef.current = img; };
  }, [videoMetadata?.background_image_base64]);

  // Auto-switch embeddings to time colour; clear highlight on unmount
  useEffect(() => {
    setGlobalColorMode('time');
    return () => setHighlightedSpatialClipIds(null);
  }, [setGlobalColorMode, setHighlightedSpatialClipIds]);

  // Sync all mutable config back to savedSpatialState after every change
  useEffect(() => {
    savedSpatialState = { videoId, numBuckets, mode, selectedClasses, committedKey };
  });

  // Reset clicked-bucket highlight whenever the displayed result changes
  useEffect(() => {
    setClickedBucket(null);
    setHighlightedSpatialClipIds(null);
  }, [committedKey, setHighlightedSpatialClipIds]);

  // ── Time range ────────────────────────────────────────────────────────────
  const { minTime, maxTime } = useMemo(() => {
    if (videoMetadata && videoMetadata.duration > 0)
      return { minTime: 0, maxTime: videoMetadata.duration };
    if (tracklets.length === 0) return { minTime: 0, maxTime: 1 };
    const starts = tracklets.map(t => t.start_timestamp);
    const ends = tracklets.map(t => t.end_timestamp);
    return { minTime: Math.min(...starts), maxTime: Math.max(...ends) };
  }, [videoMetadata, tracklets]);

  // ── Cache key ─────────────────────────────────────────────────────────────
  const currentKey = useMemo(
    () => `${videoId}|${numBuckets}|${mode}|${[...selectedClasses].sort().join(',')}`,
    [videoId, numBuckets, mode, selectedClasses],
  );
  const isStale = committedKey !== null && committedKey !== currentKey;

  // ── Draw effect — runs after canvases mount ───────────────────────────────
  useEffect(() => {
    if (!committedKey || !videoMetadata) return;
    const result = gridCache.get(committedKey);
    if (!result) return;
    const vw = videoMetadata.width;
    const vh = videoMetadata.height;

    const doDrawAll = () => drawResultOnCanvases(result, canvasRefs.current, bgImgRef.current, vw, vh);

    if (bgImgRef.current) {
      doDrawAll();
    } else if (videoMetadata.background_image_base64) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${videoMetadata.background_image_base64}`;
      img.onload = () => { bgImgRef.current = img; doDrawAll(); };
    } else {
      doDrawAll();
    }
  // numBuckets intentionally included so canvases redraw when grid changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [committedKey, videoMetadata]);

  // ── Compute ───────────────────────────────────────────────────────────────
  const handleCompute = useCallback(async () => {
    if (!videoMetadata || tracklets.length === 0) return;

    const key = currentKey;
    if (gridCache.has(key)) { setCommittedKey(key); return; }

    const vw = videoMetadata.width;
    const vh = videoMetadata.height;
    const bucketDur = (maxTime - minTime) / numBuckets;

    // 1. Ensure all bboxes are fetched
    const videoBoxes: Map<string, BoundingBox[]> = bboxCache.get(videoId) ?? new Map();
    const missing = tracklets
      .filter(t => !videoBoxes.has(t.tracklet_id))
      .map(t => t.tracklet_id);

    if (missing.length > 0) {
      setFetchProgress({ done: 0, total: missing.length });
      for (let i = 0; i < missing.length; i += CHUNK_SIZE) {
        try {
          const results = await fetchTrackletBatch(missing.slice(i, i + CHUNK_SIZE));
          for (const r of results)
            if (r.bounding_boxes?.length) videoBoxes.set(r.tracklet_id, r.bounding_boxes);
        } catch { /* skip failed chunk */ }
        setFetchProgress(prev => prev && { done: Math.min(prev.total, i + CHUNK_SIZE), total: prev.total });
      }
      bboxCache.set(videoId, videoBoxes);
    }

    // 2. Compute grids (filter by selected classes)
    const grids = Array.from({ length: numBuckets }, () => new Float32Array(GRID_W * GRID_H));
    const filtered = tracklets.filter(t => selectedClasses.has(t.class_name));
    for (const t of filtered) {
      const bboxes = videoBoxes.get(t.tracklet_id);
      if (!bboxes) continue;
      for (const box of bboxes) {
        const bi = toBucketIdx(box.timestamp, minTime, bucketDur, numBuckets);
        if (mode === 'centroid') accumCentroid(grids[bi], box, vw, vh);
        else accumBbox(grids[bi], box, vw, vh);
      }
    }
    const maxVals = grids.map(g => Math.max(...g, 1));

    gridCache.set(key, { grids, maxVals, numBuckets });
    setFetchProgress(null);
    setCommittedKey(key);
  }, [currentKey, videoId, videoMetadata, tracklets, numBuckets, mode, selectedClasses, minTime, maxTime]);

  // ── Bucket click ──────────────────────────────────────────────────────────
  // Use displayN (committed result's bucket count) so bucket boundaries align
  // with what's displayed even when the user has changed numBuckets without recomputing.
  const handleBucketClick = useCallback((bi: number, n: number) => {
    const next = clickedBucket === bi ? null : bi;
    setClickedBucket(next);

    if (next === null) {
      setHighlightedSpatialClipIds(null);
      return;
    }

    const bucketDur = (maxTime - minTime) / n;

    // Pre-compute which tracklet IDs fall in this bucket AND match selected classes
    const bucketTrackletIds = new Set(
      tracklets
        .filter(t => {
          if (!selectedClasses.has(t.class_name)) return false;
          const mid = (t.start_timestamp + t.end_timestamp) / 2;
          return toBucketIdx(mid, minTime, bucketDur, n) === next;
        })
        .map(t => t.tracklet_id),
    );

    // A global clip is highlighted if it contains at least one such tracklet
    const clipIds = new Set(
      globalClips
        .filter(c => c.tracklet_ids.some(tid => bucketTrackletIds.has(tid)))
        .map(c => c.clip_id),
    );

    // Fallback: if globalClips have no tracklet_ids info, match by clip time bucket + class
    if (clipIds.size === 0 && globalClips.some(c => !c.tracklet_ids?.length)) {
      globalClips.forEach(c => {
        const mid = (c.start_time + c.end_time) / 2;
        if (toBucketIdx(mid, minTime, bucketDur, n) === next) clipIds.add(c.clip_id);
      });
    }

    setHighlightedSpatialClipIds(clipIds);
  }, [clickedBucket, setHighlightedSpatialClipIds, tracklets, globalClips, selectedClasses, minTime, maxTime]);

  // ── Class filter helpers ──────────────────────────────────────────────────
  const toggleClass = (cls: string) =>
    setSelectedClasses(prev => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls); else next.add(cls);
      return next;
    });

  const toggleAll = () =>
    setSelectedClasses(
      selectedClasses.size === allClasses.length ? new Set() : new Set(allClasses),
    );

  // ── Render ────────────────────────────────────────────────────────────────
  if (!videoMetadata || tracklets.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">Load a video to see spatial summarizations.</p>
      </div>
    );
  }

  const committedResult = committedKey ? gridCache.get(committedKey) : undefined;
  const displayN = committedResult?.numBuckets ?? numBuckets;
  const displayBucketDur = (maxTime - minTime) / displayN;
  const cols = displayN === 1 ? 'grid-cols-1' : 'grid-cols-2';
  const isLoading = fetchProgress !== null;

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* ── Config panel ──────────────────────────────────────────────── */}
      <div className="shrink-0 flex flex-col gap-2 px-3 py-2 border-b border-gray-700 bg-gray-900">

        {/* Bucket count · mode toggle · Compute button */}
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-400 flex items-center gap-1.5">
            Buckets
            <input
              type="number" min={1} max={16} value={inputVal}
              onChange={e => {
                setInputVal(e.target.value);
                const n = parseInt(e.target.value, 10);
                if (!isNaN(n) && n >= 1 && n <= 16) setNumBuckets(n);
              }}
              className="w-12 bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-white text-center focus:outline-none focus:border-blue-500"
            />
          </label>

          <div className="flex rounded overflow-hidden border border-gray-600">
            {(['centroid', 'bbox'] as Mode[]).map(m => (
              <button key={m} onClick={() => setMode(m)}
                className={`px-2.5 py-1 text-xs capitalize transition-colors ${
                  mode === m ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >{m}</button>
            ))}
          </div>

          <button
            onClick={handleCompute}
            disabled={isLoading || selectedClasses.size === 0}
            className={`ml-auto flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition-colors ${
              isLoading || selectedClasses.size === 0
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : isStale
                  ? 'bg-yellow-600 hover:bg-yellow-500 text-white'
                  : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {isLoading
              ? <><div className="w-2.5 h-2.5 rounded-full border-2 border-white border-t-transparent animate-spin" />{fetchProgress && `${fetchProgress.done}/${fetchProgress.total}`}</>
              : <>{isStale && '⚠ '}Compute</>
            }
          </button>
        </div>

        {/* Class filter badges */}
        <div className="flex flex-wrap items-center gap-1">
          <button onClick={toggleAll}
            className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 border border-gray-700 rounded transition-colors">
            {selectedClasses.size === allClasses.length ? 'none' : 'all'}
          </button>
          {allClasses.map(cls => (
            <button key={cls} onClick={() => toggleClass(cls)}
              className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors ${
                selectedClasses.has(cls)
                  ? 'bg-blue-700 border-blue-500 text-white'
                  : 'bg-gray-800 border-gray-600 text-gray-500 hover:text-gray-300'
              }`}
            >{cls}</button>
          ))}
        </div>
      </div>

      {/* ── Heatmap grid or placeholder ───────────────────────────────── */}
      {!committedResult ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-gray-500 text-sm">
            {isLoading ? 'Computing…' : 'Select classes and click Compute.'}
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
        <div className={`p-2 grid ${cols} gap-2`}>
          {Array.from({ length: displayN }, (_, bi) => {
            const tStart = minTime + bi * displayBucketDur;
            const tEnd = tStart + displayBucketDur;
            const isActive = clickedBucket === bi;

            return (
              <div key={bi}
                className={`flex flex-col cursor-pointer rounded overflow-hidden border-2 transition-colors ${
                  isActive ? 'border-blue-400' : 'border-gray-700 hover:border-gray-500'
                }`}
                onClick={() => handleBucketClick(bi, displayN)}
              >
                <canvas
                  ref={el => { canvasRefs.current[bi] = el; }}
                  className="w-full bg-gray-900"
                  style={{ imageRendering: 'pixelated', aspectRatio: `${videoMetadata.width}/${videoMetadata.height}`, display: 'block' }}
                />
                <div className="shrink-0 px-1.5 py-0.5 bg-gray-800 text-center">
                  <p className="text-[10px] text-gray-400">
                    Bucket {bi + 1} · {formatTime(tStart)}–{formatTime(tEnd)}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
        </div>
      )}

      {/* ── Density legend ─────────────────────────────────────────────── */}
      <div className="shrink-0 px-4 pb-2 pt-1 border-t border-gray-700">
        <p className="text-[10px] text-gray-400 text-center mb-1 uppercase tracking-wide">
          {mode === 'centroid' ? 'Centroid Density' : 'Bbox Occupancy'}
        </p>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400 shrink-0">Low</span>
          <div className="flex-1 h-2.5 rounded" style={{
            background: 'linear-gradient(to right, hsl(240,100%,50%), hsl(180,100%,50%), hsl(120,100%,50%), hsl(60,100%,50%), hsl(0,100%,50%))',
            opacity: 0.85,
          }} />
          <span className="text-[10px] text-gray-400 shrink-0">High</span>
        </div>
      </div>
    </div>
  );
}
