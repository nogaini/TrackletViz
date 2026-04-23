import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../../../stores/useStore';
import { getClassColorHex } from '../../../../lib/colors';
import { videoStreamUrl } from '../../../../lib/api';
import type { GlobalClipMetadata } from '../../../../types/index';

// ── Loop modal ────────────────────────────────────────────────────────────────
interface LoopModalProps {
  clip: GlobalClipMetadata;
  streamUrl: string;
  loopStart: number;
  loopEnd: number;
  onClose: () => void;
}

function LoopModal({ clip, streamUrl, loopStart, loopEnd, onClose }: LoopModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-2xl w-[900px] max-w-[95vw] p-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-medium text-gray-200">
            Clip {clip.clip_index} — {loopStart.toFixed(1)}s – {loopEnd.toFixed(1)}s
          </span>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-lg leading-none">✕</button>
        </div>
        <video
          src={streamUrl}
          autoPlay
          controls
          className="w-full rounded"
          onLoadedMetadata={e => { e.currentTarget.currentTime = loopStart; }}
          onTimeUpdate={e => {
            const v = e.currentTarget;
            if (v.currentTime >= loopEnd) v.currentTime = loopStart;
          }}
        />
      </div>
    </div>
  );
}

// ── Chart geometry ────────────────────────────────────────────────────────────
const SVG_W = 600;
const SVG_H = 180;
const PAD = { top: 15, right: 15, bottom: 44, left: 50 };
const PLOT_W = SVG_W - PAD.left - PAD.right; // 535
const PLOT_H = SVG_H - PAD.top - PAD.bottom; // 121

// Storyboard thumbnail size
const THUMB_W = 96;
const THUMB_H = 54;

// Bucket duration presets: [label, seconds]
const PRESETS: [string, number][] = [
  ['10s', 10], ['30s', 30], ['1m', 60], ['5m', 300],
  ['10m', 600], ['30m', 1800], ['1h', 3600],
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function fmtVal(val: number, metric: 'count' | 'speed'): string {
  return metric === 'count' ? String(Math.round(val)) : val.toFixed(1);
}

// ── Module-level state persistence ────────────────────────────────────────────
let saved: {
  videoId: string;
  bucketDuration: number;
  metric: 'count' | 'speed';
  k: number;
  selectedClasses: Set<string> | null;
} = { videoId: '', bucketDuration: 10, metric: 'count', k: 1, selectedClasses: null };

// ── Component ─────────────────────────────────────────────────────────────────
export default function TemporalSubTab() {
  const { tracklets, globalClips, videoMetadata, selectedVideoId } = useStore();
  const videoId = selectedVideoId ?? '';
  const streamUrl = selectedVideoId ? videoStreamUrl(selectedVideoId) : '';

  const [modalState, setModalState] = useState<{
    clip: GlobalClipMetadata; loopStart: number; loopEnd: number;
  } | null>(null);

  const openModal = (clip: GlobalClipMetadata) => {
    const duration = videoMetadata?.duration ?? Infinity;
    setModalState({
      clip,
      loopStart: clip.start_time,
      loopEnd: Math.min(duration, clip.start_time + bucketDuration),
    });
  };
  const sameVideo = saved.videoId === videoId;

  const allClasses = useMemo(
    () => [...new Set(tracklets.map(t => t.class_name))].sort(),
    [tracklets],
  );

  // ── Config state ─────────────────────────────────────────────────────────
  const [bucketDuration, setBucketDuration] = useState(() => sameVideo ? saved.bucketDuration : 10);
  const [customInput, setCustomInput] = useState(() => String(sameVideo ? saved.bucketDuration : 10));
  const [metric, setMetric] = useState<'count' | 'speed'>(() => sameVideo ? saved.metric : 'count');
  const [k, setK] = useState(() => sameVideo ? saved.k : 1);
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(() =>
    sameVideo && saved.selectedClasses !== null ? saved.selectedClasses : new Set(allClasses),
  );

  const classesInited = useRef(sameVideo && saved.selectedClasses !== null);
  useEffect(() => {
    if (classesInited.current || allClasses.length === 0) return;
    setSelectedClasses(new Set(allClasses));
    classesInited.current = true;
  }, [allClasses]);

  useEffect(() => { saved = { videoId, bucketDuration, metric, k, selectedClasses }; });

  const [hoveredBucket, setHoveredBucket] = useState<number | null>(null);

  // ── Time range ────────────────────────────────────────────────────────────
  const { minTime, maxTime } = useMemo(() => {
    if (videoMetadata && videoMetadata.duration > 0)
      return { minTime: 0, maxTime: videoMetadata.duration };
    if (tracklets.length === 0) return { minTime: 0, maxTime: 1 };
    return {
      minTime: Math.min(...tracklets.map(t => t.start_timestamp)),
      maxTime: Math.max(...tracklets.map(t => t.end_timestamp)),
    };
  }, [videoMetadata, tracklets]);

  const numBuckets = useMemo(
    () => Math.max(1, Math.ceil((maxTime - minTime) / bucketDuration)),
    [minTime, maxTime, bucketDuration],
  );

  // ── Activity data — single pass ───────────────────────────────────────────
  const activityData = useMemo(() => {
    const sums = new Map<string, number[]>();
    const counts = new Map<string, number[]>();
    for (const cls of selectedClasses) {
      sums.set(cls, Array(numBuckets).fill(0));
      counts.set(cls, Array(numBuckets).fill(0));
    }
    for (const t of tracklets) {
      if (!selectedClasses.has(t.class_name)) continue;
      const bStart = Math.max(0, Math.floor((t.start_timestamp - minTime) / bucketDuration));
      const bEnd = Math.min(numBuckets - 1, Math.floor((t.end_timestamp - minTime) / bucketDuration));
      for (let bi = bStart; bi <= bEnd; bi++) {
        sums.get(t.class_name)![bi] += t.avg_speed;
        counts.get(t.class_name)![bi]++;
      }
    }
    const result = new Map<string, number[]>();
    for (const cls of selectedClasses) {
      const s = sums.get(cls)!;
      const c = counts.get(cls)!;
      result.set(cls, metric === 'count' ? c : c.map((ci, bi) => ci > 0 ? s[bi] / ci : 0));
    }
    return result;
  }, [tracklets, selectedClasses, numBuckets, bucketDuration, minTime, metric]);

  const maxVal = useMemo(() => {
    let m = 0;
    for (const vals of activityData.values()) m = Math.max(m, ...vals);
    return Math.max(1, m);
  }, [activityData]);

  // ── Chart coordinate helpers ──────────────────────────────────────────────
  const bx = (bi: number) => PAD.left + (bi + 0.5) * (PLOT_W / numBuckets);
  const vy = (val: number) => PAD.top + PLOT_H - (val / maxVal) * PLOT_H;

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map(f => ({
    y: vy(f * maxVal),
    label: fmtVal(f * maxVal, metric),
  }));

  const labelStep = Math.max(1, Math.ceil(numBuckets / 12));
  const rotateLabels = numBuckets > 8;

  // ── Storyboard ────────────────────────────────────────────────────────────
  const storyboard = useMemo(() =>
    Array.from({ length: numBuckets }, (_, bi) => {
      const bStart = minTime + bi * bucketDuration;
      const bEnd = bStart + bucketDuration;
      const overlapping = globalClips.filter(c => c.start_time < bEnd && c.end_time > bStart);
      const reps = overlapping.filter(c => c.is_representative);
      return { bStart, bEnd, clips: reps.slice(0, k) };
    }),
  [globalClips, numBuckets, bucketDuration, minTime, k]);

  // ── Class filter helpers ──────────────────────────────────────────────────
  const toggleClass = (cls: string) =>
    setSelectedClasses(prev => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls); else next.add(cls);
      return next;
    });

  const toggleAll = () =>
    setSelectedClasses(selectedClasses.size === allClasses.length ? new Set() : new Set(allClasses));

  // ── Empty state ───────────────────────────────────────────────────────────
  if (!videoMetadata || tracklets.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">Load a video to see temporal summarizations.</p>
      </div>
    );
  }

  const orderedClasses = [...selectedClasses].sort();
  const tooltipLines = hoveredBucket !== null
    ? orderedClasses.map(cls => ({ cls, val: activityData.get(cls)?.[hoveredBucket] ?? 0 }))
    : [];

  // Legend: show up to 8 classes to avoid overflow
  const legendClasses = orderedClasses.slice(0, 8);
  // Legend fits on the right only when there's enough horizontal room; position it at top-right
  const legendX = SVG_W - PAD.right - 2;
  const legendStartY = PAD.top + 2;

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* ── Config panel ──────────────────────────────────────────────────── */}
      <div className="shrink-0 flex flex-col gap-2 px-3 py-2 border-b border-gray-700 bg-gray-900">

        {/* Row 1: bucket presets · custom input · Y-axis · k */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-400 shrink-0">Bucket</span>
          <div className="flex rounded overflow-hidden border border-gray-600">
            {PRESETS.map(([label, secs]) => (
              <button key={secs}
                onClick={() => { setBucketDuration(secs); setCustomInput(String(secs)); }}
                className={`px-2 py-0.5 text-[10px] transition-colors ${
                  bucketDuration === secs
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >{label}</button>
            ))}
          </div>
          <input
            type="number" min={1} value={customInput}
            onChange={e => {
              setCustomInput(e.target.value);
              const n = parseFloat(e.target.value);
              if (!isNaN(n) && n > 0) setBucketDuration(n);
            }}
            className="w-14 bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-white text-center focus:outline-none focus:border-blue-500"
            title="Custom bucket duration in seconds"
          />
          <span className="text-[10px] text-gray-600 shrink-0">s · {numBuckets} buckets</span>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-gray-400 shrink-0">Y-axis</span>
            <div className="flex rounded overflow-hidden border border-gray-600">
              {(['count', 'speed'] as const).map(m => (
                <button key={m} onClick={() => setMetric(m)}
                  className={`px-2.5 py-0.5 text-xs capitalize transition-colors ${
                    metric === m ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                  }`}
                >{m === 'count' ? 'Count' : 'Speed'}</button>
              ))}
            </div>

            <span className="text-xs text-gray-400 shrink-0">k</span>
            <input
              type="number" min={1} max={5} value={k}
              onChange={e => { const n = parseInt(e.target.value, 10); if (!isNaN(n) && n >= 1) setK(n); }}
              className="w-10 bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-white text-center focus:outline-none focus:border-blue-500"
              title="Max keyframes per storyboard bucket"
            />
          </div>
        </div>

        {/* Row 2: class filter badges */}
        <div className="flex flex-wrap items-center gap-1">
          <button onClick={toggleAll}
            className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 border border-gray-700 rounded transition-colors">
            {selectedClasses.size === allClasses.length ? 'none' : 'all'}
          </button>
          {allClasses.map(cls => (
            <button key={cls} onClick={() => toggleClass(cls)}
              className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors ${
                selectedClasses.has(cls)
                  ? 'text-white border-transparent'
                  : 'bg-gray-800 border-gray-600 text-gray-500 hover:text-gray-300'
              }`}
              style={selectedClasses.has(cls)
                ? { backgroundColor: getClassColorHex(cls), borderColor: getClassColorHex(cls) }
                : {}}
            >{cls}</button>
          ))}
        </div>
      </div>

      {/* ── Activity Graph ────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-0 border-b border-gray-700">
        <div className="shrink-0 flex items-center justify-between px-3 py-1">
          <span className="text-xs font-medium text-gray-300">Activity</span>
          <span className="text-[10px] text-gray-500">
            {metric === 'count' ? 'objects active per bucket' : 'avg speed per bucket (px/s)'}
          </span>
        </div>

        <div className="relative flex-1 min-h-0 px-1 pb-1">
          <svg
            viewBox={`0 0 ${SVG_W} ${SVG_H}`}
            className="w-full h-full"
          >
            {/* Y gridlines + labels */}
            {gridLines.map((g, i) => (
              <g key={i}>
                <line x1={PAD.left} x2={SVG_W - PAD.right} y1={g.y} y2={g.y}
                  stroke="#374151" strokeWidth={0.5} />
                <text x={PAD.left - 4} y={g.y} textAnchor="end" dominantBaseline="middle"
                  fill="#9CA3AF" fontSize={9}>{g.label}</text>
              </g>
            ))}

            {/* Axes */}
            <line x1={PAD.left} x2={SVG_W - PAD.right} y1={PAD.top + PLOT_H} y2={PAD.top + PLOT_H}
              stroke="#6B7280" strokeWidth={1} />
            <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + PLOT_H}
              stroke="#6B7280" strokeWidth={1} />

            {/* Y-axis label */}
            <text x={10} y={PAD.top + PLOT_H / 2}
              transform={`rotate(-90, 10, ${PAD.top + PLOT_H / 2})`}
              textAnchor="middle" fill="#6B7280" fontSize={8}>
              {metric === 'count' ? 'Count' : 'Speed (px/s)'}
            </text>

            {/* Per-class polylines + dots */}
            {orderedClasses.map(cls => {
              const vals = activityData.get(cls) ?? Array(numBuckets).fill(0);
              const pts = vals.map((v, bi) => `${bx(bi).toFixed(1)},${vy(v).toFixed(1)}`).join(' ');
              const color = getClassColorHex(cls);
              return (
                <g key={cls}>
                  <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5}
                    strokeLinejoin="round" strokeLinecap="round" />
                  {vals.map((v, bi) => (
                    <circle key={bi} cx={bx(bi)} cy={vy(v)}
                      r={hoveredBucket === bi ? 3.5 : 2.5}
                      fill={color}
                      stroke={hoveredBucket === bi ? 'white' : 'none'}
                      strokeWidth={0.8}
                    />
                  ))}
                </g>
              );
            })}

            {/* Hover interaction rects — one per bucket */}
            {Array.from({ length: numBuckets }, (_, bi) => {
              const bw = PLOT_W / numBuckets;
              return (
                <rect key={bi}
                  x={PAD.left + bi * bw} y={PAD.top}
                  width={bw} height={PLOT_H}
                  fill={hoveredBucket === bi ? 'rgba(255,255,255,0.04)' : 'transparent'}
                  onMouseEnter={() => setHoveredBucket(bi)}
                  onMouseLeave={() => setHoveredBucket(null)}
                  style={{ cursor: 'default' }}
                />
              );
            })}

            {/* X-axis labels */}
            {Array.from({ length: numBuckets }, (_, bi) => {
              if (bi % labelStep !== 0) return null;
              const bStart = minTime + bi * bucketDuration;
              const cx = bx(bi);
              const cy = PAD.top + PLOT_H + 12;
              return (
                <text key={bi}
                  x={cx} y={cy}
                  textAnchor={rotateLabels ? 'end' : 'middle'}
                  fill="#9CA3AF" fontSize={8}
                  transform={rotateLabels ? `rotate(-35, ${cx}, ${cy})` : undefined}
                >{formatTime(bStart)}</text>
              );
            })}

            {/* Legend — top-right, up to 8 classes */}
            {legendClasses.map((cls, i) => {
              const color = getClassColorHex(cls);
              const ly = legendStartY + i * 14;
              return (
                <g key={cls} transform={`translate(${legendX}, ${ly})`}>
                  <line x1={-22} x2={-8} y1={5} y2={5} stroke={color} strokeWidth={1.5} />
                  <circle cx={-15} cy={5} r={2} fill={color} />
                  <text x={0} y={5} textAnchor="end" dominantBaseline="middle"
                    fill="#D1D5DB" fontSize={8}>{cls}</text>
                </g>
              );
            })}
          </svg>

          {/* Tooltip */}
          {hoveredBucket !== null && tooltipLines.length > 0 && (
            <div
              className="absolute top-0 pointer-events-none z-10"
              style={{
                left: `${Math.min(82, Math.max(8, (bx(hoveredBucket) / SVG_W) * 100 - 6))}%`,
                transform: 'translateX(-50%)',
              }}
            >
              <div className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-[10px] shadow-lg min-w-22.5">
                <p className="text-gray-400 mb-1">
                  {formatTime(minTime + hoveredBucket * bucketDuration)}–
                  {formatTime(Math.min(maxTime, minTime + (hoveredBucket + 1) * bucketDuration))}
                </p>
                {tooltipLines.map(({ cls, val }) => (
                  <div key={cls} className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: getClassColorHex(cls) }} />
                    <span className="text-gray-300">{cls}: {fmtVal(val, metric)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Keyframe Storyboard ───────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
        <div className="shrink-0 px-3 py-1 flex items-center justify-between border-b border-gray-800">
          <span className="text-xs font-medium text-gray-300">Keyframe Storyboard</span>
          <span className="text-[10px] text-gray-500">
            FPS representatives · up to {k} per bucket
          </span>
        </div>

        {globalClips.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-gray-600 text-xs text-center px-4">
              No global clips found.<br />Re-index this video to generate clips.
            </p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-2">
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(108px, 1fr))' }}
            >
              {storyboard.map(({ bStart, bEnd, clips }, bi) => (
                <div key={bi} className="flex flex-col gap-1">
                  {/* Bucket time label */}
                  <p className="text-[9px] text-gray-500 text-center leading-tight">
                    {formatTime(bStart)}–{formatTime(Math.min(maxTime, bEnd))}
                  </p>

                  {/* Thumbnails or placeholder */}
                  {clips.length > 0 ? (
                    clips.map(clip => (
                      <div key={clip.clip_id}
                        className="rounded overflow-hidden border border-blue-800 cursor-pointer hover:border-blue-400 transition-colors"
                        style={{ width: '100%', aspectRatio: `${THUMB_W}/${THUMB_H}` }}
                        title={`${formatTime(clip.start_time)}–${formatTime(clip.end_time)} · click to play`}
                        onClick={() => openModal(clip)}
                      >
                        {clip.thumbnail_base64 ? (
                          <img
                            src={`data:image/jpeg;base64,${clip.thumbnail_base64}`}
                            alt={`Clip ${clip.clip_index}`}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-gray-800">
                            <span className="text-[9px] text-gray-600">no img</span>
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div
                      className="rounded border border-gray-800 flex items-center justify-center bg-gray-900"
                      style={{ width: '100%', aspectRatio: `${THUMB_W}/${THUMB_H}` }}
                    >
                      <span className="text-[10px] text-gray-700">–</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Loop modal ────────────────────────────────────────────────────── */}
      {modalState && (
        <LoopModal
          clip={modalState.clip}
          streamUrl={streamUrl}
          loopStart={modalState.loopStart}
          loopEnd={modalState.loopEnd}
          onClose={() => setModalState(null)}
        />
      )}

    </div>
  );
}
