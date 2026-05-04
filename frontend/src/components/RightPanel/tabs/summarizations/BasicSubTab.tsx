import { useRef, useState, useMemo } from 'react';
import { useStore } from '../../../../stores/useStore';
import { getClassColorHex, getClusterColorHex } from '../../../../lib/colors';

// ── SVG layout constants ─────────────────────────────────────────────────────
const SVG_W = 600;
const PAD = { top: 16, right: 20, bottom: 44, left: 56 };
const PLOT_W = SVG_W - PAD.left - PAD.right; // 524 — used by histogram/labeled charts

// HorizBarChart uses wider right margin so value labels clear the scrollbar
const HBAR_PAD_RIGHT = 70;

// Donut — 300×300 viewBox
const DONUT_CX = 150;
const DONUT_CY = 150;
const DONUT_R_OUTER = 120;
const DONUT_R_INNER = 72;

const HIST_ACCENT = 'rgb(59,130,246)';

// Fixed duration bins
const DURATION_BREAKPOINTS = [5, 10, 15, 20, 25, 30];
const DURATION_LABELS = ['0–5s', '5–10s', '10–15s', '15–20s', '20–25s', '25–30s', '>30s'];

// Fixed speed bins (px/s in increments of 100)
const SPEED_BREAKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
const SPEED_LABELS = [
  '0–100', '100–200', '200–300', '300–400', '400–500',
  '500–600', '600–700', '700–800', '800–900', '900–1000', '>1000',
];

// ── Pure helper functions ─────────────────────────────────────────────────────
function formatDuration(secs: number): string {
  if (secs >= 3600) {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return `${h}h ${m}m`;
  }
  if (secs >= 60) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}m ${s}s`;
  }
  return `${secs.toFixed(1)}s`;
}

function buildFixedBins(
  values: number[],
  breakpoints: number[],
  labels: string[],
): { labels: string[]; counts: number[] } {
  const counts = Array(labels.length).fill(0) as number[];
  for (const v of values) {
    const idx = breakpoints.findIndex((bp) => v < bp);
    counts[idx === -1 ? labels.length - 1 : idx]++;
  }
  return { labels, counts };
}

function describeDonutArc(
  cx: number,
  cy: number,
  rOuter: number,
  rInner: number,
  startAngle: number,
  endAngle: number,
): string {
  let sweep = endAngle - startAngle;
  if (sweep >= 2 * Math.PI) sweep = 2 * Math.PI - 0.0001;
  const largeArc = sweep > Math.PI ? 1 : 0;

  const ox1 = cx + rOuter * Math.cos(startAngle);
  const oy1 = cy + rOuter * Math.sin(startAngle);
  const ox2 = cx + rOuter * Math.cos(startAngle + sweep);
  const oy2 = cy + rOuter * Math.sin(startAngle + sweep);
  const ix1 = cx + rInner * Math.cos(startAngle + sweep);
  const iy1 = cy + rInner * Math.sin(startAngle + sweep);
  const ix2 = cx + rInner * Math.cos(startAngle);
  const iy2 = cy + rInner * Math.sin(startAngle);

  return [
    `M ${ox1.toFixed(2)} ${oy1.toFixed(2)}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${ox2.toFixed(2)} ${oy2.toFixed(2)}`,
    `L ${ix1.toFixed(2)} ${iy1.toFixed(2)}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${ix2.toFixed(2)} ${iy2.toFixed(2)}`,
    'Z',
  ].join(' ');
}

function fmtVal(n: number, decimals = 1): string {
  return decimals === 0 ? String(Math.round(n)) : n.toFixed(decimals);
}

// ── Shared tooltip types & component ─────────────────────────────────────────
type TooltipState = { x: number; y: number; content: string } | null;

function ChartTooltip({ x, y, content }: { x: number; y: number; content: string }) {
  return (
    <div
      className="pointer-events-none absolute z-10 rounded bg-gray-700 border border-gray-600 px-2 py-1 text-xs text-white shadow-lg whitespace-nowrap"
      style={{ left: x + 12, top: y - 28 }}
    >
      {content}
    </div>
  );
}

// ── Sub-renderers ─────────────────────────────────────────────────────────────

function DonutChart({
  arcs,
  totalTracklets,
}: {
  arcs: { cls: string; fraction: number; startAngle: number; endAngle: number }[];
  totalTracklets: number;
}) {
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const handleMove = (e: React.MouseEvent, content: string) => {
    const r = wrapperRef.current!.getBoundingClientRect();
    setTooltip({ x: e.clientX - r.left, y: e.clientY - r.top, content });
  };

  return (
    <div className="flex items-center gap-4 pt-2">
      <div className="relative w-52 shrink-0" ref={wrapperRef}>
        <svg viewBox="0 0 300 300" className="w-full">
          {arcs.map((arc) => (
            <path
              key={arc.cls}
              d={describeDonutArc(
                DONUT_CX, DONUT_CY,
                DONUT_R_OUTER, DONUT_R_INNER,
                arc.startAngle, arc.endAngle,
              )}
              fill={getClassColorHex(arc.cls)}
              className="cursor-pointer"
              onMouseMove={(e) =>
                handleMove(e, `${arc.cls}: ${(arc.fraction * 100).toFixed(1)}%`)
              }
              onMouseLeave={() => setTooltip(null)}
            />
          ))}
          <text x={DONUT_CX} y={DONUT_CY - 8}
            textAnchor="middle" fontSize="24" fontWeight="bold" fill="white"
            style={{ pointerEvents: 'none' }}>
            {totalTracklets.toLocaleString()}
          </text>
          <text x={DONUT_CX} y={DONUT_CY + 14}
            textAnchor="middle" fontSize="11" fill="rgb(156,163,175)"
            style={{ pointerEvents: 'none' }}>
            tracklets
          </text>
        </svg>
        {tooltip && <ChartTooltip {...tooltip} />}
      </div>
      <ul className="flex-1 space-y-1.5">
        {arcs.map((arc) => (
          <li key={arc.cls} className="flex items-center gap-2 text-[11px] text-gray-300">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ background: getClassColorHex(arc.cls) }}
            />
            <span className="capitalize">{arc.cls}</span>
            <span className="ml-auto text-gray-400">
              {(arc.fraction * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function HorizBarChart({
  items,
  valueLabel,
  formatValue,
}: {
  items: { id: number; label: string; value: number; color: string }[];
  valueLabel: string;
  formatValue: (v: number) => string;
}) {
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  if (items.length === 0) {
    return <p className="text-xs text-gray-600 px-3 pb-3">No data.</p>;
  }

  const hbarPlotW = SVG_W - PAD.left - HBAR_PAD_RIGHT; // 474
  const total = items.reduce((s, d) => s + d.value, 0);
  const maxVal = Math.max(...items.map((d) => d.value));
  const plotH = Math.max(60, items.length * 22);
  const svgH = PAD.top + plotH + PAD.bottom;
  const barSlot = plotH / items.length;
  const barH = Math.max(8, barSlot - 4);

  const handleMove = (e: React.MouseEvent, d: typeof items[0]) => {
    const r = wrapperRef.current!.getBoundingClientRect();
    const pct = total > 0 ? (d.value / total) * 100 : 0;
    setTooltip({
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      content: `${d.label}: ${formatValue(d.value)} (${pct.toFixed(1)}%)`,
    });
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <svg viewBox={`0 0 ${SVG_W} ${svgH}`} className="w-full">
        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const x = PAD.left + t * hbarPlotW;
          return (
            <line
              key={t}
              x1={x} y1={PAD.top}
              x2={x} y2={PAD.top + plotH}
              stroke="rgb(55,65,81)" strokeWidth="1"
            />
          );
        })}
        {/* bars */}
        {items.map((d, i) => {
          const barW = maxVal > 0 ? (d.value / maxVal) * hbarPlotW : 0;
          const y = PAD.top + i * barSlot + (barSlot - barH) / 2;
          return (
            <g key={d.id}>
              <rect
                x={PAD.left} y={y}
                width={Math.max(2, barW)} height={barH}
                fill={d.color} rx="2"
                className="cursor-pointer"
                onMouseMove={(e) => handleMove(e, d)}
                onMouseLeave={() => setTooltip(null)}
              />
              <text
                x={PAD.left - 4} y={y + barH / 2 + 4}
                textAnchor="end" fontSize="10" fill="rgb(156,163,175)"
                style={{ pointerEvents: 'none' }}
              >
                {d.label}
              </text>
              <text
                x={PAD.left + barW + 5} y={y + barH / 2 + 4}
                textAnchor="start" fontSize="10" fill="rgb(209,213,219)"
                style={{ pointerEvents: 'none' }}
              >
                {formatValue(d.value)}
              </text>
            </g>
          );
        })}
        <text
          x={PAD.left + hbarPlotW / 2} y={svgH - 6}
          textAnchor="middle" fontSize="10" fill="rgb(107,114,128)"
          style={{ pointerEvents: 'none' }}
        >
          {valueLabel}
        </text>
      </svg>
      {tooltip && <ChartTooltip {...tooltip} />}
    </div>
  );
}

function LabeledBarChart({
  labels,
  counts,
}: {
  labels: string[];
  counts: number[];
}) {
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const LBAR_H = 160;
  const LBAR_PLOT_H = LBAR_H - PAD.top - PAD.bottom; // 100
  const total = counts.reduce((s, c) => s + c, 0);
  const maxCount = Math.max(...counts, 1);
  const slotW = PLOT_W / labels.length;
  const barW = Math.max(1, slotW - 6);
  const gridYs = [0, 0.33, 0.67, 1].map((t) => PAD.top + LBAR_PLOT_H * (1 - t));

  const handleMove = (e: React.MouseEvent, label: string, count: number) => {
    const r = wrapperRef.current!.getBoundingClientRect();
    const pct = total > 0 ? (count / total) * 100 : 0;
    setTooltip({
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      content: `${label}: ${count} (${pct.toFixed(1)}%)`,
    });
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <svg viewBox={`0 0 ${SVG_W} ${LBAR_H}`} className="w-full">
        {gridYs.map((y, i) => (
          <line key={i} x1={PAD.left} y1={y} x2={PAD.left + PLOT_W} y2={y}
            stroke="rgb(55,65,81)" strokeWidth="1" />
        ))}
        {gridYs.map((y, i) => {
          const val = Math.round(maxCount * [0, 0.33, 0.67, 1][i]);
          return (
            <text key={i} x={PAD.left - 5} y={y + 4}
              textAnchor="end" fontSize="9" fill="rgb(107,114,128)"
              style={{ pointerEvents: 'none' }}>{val}</text>
          );
        })}
        <text
          x={12} y={PAD.top + LBAR_PLOT_H / 2}
          textAnchor="middle" fontSize="10" fill="rgb(107,114,128)"
          transform={`rotate(-90, 12, ${PAD.top + LBAR_PLOT_H / 2})`}
          style={{ pointerEvents: 'none' }}
        >
          Count
        </text>
        {counts.map((c, i) => {
          const bH = (c / maxCount) * LBAR_PLOT_H;
          const x = PAD.left + i * slotW + (slotW - barW) / 2;
          const y = PAD.top + LBAR_PLOT_H - bH;
          return (
            <rect key={i} x={x} y={y} width={barW} height={bH}
              fill={HIST_ACCENT} rx="2"
              className="cursor-pointer"
              onMouseMove={(e) => handleMove(e, labels[i], c)}
              onMouseLeave={() => setTooltip(null)}
            />
          );
        })}
        {labels.map((lbl, i) => {
          const x = PAD.left + i * slotW + slotW / 2;
          return (
            <text key={i} x={x} y={PAD.top + LBAR_PLOT_H + 14}
              textAnchor="middle" fontSize="9" fill="rgb(107,114,128)"
              style={{ pointerEvents: 'none' }}>{lbl}</text>
          );
        })}
      </svg>
      {tooltip && <ChartTooltip {...tooltip} />}
    </div>
  );
}

function StackedClusterBar({
  items,
}: {
  items: { id: number; label: string; value: number; color: string }[];
}) {
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  if (items.length === 0) {
    return <p className="text-xs text-gray-600 px-3 pb-3">No data.</p>;
  }

  const BAR_MARGIN = 16;
  const barPlotW = SVG_W - BAR_MARGIN * 2; // 568
  const BAR_Y = 8;
  const BAR_H = 28;
  const LEG_START = BAR_Y + BAR_H + 14;
  const LEG_COLS = 3;
  const LEG_ROW_H = 16;
  const legendRows = Math.ceil(items.length / LEG_COLS);
  const svgH = LEG_START + legendRows * LEG_ROW_H + 10;
  const colW = barPlotW / LEG_COLS;

  const total = items.reduce((s, d) => s + d.value, 0);
  const widths = items.map((d) => Math.max(1, total > 0 ? (d.value / total) * barPlotW : 0));
  const segments = items.map((d, i) => ({
    ...d,
    x: BAR_MARGIN + widths.slice(0, i).reduce((s, w) => s + w, 0),
    w: widths[i],
  }));

  const handleMove = (e: React.MouseEvent, d: typeof items[0]) => {
    const r = wrapperRef.current!.getBoundingClientRect();
    const pct = total > 0 ? (d.value / total) * 100 : 0;
    setTooltip({
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      content: `${d.label}: ${d.value} (${pct.toFixed(1)}%)`,
    });
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <svg viewBox={`0 0 ${SVG_W} ${svgH}`} className="w-full">
        {segments.map((seg) => (
          <rect key={seg.id} x={seg.x} y={BAR_Y} width={seg.w} height={BAR_H}
            fill={seg.color}
            className="cursor-pointer"
            onMouseMove={(e) => handleMove(e, seg)}
            onMouseLeave={() => setTooltip(null)}
          />
        ))}
        {items.map((d, i) => {
          const col = i % LEG_COLS;
          const row = Math.floor(i / LEG_COLS);
          const x = BAR_MARGIN + col * colW;
          const y = LEG_START + row * LEG_ROW_H;
          return (
            <g key={d.id} style={{ pointerEvents: 'none' }}>
              <rect x={x} y={y - 8} width={8} height={8} fill={d.color} rx="1" />
              <text x={x + 11} y={y} fontSize="9" fill="rgb(156,163,175)">
                {d.label}: {d.value}
              </text>
            </g>
          );
        })}
      </svg>
      {tooltip && <ChartTooltip {...tooltip} />}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function BasicSubTab() {
  const { videoMetadata, tracklets, globalClips } = useStore();

  // ── All useMemo hooks (unconditional) ──
  const numClasses = useMemo(
    () => Object.keys(videoMetadata?.class_distribution ?? {}).length,
    [videoMetadata],
  );

  const numClusters = useMemo(
    () => (videoMetadata?.cluster_stats ?? []).filter((c) => c.cluster_id >= 0).length,
    [videoMetadata],
  );

  const donutArcs = useMemo(() => {
    const dist = videoMetadata?.class_distribution ?? {};
    const entries = Object.entries(dist);
    const total = entries.reduce((s, [, v]) => s + v, 0);
    if (total === 0) return [];
    let angleAcc = -Math.PI / 2;
    return entries.map(([cls, pct]) => {
      const fraction = pct / total;
      const startAngle = angleAcc;
      angleAcc += fraction * 2 * Math.PI;
      return { cls, fraction, startAngle, endAngle: angleAcc };
    });
  }, [videoMetadata]);

  const clusterSizesSorted = useMemo(
    () =>
      [...(videoMetadata?.cluster_stats ?? [])].sort(
        (a, b) => b.member_count - a.member_count,
      ),
    [videoMetadata],
  );

  const clusterSpeedsSorted = useMemo(
    () =>
      [...(videoMetadata?.cluster_stats ?? [])].sort(
        (a, b) => b.avg_speed - a.avg_speed,
      ),
    [videoMetadata],
  );

  const speedBins = useMemo(() => {
    const speeds = tracklets.map((t) => t.avg_speed).filter((v) => v >= 0);
    if (speeds.length === 0) return null;
    return buildFixedBins(speeds, SPEED_BREAKPOINTS, SPEED_LABELS);
  }, [tracklets]);

  const durationBins = useMemo(() => {
    const durs = tracklets.map((t) => t.duration).filter((v) => v > 0);
    if (durs.length === 0) return null;
    return buildFixedBins(durs, DURATION_BREAKPOINTS, DURATION_LABELS);
  }, [tracklets]);

  const globalClusterSizesSorted = useMemo(() => {
    const counts = new Map<number, number>();
    for (const clip of globalClips) {
      counts.set(clip.cluster_id, (counts.get(clip.cluster_id) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([cluster_id, member_count]) => ({ cluster_id, member_count }))
      .sort((a, b) => b.member_count - a.member_count);
  }, [globalClips]);

  const numGlobalClusters = useMemo(
    () => globalClusterSizesSorted.filter((c) => c.cluster_id >= 0).length,
    [globalClusterSizesSorted],
  );

  // ── Empty state (after all hooks) ──
  if (!videoMetadata || tracklets.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">Load a video to see basic summarizations.</p>
      </div>
    );
  }

  const clusterSizeItems = clusterSizesSorted.map((c) => ({
    id: c.cluster_id,
    label: c.cluster_id < 0 ? 'Noise' : `C${c.cluster_id}`,
    value: c.member_count,
    color: getClusterColorHex(c.cluster_id),
  }));

  const clusterSpeedItems = clusterSpeedsSorted.map((c) => ({
    id: c.cluster_id,
    label: c.cluster_id < 0 ? 'Noise' : `C${c.cluster_id}`,
    value: c.avg_speed,
    color: getClusterColorHex(c.cluster_id),
  }));

  const globalClusterSizeItems = globalClusterSizesSorted.map((c) => ({
    id: c.cluster_id,
    label: c.cluster_id < 0 ? 'Noise' : `C${c.cluster_id}`,
    value: c.member_count,
    color: getClusterColorHex(c.cluster_id),
  }));

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-gray-900">

      {/* ── 1. Stat cards ── */}
      <div className="px-3 pt-3 pb-3 border-b border-gray-800">
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'Total Tracklets', value: videoMetadata.total_tracklets.toLocaleString() },
            { label: 'Total Clips', value: globalClips.length.toLocaleString() },
            { label: 'Object Classes', value: numClasses },
            { label: 'Local Clusters', value: numClusters },
            { label: 'Global Clusters', value: numGlobalClusters },
            { label: 'Duration', value: formatDuration(videoMetadata.duration) },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-800 rounded-lg p-3 flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
              <span className="text-xl font-bold text-white">{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── 2. Class distribution donut ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Class Distribution</h3>
        <p className="text-[10px] text-gray-500">Percentage of tracklets per object class</p>
      </div>
      <div className="px-3 pb-3 border-b border-gray-800">
        {donutArcs.length === 0 ? (
          <p className="text-xs text-gray-600 py-3">No class data.</p>
        ) : (
          <DonutChart arcs={donutArcs} totalTracklets={videoMetadata.total_tracklets} />
        )}
      </div>

      {/* ── 3. Cluster sizes (local) ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Cluster Sizes</h3>
        <p className="text-[10px] text-gray-500">Number of tracklets per cluster (local view)</p>
      </div>
      <div className="px-1 pb-3 border-b border-gray-800">
        <HorizBarChart
          items={clusterSizeItems}
          valueLabel="Tracklet count"
          formatValue={(v) => fmtVal(v, 0)}
        />
      </div>

      {/* ── 4. Cluster average speed (local) ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Cluster Average Speed</h3>
        <p className="text-[10px] text-gray-500">Mean tracklet speed per cluster (local view)</p>
      </div>
      <div className="px-1 pb-3 border-b border-gray-800">
        <HorizBarChart
          items={clusterSpeedItems}
          valueLabel="avg speed (px/s)"
          formatValue={(v) => `${v.toFixed(1)}`}
        />
      </div>

      {/* ── 5. Speed distribution (fixed bins) ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Speed Distribution</h3>
        <p className="text-[10px] text-gray-500">Tracklet average speed in px/s</p>
      </div>
      <div className="px-1 pb-3 border-b border-gray-800">
        {speedBins ? (
          <LabeledBarChart labels={speedBins.labels} counts={speedBins.counts} />
        ) : (
          <p className="text-xs text-gray-600 px-3 py-3">Insufficient data.</p>
        )}
      </div>

      {/* ── 6. Tracklet duration (fixed bins) ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Tracklet Duration</h3>
        <p className="text-[10px] text-gray-500">Distribution of tracklet duration</p>
      </div>
      <div className="px-1 pb-3 border-b border-gray-800">
        {durationBins ? (
          <LabeledBarChart labels={durationBins.labels} counts={durationBins.counts} />
        ) : (
          <p className="text-xs text-gray-600 px-3 py-3">Insufficient data.</p>
        )}
      </div>

      {/* ── 7. Global cluster sizes ── */}
      <div className="shrink-0 px-3 pt-3 pb-1 border-b border-gray-800">
        <h3 className="text-xs font-medium text-gray-300">Global Cluster Sizes</h3>
        <p className="text-[10px] text-gray-500">Proportional clip count per cluster (global view)</p>
      </div>
      <div className="px-1 pb-4">
        {globalClips.length === 0 ? (
          <p className="text-xs text-gray-600 px-3 py-3">No global clips loaded.</p>
        ) : (
          <StackedClusterBar items={globalClusterSizeItems} />
        )}
      </div>

    </div>
  );
}
