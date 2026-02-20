import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { OrthographicView } from 'deck.gl';
import { ScatterplotLayer } from '@deck.gl/layers';
import { Lasso, MousePointer2, Square } from 'lucide-react';
import { useStore } from '../../stores/useStore';
import { getClassColor, getClusterColor, getClassColorHex, getClusterColorHex, timeToColor } from '../../lib/colors';
import { pointInPolygon } from '../../lib/utils';
import type { TrackletMetadata } from '../../types/index';

interface ViewState {
  target: [number, number, number];
  zoom: number;
}

interface HoverInfo {
  x: number;
  y: number;
  tracklet: TrackletMetadata;
}

export default function EmbeddingsPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [panelSize, setPanelSize] = useState({ width: 1, height: 1 });

  // Controlled viewState — allows programmatic zoom updates from SVG wheel handler
  const [viewState, setViewState] = useState<ViewState>({ target: [0, 0, 0], zoom: 1 });
  // Ref for fast access inside mouse/wheel event handlers (no re-render needed)
  const viewStateRef = useRef<ViewState>({ target: [0, 0, 0], zoom: 1 });

  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Selection shapes stored in WORLD coordinates so they survive zoom changes
  const [lassoPointsWorld, setLassoPointsWorld] = useState<[number, number][]>([]);
  const [rectStartWorld, setRectStartWorld] = useState<[number, number] | null>(null);
  const [rectCurrentWorld, setRectCurrentWorld] = useState<[number, number] | null>(null);

  const {
    tracklets,
    selectionMode,
    setSelectionMode,
    selectedTrackletIds,
    setSelectedTrackletIds,
    colorMode,
    setColorMode,
    highlightedClusterId,
    highlightedTrackletId,
  } = useStore();

  // Legend items derived from tracklets
  const classItems = useMemo(
    () => [...new Set(tracklets.map(t => t.class_name))].sort(),
    [tracklets],
  );

  const clusterItems = useMemo(
    () => [...new Set(tracklets.map(t => t.cluster_id))].sort((a, b) => a - b),
    [tracklets],
  );

  const { minTime, maxTime } = useMemo(() => {
    if (tracklets.length === 0) return { minTime: 0, maxTime: 1 };
    const times = tracklets.map(t => t.start_timestamp);
    return { minTime: Math.min(...times), maxTime: Math.max(...times) };
  }, [tracklets]);

  // Track panel dimensions
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setPanelSize({ width: width || 1, height: height || 1 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit view to tracklets whenever tracklets or panel size change
  useEffect(() => {
    if (tracklets.length === 0 || panelSize.width <= 1) return;
    const xs = tracklets.map(t => t.umap_x);
    const ys = tracklets.map(t => t.umap_y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min(panelSize.width / rangeX, panelSize.height / rangeY) * 0.85;
    const vs: ViewState = {
      target: [(minX + maxX) / 2, (minY + maxY) / 2, 0],
      zoom: Math.log2(scale),
    };
    viewStateRef.current = vs;
    setViewState(vs);
  }, [tracklets, panelSize]);

  // Convert pixel coords → UMAP world coords using current viewState ref (fast, no re-render)
  const pixelToWorld = useCallback(
    (px: number, py: number): [number, number] => {
      const vs = viewStateRef.current;
      const scale = Math.pow(2, vs.zoom);
      const wx = (px - panelSize.width / 2) / scale + vs.target[0];
      const wy = (py - panelSize.height / 2) / scale + vs.target[1];
      return [wx, wy];
    },
    [panelSize],
  );

  // Convert UMAP world coords → pixel coords using reactive viewState (triggers re-render for SVG)
  const worldToPixel = useCallback(
    (wx: number, wy: number): [number, number] => {
      const scale = Math.pow(2, viewState.zoom);
      return [
        (wx - viewState.target[0]) * scale + panelSize.width / 2,
        (wy - viewState.target[1]) * scale + panelSize.height / 2,
      ];
    },
    [panelSize, viewState],
  );

  // deck.gl layer
  const layers = useMemo(() => {
    if (tracklets.length === 0) return [];
    return [
      new ScatterplotLayer<TrackletMetadata>({
        id: 'tracklets',
        data: tracklets,
        getPosition: (d: TrackletMetadata) => [d.umap_x, d.umap_y, 0],
        getFillColor: (d: TrackletMetadata): [number, number, number, number] => {
          const [r, g, b] =
            colorMode === 'class'   ? getClassColor(d.class_name) :
            colorMode === 'cluster' ? getClusterColor(d.cluster_id) :
                                      timeToColor(d.start_timestamp, minTime, maxTime);
          const hasSelection = selectedTrackletIds.size > 0;
          const isSelected = selectedTrackletIds.has(d.tracklet_id);
          const isHighlightedCluster =
            highlightedClusterId !== null && d.cluster_id === highlightedClusterId;
          const isHighlightedTracklet = d.tracklet_id === highlightedTrackletId;
          let alpha = 220;
          if (hasSelection && !isSelected) alpha = Math.min(alpha, 40);
          if (isHighlightedCluster || isHighlightedTracklet) alpha = 255;
          return [r, g, b, alpha];
        },
        getRadius: (d: TrackletMetadata): number => {
          if (d.tracklet_id === highlightedTrackletId) return 8;
          if (highlightedClusterId !== null && d.cluster_id === highlightedClusterId) return 6;
          return 4;
        },
        radiusUnits: 'pixels',
        pickable: selectionMode === 'none',
        updateTriggers: {
          getFillColor: [colorMode, selectedTrackletIds, highlightedClusterId, highlightedTrackletId, minTime, maxTime],
          getRadius: [highlightedTrackletId, highlightedClusterId],
        },
      }),
    ];
  }, [
    tracklets,
    colorMode,
    selectedTrackletIds,
    highlightedClusterId,
    highlightedTrackletId,
    selectionMode,
    minTime,
    maxTime,
  ]);

  // deck.gl hover handler
  const handleDeckHover = useCallback(
    (info: { object?: unknown; x: number; y: number }) => {
      const obj = info.object as TrackletMetadata | null | undefined;
      if (obj && obj.tracklet_id) {
        setHoverInfo({ x: info.x, y: info.y, tracklet: obj });
      } else {
        setHoverInfo(null);
      }
    },
    [],
  );

  // Keep controlled viewState in sync when deck.gl pans/zooms normally
  const handleViewStateChange = useCallback(({ viewState: vs }: { viewState: unknown }) => {
    viewStateRef.current = vs as ViewState;
    setViewState(vs as ViewState);
  }, []);

  // Legend click handlers — toggle selection for all tracklets matching a class or cluster
  const handleClassLegendClick = useCallback(
    (className: string) => {
      const matching = new Set(
        tracklets.filter(t => t.class_name === className).map(t => t.tracklet_id),
      );
      const allSelected = matching.size > 0 && [...matching].every(id => selectedTrackletIds.has(id));
      if (allSelected) {
        const next = new Set(selectedTrackletIds);
        matching.forEach(id => next.delete(id));
        setSelectedTrackletIds(next);
      } else {
        setSelectedTrackletIds(new Set([...selectedTrackletIds, ...matching]));
      }
    },
    [tracklets, selectedTrackletIds, setSelectedTrackletIds],
  );

  const handleClusterLegendClick = useCallback(
    (clusterId: number) => {
      const matching = new Set(
        tracklets.filter(t => t.cluster_id === clusterId).map(t => t.tracklet_id),
      );
      const allSelected = matching.size > 0 && [...matching].every(id => selectedTrackletIds.has(id));
      if (allSelected) {
        const next = new Set(selectedTrackletIds);
        matching.forEach(id => next.delete(id));
        setSelectedTrackletIds(next);
      } else {
        setSelectedTrackletIds(new Set([...selectedTrackletIds, ...matching]));
      }
    },
    [tracklets, selectedTrackletIds, setSelectedTrackletIds],
  );

  // SVG overlay mouse handlers — store positions in world coords
  const getSVGCoords = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onSVGMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    e.preventDefault();
    const pos = getSVGCoords(e);
    const [wx, wy] = pixelToWorld(pos.x, pos.y);
    if (selectionMode === 'lasso') {
      setLassoPointsWorld([[wx, wy]]);
      setIsDrawing(true);
    } else if (selectionMode === 'rect') {
      setRectStartWorld([wx, wy]);
      setRectCurrentWorld([wx, wy]);
      setIsDrawing(true);
    }
  };

  const onSVGMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDrawing) return;
    const pos = getSVGCoords(e);
    const [wx, wy] = pixelToWorld(pos.x, pos.y);
    if (selectionMode === 'lasso') {
      setLassoPointsWorld(prev => [...prev, [wx, wy]]);
    } else if (selectionMode === 'rect') {
      setRectCurrentWorld([wx, wy]);
    }
  };

  const finaliseSelection = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    if (selectionMode === 'lasso' && lassoPointsWorld.length > 2) {
      const worldPoly = lassoPointsWorld.map(([wx, wy]) => ({ x: wx, y: wy }));
      const selected = tracklets.filter(t =>
        pointInPolygon(t.umap_x, t.umap_y, worldPoly),
      );
      setSelectedTrackletIds(new Set(selected.map(t => t.tracklet_id)));
    } else if (selectionMode === 'rect' && rectStartWorld && rectCurrentWorld) {
      const [wx1, wy1] = rectStartWorld;
      const [wx2, wy2] = rectCurrentWorld;
      const minX = Math.min(wx1, wx2);
      const maxX = Math.max(wx1, wx2);
      const minY = Math.min(wy1, wy2);
      const maxY = Math.max(wy1, wy2);
      const selected = tracklets.filter(
        t => t.umap_x >= minX && t.umap_x <= maxX && t.umap_y >= minY && t.umap_y <= maxY,
      );
      setSelectedTrackletIds(new Set(selected.map(t => t.tracklet_id)));
    }

    setLassoPointsWorld([]);
    setRectStartWorld(null);
    setRectCurrentWorld(null);
  };

  // Zoom-to-cursor handler for the SVG overlay
  const onSVGWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const svgRect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - svgRect.left;
    const py = e.clientY - svgRect.top;
    // World point under cursor BEFORE zoom
    const [wx, wy] = pixelToWorld(px, py);
    const vs = viewStateRef.current;
    const ZOOM_SPEED = 0.001;
    const delta = e.deltaY * (e.deltaMode === 1 ? 50 : 1); // normalize line-mode scrolling
    const newZoom = Math.max(-2, Math.min(10, vs.zoom - delta * ZOOM_SPEED));
    // Adjust target so cursor world point stays under cursor after zoom
    const newScale = Math.pow(2, newZoom);
    const newTarget: [number, number, number] = [
      wx - (px - panelSize.width / 2) / newScale,
      wy - (py - panelSize.height / 2) / newScale,
      0,
    ];
    const newVs: ViewState = { target: newTarget, zoom: newZoom };
    viewStateRef.current = newVs;
    setViewState(newVs);
  };

  // Attach native wheel listener with passive:false so e.preventDefault() works
  const isSelecting = selectionMode !== 'none';
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !isSelecting) return;
    const handler = (e: WheelEvent) => e.preventDefault();
    svg.addEventListener('wheel', handler, { passive: false });
    return () => svg.removeEventListener('wheel', handler);
  }, [isSelecting]);

  // Project world-coord lasso points back to pixel for SVG rendering
  const lassoPath =
    lassoPointsWorld.length > 0
      ? `M ${lassoPointsWorld.map(([wx, wy]) => worldToPixel(wx, wy).join(',')).join(' L ')} Z`
      : '';

  const formatTime = (s: number): string => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    return `${m}:${String(sec).padStart(2, '0')}`;
  };

  const deckViewId = tracklets.length > 0 ? (tracklets[0].video_id ?? 'empty') : 'empty';

  return (
    <div ref={containerRef} className="relative w-full h-full bg-gray-900 overflow-hidden">
      {/* deck.gl scatter plot — controlled viewState */}
      <DeckGL
        key={deckViewId}
        views={new OrthographicView({ id: 'default', flipY: true })}
        viewState={viewState}
        controller={!isSelecting}
        layers={layers}
        onHover={handleDeckHover as Parameters<typeof DeckGL>[0]['onHover']}
        onViewStateChange={handleViewStateChange as Parameters<typeof DeckGL>[0]['onViewStateChange']}
        style={{ position: 'absolute', inset: '0' }}
      />

      {/* Selection overlay SVG — intercepts pointer events; wheel zooms the scatter plot */}
      {isSelecting && (
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full"
          style={{ cursor: 'crosshair', pointerEvents: 'all' }}
          onMouseDown={onSVGMouseDown}
          onMouseMove={onSVGMouseMove}
          onMouseUp={finaliseSelection}
          onMouseLeave={finaliseSelection}
          onWheel={onSVGWheel}
        >
          {selectionMode === 'lasso' && lassoPath && (
            <path
              d={lassoPath}
              fill="rgba(59,130,246,0.15)"
              stroke="rgb(59,130,246)"
              strokeWidth={1.5}
              strokeDasharray="4,2"
            />
          )}
          {selectionMode === 'rect' && rectStartWorld && rectCurrentWorld && (() => {
            const [x1, y1] = worldToPixel(...rectStartWorld);
            const [x2, y2] = worldToPixel(...rectCurrentWorld);
            return (
              <rect
                x={Math.min(x1, x2)}
                y={Math.min(y1, y2)}
                width={Math.abs(x2 - x1)}
                height={Math.abs(y2 - y1)}
                fill="rgba(59,130,246,0.15)"
                stroke="rgb(59,130,246)"
                strokeWidth={1.5}
                strokeDasharray="4,2"
              />
            );
          })()}
        </svg>
      )}

      {/* Selection mode buttons */}
      <div className="absolute top-3 left-3 z-10 flex gap-1 bg-gray-800/90 backdrop-blur rounded-lg p-1">
        <button
          onClick={() => setSelectionMode('none')}
          title="Pan / Zoom"
          className={`p-1.5 rounded transition-colors ${
            selectionMode === 'none' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          <MousePointer2 size={16} />
        </button>
        <button
          onClick={() => setSelectionMode('rect')}
          title="Rectangle selection"
          className={`p-1.5 rounded transition-colors ${
            selectionMode === 'rect' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          <Square size={16} />
        </button>
        <button
          onClick={() => setSelectionMode('lasso')}
          title="Lasso selection"
          className={`p-1.5 rounded transition-colors ${
            selectionMode === 'lasso' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          <Lasso size={16} />
        </button>
      </div>

      {/* Color mode toggle */}
      <div className="absolute top-3 right-3 z-10 flex gap-1 bg-gray-800/90 backdrop-blur rounded-lg p-1">
        <button
          onClick={() => setColorMode('class')}
          className={`px-3 py-1 text-xs rounded transition-colors ${
            colorMode === 'class' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'
          }`}
        >
          By Class
        </button>
        <button
          onClick={() => setColorMode('cluster')}
          className={`px-3 py-1 text-xs rounded transition-colors ${
            colorMode === 'cluster' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'
          }`}
        >
          By Cluster
        </button>
        <button
          onClick={() => setColorMode('time')}
          className={`px-3 py-1 text-xs rounded transition-colors ${
            colorMode === 'time' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'
          }`}
        >
          By Time
        </button>
      </div>

      {/* Select all / Clear selection */}
      {tracklets.length > 0 && (
        <div className="absolute bottom-3 left-3 z-10 flex gap-2">
          {selectedTrackletIds.size < tracklets.length && (
            <button
              onClick={() => setSelectedTrackletIds(new Set(tracklets.map(t => t.tracklet_id)))}
              className="bg-gray-800/90 backdrop-blur text-xs text-gray-300 hover:text-white px-3 py-1.5 rounded-lg border border-gray-600 transition-colors"
            >
              Select all ({tracklets.length})
            </button>
          )}
          {selectedTrackletIds.size > 0 && (
            <button
              onClick={() => setSelectedTrackletIds(new Set())}
              className="bg-gray-800/90 backdrop-blur text-xs text-gray-300 hover:text-white px-3 py-1.5 rounded-lg border border-gray-600 transition-colors"
            >
              Clear selection ({selectedTrackletIds.size})
            </button>
          )}
        </div>
      )}

      {/* Thumbnail tooltip */}
      {hoverInfo && hoverInfo.tracklet.thumbnail_base64 && (
        <div
          className="absolute z-20 pointer-events-none"
          style={{ left: hoverInfo.x + 14, top: hoverInfo.y - 14 }}
        >
          <div className="bg-gray-900 border border-gray-600 rounded-lg p-1.5 shadow-xl">
            <img
              src={`data:image/jpeg;base64,${hoverInfo.tracklet.thumbnail_base64}`}
              alt="tracklet thumbnail"
              className="w-24 h-24 object-cover rounded"
            />
            <p className="text-[10px] text-gray-400 mt-1 capitalize text-center">
              {hoverInfo.tracklet.class_name}
            </p>
          </div>
        </div>
      )}

      {/* Legend */}
      {tracklets.length > 0 && (
        <div className={`absolute bottom-3 right-3 z-10 bg-gray-800/90 backdrop-blur rounded-lg p-2 ${colorMode === 'time' ? 'w-[220px]' : 'max-w-[180px]'}`}>
          <p className="text-[10px] text-gray-400 uppercase tracking-wide mb-1.5 text-center">
            {colorMode === 'class' ? 'Classes' : colorMode === 'cluster' ? 'Clusters' : 'Time'}
          </p>
          {colorMode === 'time' ? (
            <div className="px-0.5">
              <div
                className="rounded"
                style={{
                  width: '100%',
                  height: 12,
                  background: 'linear-gradient(to right, rgb(13,8,135), rgb(126,3,168), rgb(240,100,61), rgb(240,249,33))',
                }}
              />
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-gray-400">{formatTime(minTime)}</span>
                <span className="text-[9px] text-gray-400">{formatTime(maxTime)}</span>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap gap-1 max-h-40 overflow-y-auto p-0.5">
              {colorMode === 'class'
                ? classItems.map(cls => {
                    const isActive =
                      tracklets.some(t => t.class_name === cls) &&
                      tracklets.filter(t => t.class_name === cls).every(t => selectedTrackletIds.has(t.tracklet_id));
                    return (
                      <button
                        key={cls}
                        onClick={() => handleClassLegendClick(cls)}
                        className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-all ${
                          isActive ? 'ring-1 ring-white bg-gray-600' : 'hover:bg-gray-700'
                        }`}
                      >
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: getClassColorHex(cls) }}
                        />
                        <span className="text-gray-200 capitalize">{cls}</span>
                      </button>
                    );
                  })
                : clusterItems.map(id => {
                    const isActive =
                      tracklets.some(t => t.cluster_id === id) &&
                      tracklets.filter(t => t.cluster_id === id).every(t => selectedTrackletIds.has(t.tracklet_id));
                    return (
                      <button
                        key={id}
                        onClick={() => handleClusterLegendClick(id)}
                        className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-all ${
                          isActive ? 'ring-1 ring-white bg-gray-600' : 'hover:bg-gray-700'
                        }`}
                      >
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: getClusterColorHex(id) }}
                        />
                        <span className="text-gray-200">
                          {id < 0 ? 'Noise' : `C${id}`}
                        </span>
                      </button>
                    );
                  })}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {tracklets.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-gray-500 text-sm">Select a video to view embeddings</p>
        </div>
      )}
    </div>
  );
}
