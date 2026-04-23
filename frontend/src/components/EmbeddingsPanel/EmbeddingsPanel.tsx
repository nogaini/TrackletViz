import { ScatterplotLayer } from "@deck.gl/layers";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "deck.gl";
import { Lasso, MousePointer2, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getClassColor,
  getClassColorHex,
  getClusterColor,
  getClusterColorHex,
  timeToColor,
} from "../../lib/colors";
import { pointInPolygon } from "../../lib/utils";
import { useStore } from "../../stores/useStore";
import type { GlobalClipMetadata, TrackletMetadata } from "../../types/index";
import { useThumbnail } from "../shared/LazyThumbnail";

const HIGHLIGHT_COLOR: [number, number, number, number] = [255, 255, 255, 255];
const DIM_ALPHA = 35; // dimmed but visible when legend focus is active
const TWOPOINT_COLOR: [number, number, number, number] = [251, 191, 36, 255]; // amber-400

interface ViewState {
  target: [number, number, number];
  zoom: number;
}

interface LocalHoverInfo {
  x: number;
  y: number;
  tracklet: TrackletMetadata;
}

interface GlobalHoverInfo {
  x: number;
  y: number;
  clip: GlobalClipMetadata;
}

export default function EmbeddingsPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [panelSize, setPanelSize] = useState({ width: 1, height: 1 });

  const [viewState, setViewState] = useState<ViewState>({
    target: [0, 0, 0],
    zoom: 1,
  });
  const viewStateRef = useRef<ViewState>({ target: [0, 0, 0], zoom: 1 });
  const rightDragRef = useRef<{ lastX: number; lastY: number } | null>(null);

  const [localHoverInfo, setLocalHoverInfo] = useState<LocalHoverInfo | null>(
    null,
  );
  const [globalHoverInfo, setGlobalHoverInfo] =
    useState<GlobalHoverInfo | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Mouse position tracking for the 2-point mode pending line
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);

  const hoverThumbnail = useThumbnail(
    localHoverInfo?.tracklet.thumbnail_base64
      ? null
      : (localHoverInfo?.tracklet.tracklet_id ?? null),
  );
  const localTooltipThumb =
    localHoverInfo?.tracklet.thumbnail_base64 ?? hoverThumbnail;

  const [lassoPointsWorld, setLassoPointsWorld] = useState<[number, number][]>(
    [],
  );
  const [rectStartWorld, setRectStartWorld] = useState<[number, number] | null>(
    null,
  );
  const [rectCurrentWorld, setRectCurrentWorld] = useState<
    [number, number] | null
  >(null);

  const {
    // local
    tracklets,
    selectionMode,
    setSelectionMode,
    selectedTrackletIds,
    setSelectedTrackletIds,
    colorMode,
    setColorMode,
    highlightedClusterId,
    highlightedTrackletId,
    // legend focus
    legendFocus,
    globalLegendFocus,
    setLegendFocus,
    setGlobalLegendFocus,
    // view
    viewMode,
    // global
    globalClips,
    selectedClipIds,
    setSelectedClipIds,
    globalSelectionMode,
    setGlobalSelectionMode,
    globalColorMode,
    setGlobalColorMode,
    twoPointSelection,
    setTwoPointSelection,
    twoPointPending,
    setTwoPointPending,
    highlightedGlobalClusterId,
    highlightedClipId,
    highlightedSpatialClipIds,
  } = useStore();

  // Derived data
  const isGlobal = viewMode === "global";

  const classItems = useMemo(
    () => [...new Set(tracklets.map((t) => t.class_name))].sort(),
    [tracklets],
  );

  const clusterItems = useMemo(
    () =>
      [...new Set(tracklets.map((t) => t.cluster_id))].sort((a, b) => a - b),
    [tracklets],
  );

  const globalClusterItems = useMemo(
    () =>
      [...new Set(globalClips.map((c) => c.cluster_id))].sort((a, b) => a - b),
    [globalClips],
  );

  const { minTime: localMinTime, maxTime: localMaxTime } = useMemo(() => {
    if (tracklets.length === 0) return { minTime: 0, maxTime: 1 };
    const times = tracklets.map((t) => t.start_timestamp);
    return { minTime: Math.min(...times), maxTime: Math.max(...times) };
  }, [tracklets]);

  const { minTime: globalMinTime, maxTime: globalMaxTime } = useMemo(() => {
    if (globalClips.length === 0) return { minTime: 0, maxTime: 1 };
    const times = globalClips.map((c) => c.start_time);
    return { minTime: Math.min(...times), maxTime: Math.max(...times) };
  }, [globalClips]);

  // Active data for the current view
  const activeData = isGlobal ? globalClips : tracklets;
  const activeSelMode = isGlobal ? globalSelectionMode : selectionMode;
  const isSelecting = activeSelMode !== "none" && activeSelMode !== "twopoint";
  const isTwoPoint = isGlobal && globalSelectionMode === "twopoint";

  // Track panel dimensions
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setPanelSize({ width: width || 1, height: height || 1 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit view whenever data or panel size changes
  useEffect(() => {
    if (activeData.length === 0 || panelSize.width <= 1) return;
    const xs = activeData.map((d) => d.umap_x);
    const ys = activeData.map((d) => d.umap_y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale =
      Math.min(panelSize.width / rangeX, panelSize.height / rangeY) * 0.85;
    const vs: ViewState = {
      target: [(minX + maxX) / 2, (minY + maxY) / 2, 0],
      zoom: Math.log2(scale),
    };
    viewStateRef.current = vs;
    setViewState(vs);
  }, [activeData, panelSize]);

  const pixelToWorld = useCallback(
    (px: number, py: number): [number, number] => {
      const vs = viewStateRef.current;
      const scale = Math.pow(2, vs.zoom);
      return [
        (px - panelSize.width / 2) / scale + vs.target[0],
        (py - panelSize.height / 2) / scale + vs.target[1],
      ];
    },
    [panelSize],
  );

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

  // ── Layers ─────────────────────────────────────────────────────────────

  const layers = useMemo(() => {
    if (activeData.length === 0) return [];

    if (isGlobal) {
      const clips = globalClips;
      return [
        new ScatterplotLayer<GlobalClipMetadata>({
          id: "global-clips",
          data: clips,
          getPosition: (d: GlobalClipMetadata) => [d.umap_x, d.umap_y, 0],
          getFillColor: (
            d: GlobalClipMetadata,
          ): [number, number, number, number] => {
            const isTwoPt1 = twoPointPending?.clip_id === d.clip_id;
            const isTwoPt2a = twoPointSelection?.clip1.clip_id === d.clip_id;
            const isTwoPt2b = twoPointSelection?.clip2.clip_id === d.clip_id;
            if (isTwoPt1 || isTwoPt2a || isTwoPt2b) return TWOPOINT_COLOR;

            const [r, g, b] =
              globalColorMode === "cluster"
                ? getClusterColor(d.cluster_id)
                : timeToColor(d.start_time, globalMinTime, globalMaxTime);

            // Legend focus: dim non-matching points
            if (globalLegendFocus !== null) {
              const matches = d.cluster_id === globalLegendFocus.id;
              if (!matches) return [r, g, b, DIM_ALPHA];
              if (highlightedClipId === d.clip_id) return HIGHLIGHT_COLOR;
              const hasSelectionLF = selectedClipIds.size > 0;
              if (hasSelectionLF && selectedClipIds.has(d.clip_id)) return [255, 255, 255, 230];
              return [r, g, b, 220];
            }

            if (highlightedClipId === d.clip_id) return HIGHLIGHT_COLOR;

            // Brighten noise-cluster reps in 2-point mode so they are visually distinct
            if (globalSelectionMode === "twopoint" && d.is_representative && d.cluster_id < 0) {
              return [210, 210, 210, 240];
            }

            // Dim non-representatives in 2-point mode
            if (globalSelectionMode === "twopoint" && !d.is_representative) {
              return [r, g, b, 25];
            }

            const hasSelection = selectedClipIds.size > 0;
            const isSelected = selectedClipIds.has(d.clip_id);
            const isHighlightedCluster =
              highlightedGlobalClusterId !== null &&
              d.cluster_id === highlightedGlobalClusterId;
            // White fill for selected clips (mirrors local tracklet behavior)
            if (globalSelectionMode !== "twopoint" && hasSelection && isSelected) return [255, 255, 255, 230];
            let alpha = 220;
            if (globalSelectionMode !== "twopoint" && hasSelection && !isSelected) alpha = Math.min(alpha, 40);
            if (isHighlightedCluster) alpha = 255;

            // Spatial bucket highlight — white for matching clips, dim for the rest
            if (highlightedSpatialClipIds !== null) {
              if (highlightedSpatialClipIds.has(d.clip_id)) return [255, 255, 255, 230];
              return [r, g, b, DIM_ALPHA];
            }

            return [r, g, b, alpha];
          },
          getRadius: (d: GlobalClipMetadata): number => {
            if (d.clip_id === highlightedClipId) return 14;
            if (globalSelectionMode === "twopoint" && d.is_representative)
              return 9;
            if (
              highlightedGlobalClusterId !== null &&
              d.cluster_id === highlightedGlobalClusterId
            )
              return 6;
            return 5;
          },
          radiusUnits: "pixels",
          pickable: true,
          onClick: (info: { object?: unknown }) => {
            if (globalSelectionMode !== "twopoint") return;
            const obj = info.object as GlobalClipMetadata | undefined;
            if (!obj) return;
            if (!obj.is_representative) return; // block clicks on non-representatives
            if (!twoPointPending) {
              setTwoPointPending(obj);
              setSelectedClipIds(new Set([obj.clip_id]));
            } else {
              setTwoPointSelection({ clip1: twoPointPending, clip2: obj });
              setTwoPointPending(null);
              setSelectedClipIds(
                new Set([twoPointPending.clip_id, obj.clip_id]),
              );
            }
          },
          onHover: (info: { object?: unknown; x: number; y: number }) => {
            const obj = info.object as GlobalClipMetadata | null | undefined;
            if (obj && obj.clip_id) {
              if (globalLegendFocus !== null && obj.cluster_id !== globalLegendFocus.id) {
                setGlobalHoverInfo(null);
                return;
              }
              setGlobalHoverInfo({ x: info.x, y: info.y, clip: obj });
            } else {
              setGlobalHoverInfo(null);
            }
          },
          updateTriggers: {
            getFillColor: [
              globalColorMode,
              globalSelectionMode,
              selectedClipIds,
              highlightedGlobalClusterId,
              highlightedClipId,
              twoPointPending,
              twoPointSelection,
              globalMinTime,
              globalMaxTime,
              globalLegendFocus,
              highlightedSpatialClipIds,
            ],
            getRadius: [highlightedClipId, highlightedGlobalClusterId, globalSelectionMode],
          },
        }),
      ];
    }

    // Local mode
    return [
      new ScatterplotLayer<TrackletMetadata>({
        id: "tracklets",
        data: tracklets,
        getPosition: (d: TrackletMetadata) => [d.umap_x, d.umap_y, 0],
        getFillColor: (
          d: TrackletMetadata,
        ): [number, number, number, number] => {
          const [r, g, b] =
            colorMode === "class"
              ? getClassColor(d.class_name)
              : colorMode === "cluster"
                ? getClusterColor(d.cluster_id)
                : timeToColor(d.start_timestamp, localMinTime, localMaxTime);
          const isHighlightedTracklet = d.tracklet_id === highlightedTrackletId;

          // Legend focus: dim non-matching points (independent of selection)
          if (legendFocus !== null) {
            const matches =
              legendFocus.type === "class"
                ? d.class_name === legendFocus.id
                : d.cluster_id === legendFocus.id;
            if (!matches) return [r, g, b, DIM_ALPHA];
            if (isHighlightedTracklet) return HIGHLIGHT_COLOR;
            const hasSelectionLF = selectedTrackletIds.size > 0;
            if (hasSelectionLF && selectedTrackletIds.has(d.tracklet_id)) return [255, 255, 255, 230];
            return [r, g, b, 220];
          }

          if (isHighlightedTracklet) return HIGHLIGHT_COLOR;
          const hasSelection = selectedTrackletIds.size > 0;
          const isSelected = selectedTrackletIds.has(d.tracklet_id);
          const isHighlightedCluster =
            highlightedClusterId !== null &&
            d.cluster_id === highlightedClusterId;
          if (hasSelection && isSelected) return [255, 255, 255, 230];
          let alpha = 220;
          if (hasSelection && !isSelected) alpha = Math.min(alpha, 40);
          if (isHighlightedCluster) alpha = 255;
          return [r, g, b, alpha];
        },
        getRadius: (d: TrackletMetadata): number => {
          if (d.tracklet_id === highlightedTrackletId) return 14;
          if (
            highlightedClusterId !== null &&
            d.cluster_id === highlightedClusterId
          )
            return 6;
          return 4;
        },
        radiusUnits: "pixels",
        pickable: selectionMode === "none",
        updateTriggers: {
          getFillColor: [
            colorMode,
            selectedTrackletIds,
            highlightedClusterId,
            highlightedTrackletId,
            localMinTime,
            localMaxTime,
            legendFocus,
          ],
          getRadius: [highlightedTrackletId, highlightedClusterId],
        },
      }),
    ];
  }, [
    activeData,
    isGlobal,
    globalClips,
    tracklets,
    colorMode,
    globalColorMode,
    selectedTrackletIds,
    selectedClipIds,
    highlightedClusterId,
    highlightedTrackletId,
    highlightedGlobalClusterId,
    highlightedClipId,
    selectionMode,
    globalSelectionMode,
    twoPointPending,
    twoPointSelection,
    localMinTime,
    localMaxTime,
    globalMinTime,
    globalMaxTime,
    legendFocus,
    globalLegendFocus,
    highlightedSpatialClipIds,
    setTwoPointPending,
    setSelectedClipIds,
    setTwoPointSelection,
  ]);

  // ── Hover handlers ─────────────────────────────────────────────────────

  const handleDeckHover = useCallback(
    (info: { object?: unknown; x: number; y: number }) => {
      if (isGlobal) return; // global hover handled per-layer
      const obj = info.object as TrackletMetadata | null | undefined;
      if (obj && obj.tracklet_id) {
        if (legendFocus !== null) {
          const visible =
            legendFocus.type === "class"
              ? obj.class_name === legendFocus.id
              : obj.cluster_id === legendFocus.id;
          if (!visible) {
            setLocalHoverInfo(null);
            return;
          }
        }
        setLocalHoverInfo({ x: info.x, y: info.y, tracklet: obj });
      } else {
        setLocalHoverInfo(null);
      }
    },
    [isGlobal, legendFocus],
  );

  const handleViewStateChange = useCallback(
    ({ viewState: vs }: { viewState: unknown }) => {
      viewStateRef.current = vs as ViewState;
      setViewState(vs as ViewState);
    },
    [],
  );

  // ── Legend click handlers ──────────────────────────────────────────────

  const handleClassLegendClick = useCallback(
    (className: string) => {
      if (legendFocus?.type === "class" && legendFocus.id === className) {
        setLegendFocus(null);
      } else {
        setLegendFocus({ type: "class", id: className });
      }
    },
    [legendFocus, setLegendFocus],
  );

  const handleClusterLegendClick = useCallback(
    (clusterId: number) => {
      if (legendFocus?.type === "cluster" && legendFocus.id === clusterId) {
        setLegendFocus(null);
      } else {
        setLegendFocus({ type: "cluster", id: clusterId });
      }
    },
    [legendFocus, setLegendFocus],
  );

  const handleGlobalClusterLegendClick = useCallback(
    (clusterId: number) => {
      if (globalLegendFocus?.id === clusterId) {
        setGlobalLegendFocus(null);
      } else {
        setGlobalLegendFocus({ type: "cluster", id: clusterId });
      }
    },
    [globalLegendFocus, setGlobalLegendFocus],
  );

  // ── Right-click drag to pan (always active) ────────────────────────────

  const onContainerMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 2) return;
    e.preventDefault();
    rightDragRef.current = { lastX: e.clientX, lastY: e.clientY };
  }, []);

  const onContainerContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!rightDragRef.current) return;
      const dx = e.clientX - rightDragRef.current.lastX;
      const dy = e.clientY - rightDragRef.current.lastY;
      rightDragRef.current = { lastX: e.clientX, lastY: e.clientY };

      const vs = viewStateRef.current;
      const scale = Math.pow(2, vs.zoom);
      const newTarget: [number, number, number] = [
        vs.target[0] - dx / scale,
        vs.target[1] - dy / scale,
        0,
      ];
      const newVs = { ...vs, target: newTarget };
      viewStateRef.current = newVs;
      setViewState(newVs);
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (e.button === 2) rightDragRef.current = null;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // ── SVG overlay mouse handlers ─────────────────────────────────────────

  const getSVGCoords = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onSVGMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    e.preventDefault();
    const pos = getSVGCoords(e);
    const [wx, wy] = pixelToWorld(pos.x, pos.y);
    if (activeSelMode === "lasso") {
      setLassoPointsWorld([[wx, wy]]);
      setIsDrawing(true);
    } else if (activeSelMode === "rect") {
      setRectStartWorld([wx, wy]);
      setRectCurrentWorld([wx, wy]);
      setIsDrawing(true);
    }
  };

  const onSVGMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDrawing) return;
    const pos = getSVGCoords(e);
    const [wx, wy] = pixelToWorld(pos.x, pos.y);
    if (activeSelMode === "lasso") {
      setLassoPointsWorld((prev) => [...prev, [wx, wy]]);
    } else if (activeSelMode === "rect") {
      setRectCurrentWorld([wx, wy]);
    }
  };

  const isLegendVisible = (t: TrackletMetadata) => {
    if (legendFocus === null) return true;
    return legendFocus.type === "class"
      ? t.class_name === legendFocus.id
      : t.cluster_id === legendFocus.id;
  };

  const finaliseSelection = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    if (isGlobal) {
      if (activeSelMode === "lasso" && lassoPointsWorld.length > 2) {
        const poly = lassoPointsWorld.map(([wx, wy]) => ({ x: wx, y: wy }));
        const selected = globalClips.filter((c) =>
          (globalLegendFocus === null || c.cluster_id === globalLegendFocus.id) &&
          pointInPolygon(c.umap_x, c.umap_y, poly),
        );
        setSelectedClipIds(new Set(selected.map((c) => c.clip_id)));
      } else if (
        activeSelMode === "rect" &&
        rectStartWorld &&
        rectCurrentWorld
      ) {
        const [wx1, wy1] = rectStartWorld;
        const [wx2, wy2] = rectCurrentWorld;
        const minX = Math.min(wx1, wx2);
        const maxX = Math.max(wx1, wx2);
        const minY = Math.min(wy1, wy2);
        const maxY = Math.max(wy1, wy2);
        const selected = globalClips.filter(
          (c) =>
            (globalLegendFocus === null || c.cluster_id === globalLegendFocus.id) &&
            c.umap_x >= minX &&
            c.umap_x <= maxX &&
            c.umap_y >= minY &&
            c.umap_y <= maxY,
        );
        setSelectedClipIds(new Set(selected.map((c) => c.clip_id)));
      }
    } else {
      if (selectionMode === "lasso" && lassoPointsWorld.length > 2) {
        const worldPoly = lassoPointsWorld.map(([wx, wy]) => ({
          x: wx,
          y: wy,
        }));
        const selected = tracklets.filter((t) =>
          isLegendVisible(t) && pointInPolygon(t.umap_x, t.umap_y, worldPoly),
        );
        setSelectedTrackletIds(new Set(selected.map((t) => t.tracklet_id)));
      } else if (
        selectionMode === "rect" &&
        rectStartWorld &&
        rectCurrentWorld
      ) {
        const [wx1, wy1] = rectStartWorld;
        const [wx2, wy2] = rectCurrentWorld;
        const minX = Math.min(wx1, wx2);
        const maxX = Math.max(wx1, wx2);
        const minY = Math.min(wy1, wy2);
        const maxY = Math.max(wy1, wy2);
        const selected = tracklets.filter(
          (t) =>
            isLegendVisible(t) &&
            t.umap_x >= minX &&
            t.umap_x <= maxX &&
            t.umap_y >= minY &&
            t.umap_y <= maxY,
        );
        setSelectedTrackletIds(new Set(selected.map((t) => t.tracklet_id)));
      }
    }

    setLassoPointsWorld([]);
    setRectStartWorld(null);
    setRectCurrentWorld(null);
  };

  // SVG wheel — zoom-to-cursor
  const onSVGWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const svgRect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - svgRect.left;
    const py = e.clientY - svgRect.top;
    const [wx, wy] = pixelToWorld(px, py);
    const vs = viewStateRef.current;
    const ZOOM_SPEED = 0.001;
    const delta = e.deltaY * (e.deltaMode === 1 ? 50 : 1);
    const newZoom = Math.max(-2, Math.min(10, vs.zoom - delta * ZOOM_SPEED));
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

  // Prevent default scroll when in selection mode
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !isSelecting) return;
    const handler = (e: WheelEvent) => e.preventDefault();
    svg.addEventListener("wheel", handler, { passive: false });
    return () => svg.removeEventListener("wheel", handler);
  }, [isSelecting]);

  // Escape key cancels 2-point first-click
  useEffect(() => {
    if (!isTwoPoint) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setTwoPointPending(null);
        setGlobalSelectionMode("none");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isTwoPoint, setTwoPointPending, setGlobalSelectionMode]);

  // Mouse move tracking for 2-point pending line
  const onContainerMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isTwoPoint || !twoPointPending) return;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      setMousePos([e.clientX - rect.left, e.clientY - rect.top]);
    },
    [isTwoPoint, twoPointPending],
  );

  const onContainerMouseLeave = useCallback(() => {
    setMousePos(null);
  }, []);

  // SVG projection for 2-point overlay
  const twoPointLine = useMemo(() => {
    if (!isGlobal) return null;
    if (twoPointSelection) {
      const [x1, y1] = worldToPixel(
        twoPointSelection.clip1.umap_x,
        twoPointSelection.clip1.umap_y,
      );
      const [x2, y2] = worldToPixel(
        twoPointSelection.clip2.umap_x,
        twoPointSelection.clip2.umap_y,
      );
      return { x1, y1, x2, y2, fixed: true };
    }
    if (twoPointPending && mousePos) {
      const [x1, y1] = worldToPixel(
        twoPointPending.umap_x,
        twoPointPending.umap_y,
      );
      return { x1, y1, x2: mousePos[0], y2: mousePos[1], fixed: false };
    }
    return null;
  }, [isGlobal, twoPointSelection, twoPointPending, mousePos, worldToPixel]);

  // Lasso SVG path
  const lassoPath =
    lassoPointsWorld.length > 0
      ? `M ${lassoPointsWorld.map(([wx, wy]) => worldToPixel(wx, wy).join(",")).join(" L ")} Z`
      : "";

  const formatTime = (s: number): string => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0)
      return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    return `${m}:${String(sec).padStart(2, "0")}`;
  };

  const deckViewId = isGlobal
    ? globalClips.length > 0
      ? globalClips[0].video_id + "-global"
      : "global-empty"
    : tracklets.length > 0
      ? (tracklets[0].video_id ?? "empty")
      : "empty";

  // deck.gl controller: disabled during lasso/rect selection;
  // in 2-point mode deck.gl handles clicks via per-layer onClick
  const deckController = isSelecting ? false : true;

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      onMouseMove={onContainerMouseMove}
      onMouseLeave={onContainerMouseLeave}
      onMouseDown={onContainerMouseDown}
      onContextMenu={onContainerContextMenu}
    >
      {/* deck.gl scatter plot */}
      <DeckGL
        key={deckViewId}
        views={new OrthographicView({ id: "default", flipY: true })}
        viewState={viewState}
        controller={deckController}
        layers={layers}
        onHover={handleDeckHover as Parameters<typeof DeckGL>[0]["onHover"]}
        onViewStateChange={
          handleViewStateChange as Parameters<
            typeof DeckGL
          >[0]["onViewStateChange"]
        }
        onClick={(info) => {
          if (!info.object) {
            setLegendFocus(null);
            setGlobalLegendFocus(null);
          }
        }}
        style={{ position: "absolute", inset: "0" }}
      />

      {/* Selection overlay SVG (lasso / rect) */}
      {isSelecting && (
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full"
          style={{ cursor: "crosshair", pointerEvents: "all" }}
          onMouseDown={onSVGMouseDown}
          onMouseMove={onSVGMouseMove}
          onMouseUp={finaliseSelection}
          onMouseLeave={finaliseSelection}
          onWheel={onSVGWheel}
        >
          {activeSelMode === "lasso" && lassoPath && (
            <path
              d={lassoPath}
              fill="rgba(59,130,246,0.15)"
              stroke="rgb(59,130,246)"
              strokeWidth={1.5}
              strokeDasharray="4,2"
            />
          )}
          {activeSelMode === "rect" &&
            rectStartWorld &&
            rectCurrentWorld &&
            (() => {
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

      {/* 2-point selection SVG overlay (pointer-events: none so deck.gl handles clicks) */}
      {isGlobal && twoPointLine && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 5 }}
        >
          <line
            x1={twoPointLine.x1}
            y1={twoPointLine.y1}
            x2={twoPointLine.x2}
            y2={twoPointLine.y2}
            stroke="#FBBF24"
            strokeWidth={2}
            strokeDasharray="6,3"
          />
          {/* Fixed endpoint circles */}
          {twoPointLine.fixed && (
            <>
              <circle
                cx={twoPointLine.x1}
                cy={twoPointLine.y1}
                r={6}
                fill="#FBBF24"
              />
              <circle
                cx={twoPointLine.x2}
                cy={twoPointLine.y2}
                r={6}
                fill="#FBBF24"
              />
            </>
          )}
          {!twoPointLine.fixed && (
            <circle
              cx={twoPointLine.x1}
              cy={twoPointLine.y1}
              r={6}
              fill="#FBBF24"
            />
          )}
        </svg>
      )}

      {/* 2-point mode hint */}
      {isGlobal && isTwoPoint && twoPointPending && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-20 bg-amber-900/90 border border-amber-500 text-amber-200 text-xs px-3 py-1.5 rounded-lg pointer-events-none">
          Click a second clip to compare • Esc to cancel
        </div>
      )}

      {/* Selection mode buttons */}
      <div className="absolute top-3 left-3 z-10 flex gap-1 bg-gray-800/90 backdrop-blur rounded-lg p-1">
        {/* Pan button */}
        <button
          onClick={() => {
            if (isGlobal) { setGlobalSelectionMode("none"); setSelectedClipIds(new Set()); setTwoPointSelection(null); setTwoPointPending(null); }
            else { setSelectionMode("none"); setSelectedTrackletIds(new Set()); }
          }}
          title="Pan / Zoom"
          className={`p-1.5 rounded transition-colors ${
            activeSelMode === "none"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          <MousePointer2 size={16} />
        </button>
        {/* Rect selection */}
        <button
          onClick={() => {
            if (isGlobal) { setGlobalSelectionMode("rect"); setSelectedClipIds(new Set()); setTwoPointSelection(null); setTwoPointPending(null); }
            else { setSelectionMode("rect"); setSelectedTrackletIds(new Set()); }
          }}
          title="Rectangle selection"
          className={`p-1.5 rounded transition-colors ${
            activeSelMode === "rect"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          <Square size={16} />
        </button>
        {/* Lasso selection */}
        <button
          onClick={() => {
            if (isGlobal) { setGlobalSelectionMode("lasso"); setSelectedClipIds(new Set()); setTwoPointSelection(null); setTwoPointPending(null); }
            else { setSelectionMode("lasso"); setSelectedTrackletIds(new Set()); }
          }}
          title="Lasso selection"
          className={`p-1.5 rounded transition-colors ${
            activeSelMode === "lasso"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white"
          }`}
        >
          <Lasso size={16} />
        </button>
        {/* 2-point selection (global only) */}
        {isGlobal && (
          <button
            onClick={() => { setGlobalSelectionMode("twopoint"); setSelectedClipIds(new Set()); setTwoPointSelection(null); setTwoPointPending(null); }}
            title="Compare two clips (2-point selection)"
            className={`p-1.5 rounded transition-colors ${
              globalSelectionMode === "twopoint"
                ? "bg-blue-600 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="2" cy="14" r="2" fill="currentColor" />
                <line x1="3.4" y1="12.6" x2="12.6" y2="3.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                <circle cx="14" cy="2" r="2" fill="currentColor" />
              </svg>
          </button>
        )}
      </div>

      {/* Color mode toggle */}
      <div className="absolute top-3 right-3 z-10 flex gap-1 bg-gray-800/90 backdrop-blur rounded-lg p-1">
        {isGlobal ? (
          <>
            <button
              onClick={() => setGlobalColorMode("cluster")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                globalColorMode === "cluster"
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              By Cluster
            </button>
            <button
              onClick={() => setGlobalColorMode("time")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                globalColorMode === "time"
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              By Time
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => setColorMode("class")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                colorMode === "class"
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              By Class
            </button>
            <button
              onClick={() => setColorMode("cluster")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                colorMode === "cluster"
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              By Cluster
            </button>
            <button
              onClick={() => setColorMode("time")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                colorMode === "time"
                  ? "bg-blue-600 text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              By Time
            </button>
          </>
        )}
      </div>

      {/* Select all / Clear selection */}
      {activeData.length > 0 && (
        <div className="absolute bottom-3 left-3 z-10 flex gap-2">
          {isGlobal ? (
            <>
              {selectedClipIds.size < globalClips.length && (
                <button
                  onClick={() =>
                    setSelectedClipIds(
                      new Set(globalClips.map((c) => c.clip_id)),
                    )
                  }
                  className="bg-gray-800/90 backdrop-blur text-xs text-gray-300 hover:text-white px-3 py-1.5 rounded-lg border border-gray-600 transition-colors"
                >
                  Select all ({globalClips.length})
                </button>
              )}
              {selectedClipIds.size > 0 && (
                <button
                  onClick={() => {
                    setSelectedClipIds(new Set());
                    setTwoPointSelection(null);
                    setTwoPointPending(null);
                  }}
                  className="bg-gray-800/90 backdrop-blur text-xs text-gray-300 hover:text-white px-3 py-1.5 rounded-lg border border-gray-600 transition-colors"
                >
                  Clear all ({selectedClipIds.size})
                </button>
              )}
            </>
          ) : (
            <>
              {selectedTrackletIds.size < tracklets.length && (
                <button
                  onClick={() =>
                    setSelectedTrackletIds(
                      new Set(tracklets.map((t) => t.tracklet_id)),
                    )
                  }
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
                  Clear all ({selectedTrackletIds.size})
                </button>
              )}
            </>
          )}
        </div>
      )}

      {/* Tooltips */}
      {!isGlobal && localHoverInfo && (
        <div
          className="absolute z-20 pointer-events-none"
          style={{ left: localHoverInfo.x + 14, top: localHoverInfo.y - 14 }}
        >
          <div className="bg-gray-900 border border-gray-600 rounded-lg p-1.5 shadow-xl">
            {localTooltipThumb ? (
              <img
                src={`data:image/jpeg;base64,${localTooltipThumb}`}
                alt="tracklet thumbnail"
                className="w-24 h-24 object-cover rounded"
              />
            ) : (
              <div className="w-24 h-24 bg-gray-700 rounded flex items-center justify-center">
                <span className="text-[10px] text-gray-500">…</span>
              </div>
            )}
            <p className="text-[10px] text-gray-400 mt-1 capitalize text-center">
              {localHoverInfo.tracklet.class_name}
            </p>
          </div>
        </div>
      )}
      {isGlobal && globalHoverInfo && (
        <div
          className="absolute z-20 pointer-events-none"
          style={{ left: globalHoverInfo.x + 14, top: globalHoverInfo.y - 14 }}
        >
          <div className="bg-gray-900 border border-gray-600 rounded-lg p-1.5 shadow-xl">
            {globalHoverInfo.clip.thumbnail_base64 ? (
              <img
                src={`data:image/jpeg;base64,${globalHoverInfo.clip.thumbnail_base64}`}
                alt="clip thumbnail"
                className="w-40 h-auto rounded"
              />
            ) : (
              <div className="w-40 h-24 bg-gray-700 rounded flex items-center justify-center">
                <span className="text-[10px] text-gray-500">…</span>
              </div>
            )}
            <p className="text-[10px] text-gray-400 mt-1 text-center">
              Clip {globalHoverInfo.clip.clip_index} ·{" "}
              {formatTime(globalHoverInfo.clip.start_time)}–
              {formatTime(globalHoverInfo.clip.end_time)}
            </p>
          </div>
        </div>
      )}

      {/* Legend */}
      {activeData.length > 0 && (
        <div
          className={`absolute bottom-3 right-3 z-10 bg-gray-800/90 backdrop-blur rounded-lg p-2 ${
            (!isGlobal && colorMode === "time") ||
            (isGlobal && globalColorMode === "time")
              ? "w-[220px]"
              : "max-w-[180px]"
          }`}
        >
          <p className="text-[10px] text-gray-400 uppercase tracking-wide mb-1.5 text-center">
            {isGlobal
              ? globalColorMode === "cluster"
                ? "Clusters"
                : "Time"
              : colorMode === "class"
                ? "Classes"
                : colorMode === "cluster"
                  ? "Clusters"
                  : "Time"}
          </p>

          {(isGlobal ? globalColorMode === "time" : colorMode === "time") ? (
            <div className="px-0.5">
              <div
                className="rounded"
                style={{
                  width: "100%",
                  height: 12,
                  background:
                    "linear-gradient(to right, rgb(13,8,135), rgb(126,3,168), rgb(240,100,61), rgb(240,249,33))",
                }}
              />
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-gray-400">
                  {formatTime(isGlobal ? globalMinTime : localMinTime)}
                </span>
                <span className="text-[9px] text-gray-400">
                  {formatTime(isGlobal ? globalMaxTime : localMaxTime)}
                </span>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap gap-1 max-h-40 overflow-y-auto p-0.5">
              {isGlobal
                ? globalClusterItems.map((id) => {
                    const isActive = globalLegendFocus?.id === id;
                    return (
                      <button
                        key={id}
                        onClick={() => handleGlobalClusterLegendClick(id)}
                        className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-all ${
                          isActive
                            ? "ring-1 ring-white bg-gray-600"
                            : "hover:bg-gray-700"
                        }`}
                      >
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: getClusterColorHex(id) }}
                        />
                        <span className="text-gray-200">
                          {id < 0 ? "Noise" : `C${id}`}
                        </span>
                      </button>
                    );
                  })
                : colorMode === "class"
                  ? classItems.map((cls) => {
                      const isActive =
                        legendFocus?.type === "class" && legendFocus.id === cls;
                      return (
                        <button
                          key={cls}
                          onClick={() => handleClassLegendClick(cls)}
                          className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-all ${
                            isActive
                              ? "ring-1 ring-white bg-gray-600"
                              : "hover:bg-gray-700"
                          }`}
                        >
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ backgroundColor: getClassColorHex(cls) }}
                          />
                          <span className="text-gray-200 capitalize">
                            {cls}
                          </span>
                        </button>
                      );
                    })
                  : clusterItems.map((id) => {
                      const isActive =
                        legendFocus?.type === "cluster" && legendFocus.id === id;
                      return (
                        <button
                          key={id}
                          onClick={() => handleClusterLegendClick(id)}
                          className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-all ${
                            isActive
                              ? "ring-1 ring-white bg-gray-600"
                              : "hover:bg-gray-700"
                          }`}
                        >
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ backgroundColor: getClusterColorHex(id) }}
                          />
                          <span className="text-gray-200">
                            {id < 0 ? "Noise" : `C${id}`}
                          </span>
                        </button>
                      );
                    })}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {activeData.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-gray-500 text-sm">
            {isGlobal
              ? "Select a video to view global clip embeddings"
              : "Select a video to view embeddings"}
          </p>
        </div>
      )}
    </div>
  );
}
