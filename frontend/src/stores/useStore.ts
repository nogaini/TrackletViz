import { create } from 'zustand';
import type {
  VideoSummary,
  VideoMetadata,
  TrackletMetadata,
  GlobalClipMetadata,
  TwoPointSelection,
} from '../types/index';

interface AppState {
  // ── Local view ────────────────────────────────────────────────────────
  videos: VideoSummary[];
  selectedVideoId: string | null;
  videoMetadata: VideoMetadata | null;
  tracklets: TrackletMetadata[];
  selectedTrackletIds: Set<string>;
  selectionMode: 'lasso' | 'rect' | 'none';
  colorMode: 'class' | 'cluster' | 'time';
  highlightedClusterId: number | null;
  highlightedTrackletId: string | null;
  activeTabIndex: number;

  // ── Legend focus (visibility filter — independent of selection) ──────
  legendFocus: { type: 'cluster'; id: number } | { type: 'class'; id: string } | null;
  globalLegendFocus: { type: 'cluster'; id: number } | null;

  // ── View mode ─────────────────────────────────────────────────────────
  viewMode: 'local' | 'global';

  // ── Global view ───────────────────────────────────────────────────────
  globalClips: GlobalClipMetadata[];
  selectedClipIds: Set<string>;
  globalSelectionMode: 'lasso' | 'rect' | 'twopoint' | 'none';
  globalColorMode: 'cluster' | 'time';
  twoPointSelection: TwoPointSelection | null;
  twoPointPending: GlobalClipMetadata | null;
  highlightedGlobalClusterId: number | null;
  highlightedClipId: string | null;
  activeGlobalTabIndex: number;
  highlightedSpatialClipIds: Set<string> | null;

  // ── Actions — local ───────────────────────────────────────────────────
  setVideos: (videos: VideoSummary[]) => void;
  setSelectedVideoId: (id: string | null) => void;
  setVideoMetadata: (meta: VideoMetadata | null) => void;
  setTracklets: (tracklets: TrackletMetadata[]) => void;
  setSelectedTrackletIds: (ids: Set<string>) => void;
  setSelectionMode: (mode: 'lasso' | 'rect' | 'none') => void;
  setColorMode: (mode: 'class' | 'cluster' | 'time') => void;
  setHighlightedClusterId: (id: number | null) => void;
  setHighlightedTrackletId: (id: string | null) => void;
  setActiveTabIndex: (index: number) => void;

  setLegendFocus: (f: AppState['legendFocus']) => void;
  setGlobalLegendFocus: (f: AppState['globalLegendFocus']) => void;

  // ── Actions — view mode ───────────────────────────────────────────────
  setViewMode: (mode: 'local' | 'global') => void;

  // ── Actions — global ──────────────────────────────────────────────────
  setGlobalClips: (clips: GlobalClipMetadata[]) => void;
  setSelectedClipIds: (ids: Set<string>) => void;
  setGlobalSelectionMode: (mode: 'lasso' | 'rect' | 'twopoint' | 'none') => void;
  setGlobalColorMode: (mode: 'cluster' | 'time') => void;
  setTwoPointSelection: (sel: TwoPointSelection | null) => void;
  setTwoPointPending: (clip: GlobalClipMetadata | null) => void;
  setHighlightedGlobalClusterId: (id: number | null) => void;
  setHighlightedClipId: (id: string | null) => void;
  setActiveGlobalTabIndex: (index: number) => void;
  setHighlightedSpatialClipIds: (ids: Set<string> | null) => void;
}

export const useStore = create<AppState>((set) => ({
  // ── Local view defaults ───────────────────────────────────────────────
  videos: [],
  selectedVideoId: null,
  videoMetadata: null,
  tracklets: [],
  selectedTrackletIds: new Set(),
  selectionMode: 'none',
  colorMode: 'class',
  highlightedClusterId: null,
  highlightedTrackletId: null,
  activeTabIndex: 0,

  // ── Legend focus defaults ─────────────────────────────────────────────
  legendFocus: null,
  globalLegendFocus: null,

  // ── View mode default ─────────────────────────────────────────────────
  viewMode: 'local',

  // ── Global view defaults ──────────────────────────────────────────────
  globalClips: [],
  selectedClipIds: new Set(),
  globalSelectionMode: 'none',
  globalColorMode: 'cluster',
  twoPointSelection: null,
  twoPointPending: null,
  highlightedGlobalClusterId: null,
  highlightedClipId: null,
  activeGlobalTabIndex: 0,
  highlightedSpatialClipIds: null,

  // ── Actions — local ───────────────────────────────────────────────────
  setVideos: (videos) => set({ videos }),
  setSelectedVideoId: (selectedVideoId) => set({ selectedVideoId }),
  setVideoMetadata: (videoMetadata) => set({ videoMetadata }),
  setTracklets: (tracklets) => set({ tracklets }),
  setSelectedTrackletIds: (selectedTrackletIds) => set({ selectedTrackletIds }),
  setSelectionMode: (selectionMode) => set({ selectionMode }),
  setColorMode: (colorMode) => set({ colorMode }),
  setHighlightedClusterId: (highlightedClusterId) => set({ highlightedClusterId }),
  setHighlightedTrackletId: (highlightedTrackletId) => set({ highlightedTrackletId }),
  setActiveTabIndex: (activeTabIndex) => set({ activeTabIndex }),
  setLegendFocus: (legendFocus) => set({ legendFocus }),
  setGlobalLegendFocus: (globalLegendFocus) => set({ globalLegendFocus }),

  // ── Actions — view mode ───────────────────────────────────────────────
  setViewMode: (viewMode) =>
    set({
      viewMode,
      // Reset all selection state when switching views
      selectedTrackletIds: new Set(),
      selectionMode: 'none',
      selectedClipIds: new Set(),
      globalSelectionMode: 'none',
      twoPointSelection: null,
      twoPointPending: null,
      legendFocus: null,
      globalLegendFocus: null,
      activeTabIndex: 0,
      activeGlobalTabIndex: 0,
    }),

  // ── Actions — global ──────────────────────────────────────────────────
  setGlobalClips: (globalClips) => set({ globalClips }),
  setSelectedClipIds: (selectedClipIds) => set({ selectedClipIds }),
  setGlobalSelectionMode: (globalSelectionMode) => set({ globalSelectionMode }),
  setGlobalColorMode: (globalColorMode) => set({ globalColorMode }),
  setTwoPointSelection: (twoPointSelection) => set({ twoPointSelection }),
  setTwoPointPending: (twoPointPending) => set({ twoPointPending }),
  setHighlightedGlobalClusterId: (highlightedGlobalClusterId) =>
    set({ highlightedGlobalClusterId }),
  setHighlightedClipId: (highlightedClipId) => set({ highlightedClipId }),
  setActiveGlobalTabIndex: (activeGlobalTabIndex) => set({ activeGlobalTabIndex }),
  setHighlightedSpatialClipIds: (highlightedSpatialClipIds) => set({ highlightedSpatialClipIds }),
}));
