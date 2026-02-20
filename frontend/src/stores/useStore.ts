import { create } from 'zustand';
import type { VideoSummary, VideoMetadata, TrackletMetadata } from '../types/index';

interface AppState {
  // Data
  videos: VideoSummary[];
  selectedVideoId: string | null;
  videoMetadata: VideoMetadata | null;
  tracklets: TrackletMetadata[];
  // Selection
  selectedTrackletIds: Set<string>;
  selectionMode: 'lasso' | 'rect' | 'none';
  // Display
  colorMode: 'class' | 'cluster' | 'time';
  highlightedClusterId: number | null;
  highlightedTrackletId: string | null;
  activeTabIndex: number;
  // Actions
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
}

export const useStore = create<AppState>((set) => ({
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
}));
