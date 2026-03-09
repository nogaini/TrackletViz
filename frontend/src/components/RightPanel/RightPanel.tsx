import { useMemo } from 'react';
import { useStore } from '../../stores/useStore';
import VideoPlayerTab from './tabs/VideoPlayerTab';
import HeatmapTab from './tabs/HeatmapTab';
import TrackListTab from './tabs/TrackListTab';
import ClusterSummariesTab from './tabs/ClusterSummariesTab';
import TextSearchTab from './tabs/TextSearchTab';
import GlobalVideoTab from './tabs/GlobalVideoTab';
import GlobalHeatmapTab from './tabs/GlobalHeatmapTab';
import GlobalClusterTab from './tabs/GlobalClusterTab';
import GlobalSearchTab from './tabs/GlobalSearchTab';

const LOCAL_TABS = ['Video', 'Heatmap', 'Tracks', 'Clusters', 'Search'];
const GLOBAL_TABS = ['Video', 'Heatmap', 'Clusters', 'Search'];

export default function RightPanel() {
  const {
    viewMode,
    activeTabIndex,
    setActiveTabIndex,
    activeGlobalTabIndex,
    setActiveGlobalTabIndex,
    tracklets,
    selectedTrackletIds,
    globalClips,
    selectedClipIds,
  } = useStore();

  const isGlobal = viewMode === 'global';

  const selectedTracklets = useMemo(
    () => tracklets.filter(t => selectedTrackletIds.has(t.tracklet_id)),
    [tracklets, selectedTrackletIds],
  );

  const selectedClips = useMemo(
    () => globalClips.filter(c => selectedClipIds.has(c.clip_id)),
    [globalClips, selectedClipIds],
  );

  if (isGlobal) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex border-b border-gray-700 shrink-0 bg-gray-900">
          {GLOBAL_TABS.map((label, i) => (
            <button
              key={label}
              onClick={() => setActiveGlobalTabIndex(i)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
                activeGlobalTabIndex === i
                  ? 'border-blue-500 text-white'
                  : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex-1 overflow-hidden relative">
          {activeGlobalTabIndex === 0 && <GlobalVideoTab selectedClips={selectedClips} />}
          {activeGlobalTabIndex === 1 && <GlobalHeatmapTab />}
          {activeGlobalTabIndex === 2 && <GlobalClusterTab />}
          {activeGlobalTabIndex === 3 && <GlobalSearchTab />}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex border-b border-gray-700 shrink-0 bg-gray-900">
        {LOCAL_TABS.map((label, i) => (
          <button
            key={label}
            onClick={() => setActiveTabIndex(i)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
              activeTabIndex === i
                ? 'border-blue-500 text-white'
                : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-hidden relative">
        {activeTabIndex === 0 && <VideoPlayerTab selectedTracklets={selectedTracklets} />}
        {activeTabIndex === 1 && <HeatmapTab selectedTracklets={selectedTracklets} />}
        {activeTabIndex === 2 && <TrackListTab selectedTracklets={selectedTracklets} />}
        {activeTabIndex === 3 && <ClusterSummariesTab selectedTracklets={selectedTracklets} />}
        {activeTabIndex === 4 && <TextSearchTab />}
      </div>
    </div>
  );
}
