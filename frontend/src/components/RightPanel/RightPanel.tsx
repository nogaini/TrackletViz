import { useMemo } from 'react';
import { useStore } from '../../stores/useStore';
import VideoPlayerTab from './tabs/VideoPlayerTab';
import HeatmapTab from './tabs/HeatmapTab';
import TrackListTab from './tabs/TrackListTab';
import ClusterSummariesTab from './tabs/ClusterSummariesTab';
import TextSearchTab from './tabs/TextSearchTab';

const TAB_LABELS = ['Video', 'Heatmap', 'Tracks', 'Clusters', 'Search'];

export default function RightPanel() {
  const { activeTabIndex, setActiveTabIndex, tracklets, selectedTrackletIds } = useStore();

  const selectedTracklets = useMemo(
    () => tracklets.filter(t => selectedTrackletIds.has(t.tracklet_id)),
    [tracklets, selectedTrackletIds],
  );

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-gray-700 shrink-0 bg-gray-900">
        {TAB_LABELS.map((label, i) => (
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

      {/* Tab content */}
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
