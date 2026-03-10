import { useStore } from '../../stores/useStore';
import { useVideos } from '../../hooks/useVideos';

export default function Header() {
  const {
    selectedVideoId,
    setSelectedVideoId,
    setSelectedTrackletIds,
    setSelectedClipIds,
    setActiveTabIndex,
    setLegendFocus,
    setGlobalLegendFocus,
    viewMode,
    setViewMode,
  } = useStore();
  const { data: videos } = useVideos();

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value || null;
    setSelectedVideoId(id);
    setSelectedTrackletIds(new Set());
    setSelectedClipIds(new Set());
    setLegendFocus(null);
    setGlobalLegendFocus(null);
    setActiveTabIndex(0);
  };

  return (
    <header className="flex items-center gap-4 px-4 h-12 bg-gray-900 border-b border-gray-700 shrink-0">
      <div className="flex items-center gap-2">
        <img src="/logo.svg" alt="TrackletViz logo" className="h-7 w-7" />
        <span className="font-bold text-base tracking-tight text-white">TrackletViz</span>
      </div>

      <div className="flex items-center gap-0.5 bg-gray-800 rounded-lg p-1 border border-gray-700">
        <button
          onClick={() => setViewMode('local')}
          className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
            viewMode === 'local' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Local
        </button>
        <button
          onClick={() => setViewMode('global')}
          className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
            viewMode === 'global' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Global
        </button>
      </div>

      <select
        className="ml-auto bg-gray-800 text-gray-200 text-sm border border-gray-600 rounded px-2 py-1 focus:outline-none focus:border-blue-500 cursor-pointer"
        value={selectedVideoId ?? ''}
        onChange={handleChange}
      >
        <option value="">— select a video —</option>
        {(videos ?? []).map(v => (
          <option key={v.video_id} value={v.video_id}>
            {v.tag ?? v.video_id} ({v.total_tracklets} tracklets, {v.duration?.toFixed(1)}s)
          </option>
        ))}
      </select>
    </header>
  );
}
