import { useStore } from '../../stores/useStore';
import { useVideos } from '../../hooks/useVideos';

export default function Header() {
  const { selectedVideoId, setSelectedVideoId, setSelectedTrackletIds, setActiveTabIndex } =
    useStore();
  const { data: videos } = useVideos();

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value || null;
    setSelectedVideoId(id);
    setSelectedTrackletIds(new Set());
    setActiveTabIndex(0);
  };

  return (
    <header className="flex items-center gap-4 px-4 h-12 bg-gray-900 border-b border-gray-700 shrink-0">
      <span className="font-bold text-base tracking-tight text-white">TrackletViz</span>
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
