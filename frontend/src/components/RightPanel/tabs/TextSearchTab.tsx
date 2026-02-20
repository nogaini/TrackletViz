import { useCallback, useRef, useState } from 'react';
import { useStore } from '../../../stores/useStore';
import { videoStreamUrl, searchText } from '../../../lib/api';
import { BBOX_COLOR } from '../../../lib/colors';
import LazyThumbnail from '../../shared/LazyThumbnail';
import type { SearchResult, TrackletMetadata } from '../../../types/index';

function TrackletModal({ tracklet, onClose }: { tracklet: TrackletMetadata; onClose: () => void }) {
  const { selectedVideoId, videoMetadata } = useStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.currentTime >= tracklet.end_timestamp) {
      video.currentTime = tracklet.start_timestamp;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const boxes = tracklet.bounding_boxes;
    if (boxes.length === 0) return;
    let best = boxes[0];
    let bestDiff = Math.abs(boxes[0].timestamp - video.currentTime);
    for (const b of boxes) {
      const d = Math.abs(b.timestamp - video.currentTime);
      if (d < bestDiff) { bestDiff = d; best = b; }
    }
    const sx = canvas.width / (videoMetadata?.width ?? canvas.width);
    const sy = canvas.height / (videoMetadata?.height ?? canvas.height);
    ctx.strokeStyle = BBOX_COLOR;
    ctx.lineWidth = 2;
    ctx.strokeRect(best.x1 * sx, best.y1 * sy, (best.x2 - best.x1) * sx, (best.y2 - best.y1) * sy);
  };

  const handleLoaded = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = tracklet.start_timestamp;
    void video.play();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-2xl w-[900px] max-w-[95vw] max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
          <span className="text-sm text-gray-200 capitalize">
            {tracklet.class_name} — #{tracklet.tracklet_id}
          </span>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-lg leading-none"
          >
            ×
          </button>
        </div>
        <div className="relative bg-black aspect-video">
          {selectedVideoId && (
            <>
              <video
                ref={videoRef}
                src={videoStreamUrl(selectedVideoId)}
                className="w-full h-full object-contain"
                onTimeUpdate={handleTimeUpdate}
                onLoadedData={handleLoaded}
                muted
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                width={videoMetadata?.width ?? 1280}
                height={videoMetadata?.height ?? 720}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TextSearchTab() {
  const { selectedVideoId, setHighlightedTrackletId } = useStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modal, setModal] = useState<TrackletMetadata | null>(null);

  const handleSearch = async () => {
    if (!selectedVideoId || !query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await searchText(selectedVideoId, query.trim(), 20);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') void handleSearch();
  };

  const handleThumbMouseEnter = useCallback(
    (trackletId: string) => setHighlightedTrackletId(trackletId),
    [setHighlightedTrackletId],
  );
  const handleThumbMouseLeave = useCallback(
    () => setHighlightedTrackletId(null),
    [setHighlightedTrackletId],
  );

  return (
    <>
      <div className="flex flex-col h-full overflow-hidden">
        {/* Search input */}
        <div className="p-3 border-b border-gray-700 shrink-0">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="e.g. person walking fast..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 bg-gray-800 text-gray-200 text-sm border border-gray-600 rounded px-3 py-1.5 focus:outline-none focus:border-blue-500 placeholder-gray-500"
            />
            <button
              onClick={() => void handleSearch()}
              disabled={loading || !selectedVideoId || !query.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm px-4 py-1.5 rounded transition-colors"
            >
              {loading ? '…' : 'Search'}
            </button>
          </div>
          {!selectedVideoId && (
            <p className="text-[10px] text-gray-500 mt-1">Select a video first.</p>
          )}
          {error && <p className="text-[10px] text-red-400 mt-1">{error}</p>}
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto px-3 pb-3 pt-10">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <span className="text-gray-400 text-sm">Searching…</span>
            </div>
          )}
          {!loading && results.length === 0 && query && (
            <p className="text-gray-500 text-sm text-center py-8">No results found.</p>
          )}
          {!loading && results.length === 0 && !query && (
            <p className="text-gray-500 text-sm text-center py-8">
              Enter a description to find matching tracklets.
            </p>
          )}
          <div className="grid grid-cols-10 gap-1">
            {results.map((r, i) => (
              <div key={`${r.tracklet.tracklet_id}-${i}`} className="relative group">
                <button
                  onClick={() => setModal(r.tracklet)}
                  onMouseEnter={() => handleThumbMouseEnter(r.tracklet.tracklet_id)}
                  onMouseLeave={handleThumbMouseLeave}
                  className="w-full bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-500 rounded-lg p-1 text-left transition-colors"
                >
                  <LazyThumbnail
                    trackletId={r.tracklet.tracklet_id}
                    srcOverride={r.tracklet.thumbnail_base64}
                    className="w-full h-auto rounded"
                    alt={r.tracklet.class_name}
                  />
                </button>
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-10 hidden group-hover:block bg-gray-900 border border-gray-600 rounded px-2 py-1 text-[10px] text-gray-200 whitespace-nowrap pointer-events-none shadow-lg">
                  <span className="capitalize">{r.tracklet.class_name}</span>
                  <span className="text-blue-400 font-mono ml-1">{(r.score * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {modal && <TrackletModal tracklet={modal} onClose={() => setModal(null)} />}
    </>
  );
}
