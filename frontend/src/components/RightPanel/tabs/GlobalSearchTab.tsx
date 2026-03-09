import { useState } from "react";
import { searchClips } from "../../../lib/api";
import { useStore } from "../../../stores/useStore";
import type {
  ClipSearchResult,
  GlobalClipMetadata,
} from "../../../types/index";

interface LoopModalProps {
  clip: GlobalClipMetadata;
  streamUrl: string;
  onClose: () => void;
}

function LoopModal({ clip, streamUrl, onClose }: LoopModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-2xl w-[1500px] max-w-[95vw] max-h-[95vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-medium text-gray-200">
            Clip {clip.clip_index} — {clip.start_time.toFixed(1)}s –{" "}
            {clip.end_time.toFixed(1)}s
          </span>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            ✕
          </button>
        </div>
        <video
          src={streamUrl}
          autoPlay
          controls
          className="w-full rounded"
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            v.currentTime = clip.start_time;
          }}
          onTimeUpdate={(e) => {
            const v = e.currentTarget;
            if (v.currentTime >= clip.end_time) v.currentTime = clip.start_time;
          }}
        />
      </div>
    </div>
  );
}

export default function GlobalSearchTab() {
  const { selectedVideoId } = useStore();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ClipSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loopClip, setLoopClip] = useState<GlobalClipMetadata | null>(null);

  const handleSearch = async () => {
    if (!selectedVideoId || !query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await searchClips(selectedVideoId, query.trim());
      setResults(res);
    } catch (e: unknown) {
      setError(String(e));
    }
    setLoading(false);
  };

  const streamUrl = selectedVideoId
    ? `/api/videos/${selectedVideoId}/stream`
    : "";

  return (
    <div className="flex flex-col h-full">
      {/* Search input */}
      <div className="p-3 border-b border-gray-700 shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="e.g., people standing in queue"
            className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !selectedVideoId}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm px-4 py-1.5 rounded transition-colors"
          >
            {loading ? "…" : "Search"}
          </button>
        </div>
        {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-3">
        {results.length === 0 && !loading && (
          <p className="text-gray-500 text-sm text-center mt-8">
            {query ? "No results" : "Enter a query to search clips"}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {results.map(({ clip, score }) => (
            <button
              key={clip.clip_id}
              onClick={() => setLoopClip(clip)}
              className="flex items-center gap-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg p-2 transition-colors text-left"
            >
              {clip.thumbnail_base64 ? (
                <img
                  src={`data:image/jpeg;base64,${clip.thumbnail_base64}`}
                  alt=""
                  className="w-28 h-auto rounded flex-shrink-0"
                />
              ) : (
                <div className="w-28 h-16 bg-gray-700 rounded flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-200">Clip {clip.clip_index}</p>
                <p className="text-xs text-gray-400">
                  {clip.start_time.toFixed(1)}s – {clip.end_time.toFixed(1)}s
                </p>
                <p className="text-xs text-blue-400 mt-1">
                  Score: {(score * 100).toFixed(1)}%
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Loop modal */}
      {loopClip && (
        <LoopModal
          clip={loopClip}
          streamUrl={streamUrl}
          onClose={() => setLoopClip(null)}
        />
      )}
    </div>
  );
}
