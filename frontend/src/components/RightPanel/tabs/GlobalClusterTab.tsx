import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useStore } from '../../../stores/useStore';
import { fetchGlobalClusterStats } from '../../../lib/api';
import { getClusterColorHex } from '../../../lib/colors';
import type { GlobalClipMetadata, GlobalClusterStatistics } from '../../../types/index';

// Loop modal for playing a clip segment
interface LoopModalProps {
  clip: GlobalClipMetadata;
  streamUrl: string;
  onClose: () => void;
}

function LoopModal({ clip, streamUrl, onClose }: LoopModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-2xl w-[1500px] max-w-[95vw] max-h-[95vh] overflow-y-auto p-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-medium text-gray-200">
            Clip {clip.clip_index} — {clip.start_time.toFixed(1)}s – {clip.end_time.toFixed(1)}s
          </span>
          <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
        </div>
        <video
          src={streamUrl}
          autoPlay
          loop
          controls
          className="w-full rounded"
          onLoadedMetadata={e => {
            const v = e.currentTarget;
            v.currentTime = clip.start_time;
          }}
          onTimeUpdate={e => {
            const v = e.currentTarget;
            if (v.currentTime >= clip.end_time) v.currentTime = clip.start_time;
          }}
        />
      </div>
    </div>
  );
}

export default function GlobalClusterTab() {
  const {
    selectedVideoId,
    globalClips,
    tracklets,
    setHighlightedGlobalClusterId,
    setHighlightedClipId,
  } = useStore();

  const [loopClip, setLoopClip] = useState<GlobalClipMetadata | null>(null);

  const { data: statsResponse, isLoading } = useQuery({
    queryKey: ['global-cluster-stats', selectedVideoId],
    queryFn: () => fetchGlobalClusterStats(selectedVideoId!),
    enabled: !!selectedVideoId,
    staleTime: Infinity,
  });

  const clusterStats: GlobalClusterStatistics[] = statsResponse?.clusters ?? [];
  const metaSummary = statsResponse?.meta_summary;

  // Build clip map for fast lookup
  const clipMap = new Map(globalClips.map(c => [c.clip_id, c]));

  // Build tracklet map for per-cluster stats derivation
  const trackletMap = new Map(tracklets.map(t => [t.tracklet_id, t]));

  // Collect all tracklet IDs per cluster from globalClips
  const clusterTrackletSets = new Map<number, Set<string>>();
  for (const clip of globalClips) {
    const cid = clip.cluster_id ?? -1;
    if (!clusterTrackletSets.has(cid)) clusterTrackletSets.set(cid, new Set());
    for (const tid of clip.tracklet_ids ?? []) {
      clusterTrackletSets.get(cid)!.add(tid);
    }
  }

  const streamUrl = selectedVideoId
    ? `/api/videos/${selectedVideoId}/stream`
    : '';

  if (!selectedVideoId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Select a video to see global clusters
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Loading cluster stats…
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto p-3 gap-3">
      {metaSummary && (
        <div className="p-2 rounded bg-gray-800 border border-gray-700 text-xs text-gray-300 italic">
          {metaSummary}
        </div>
      )}
      {clusterStats.map(stat => {
        const borderColor = getClusterColorHex(stat.cluster_id);
        const label = stat.cluster_id < 0 ? 'Noise' : `Cluster ${stat.cluster_id}`;

        // Derive avg speed and class distribution from tracklets in this cluster
        const tids = clusterTrackletSets.get(stat.cluster_id) ?? new Set<string>();
        const clusterTracklets = Array.from(tids)
          .map(tid => trackletMap.get(tid))
          .filter(Boolean) as typeof tracklets;

        const avgSpeed = clusterTracklets.length > 0
          ? clusterTracklets.reduce((s, t) => s + t.avg_speed, 0) / clusterTracklets.length
          : 0;

        const classCounts: Record<string, number> = {};
        for (const t of clusterTracklets) {
          classCounts[t.class_name] = (classCounts[t.class_name] ?? 0) + 1;
        }
        const total = clusterTracklets.length || 1;
        const classDist: Record<string, number> = {};
        for (const [cls, cnt] of Object.entries(classCounts)) {
          classDist[cls] = (cnt / total) * 100;
        }

        const repClips = stat.representative_clip_ids
          .map(cid => clipMap.get(cid))
          .filter(Boolean) as GlobalClipMetadata[];

        return (
          <div
            key={stat.cluster_id}
            className="bg-gray-800 rounded-lg p-3 border-l-4 transition-all hover:bg-gray-750"
            style={{ borderLeftColor: borderColor }}
            onMouseEnter={() => setHighlightedGlobalClusterId(stat.cluster_id)}
            onMouseLeave={() => setHighlightedGlobalClusterId(null)}
          >
            {/* Header */}
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium text-white">{label}</span>
              <span className="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">
                {stat.member_count} clips
              </span>
            </div>

            {/* Stats */}
            <div className="flex gap-4 text-[11px] text-gray-400 mb-2">
              <span>
                Avg speed: <span className="text-gray-200">{avgSpeed.toFixed(1)} px/s</span>
              </span>
            </div>

            {/* Class distribution */}
            {Object.keys(classDist).length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {Object.entries(classDist).map(([cls, pct]) => (
                  <span
                    key={cls}
                    className="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full capitalize"
                  >
                    {cls}: {pct.toFixed(0)}%
                  </span>
                ))}
              </div>
            )}

            {/* MLLM description */}
            {stat.description && (
              <p className="text-[11px] text-gray-400 italic mb-2">{stat.description}</p>
            )}

            {/* Representative thumbnails */}
            {repClips.length > 0 && (
              <div>
                <p className="text-[10px] text-gray-500 mb-1">
                  Showing {repClips.length} representatives
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {repClips.map(clip => (
                    <button
                      key={clip.clip_id}
                      className="border border-gray-600 hover:border-gray-400 rounded overflow-hidden transition-colors"
                      onMouseEnter={() => setHighlightedClipId(clip.clip_id)}
                      onMouseLeave={() => setHighlightedClipId(null)}
                      onClick={() => setLoopClip(clip)}
                      title={`Clip ${clip.clip_index} — click to loop`}
                    >
                      {clip.thumbnail_base64 ? (
                        <img
                          src={`data:image/jpeg;base64,${clip.thumbnail_base64}`}
                          alt=""
                          className="w-24 h-auto"
                        />
                      ) : (
                        <div className="w-24 h-14 bg-gray-700 flex items-center justify-center">
                          <span className="text-[10px] text-gray-500">Clip {clip.clip_index}</span>
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}

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
