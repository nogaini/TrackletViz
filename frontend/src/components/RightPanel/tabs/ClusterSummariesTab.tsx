import { useCallback, useRef, useState } from "react";
import { videoStreamUrl } from "../../../lib/api";
import { getClusterColorHex, BBOX_COLOR, speedToColor } from "../../../lib/colors";
import { useStore } from "../../../stores/useStore";
import type { TrackletMetadata } from "../../../types/index";

interface Props {
  selectedTracklets: TrackletMetadata[];
}

interface ModalState {
  tracklet: TrackletMetadata;
}

function TrackletModal({
  tracklet,
  onClose,
}: {
  tracklet: TrackletMetadata;
  onClose: () => void;
}) {
  const { selectedVideoId, videoMetadata } = useStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.currentTime >= tracklet.end_timestamp) {
      video.currentTime = tracklet.start_timestamp;
    }
    // Draw bbox
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const boxes = tracklet.bounding_boxes;
    if (boxes.length === 0) return;
    let best = boxes[0];
    let bestDiff = Math.abs(boxes[0].timestamp - video.currentTime);
    for (const b of boxes) {
      const d = Math.abs(b.timestamp - video.currentTime);
      if (d < bestDiff) {
        bestDiff = d;
        best = b;
      }
    }
    const sx = canvas.width / (videoMetadata?.width ?? canvas.width);
    const sy = canvas.height / (videoMetadata?.height ?? canvas.height);
    ctx.strokeStyle = BBOX_COLOR;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      best.x1 * sx,
      best.y1 * sy,
      (best.x2 - best.x1) * sx,
      (best.y2 - best.y1) * sy,
    );

    // Draw speed-colored track line
    const centers = tracklet.bbox_centers;
    const speeds = tracklet.bounding_boxes.map((b) => b.speed);
    const maxSpeed = Math.max(...speeds, 1);
    for (let i = 1; i < centers.length; i++) {
      const [r, g, b] = speedToColor(speeds[i], maxSpeed);
      ctx.strokeStyle = `rgba(${r},${g},${b},0.75)`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(centers[i - 1][0] * sx, centers[i - 1][1] * sy);
      ctx.lineTo(centers[i][0] * sx, centers[i][1] * sy);
      ctx.stroke();
    }
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
        onClick={(e) => e.stopPropagation()}
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

export default function ClusterSummariesTab({
  selectedTracklets: _selectedTracklets,
}: Props) {
  const {
    videoMetadata,
    setHighlightedClusterId,
    setHighlightedTrackletId,
    tracklets,
  } = useStore();
  const [modal, setModal] = useState<ModalState | null>(null);
  const [tooltip, setTooltip] = useState<{
    tracklet: TrackletMetadata;
    x: number;
    y: number;
  } | null>(null);

  const handleCardMouseEnter = useCallback(
    (clusterId: number) => setHighlightedClusterId(clusterId),
    [setHighlightedClusterId],
  );
  const handleCardMouseLeave = useCallback(
    () => setHighlightedClusterId(null),
    [setHighlightedClusterId],
  );
  const handleThumbMouseEnter = useCallback(
    (trackletId: string) => setHighlightedTrackletId(trackletId),
    [setHighlightedTrackletId],
  );
  const handleThumbMouseLeave = useCallback(
    () => setHighlightedTrackletId(null),
    [setHighlightedTrackletId],
  );

  const clusterStats = videoMetadata?.cluster_stats ?? [];

  if (clusterStats.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">No cluster data available.</p>
      </div>
    );
  }

  return (
    <>
      <div className="h-full overflow-y-auto p-3 space-y-3">
        {clusterStats.map((cluster) => {
          const repTracklets = cluster.representative_tracklet_ids
            .map((id) => tracklets.find((t) => t.tracklet_id === id))
            .filter((t): t is TrackletMetadata => t !== undefined);

          const borderColor = getClusterColorHex(cluster.cluster_id);
          const label =
            cluster.cluster_id < 0 ? "Noise" : `Cluster ${cluster.cluster_id}`;

          return (
            <div
              key={cluster.cluster_id}
              className="bg-gray-800 rounded-lg p-3 border-l-4 transition-all hover:bg-gray-750"
              style={{ borderLeftColor: borderColor }}
              onMouseEnter={() => handleCardMouseEnter(cluster.cluster_id)}
              onMouseLeave={handleCardMouseLeave}
            >
              {/* Header */}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-white">{label}</span>
                <span className="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">
                  {cluster.member_count} tracklets
                </span>
              </div>

              {/* Stats */}
              <div className="flex gap-4 text-[11px] text-gray-400 mb-2">
                <span>
                  Avg speed:{" "}
                  <span className="text-gray-200">
                    {cluster.avg_speed.toFixed(1)} px/s
                  </span>
                </span>
              </div>

              {/* Class distribution */}
              <div className="flex flex-wrap gap-1 mb-3">
                {Object.entries(cluster.class_distribution).map(
                  ([cls, pct]) => (
                    <span
                      key={cls}
                      className="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full capitalize"
                    >
                      {cls}: {pct.toFixed(0)}%
                    </span>
                  ),
                )}
              </div>

              {/* Representative thumbnails */}
              {repTracklets.length > 0 && (
                <div>
                  <p className="text-[10px] text-gray-500 mb-1">
                    Showing {repTracklets.length} representatives
                  </p>
                  <div className="flex gap-1.5 flex-wrap">
                    {repTracklets.map((t) => (
                      <button
                        key={t.tracklet_id}
                        className="relative rounded overflow-hidden border border-gray-600 hover:border-blue-500 transition-colors"
                        onMouseEnter={(e) => {
                          handleThumbMouseEnter(t.tracklet_id);
                          setTooltip({ tracklet: t, x: e.clientX, y: e.clientY });
                        }}
                        onMouseMove={(e) => {
                          setTooltip((prev) =>
                            prev ? { ...prev, x: e.clientX, y: e.clientY } : prev,
                          );
                        }}
                        onMouseLeave={() => {
                          handleThumbMouseLeave();
                          setTooltip(null);
                        }}
                        onClick={() => setModal({ tracklet: t })}
                      >
                        {t.thumbnail_base64 ? (
                          <img
                            src={`data:image/jpeg;base64,${t.thumbnail_base64}`}
                            alt={t.class_name}
                            className="w-14 h-14 object-cover"
                          />
                        ) : (
                          <div className="w-14 h-14 bg-gray-700 flex items-center justify-center">
                            <span className="text-[10px] text-gray-400">
                              No img
                            </span>
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
      </div>

      {modal && (
        <TrackletModal
          tracklet={modal.tracklet}
          onClose={() => setModal(null)}
        />
      )}

      {tooltip && (
        <div
          className="fixed z-50 bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-xs text-gray-200 shadow-xl pointer-events-none space-y-0.5"
          style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}
        >
          <div className="font-medium capitalize">{tooltip.tracklet.class_name}</div>
          <div className="text-gray-400">
            Avg speed:{" "}
            <span className="text-gray-200">
              {tooltip.tracklet.avg_speed.toFixed(1)} px/s
            </span>
          </div>
          <div className="text-gray-400">
            Duration:{" "}
            <span className="text-gray-200">
              {tooltip.tracklet.duration.toFixed(1)} s
            </span>
          </div>
          <div className="text-gray-400">
            Frames:{" "}
            <span className="text-gray-200">{tooltip.tracklet.point_count}</span>
          </div>
        </div>
      )}
    </>
  );
}
