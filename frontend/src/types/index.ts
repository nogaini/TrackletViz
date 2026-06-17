export interface BoundingBox {
  frame_num: number;
  timestamp: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  center_x: number;
  center_y: number;
  width: number;
  height: number;
  speed: number;
}

export interface TrackletMetadata {
  tracklet_id: string;
  video_id: string;
  class_name: string;
  class_id: number;
  bounding_boxes?: BoundingBox[];
  bbox_centers?: [number, number][];
  start_timestamp: number;
  end_timestamp: number;
  duration: number;
  avg_speed: number;
  max_speed: number;
  point_count: number;
  thumbnail_base64?: string;
  cluster_id: number;
  umap_x: number;
  umap_y: number;
}

export interface ClusterStatistics {
  cluster_id: number;
  member_count: number;
  avg_speed: number;
  class_distribution: Record<string, number>;
  representative_tracklet_ids: string[];
  description?: string;
}

export interface VideoMetadata {
  video_id: string;
  video_path: string;
  fps: number;
  width: number;
  height: number;
  duration: number;
  total_frames: number;
  background_image_base64: string;
  cluster_stats: ClusterStatistics[];
  total_tracklets: number;
  class_distribution: Record<string, number>;
  tag?: string;
  cluster_meta_summary?: string;
  global_cluster_meta_summary?: string;
}

export interface VideoSummary {
  video_id: string;
  video_path: string;
  duration: number;
  total_tracklets: number;
  tag?: string;
}

export interface SearchResult {
  tracklet: TrackletMetadata;
  score: number;
}

export interface GlobalClipMetadata {
  clip_id: string;
  video_id: string;
  clip_index: number;
  start_time: number;
  end_time: number;
  cluster_id: number;
  umap_x: number;
  umap_y: number;
  thumbnail_base64?: string;
  median_frame_b64?: string;
  optical_flow_b64?: string;
  flow_width: number;
  flow_height: number;
  tracklet_ids: string[];
  is_representative?: boolean;
}

export interface GlobalClusterStatistics {
  cluster_id: number;
  member_count: number;
  representative_clip_ids: string[];
  description?: string;
}

export interface GlobalClusterStatsResponse {
  clusters: GlobalClusterStatistics[];
  meta_summary?: string;
}

export interface ClipSearchResult {
  clip: GlobalClipMetadata;
  score: number;
}

export interface TwoPointSelection {
  clip1: GlobalClipMetadata;
  clip2: GlobalClipMetadata;
}
