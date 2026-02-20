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
  bounding_boxes: BoundingBox[];
  bbox_centers: [number, number][];
  start_timestamp: number;
  end_timestamp: number;
  duration: number;
  avg_speed: number;
  max_speed: number;
  point_count: number;
  thumbnail_base64: string;
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
