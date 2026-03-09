"""
Pydantic v2 data models for TrackletViz.

All pipeline modules import from here to ensure consistent data structures
across detection, tracking, embedding, clustering, and storage layers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel


# ── Shared low-level models ────────────────────────────────────────────────

class TrajectoryPoint(BaseModel):
    """A single point in a tracked object's trajectory."""
    track_id: int
    class_name: str
    frame_num: int
    timestamp: float          # seconds from video start
    center_x: float
    center_y: float
    bbox_w: float
    bbox_h: float
    speed: Optional[float] = None   # pixels/second; filled in by SpeedEstimator


class ProcessingConfig(BaseModel):
    """Detection and tracking configuration (passed to VideoProcessor)."""
    yolo_model: str = "yolo11n.pt"
    tracker: str = "botsort"
    confidence_threshold: float = 0.3
    target_classes: List[int] = [0, 1, 2, 3, 5, 7]
    min_tracklet_frames: int = 16
    device: str = "cuda"


# ── Bounding box ───────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """A single bounding box observation for a tracklet."""
    frame_num: int
    timestamp: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    speed: Optional[float] = None


# ── Tracklet ───────────────────────────────────────────────────────────────

class TrackletMetadata(BaseModel):
    """Full metadata for a single tracked object (one track across frames)."""
    tracklet_id: str                        # "{video_id}_{track_id}"
    video_id: str
    class_name: str
    class_id: int
    bounding_boxes: List[BoundingBox]
    bbox_centers: List[Tuple[float, float]]  # [(cx, cy), ...] for rendering
    start_timestamp: float
    end_timestamp: float
    duration: float
    avg_speed: float
    max_speed: float
    point_count: int
    thumbnail_base64: Optional[str] = None
    cluster_id: int = -1
    umap_x: float = 0.0
    umap_y: float = 0.0
    start_world_time: Optional[str] = None  # ISO format: "YYYY-MM-DDTHH:MM:SS"
    end_world_time: Optional[str] = None


# ── Cluster ────────────────────────────────────────────────────────────────

class ClusterStatistics(BaseModel):
    """Aggregate statistics for a single HDBSCAN cluster."""
    cluster_id: int
    member_count: int
    avg_speed: float
    class_distribution: Dict[str, float]        # {"person": 60.0, "car": 40.0}
    representative_tracklet_ids: List[str]


# ── Video ──────────────────────────────────────────────────────────────────

class VideoMetadata(BaseModel):
    """Top-level metadata for an indexed video."""
    video_id: str
    video_path: str
    fps: float
    width: int
    height: int
    duration: float
    total_frames: int
    background_image_base64: Optional[str] = None
    cluster_stats: List[ClusterStatistics] = []
    total_tracklets: int = 0
    class_distribution: Dict[str, float] = {}   # percentages across all tracklets
    video_start_time: Optional[str] = None  # ISO format: "YYYY-MM-DDTHH:MM:SS"
    tag: Optional[str] = None               # human-readable label set at index time


# ── Global Clips ────────────────────────────────────────────────────────────

class GlobalClipMetadata(BaseModel):
    """Metadata for a single non-overlapping full-scene video clip."""
    clip_id: str                    # "{video_id}_clip_{index:04d}"
    video_id: str
    clip_index: int                 # 0-based
    start_time: float               # seconds from video start
    end_time: float
    cluster_id: int = -1
    umap_x: float = 0.0
    umap_y: float = 0.0
    thumbnail_base64: Optional[str] = None   # thumbnail_widthxH middle frame, full-scene JPEG
    median_frame_b64: Optional[str] = None   # native-res median frame for heatmaps (reps only)
    optical_flow_b64: Optional[str] = None   # base64-encoded raw float32 bytes (fh,fw,2) (reps only)
    flow_width: int = 0             # native video width at index time
    flow_height: int = 0            # native video height at index time
    tracklet_ids: List[str] = []    # tracklets overlapping [start_time, end_time]
    is_representative: bool = False  # True for FPS-selected cluster representatives


class GlobalClusterStatistics(BaseModel):
    """Aggregate statistics for a single HDBSCAN cluster of global clips."""
    cluster_id: int
    member_count: int
    representative_clip_ids: List[str]   # FPS-selected clip IDs
