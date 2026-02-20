"""
TrackletViz Indexing Pipeline — entry point.

Usage:
    python indexer/main.py --video /path/to/video.mp4 [--config config/default.yaml]

Run from the backend/ directory (or set PYTHONPATH=backend/).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Ensure the backend package root is on sys.path regardless of CWD
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from indexer.clustering import TrackletClusterer
from indexer.config import load_config
from indexer.detector import VideoProcessor
from indexer.embeddings import TrackletEmbedder
from indexer.speed import SpeedEstimator
from indexer.storage import QdrantStorage
from indexer.thumbnails import ThumbnailGenerator
from indexer.trajectory import TrajectoryExtractor
from models.schemas import BoundingBox, TrackletMetadata, VideoMetadata


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_video_id(video_path: str) -> str:
    """Generate a stable 16-char ID from the file path and modification time."""
    mtime = str(os.path.getmtime(video_path))
    raw = f"{os.path.abspath(video_path)}:{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_tracklet_metadata(
    video_id: str,
    track_id: int,
    trajectory_extractor: TrajectoryExtractor,
    speeds: Dict[int, List[float]],
    umap_coords: Dict[str, Tuple[float, float]],
    cluster_labels: Dict[str, int],
    video_start_dt: Optional[datetime] = None,
) -> TrackletMetadata:
    """Assemble a TrackletMetadata from all extracted data for one track."""
    tracklet_id = f"{video_id}_{track_id}"
    points = trajectory_extractor.get_track_history(track_id)
    track_speeds = speeds.get(track_id, [0.0] * len(points))

    bounding_boxes: List[BoundingBox] = []
    bbox_centers: List[Tuple[float, float]] = []

    for i, pt in enumerate(points):
        x1 = pt.center_x - pt.bbox_w / 2
        y1 = pt.center_y - pt.bbox_h / 2
        x2 = pt.center_x + pt.bbox_w / 2
        y2 = pt.center_y + pt.bbox_h / 2
        spd = track_speeds[i] if i < len(track_speeds) else 0.0

        bounding_boxes.append(
            BoundingBox(
                frame_num=pt.frame_num,
                timestamp=pt.timestamp,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=pt.center_x,
                center_y=pt.center_y,
                width=pt.bbox_w,
                height=pt.bbox_h,
                speed=spd,
            )
        )
        bbox_centers.append((pt.center_x, pt.center_y))

    non_zero = [s for s in track_speeds if s > 0]
    avg_speed = float(np.mean(non_zero)) if non_zero else 0.0
    max_speed = float(np.max(non_zero)) if non_zero else 0.0

    umap_x, umap_y = umap_coords.get(tracklet_id, (0.0, 0.0))
    cluster_id = cluster_labels.get(tracklet_id, -1)

    start_ts = points[0].timestamp
    end_ts = points[-1].timestamp

    start_world_time = None
    end_world_time = None
    if video_start_dt is not None:
        start_world_time = (video_start_dt + timedelta(seconds=start_ts)).strftime("%Y-%m-%dT%H:%M:%S")
        end_world_time = (video_start_dt + timedelta(seconds=end_ts)).strftime("%Y-%m-%dT%H:%M:%S")

    return TrackletMetadata(
        tracklet_id=tracklet_id,
        video_id=video_id,
        class_name=points[0].class_name,
        class_id=0,          # class_id is not stored per-point; resolved below if needed
        bounding_boxes=bounding_boxes,
        bbox_centers=bbox_centers,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        duration=end_ts - start_ts,
        avg_speed=avg_speed,
        max_speed=max_speed,
        point_count=len(points),
        umap_x=umap_x,
        umap_y=umap_y,
        cluster_id=cluster_id,
        start_world_time=start_world_time,
        end_world_time=end_world_time,
    )


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_pipeline(video_path: str, config_path: str, start_time: Optional[str] = None, tag: Optional[str] = None):
    logger.info("=" * 60)
    logger.info(f"TrackletViz Indexing Pipeline")
    logger.info(f"  video : {video_path}")
    logger.info(f"  config: {config_path}")
    if start_time:
        logger.info(f"  start-time: {start_time}")
    if tag:
        logger.info(f"  tag: {tag}")
    logger.info("=" * 60)

    video_start_dt: Optional[datetime] = None
    if start_time:
        video_start_dt = datetime.strptime(start_time, "%Y%m%dT%H%M%S")

    # ── Step 1: Load config ────────────────────────────────────────────────
    logger.info("[1/13] Loading configuration")
    config = load_config(config_path)

    # ── Step 2: Generate video_id ──────────────────────────────────────────
    logger.info("[2/13] Generating video_id")
    video_id = _make_video_id(video_path)
    logger.info(f"  video_id = {video_id}")

    # ── Step 3: Video metadata + background frame ──────────────────────────
    logger.info("[3/13] Extracting video metadata and background frame")
    processor = VideoProcessor(config.processing)
    processor.setup()

    meta_raw = processor.get_video_metadata(video_path)
    fps = meta_raw["fps"]
    background_frame = processor.extract_background_frame(video_path)

    # ── Step 4: Detection + tracking ──────────────────────────────────────
    logger.info("[4/13] Running detection and tracking")
    extractor = TrajectoryExtractor(fps=fps, class_names=processor.class_names)

    for frame_num, frame, _yolo_results, tracks in processor.process_video(video_path):
        extractor.extract_from_tracks(frame_num, tracks)
        if frame_num % 100 == 0:
            logger.info(f"  Processed frame {frame_num} …")

    stats = extractor.get_statistics()
    logger.info(f"  Tracks found: {stats['unique_tracks']}  |  Points: {stats['total_points']}")

    # ── Step 5: Filter tracklets ───────────────────────────────────────────
    logger.info("[5/13] Filtering short tracklets")
    valid_track_ids = extractor.get_filtered_track_ids(config.processing.min_tracklet_frames)
    if not valid_track_ids:
        logger.warning("No valid tracklets found after filtering. Exiting.")
        return

    # ── Step 6: Speed calculation ──────────────────────────────────────────
    logger.info("[6/13] Calculating speeds")
    estimator = SpeedEstimator(fps=fps)
    track_history = {
        tid: extractor.get_track_history(tid) for tid in valid_track_ids
    }
    speeds_raw = estimator.calculate_speeds(track_history)
    speed_stats = estimator.get_speed_statistics(speeds_raw)
    max_speed = speed_stats["max_speed"] * 2 if speed_stats["max_speed"] > 0 else 1e6
    speeds = estimator.filter_outlier_speeds(speeds_raw, max_speed)
    estimator.apply_speeds(extractor.get_trajectories(), speeds)
    logger.info(
        f"  Avg speed: {speed_stats['avg_speed']:.1f}  "
        f"Max speed: {speed_stats['max_speed']:.1f} px/s"
    )

    # ── Step 7: Build tracklet info for embedding extraction ───────────────
    logger.info("[7/13] Building tracklet clips for embedding")
    tracklet_infos = []
    for tid in valid_track_ids:
        tracklet_id = f"{video_id}_{tid}"
        points = extractor.get_track_history(tid)
        frame_nums = [p.frame_num for p in points]
        bboxes = [
            (
                p.center_x - p.bbox_w / 2,
                p.center_y - p.bbox_h / 2,
                p.center_x + p.bbox_w / 2,
                p.center_y + p.bbox_h / 2,
            )
            for p in points
        ]
        tracklet_infos.append(
            {"tracklet_id": tracklet_id, "frame_nums": frame_nums, "bboxes": bboxes}
        )

    # ── Step 8: Embed tracklets ────────────────────────────────────────────
    logger.info("[8/13] Extracting VideoPrism embeddings")
    # Free PyTorch GPU memory before JAX takes over
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("  PyTorch GPU cache cleared before embedding")
    embedder = TrackletEmbedder(config.videoprism)
    embedder.setup()
    embeddings = embedder.embed_tracklets(video_path, tracklet_infos)

    # ── Step 9: UMAP + HDBSCAN clustering ─────────────────────────────────
    logger.info("[9/13] Running UMAP + HDBSCAN clustering")
    clusterer = TrackletClusterer(config.clustering)
    umap_coords, cluster_labels = clusterer.fit(embeddings)

    # ── Step 10: Thumbnails ────────────────────────────────────────────────
    logger.info("[10/13] Generating thumbnails")
    thumb_gen = ThumbnailGenerator(config.thumbnails)

    thumb_requests = []
    for tid in valid_track_ids:
        tracklet_id = f"{video_id}_{tid}"
        points = extractor.get_track_history(tid)
        mid = len(points) // 2
        mid_pt = points[mid]
        thumb_requests.append(
            {
                "tracklet_id": tracklet_id,
                "frame_num": mid_pt.frame_num,
                "bbox": (
                    mid_pt.center_x - mid_pt.bbox_w / 2,
                    mid_pt.center_y - mid_pt.bbox_h / 2,
                    mid_pt.center_x + mid_pt.bbox_w / 2,
                    mid_pt.center_y + mid_pt.bbox_h / 2,
                ),
            }
        )

    thumbnails = thumb_gen.generate_thumbnails_batch(video_path, thumb_requests)
    background_b64 = thumb_gen.generate_background(background_frame)

    # ── Step 11: Build TrackletMetadata objects ────────────────────────────
    logger.info("[11/13] Building TrackletMetadata objects")
    tracklets: List[TrackletMetadata] = []
    class_name_to_id = {v: k for k, v in processor.class_names.items()}

    for tid in valid_track_ids:
        tm = _build_tracklet_metadata(
            video_id, tid, extractor, speeds, umap_coords, cluster_labels,
            video_start_dt=video_start_dt,
        )
        tm.thumbnail_base64 = thumbnails.get(tm.tracklet_id)
        # Resolve class_id from class_name
        tm.class_id = class_name_to_id.get(tm.class_name, -1)
        tracklets.append(tm)

    # ── Step 12: Cluster statistics ────────────────────────────────────────
    logger.info("[12/13] Computing cluster statistics")
    cluster_stats = clusterer.compute_cluster_stats(tracklets, cluster_labels, embeddings)

    # Video-level class distribution
    class_counts: Dict[str, int] = defaultdict(int)
    for t in tracklets:
        class_counts[t.class_name] += 1
    total_t = sum(s.member_count for s in cluster_stats)
    class_dist = {
        cls: round(cnt / total_t * 100.0, 1) for cls, cnt in class_counts.items()
    }

    video_meta = VideoMetadata(
        video_id=video_id,
        video_path=os.path.abspath(video_path),
        fps=fps,
        width=meta_raw["width"],
        height=meta_raw["height"],
        duration=meta_raw["duration"],
        total_frames=meta_raw["total_frames"],
        background_image_base64=background_b64,
        cluster_stats=cluster_stats,
        total_tracklets=total_t,
        class_distribution=class_dist,
        video_start_time=video_start_dt.strftime("%Y-%m-%dT%H:%M:%S") if video_start_dt else None,
        tag=tag,
    )

    # ── Step 13: Store in Qdrant ───────────────────────────────────────────
    logger.info("[13/13] Storing in Qdrant")
    try:
        storage = QdrantStorage(config.qdrant)
        storage.setup_collections()
        storage.upsert_video(video_id, tracklets, video_meta, embeddings)
    except Exception as exc:
        logger.error(
            f"Failed to connect to Qdrant at "
            f"{config.qdrant.host}:{config.qdrant.port} — {exc}\n"
            "Make sure Qdrant is running (e.g. docker run -p 6333:6333 qdrant/qdrant)"
        )
        raise

    logger.success("=" * 60)
    logger.success("Indexing complete!")
    logger.success(f"  video_id      : {video_id}")
    logger.success(f"  tracklets     : {total_t}")
    logger.success(f"  clusters      : {len([s for s in cluster_stats if s.cluster_id >= 0])}")
    logger.success(f"  class dist    : {class_dist}")
    logger.success("=" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TrackletViz: Index a video into Qdrant"
    )
    parser.add_argument("--video", required=False, default=None, help="Path to input video file")
    parser.add_argument(
        "--delete-video",
        type=str,
        default=None,
        metavar="VIDEO_ID",
        help="Delete all Qdrant data for VIDEO_ID and exit (does not re-index)",
    )
    parser.add_argument(
        "--delete-tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Delete all Qdrant data for the video with this tag and exit",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_BACKEND_DIR, "config", "default.yaml"),
        help="Path to YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="World start time of the video (format: YYYYMMDDTHHMMSS, e.g. 20240315T143022)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Human-readable label for this video (shown in the UI dropdown instead of video_id)",
    )
    args = parser.parse_args()

    if args.delete_video:
        config = load_config(args.config)
        storage = QdrantStorage(config.qdrant)
        storage.delete_video(args.delete_video)
        sys.exit(0)

    if args.delete_tag:
        config = load_config(args.config)
        storage = QdrantStorage(config.qdrant)
        video_id = storage.get_video_id_by_tag(args.delete_tag)
        if video_id is None:
            logger.error(f"No video found with tag='{args.delete_tag}'")
            sys.exit(1)
        storage.delete_video(video_id)
        sys.exit(0)

    if not args.video:
        parser.error("one of --video, --delete-video, or --delete-tag is required")

    if not os.path.isfile(args.video):
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if args.start_time:
        try:
            datetime.strptime(args.start_time, "%Y%m%dT%H%M%S")
        except ValueError:
            logger.error(f"Invalid --start-time format: '{args.start_time}'. Expected YYYYMMDDTHHMMSS.")
            sys.exit(1)

    run_pipeline(args.video, args.config, start_time=args.start_time, tag=args.tag)


if __name__ == "__main__":
    main()
