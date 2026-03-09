"""
TrackletViz Indexing Pipeline — entry point.

Usage:
    python indexer/main.py --video /path/to/video.mp4 [--config config/default.yaml]
    python indexer/main.py --video /path/to/video.mp4 --force   # clear checkpoints

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

import cv2
import numpy as np
from loguru import logger

# Ensure the backend package root is on sys.path regardless of CWD
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from indexer.checkpoint import CheckpointManager
from indexer.clustering import TrackletClusterer
from indexer.config import ClusteringConfig, load_config
from indexer.detector import VideoProcessor
from indexer.embeddings import TrackletEmbedder
from indexer.global_clips import GlobalClipProcessor
from indexer.speed import SpeedEstimator
from indexer.storage import QdrantStorage
from indexer.thumbnails import ThumbnailGenerator
from indexer.trajectory import TrajectoryExtractor
from models.schemas import BoundingBox, GlobalClipMetadata, TrackletMetadata, VideoMetadata


_RECOMPUTE_OPTIONS = [
    "optical-flow", "median-frames", "clip-thumbnails",
    "tracklet-thumbnails", "local-embeddings", "global-embeddings",
]


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

def run_pipeline(
    video_path: str,
    config_path: str,
    start_time: Optional[str] = None,
    tag: Optional[str] = None,
    force: bool = False,
):
    logger.info("=" * 60)
    logger.info(f"TrackletViz Indexing Pipeline")
    logger.info(f"  video : {video_path}")
    logger.info(f"  config: {config_path}")
    if start_time:
        logger.info(f"  start-time: {start_time}")
    if tag:
        logger.info(f"  tag: {tag}")
    if force:
        logger.info(f"  --force: will clear checkpoints and rerun from scratch")
    logger.info("=" * 60)

    video_start_dt: Optional[datetime] = None
    if start_time:
        video_start_dt = datetime.strptime(start_time, "%Y%m%dT%H%M%S")

    # ── Step 1: Load config ────────────────────────────────────────────────
    logger.info("[1/21] Loading configuration")
    config = load_config(config_path)

    # ── Step 2: Generate video_id ──────────────────────────────────────────
    logger.info("[2/21] Generating video_id")
    video_id = _make_video_id(video_path)
    logger.info(f"  video_id = {video_id}")

    # ── Initialize CheckpointManager ──────────────────────────────────────
    cm = CheckpointManager(cache_dir=config.cache_dir, video_id=video_id)
    if force:
        logger.info("  --force: clearing all checkpoints")
        cm.clear()

    # ── Qdrant pre-check: sync checkpoint flags from existing indexed data ─
    # Runs even if no local checkpoint files exist (e.g. different machine,
    # cache cleared) so that already-indexed videos are never recomputed.
    if not force:
        try:
            _storage_probe = QdrantStorage(config.qdrant)
            _storage_probe.setup_collections()

            if not cm.is_done("tracklets_indexed"):
                _vm = _storage_probe.get_video_metadata(video_id)
                if _vm is not None:
                    logger.info("  Found existing tracklets in Qdrant — syncing checkpoint flags")
                    if not cm.has("tracklet_metadata"):
                        _raw_tracklets = _storage_probe.get_tracklets_for_video(
                            video_id, include_thumbnails=False
                        )
                        _tracklets_from_qdrant = [
                            TrackletMetadata.model_validate(t) for t in _raw_tracklets
                        ]
                        _meta_raw_from_qdrant = {
                            "fps": _vm.get("fps", 0.0),
                            "width": _vm.get("width", 0),
                            "height": _vm.get("height", 0),
                            "duration": _vm.get("duration", 0.0),
                            "total_frames": _vm.get("total_frames", 0),
                        }
                        cm.save("tracklet_metadata", {
                            "tracklets": _tracklets_from_qdrant,
                            "video_meta": VideoMetadata.model_validate(_vm),
                            "cluster_stats": VideoMetadata.model_validate(_vm).cluster_stats,
                            "class_dist": _vm.get("class_distribution", {}),
                            "meta_raw": _meta_raw_from_qdrant,
                            "fps": _vm.get("fps", 0.0),
                        })
                    cm.mark_done("tracklets_indexed")

            if not cm.is_done("clips_indexed"):
                if _storage_probe.video_clips_indexed(video_id):
                    logger.info("  Found existing global clips in Qdrant — syncing checkpoint flag")
                    cm.mark_done("clips_indexed")

        except Exception as _e:
            logger.debug(f"  Qdrant pre-check skipped: {_e}")

    # ── Initialize pipeline-scope variables ───────────────────────────────
    # These are populated by whichever checkpoint branch runs below.
    meta_raw: dict = {}
    fps: float = 0.0
    valid_track_ids: list = []
    tracklet_infos: list = []
    speeds: dict = {}
    background_frame = None
    class_name_to_id: dict = {}
    extractor: Optional[TrajectoryExtractor] = None
    embeddings: dict = {}
    embedder: Optional[TrackletEmbedder] = None
    tracklets: List[TrackletMetadata] = []
    video_meta: Optional[VideoMetadata] = None
    cluster_stats: list = []
    class_dist: dict = {}
    storage: Optional[QdrantStorage] = None
    clip_infos: list = []
    gc_processor: Optional[GlobalClipProcessor] = None
    clip_embeddings: dict = {}
    unique_gc_clusters: set = set()
    global_clips_meta: List[GlobalClipMetadata] = []

    # ── Steps 3-13: Tracking, embedding, clustering, storing tracklets ─────

    if cm.is_done("tracklets_indexed"):
        logger.info("[3-13/21] Tracklets already indexed — loading from checkpoint")
        td2 = cm.load("tracklet_metadata")
        tracklets = td2["tracklets"]
        video_meta = td2["video_meta"]
        cluster_stats = td2["cluster_stats"]
        class_dist = td2["class_dist"]
        meta_raw = td2["meta_raw"]
        fps = meta_raw["fps"]
        storage = QdrantStorage(config.qdrant)
        storage.setup_collections()

    else:
        # ── Steps 3-7: Detection, tracking, speed ─────────────────────────

        if cm.has("tracking"):
            logger.info("[3-7/21] Loaded tracking checkpoint")
            td = cm.load("tracking")
            tracklet_infos   = td["tracklet_infos"]
            valid_track_ids  = td["valid_track_ids"]
            speeds           = td["speeds"]
            meta_raw         = td["meta_raw"]
            fps              = td["fps"]
            background_frame = td["background_frame"]
            class_name_to_id = td["class_name_to_id"]
            # Reconstruct extractor so _build_tracklet_metadata can call get_track_history()
            extractor = TrajectoryExtractor(fps=fps, class_names={})
            for tid, pts in td["trajectories"].items():
                extractor.track_history[tid] = pts

        else:
            # ── Step 3: Video metadata + background frame ──────────────────
            logger.info("[3/21] Extracting video metadata and background frame")
            processor = VideoProcessor(config.processing)
            processor.setup()

            meta_raw = processor.get_video_metadata(video_path)
            fps = meta_raw["fps"]
            background_frame = processor.extract_background_frame(video_path)

            # ── Step 4: Detection + tracking ──────────────────────────────
            logger.info("[4/21] Running detection and tracking")
            extractor = TrajectoryExtractor(fps=fps, class_names=processor.class_names)

            for frame_num, frame, _yolo_results, tracks in processor.process_video(video_path):
                extractor.extract_from_tracks(frame_num, tracks)
                if frame_num % 100 == 0:
                    logger.info(f"  Processed frame {frame_num} …")

            stats = extractor.get_statistics()
            logger.info(f"  Tracks found: {stats['unique_tracks']}  |  Points: {stats['total_points']}")

            # ── Step 5: Filter tracklets ───────────────────────────────────
            logger.info("[5/21] Filtering short tracklets")
            valid_track_ids = extractor.get_filtered_track_ids(config.processing.min_tracklet_frames)
            if not valid_track_ids:
                logger.warning("No valid tracklets found after filtering. Exiting.")
                return

            # ── Step 6: Speed calculation ──────────────────────────────────
            logger.info("[6/21] Calculating speeds")
            estimator = SpeedEstimator(fps=fps)
            track_history = {
                tid: extractor.get_track_history(tid) for tid in valid_track_ids
            }
            speeds_raw = estimator.calculate_speeds(track_history)
            speed_stats = estimator.get_speed_statistics(speeds_raw)
            max_spd = speed_stats["max_speed"] * 2 if speed_stats["max_speed"] > 0 else 1e6
            speeds = estimator.filter_outlier_speeds(speeds_raw, max_spd)
            estimator.apply_speeds(extractor.get_trajectories(), speeds)
            logger.info(
                f"  Avg speed: {speed_stats['avg_speed']:.1f}  "
                f"Max speed: {speed_stats['max_speed']:.1f} px/s"
            )

            # ── Step 7: Build tracklet info for embedding extraction ───────
            logger.info("[7/21] Building tracklet clips for embedding")
            class_name_to_id = {v: k for k, v in processor.class_names.items()}
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

            cm.save("tracking", {
                "tracklet_infos":   tracklet_infos,
                "valid_track_ids":  valid_track_ids,
                "speeds":           speeds,
                "trajectories":     {tid: extractor.get_track_history(tid) for tid in valid_track_ids},
                "meta_raw":         meta_raw,
                "fps":              fps,
                "background_frame": background_frame,
                "class_name_to_id": class_name_to_id,
            })

        # ── Step 8: Embed tracklets ────────────────────────────────────────

        if cm.has("embeddings"):
            logger.info("[8/21] Loaded embeddings checkpoint")
            embeddings = cm.load("embeddings")
            # embedder left as None; lazy-initialized before steps 16-17 if needed

        else:
            logger.info("[8/21] Extracting VideoPrism embeddings")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  PyTorch GPU cache cleared before embedding")
            embedder = TrackletEmbedder(config.videoprism)
            embedder.setup()
            embeddings = embedder.embed_tracklets(video_path, tracklet_infos)
            cm.save("embeddings", embeddings)

        # ── Steps 9-13: Cluster, thumbnails, build metadata, store ────────

        # ── Step 9: UMAP + HDBSCAN clustering ─────────────────────────────
        logger.info("[9/21] Running UMAP + HDBSCAN clustering")
        clusterer = TrackletClusterer(config.clustering)
        umap_coords, cluster_labels = clusterer.fit(embeddings)

        # ── Step 10: Thumbnails ────────────────────────────────────────────
        logger.info("[10/21] Generating thumbnails")
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

        # ── Step 11: Build TrackletMetadata objects ────────────────────────
        logger.info("[11/21] Building TrackletMetadata objects")
        tracklets = []

        for tid in valid_track_ids:
            tm = _build_tracklet_metadata(
                video_id, tid, extractor, speeds, umap_coords, cluster_labels,
                video_start_dt=video_start_dt,
            )
            tm.thumbnail_base64 = thumbnails.get(tm.tracklet_id)
            tm.class_id = class_name_to_id.get(tm.class_name, -1)
            tracklets.append(tm)

        # ── Step 12: Cluster statistics ────────────────────────────────────
        logger.info("[12/21] Computing cluster statistics")
        cluster_stats = clusterer.compute_cluster_stats(tracklets, cluster_labels, embeddings)

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

        # ── Step 13: Store in Qdrant ───────────────────────────────────────
        logger.info("[13/21] Storing tracklets/video in Qdrant")
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

        cm.save("tracklet_metadata", {
            "tracklets":     tracklets,
            "video_meta":    video_meta,
            "cluster_stats": cluster_stats,
            "class_dist":    class_dist,
            "meta_raw":      meta_raw,
            "fps":           fps,
        })
        cm.mark_done("tracklets_indexed")

    # ── Step 14: Setup global clips collection ────────────────────────────
    logger.info("[14/21] Setting up global clips collection")
    storage.setup_global_clips_collection()

    # ── Step 15: Compute clip segments ────────────────────────────────────

    if cm.has("clip_infos"):
        logger.info("[15/21] Loaded clip_infos checkpoint")
        clip_infos = cm.load("clip_infos")
    else:
        logger.info("[15/21] Computing global clip segments")
        gc_processor = GlobalClipProcessor(config.global_clips)
        clip_infos = gc_processor.compute_clips(
            video_path=video_path,
            fps=fps,
            total_frames=meta_raw["total_frames"],
            video_id=video_id,
        )
        cm.save("clip_infos", clip_infos)

    # Ensure gc_processor is always initialized (needed for steps 16-17 and 20)
    if gc_processor is None:
        gc_processor = GlobalClipProcessor(config.global_clips)

    # ── Steps 16-17: Embed global clips (streaming, memory-efficient) ──────

    if cm.has("clip_embeddings"):
        logger.info("[16-17/21] Loaded clip_embeddings checkpoint")
        clip_embeddings = cm.load("clip_embeddings")
    else:
        logger.info("[16-17/21] Extracting and embedding global clips (streaming)")
        # Lazy-init embedder if it was skipped (loaded from checkpoint in step 8)
        if embedder is None:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  PyTorch GPU cache cleared before embedding")
            embedder = TrackletEmbedder(config.videoprism)
            embedder.setup()

        cap_embed = cv2.VideoCapture(video_path)
        try:
            clip_embeddings = embedder.embed_global_clips_streaming(
                clip_infos=clip_infos,
                cap=cap_embed,
                gc_processor=gc_processor,
            )
        finally:
            cap_embed.release()

        cm.save("clip_embeddings", clip_embeddings)

    # ── Steps 18-21: Cluster clips, build metadata, store in Qdrant ───────

    if cm.is_done("clips_indexed"):
        logger.info("[18-21/21] Global clips already indexed, skipping")
        unique_gc_clusters = set()

    else:
        # ── Step 18: UMAP + HDBSCAN on global clips ───────────────────────
        logger.info("[18/21] Running UMAP + HDBSCAN on global clips")
        gc_cfg = config.global_clips
        gc_clustering_cfg = ClusteringConfig(
            umap=gc_cfg.umap,
            hdbscan=gc_cfg.hdbscan,
            fps_representatives=gc_cfg.fps_representatives,
        )
        gc_umap_coords: Dict[str, Tuple[float, float]] = {}
        gc_cluster_labels: Dict[str, int] = {}
        if len(clip_embeddings) >= 2:
            gc_clusterer = TrackletClusterer(gc_clustering_cfg)
            gc_umap_coords, gc_cluster_labels = gc_clusterer.fit(clip_embeddings)
        else:
            logger.info("  Fewer than 2 clips — skipping UMAP/HDBSCAN, assigning cluster_id=-1")
            for cid in clip_embeddings:
                gc_umap_coords[cid] = (0.0, 0.0)
                gc_cluster_labels[cid] = -1

        # ── Step 19: FPS representatives per cluster ───────────────────────
        logger.info("[19/21] Computing FPS representatives for global clusters")
        unique_gc_clusters = set(gc_cluster_labels.values())
        global_cluster_rep_ids: Dict[int, List[str]] = {}
        for cluster_id in unique_gc_clusters:
            reps = TrackletClusterer.get_fps_representatives(
                clip_embeddings, gc_cluster_labels, cluster_id, gc_cfg.fps_representatives
            )
            global_cluster_rep_ids[cluster_id] = reps

        all_rep_ids: set = set()
        for reps in global_cluster_rep_ids.values():
            all_rep_ids.update(reps)

        # ── Step 20: Generate clip thumbnails, median frames, optical flow ─
        logger.info("[20/21] Generating clip thumbnails, median frames, optical flow")
        vid_w = meta_raw["width"]
        vid_h = meta_raw["height"]
        batch_size = config.videoprism.batch_size

        overlapping = GlobalClipProcessor.compute_tracklet_ids_for_clips(clip_infos, tracklets)

        total_clips = len(clip_infos)
        global_clips_meta = []
        cap_feat = cv2.VideoCapture(video_path)
        try:
            for i, clip in enumerate(clip_infos):
                cid = clip["clip_id"]
                is_rep = cid in all_rep_ids

                thumbnail_b64 = gc_processor.generate_clip_thumbnail(
                    cap_feat,
                    clip["start_frame"],
                    clip["end_frame"],
                    gc_cfg.thumbnail_width,
                    vid_h,
                    vid_w,
                )

                if is_rep:
                    median_b64 = gc_processor.compute_median_frame(
                        cap_feat,
                        clip["start_frame"],
                        clip["end_frame"],
                        vid_h,
                        vid_w,
                    )
                    flow_b64, fw, fh = gc_processor.compute_optical_flow(
                        cap_feat,
                        clip["sample_frame_nums"],
                        vid_h,
                        vid_w,
                    )
                else:
                    median_b64 = None
                    flow_b64 = None
                    fw, fh = vid_w, vid_h

                ux, uy = gc_umap_coords.get(cid, (0.0, 0.0))
                c_label = gc_cluster_labels.get(cid, -1)
                payload = GlobalClipMetadata(
                    clip_id=cid,
                    video_id=video_id,
                    clip_index=clip["clip_index"],
                    start_time=clip["start_time"],
                    end_time=clip["end_time"],
                    cluster_id=c_label,
                    umap_x=ux,
                    umap_y=uy,
                    thumbnail_base64=thumbnail_b64,
                    median_frame_b64=median_b64,
                    optical_flow_b64=flow_b64,
                    flow_width=fw,
                    flow_height=fh,
                    tracklet_ids=overlapping.get(cid, []),
                    is_representative=is_rep,
                )
                global_clips_meta.append(payload)
                done = i + 1
                if done % batch_size == 0 or done == total_clips:
                    logger.info(f"  Generated features for {done} / {total_clips} clips")
        finally:
            cap_feat.release()

        # ── Step 21: Store global clips in Qdrant ─────────────────────────
        logger.info("[21/21] Storing global clips in Qdrant")
        storage.upsert_global_clips(video_id, global_clips_meta, clip_embeddings)
        cm.mark_done("clips_indexed")

    # ── Final summary ──────────────────────────────────────────────────────
    n_tracklets = len(tracklets)
    n_clips = len(global_clips_meta) if global_clips_meta else len(clip_infos)
    n_clusters = len([s for s in cluster_stats if s.cluster_id >= 0])
    n_gc_clusters = len([c for c in unique_gc_clusters if c >= 0])

    logger.success("=" * 60)
    logger.success("Indexing complete!")
    logger.success(f"  video_id       : {video_id}")
    logger.success(f"  tracklets      : {n_tracklets}")
    logger.success(f"  clusters       : {n_clusters}")
    logger.success(f"  class dist     : {class_dist}")
    logger.success(f"  global clips   : {n_clips}")
    logger.success(f"  global clusters: {n_gc_clusters}")
    logger.success("=" * 60)


# ── Selective recomputation ────────────────────────────────────────────────

def run_recompute(video_path: str, config_path: str, components: List[str]) -> None:
    """Recompute specific pipeline components for an already-indexed video."""
    logger.info("=" * 60)
    logger.info("TrackletViz — Selective Recomputation")
    logger.info(f"  video      : {video_path}")
    logger.info(f"  components : {components}")
    logger.info("=" * 60)

    config = load_config(config_path)
    video_id = _make_video_id(video_path)
    logger.info(f"  video_id = {video_id}")
    cm = CheckpointManager(cache_dir=config.cache_dir, video_id=video_id)

    # ── Determine which components are surgical vs. pipeline-reset ──────────
    _PIPELINE_RESET = {
        "local-embeddings":  ["embeddings", "tracklets_indexed"],
        "global-embeddings": ["clip_embeddings", "clips_indexed"],
    }
    pipeline_comps = [c for c in components if c in _PIPELINE_RESET]
    surgical_comps = [c for c in components if c not in _PIPELINE_RESET]

    # Suppress redundant surgical work if the pipeline will redo it anyway
    if "global-embeddings" in pipeline_comps:
        surgical_comps = [c for c in surgical_comps
                          if c not in {"optical-flow", "median-frames", "clip-thumbnails"}]
    if "local-embeddings" in pipeline_comps:
        surgical_comps = [c for c in surgical_comps if c != "tracklet-thumbnails"]

    # ── Surgical updates ────────────────────────────────────────────────────
    clip_surgical = {c for c in surgical_comps
                     if c in {"optical-flow", "median-frames", "clip-thumbnails"}}
    tracklet_surgical = "tracklet-thumbnails" in surgical_comps

    if clip_surgical:
        if not cm.has("clip_infos"):
            logger.error("clip_infos checkpoint not found — cannot do surgical clip updates")
        else:
            clip_infos: List[dict] = cm.load("clip_infos")
            clip_lookup = {c["clip_id"]: c for c in clip_infos}

            storage = QdrantStorage(config.qdrant)
            storage.setup_collections()
            meta_raw = storage.get_video_metadata(video_id)
            if meta_raw is None:
                logger.error("No video metadata in Qdrant — run full pipeline first")
            else:
                vid_w, vid_h = meta_raw["width"], meta_raw["height"]
                gc_processor = GlobalClipProcessor(config.global_clips)

                # optical-flow and median-frames: only representative clips
                if {"optical-flow", "median-frames"} & clip_surgical:
                    rep_clips = storage.get_representative_clips(video_id)
                    logger.info(f"  {len(rep_clips)} representative clips for flow/median update")
                    cap = cv2.VideoCapture(video_path)
                    try:
                        for i, clip in enumerate(rep_clips):
                            cid = clip["clip_id"]
                            info = clip_lookup.get(cid)
                            if info is None:
                                logger.warning(f"  {cid} not in checkpoint — skipping")
                                continue
                            fields: dict = {}
                            if "optical-flow" in clip_surgical:
                                b64, fw, fh = gc_processor.compute_optical_flow(
                                    cap, info["sample_frame_nums"], vid_h, vid_w
                                )
                                fields.update({"optical_flow_b64": b64,
                                               "flow_width": fw, "flow_height": fh})
                            if "median-frames" in clip_surgical:
                                med = gc_processor.compute_median_frame(
                                    cap, info["start_frame"], info["end_frame"], vid_h, vid_w
                                )
                                fields["median_frame_b64"] = med
                            storage.patch_global_clip(cid, fields)
                            logger.info(f"  [{i+1}/{len(rep_clips)}] Patched {cid}")
                    finally:
                        cap.release()

                # clip-thumbnails: all clips
                if "clip-thumbnails" in clip_surgical:
                    all_clips = storage.get_global_clips_for_video(video_id)
                    logger.info(f"  {len(all_clips)} clips for thumbnail update")
                    cap = cv2.VideoCapture(video_path)
                    try:
                        for i, clip in enumerate(all_clips):
                            cid = clip["clip_id"]
                            info = clip_lookup.get(cid)
                            if info is None:
                                logger.warning(f"  {cid} not in checkpoint — skipping")
                                continue
                            thumb = gc_processor.generate_clip_thumbnail(
                                cap, info["start_frame"], info["end_frame"],
                                config.global_clips.thumbnail_width, vid_h, vid_w
                            )
                            storage.patch_global_clip(cid, {"thumbnail_base64": thumb})
                            done = i + 1
                            if done % 50 == 0 or done == len(all_clips):
                                logger.info(f"  Thumbnails: {done}/{len(all_clips)}")
                    finally:
                        cap.release()

    if tracklet_surgical:
        if not cm.has("tracklet_metadata"):
            logger.error("tracklet_metadata checkpoint not found — cannot redo tracklet thumbnails")
        else:
            data = cm.load("tracklet_metadata")
            tracklets_list: List[TrackletMetadata] = data["tracklets"]
            tn_gen = ThumbnailGenerator(config.thumbnails)
            requests = []
            for t in tracklets_list:
                mid = len(t.bounding_boxes) // 2
                bb = t.bounding_boxes[mid]
                requests.append({
                    "tracklet_id": t.tracklet_id,
                    "frame_num": bb.frame_num,
                    "bbox": (bb.x1, bb.y1, bb.x2, bb.y2),
                })
            thumbnails = tn_gen.generate_thumbnails_batch(video_path, requests)
            storage = QdrantStorage(config.qdrant)
            storage.setup_collections()
            for tid, thumb_b64 in thumbnails.items():
                storage.patch_tracklet(tid, {"thumbnail_base64": thumb_b64})
            logger.info(f"  Updated thumbnails for {len(thumbnails)} tracklets")

    # ── Pipeline resets ─────────────────────────────────────────────────────
    if pipeline_comps:
        for comp in pipeline_comps:
            for ckpt_name in _PIPELINE_RESET[comp]:
                cm.delete(ckpt_name)
                logger.info(f"  Deleted checkpoint '{ckpt_name}' for {comp}")
        logger.info("  Checkpoints cleared — resuming pipeline...")
        run_pipeline(video_path, config_path)

    logger.success("Recomputation complete")


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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Clear checkpoints for this video and rerun from scratch",
    )
    parser.add_argument(
        "--recompute",
        nargs="+",
        choices=_RECOMPUTE_OPTIONS,
        metavar="COMPONENT",
        default=None,
        help=(
            "Recompute specific components for an already-indexed video. "
            f"Choices: {', '.join(_RECOMPUTE_OPTIONS)}"
        ),
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

    if args.recompute:
        if not args.video:
            parser.error("--recompute requires --video")
        run_recompute(args.video, args.config, args.recompute)
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

    run_pipeline(
        args.video,
        args.config,
        start_time=args.start_time,
        tag=args.tag,
        force=args.force,
    )


if __name__ == "__main__":
    main()
