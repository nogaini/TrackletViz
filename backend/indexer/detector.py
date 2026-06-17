"""
Video processing with YOLO object detection and BoxMOT tracking.

Adapted from existing_code/detector.py:
  - Config driven (uses AppConfig instead of hardcoded values)
  - Added extract_background_frame() for median background computation
  - Device selection respects config.processing.device
"""

import subprocess
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger

from models.schemas import ProcessingConfig

# Trackers that don't need a ReID model
_NO_REID_TRACKERS = {"bytetrack", "ocsort", "sfsort"}


class VideoProcessor:
    """
    Video processor integrating YOLO detection and BoxMOT tracking.

    Yields per-frame (frame_num, frame, detections, tracks) tuples for
    memory-efficient processing of long videos.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.tracker = None
        self.class_names: Dict[int, str] = {}

    def setup(self):
        """Initialize YOLO model and BoxMOT tracker."""
        logger.info(f"Loading YOLO model: {self.config.yolo_model}")
        self.model = YOLO(self.config.yolo_model)

        self.class_names = {
            class_id: self.model.names[class_id]
            for class_id in self.config.target_classes
        }
        logger.info(f"Target classes: {self.class_names}")

        logger.info(f"Initializing {self.config.tracker} tracker")
        self.tracker = self._create_tracker()

        logger.success("Video processor initialized successfully")

    def _create_tracker(self):
        """
        Instantiate the configured BoxMOT tracker.

        For trackers that require a ReID model (e.g. botsort, strongsort),
        we disable ReID when no weights file is present, falling back to
        appearance-free tracking.
        """
        tracker_type = self.config.tracker.lower()
        if self.config.device == "cuda":
            device = 0
        half = self.config.device != "cpu"

        if tracker_type in _NO_REID_TRACKERS:
            from boxmot import create_tracker
            return create_tracker(
                tracker_type=tracker_type,
                reid_weights=None,
                device=device,
                half=half,
            )

        # ReID-capable trackers: use a dummy weight path with ReID disabled
        # so we don't need to download the ReID model.
        _DUMMY_REID = Path("lmbn_n_duke.pt")

        if tracker_type == "botsort":
            from boxmot import BotSort
            return BotSort(
                reid_weights=_DUMMY_REID,
                device=device,
                half=half,
                with_reid=False,
            )
        if tracker_type == "strongsort":
            from boxmot import StrongSort
            return StrongSort(
                reid_weights=_DUMMY_REID,
                device=device,
                half=half,
                with_reid=False,
            )
        if tracker_type == "deepocsort":
            from boxmot import DeepOcSort
            return DeepOcSort(
                reid_weights=_DUMMY_REID,
                device=device,
                half=half,
                with_reid=False,
            )
        # Fallback: try generic create_tracker
        from boxmot import create_tracker
        return create_tracker(
            tracker_type=tracker_type,
            reid_weights=None,
            device=device,
            half=half,
        )

    def _build_pts_list(self, video_path: str) -> Optional[List[float]]:
        """
        Read actual per-frame presentation timestamps via ffprobe.

        Returns a sorted (display-order) list of PTS values in seconds, one entry
        per video frame, or None if ffprobe is unavailable or fails.  Sorting by
        pts_time makes the result correct even for B-frame streams where packet
        order differs from display order.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "v:0",
                    "-show_packets",
                    "-show_entries", "packet=pts_time",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            pts: List[float] = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and line != "N/A":
                    try:
                        pts.append(float(line))
                    except ValueError:
                        pass
            if pts:
                pts.sort()
                logger.info(f"ffprobe: {len(pts)} frame PTS values, last={pts[-1]:.3f}s")
                return pts
        except Exception as exc:
            logger.warning(f"ffprobe unavailable ({exc}) — falling back to frame_num/fps timestamps")
        return None

    def process_video(
        self, video_path: str
    ) -> Generator[Tuple[int, np.ndarray, object, np.ndarray, float], None, None]:
        """
        Process video frame by frame with detection and tracking.

        Yields:
            (frame_num, frame, yolo_results, tracks, pts_s)
            tracks shape: (N, 7+) — [x1, y1, x2, y2, track_id, conf, class_id, ...]
            pts_s: actual presentation timestamp in seconds from container
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Pre-read actual container PTS for every frame via ffprobe.  This is the
        # only reliable source: cv2.CAP_PROP_POS_MSEC computes frame_count/r_fps
        # (not actual PTS) and avg_fps diverges for VFR recordings.
        fps_fallback = cap.get(cv2.CAP_PROP_FPS)
        pts_list = self._build_pts_list(video_path)

        frame_num = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pts_s = (
                    pts_list[frame_num]
                    if pts_list and frame_num < len(pts_list)
                    else frame_num / fps_fallback
                )

                results = self.model.predict(
                    frame,
                    conf=self.config.confidence_threshold,
                    classes=self.config.target_classes,
                    verbose=False,
                    device=self.config.device,
                )

                detections = self._yolo_to_boxmot(results[0])

                if len(detections) > 0:
                    tracks = self.tracker.update(detections, frame)
                else:
                    tracks = np.empty((0, 7))

                yield frame_num, frame, results[0], tracks, pts_s
                frame_num += 1

        finally:
            cap.release()
            logger.info(f"Processed {frame_num} frames from {video_path}")

    def _yolo_to_boxmot(self, yolo_results) -> np.ndarray:
        """Convert Ultralytics YOLO results to BoxMOT [x1,y1,x2,y2,conf,cls] format."""
        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            return np.empty((0, 6))

        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        confidences = yolo_results.boxes.conf.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy()

        return np.column_stack((boxes, confidences, class_ids))

    def get_video_metadata(self, video_path: str) -> Dict:
        """
        Extract basic metadata from a video file.

        Returns:
            Dict with fps, total_frames, width, height, duration.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        metadata["duration"] = metadata["total_frames"] / metadata["fps"]
        cap.release()
        return metadata

    def extract_background_frame(self, video_path: str, n_samples: int = 20) -> np.ndarray:
        """
        Compute a representative background by taking a pixel-wise median
        across evenly-spaced sample frames.

        Args:
            video_path: Path to the video file.
            n_samples: Number of frames to sample (default 20).

        Returns:
            Background image as a uint8 numpy array (H, W, 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, min(n_samples, total_frames), dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(f"Could not read any frames from {video_path}")

        stacked = np.stack(frames, axis=0).astype(np.float32)
        background = np.median(stacked, axis=0).astype(np.uint8)
        logger.info(f"Background frame computed from {len(frames)} samples")
        return background
