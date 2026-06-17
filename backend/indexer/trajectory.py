"""
Trajectory extraction and storage.

Adapted from existing_code/trajectory.py:
  - Added get_filtered_track_ids(min_frames) to drop short tracklets
  - Added get_frame_range(track_id) for thumbnail / embedding extraction
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from loguru import logger

from models.schemas import TrajectoryPoint


class TrajectoryExtractor:
    """
    Extract and accumulate trajectory data from BoxMOT tracked detections.

    Maintains a flat list of all TrajectoryPoints and a per-track history
    dict for efficient lookup during speed calculation and filtering.
    """

    def __init__(self, fps: float, class_names: Dict[int, str]):
        """
        Args:
            fps: Video frames per second (used for timestamp calculation).
            class_names: Mapping from integer class ID to string class name.
        """
        self.fps = fps
        self.class_names = class_names
        self.trajectories: List[TrajectoryPoint] = []
        self.track_history: Dict[int, List[TrajectoryPoint]] = defaultdict(list)

    def extract_from_tracks(
        self, frame_num: int, tracks: np.ndarray, timestamp_s: Optional[float] = None
    ):
        """
        Convert one frame's BoxMOT tracks into TrajectoryPoint objects.

        Args:
            frame_num: Zero-based frame index.
            tracks: BoxMOT output array [x1, y1, x2, y2, track_id, conf, class_id, ...]
                    shape (N, 7+).
            timestamp_s: Actual presentation timestamp in seconds. When provided,
                         used directly instead of computing frame_num / fps (which
                         can drift on videos where avg_fps ≠ r_frame_rate).
        """
        if len(tracks) == 0:
            return

        ts = timestamp_s if timestamp_s is not None else frame_num / self.fps

        for track in tracks:
            x1, y1, x2, y2, track_id, conf, class_id = track[:7]

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            point = TrajectoryPoint(
                track_id=int(track_id),
                class_name=self.class_names.get(int(class_id), "unknown"),
                frame_num=frame_num,
                timestamp=ts,
                center_x=float(center_x),
                center_y=float(center_y),
                bbox_w=float(bbox_w),
                bbox_h=float(bbox_h),
                speed=None,
            )

            self.trajectories.append(point)
            self.track_history[int(track_id)].append(point)

    # ── Original accessors ──────────────────────────────────────────────────

    def get_trajectories(self) -> List[TrajectoryPoint]:
        return self.trajectories

    def get_track_history(self, track_id: int) -> List[TrajectoryPoint]:
        return self.track_history[track_id]

    def get_all_track_ids(self) -> List[int]:
        return list(self.track_history.keys())

    def get_track_count(self) -> int:
        return len(self.track_history)

    def get_point_count(self) -> int:
        return len(self.trajectories)

    def get_statistics(self) -> Dict:
        if not self.trajectories:
            return {
                "total_points": 0,
                "unique_tracks": 0,
                "class_distribution": {},
                "avg_points_per_track": 0.0,
            }

        class_counts: Dict[str, int] = defaultdict(int)
        for point in self.trajectories:
            class_counts[point.class_name] += 1

        avg_points = (
            len(self.trajectories) / len(self.track_history) if self.track_history else 0
        )

        return {
            "total_points": len(self.trajectories),
            "unique_tracks": len(self.track_history),
            "class_distribution": dict(class_counts),
            "avg_points_per_track": avg_points,
        }

    # ── New helpers ─────────────────────────────────────────────────────────

    def get_filtered_track_ids(self, min_frames: int) -> List[int]:
        """
        Return track IDs whose history contains at least *min_frames* points.

        Args:
            min_frames: Minimum number of observation frames.

        Returns:
            List of qualifying track IDs.
        """
        qualified = [
            track_id
            for track_id, points in self.track_history.items()
            if len(points) >= min_frames
        ]
        logger.info(
            f"Filtered tracklets: {len(qualified)} / {len(self.track_history)} "
            f"have >= {min_frames} frames"
        )
        return qualified

    def get_frame_range(self, track_id: int) -> Tuple[int, int]:
        """
        Return the (start_frame, end_frame) inclusive for a given track.

        Args:
            track_id: Track to query.

        Returns:
            (start_frame_num, end_frame_num) tuple.
        """
        points = self.track_history[track_id]
        if not points:
            raise ValueError(f"No points for track_id={track_id}")
        return points[0].frame_num, points[-1].frame_num
