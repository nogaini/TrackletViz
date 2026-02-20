"""
Speed estimation for tracked objects.

Copied from existing_code/speed.py with no functional changes.
Calculates instantaneous speed for each trajectory point based on
Euclidean displacement between consecutive frames.
"""

import math
from typing import Dict, List
from collections import defaultdict
from loguru import logger

from models.schemas import TrajectoryPoint


class SpeedEstimator:
    """
    Calculate pixel-space speed for tracked objects.

    Computes instantaneous speed based on Euclidean distance between
    consecutive trajectory points and their time delta.
    """

    def __init__(self, fps: float, mode: str = "pixel"):
        """
        Args:
            fps: Video frames per second.
            mode: Speed calculation mode ("pixel" only currently implemented).
        """
        self.fps = fps
        self.mode = mode

        if mode not in ["pixel", "calibrated"]:
            raise ValueError(f"Unknown speed mode: {mode}. Use 'pixel' or 'calibrated'")

    def calculate_speeds(
        self, track_history: Dict[int, List[TrajectoryPoint]]
    ) -> Dict[int, List[float]]:
        """
        Calculate speed for every trajectory point.

        First point of each track receives speed=0. All subsequent points
        receive speed = displacement / time_delta (pixels/second).

        Returns:
            Dict mapping track_id → list of speeds (same length as trajectory).
        """
        speeds: Dict[int, List[float]] = {}

        for track_id, points in track_history.items():
            track_speeds = []

            for i, point in enumerate(points):
                if i == 0:
                    track_speeds.append(0.0)
                else:
                    prev_point = points[i - 1]
                    dx = point.center_x - prev_point.center_x
                    dy = point.center_y - prev_point.center_y
                    displacement = math.sqrt(dx**2 + dy**2)

                    dt = point.timestamp - prev_point.timestamp
                    if dt > 0:
                        speed = displacement / dt
                    else:
                        speed = 0.0
                        logger.warning(
                            f"Zero time delta for track {track_id} "
                            f"between frames {prev_point.frame_num} and {point.frame_num}"
                        )

                    track_speeds.append(speed)

            speeds[track_id] = track_speeds

        return speeds

    def apply_speeds(
        self,
        trajectories: List[TrajectoryPoint],
        speeds: Dict[int, List[float]],
    ):
        """
        Update trajectory points in-place with their calculated speeds.

        Args:
            trajectories: Flat list of all TrajectoryPoint objects.
            speeds: Dict mapping track_id → list of speed values.
        """
        track_indices: Dict[int, List[int]] = defaultdict(list)
        for i, point in enumerate(trajectories):
            track_indices[point.track_id].append(i)

        for track_id, indices in track_indices.items():
            if track_id not in speeds:
                logger.warning(f"No speed data for track {track_id}")
                continue

            track_speeds = speeds[track_id]

            if len(indices) != len(track_speeds):
                logger.warning(
                    f"Mismatch for track {track_id}: "
                    f"{len(indices)} points vs {len(track_speeds)} speeds"
                )
                length = min(len(indices), len(track_speeds))
            else:
                length = len(indices)

            for i in range(length):
                trajectories[indices[i]].speed = track_speeds[i]

    def get_speed_statistics(self, speeds: Dict[int, List[float]]) -> Dict:
        """Compute summary statistics (min/max/avg/median) across all tracks."""
        all_speeds = [s for track_speeds in speeds.values() for s in track_speeds if s > 0]

        if not all_speeds:
            return {
                "min_speed": 0.0,
                "max_speed": 0.0,
                "avg_speed": 0.0,
                "median_speed": 0.0,
                "total_measurements": 0,
            }

        all_speeds_sorted = sorted(all_speeds)
        n = len(all_speeds_sorted)

        return {
            "min_speed": min(all_speeds),
            "max_speed": max(all_speeds),
            "avg_speed": sum(all_speeds) / n,
            "median_speed": all_speeds_sorted[n // 2],
            "total_measurements": n,
        }

    def filter_outlier_speeds(
        self, speeds: Dict[int, List[float]], max_speed: float
    ) -> Dict[int, List[float]]:
        """
        Cap all speed values at *max_speed* to remove tracking outliers.

        Args:
            speeds: Dict of track_id → speed list.
            max_speed: Maximum allowed speed in pixels/second.

        Returns:
            New dict with capped speeds.
        """
        return {
            track_id: [min(s, max_speed) for s in track_speeds]
            for track_id, track_speeds in speeds.items()
        }
