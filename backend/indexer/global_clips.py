"""
Global clip processing for TrackletViz.

Segments a video into non-overlapping fixed-duration clips, then for each clip:
  - Extracts sampled frames for VideoPrism embedding
  - Computes a median frame (for heatmap / illumination analysis)
  - Generates a representative thumbnail (middle frame)
  - Computes average optical flow (for activity shift analysis)
  - Associates overlapping tracklet IDs
"""

from __future__ import annotations

import base64
import io
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from indexer.config import GlobalClipsConfig
from models.schemas import TrackletMetadata


class GlobalClipProcessor:
    """Extract per-clip features from a video."""

    def __init__(self, config: GlobalClipsConfig):
        self.config = config

    # ── Clip segmentation ─────────────────────────────────────────────────

    def compute_clips(
        self,
        video_path: str,
        fps: float,
        total_frames: int,
        video_id: str,
    ) -> List[dict]:
        """
        Segment a video into non-overlapping clip_duration-second windows.

        Returns a list of dicts:
            clip_id, clip_index, start_time, end_time,
            start_frame, end_frame, sample_frame_nums
        """
        clip_frames = int(round(self.config.clip_duration * fps))
        if clip_frames < 1:
            clip_frames = 1

        clips: List[dict] = []
        idx = 0
        start_frame = 0
        while start_frame < total_frames:
            end_frame = min(start_frame + clip_frames - 1, total_frames - 1)
            start_time = start_frame / fps
            end_time = end_frame / fps

            n_frames = self.config.num_frames
            n_available = end_frame - start_frame + 1
            sample_frames = self._sample_indices_in_range(
                start_frame, end_frame, n_frames
            )

            clip_id = f"{video_id}_clip_{idx:04d}"
            clips.append(
                {
                    "clip_id": clip_id,
                    "clip_index": idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "sample_frame_nums": sample_frames,
                    "n_available": n_available,
                }
            )
            idx += 1
            start_frame += clip_frames

        logger.info(f"  Computed {len(clips)} global clips ({self.config.clip_duration}s each)")
        return clips

    @staticmethod
    def _sample_indices_in_range(start: int, end: int, n: int) -> List[int]:
        """
        Sample n evenly-spaced frame indices from [start, end].
        Pads with last frame if the clip has fewer than n frames.
        """
        total = end - start + 1
        if total <= 0:
            return [start] * n
        if total >= n:
            return [start + int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        indices = list(range(start, end + 1))
        while len(indices) < n:
            indices.append(end)
        return indices

    # ── Frame extraction ──────────────────────────────────────────────────

    def extract_frames_for_clip(
        self,
        cap: cv2.VideoCapture,
        sample_frame_nums: List[int],
    ) -> np.ndarray:
        """
        Read the specified frame numbers from cap and return
        a (T, 288, 288, 3) float32 RGB array suitable for VideoPrism.
        """
        frames: List[np.ndarray] = []
        for fn in sample_frame_nums:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((288, 288, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (288, 288))
            # BGR → RGB, normalize
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(rgb)
        return np.stack(frames)  # (T, 288, 288, 3)

    # ── Median frame ──────────────────────────────────────────────────────

    def compute_median_frame(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        video_height: int,
        video_width: int,
    ) -> str:
        """
        Subsample up to 30 frames from [start_frame, end_frame], compute
        per-pixel median, downscale to median_frame_width, and return as
        base64 JPEG.
        """
        total = end_frame - start_frame + 1
        step = max(1, total // 30)
        sampled_fn = list(range(start_frame, end_frame + 1, step))[:30]

        target_w = self.config.median_frame_width
        target_h = int(round(target_w * video_height / max(video_width, 1)))

        bgr_stack: List[np.ndarray] = []
        for fn in sampled_fn:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue
            # Resize before stacking to keep memory usage low
            frame_small = cv2.resize(frame, (target_w, target_h))
            bgr_stack.append(frame_small.astype(np.float32))

        if not bgr_stack:
            placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            return _encode_jpeg(placeholder)

        median_bgr = np.median(np.stack(bgr_stack, axis=0), axis=0).astype(np.uint8)
        return _encode_jpeg(median_bgr)

    # ── Thumbnail ─────────────────────────────────────────────────────────

    def generate_clip_thumbnail(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        target_width: int,
        video_height: int,
        video_width: int,
    ) -> str:
        """Read the middle frame of the clip and return as base64 JPEG."""
        mid = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((target_width, target_width, 3), dtype=np.uint8)
        else:
            target_height = int(round(target_width * video_height / max(video_width, 1)))
            frame = cv2.resize(frame, (target_width, target_height))
        return _encode_jpeg(frame)

    # ── Optical flow ──────────────────────────────────────────────────────

    def compute_optical_flow(
        self,
        cap: cv2.VideoCapture,
        sample_frame_nums: List[int],
        video_height: int,
        video_width: int,
    ) -> Tuple[str, int, int]:
        """
        Compute average optical flow across sampled frame pairs.

        Flow is computed at native resolution then downscaled to a small grid
        (flow_width × proportional height) to keep the payload small (~154 KB
        instead of ~22 MB for 1080p).

        Returns (base64_string, flow_width, flow_height) where the b64 string
        encodes a raw float32 array of shape (flow_height, flow_width, 2).
        """
        # Output grid dimensions
        fw_out = self.config.flow_width
        fh_out = int(round(fw_out * video_height / max(video_width, 1)))

        gray_frames: List[np.ndarray] = []
        for fn in sample_frame_nums:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                gray_frames.append(np.zeros((video_height, video_width), dtype=np.uint8))
            else:
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        n_pairs = len(gray_frames) - 1
        if n_pairs <= 0:
            flow_arr = np.zeros((fh_out, fw_out, 2), dtype=np.float32)
            b64 = base64.b64encode(flow_arr.tobytes()).decode("ascii")
            return b64, fw_out, fh_out

        sum_flow = np.zeros((video_height, video_width, 2), dtype=np.float64)
        for i in range(n_pairs):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i],
                gray_frames[i + 1],
                None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            sum_flow += flow

        avg_flow = (sum_flow / n_pairs).astype(np.float32)
        # Downscale to the small output grid
        avg_flow_small = cv2.resize(avg_flow, (fw_out, fh_out), interpolation=cv2.INTER_LINEAR)
        b64 = base64.b64encode(avg_flow_small.tobytes()).decode("ascii")
        return b64, fw_out, fh_out

    # ── Tracklet association ───────────────────────────────────────────────

    @staticmethod
    def compute_tracklet_ids_for_clips(
        clip_infos: List[dict],
        tracklets: List[TrackletMetadata],
    ) -> Dict[str, List[str]]:
        """
        For each clip, find tracklets that temporally overlap [start_time, end_time].

        Returns Dict[clip_id → List[tracklet_id]].
        """
        result: Dict[str, List[str]] = {c["clip_id"]: [] for c in clip_infos}
        for clip in clip_infos:
            cid = clip["clip_id"]
            cs = clip["start_time"]
            ce = clip["end_time"]
            for t in tracklets:
                if t.start_timestamp < ce and t.end_timestamp > cs:
                    result[cid].append(t.tracklet_id)
        return result


# ── Utilities ──────────────────────────────────────────────────────────────

def _encode_jpeg(bgr: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR uint8 image to base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
