"""
Thumbnail generation for tracklets and background frames.

Produces base64-encoded JPEG images for storage in Qdrant payloads and
transmission to the frontend without requiring additional file I/O.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from indexer.config import ThumbnailConfig


class ThumbnailGenerator:
    """
    Generate compact JPEG thumbnails for individual tracklets and background frames.
    """

    def __init__(self, config: ThumbnailConfig):
        self.config = config

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encode_image(self, img: np.ndarray) -> str:
        """Encode a BGR numpy array as a base64 JPEG string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
        ret, buf = cv2.imencode(".jpg", img, encode_params)
        if not ret:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _crop_bbox(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        """Crop a padded bounding-box region and resize to thumbnail dimensions."""
        h, w = frame.shape[:2]
        pad = self.config.padding
        x1i = max(0, int(x1) - pad)
        y1i = max(0, int(y1) - pad)
        x2i = min(w, int(x2) + pad)
        y2i = min(h, int(y2) + pad)

        crop = frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            crop = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        else:
            h_crop, w_crop = crop.shape[:2]
            max_dim = max(self.config.width, self.config.height)  # 128
            scale = max_dim / max(h_crop, w_crop)
            new_w = max(1, int(w_crop * scale))
            new_h = max(1, int(h_crop * scale))
            crop = cv2.resize(crop, (new_w, new_h))
        return crop

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_thumbnail(
        self,
        video_path: str,
        frame_num: int,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[str]:
        """
        Generate a thumbnail for a tracklet at the given frame.

        Args:
            video_path: Path to the source video.
            frame_num: Frame number to read (0-based).
            bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.

        Returns:
            Base64-encoded JPEG string, or None if the frame could not be read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for thumbnail: {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.warning(f"Could not read frame {frame_num} from {video_path}")
            return None

        x1, y1, x2, y2 = bbox
        crop = self._crop_bbox(frame, x1, y1, x2, y2)
        return self._encode_image(crop)

    def generate_thumbnails_batch(
        self,
        video_path: str,
        requests: List[Dict],
    ) -> Dict[str, Optional[str]]:
        """
        Generate thumbnails for multiple tracklets efficiently (single cap open).

        Args:
            video_path: Path to the source video.
            requests: List of dicts, each with:
                - "tracklet_id": str
                - "frame_num": int   (middle frame of the tracklet)
                - "bbox": (x1, y1, x2, y2)

        Returns:
            Dict mapping tracklet_id → base64 JPEG string (or None on failure).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for thumbnails: {video_path}")
            return {req["tracklet_id"]: None for req in requests}

        results: Dict[str, Optional[str]] = {}

        for req in requests:
            tid = req["tracklet_id"]
            fn = req["frame_num"]
            bbox = req["bbox"]

            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {fn} for tracklet {tid}")
                results[tid] = None
                continue

            x1, y1, x2, y2 = bbox
            crop = self._crop_bbox(frame, x1, y1, x2, y2)
            results[tid] = self._encode_image(crop)

        cap.release()
        logger.info(f"Generated {sum(v is not None for v in results.values())} thumbnails")
        return results

    def generate_background(self, background_frame: np.ndarray) -> str:
        """
        Encode a pre-computed background frame as a base64 JPEG string.

        Args:
            background_frame: BGR uint8 numpy array from VideoProcessor.

        Returns:
            Base64-encoded JPEG string.
        """
        resized = cv2.resize(background_frame, (self.config.width * 4, self.config.height * 4))
        return self._encode_image(resized)
