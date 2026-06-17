"""
VideoPrism embedding extraction for tracklets.

Uses JAX/Flax-based VideoPrism (LvT variant) for both video and text embeddings.
The LvT (Language-Video Transformer) model supports cross-modal retrieval.

JAX is imported after PyTorch has finished and torch.cuda.empty_cache() has been
called (step 8 of the pipeline), so JAX's default memory preallocation does not
conflict with PyTorch.
"""

from __future__ import annotations

import os

import ctypes
import importlib
import pathlib
import sys

# Pre-load all CUDA sub-libraries from the venv's nvidia packages using absolute
# paths + RTLD_GLOBAL. glibc caches LD_LIBRARY_PATH at process startup, so
# os.environ changes after startup have no effect on dlopen searches. Forcing
# the correct venv libraries into the process's loaded-library table ensures JAX's
# CUDA plugin finds them when it calls cusparseGetProperty, cublasGetVersion, etc.
def _preload_venv_cuda_libs() -> None:
    try:
        _nvidia_root = pathlib.Path(importlib.import_module("nvidia").__path__[0])
    except Exception:
        return

    # Load in dependency order: primitive libs first, then those that depend on them.
    # nvJitLink is needed by cublas/cusparse/cusolver; CuDNN has its own sub-lib order.
    _ordered_pkgs: list[tuple[str, list[str] | None]] = [
        # (nvidia sub-package, [ordered .so filenames] or None to glob all *.so.*)
        ("nvjitlink",  ["libnvJitLink.so.12"]),
        ("cublas",     ["libcublasLt.so.12", "libcublas.so.12"]),
        ("cusparse",   ["libcusparse.so.12"]),
        ("cufft",      ["libcufft.so.11", "libcufftw.so.11"]),
        ("cusolver",   ["libcusolver.so.11", "libcusolverMg.so.11"]),
        ("cudnn",      [
            "libcudnn_graph.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9",
        ]),
    ]

    for _pkg, _names in _ordered_pkgs:
        _lib_dir = _nvidia_root / _pkg / "lib"
        if not _lib_dir.is_dir():
            continue
        if _names is None:
            _names = [p.name for p in sorted(_lib_dir.glob("*.so.*"))]
        for _name in _names:
            _p = _lib_dir / _name
            if _p.exists():
                try:
                    ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass  # Non-fatal


_preload_venv_cuda_libs()
del _preload_venv_cuda_libs

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from indexer.global_clips import GlobalClipProcessor

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from indexer.config import VideoPrismConfig


# ── Helpers ────────────────────────────────────────────────────────────────

def _sample_indices(n_total: int, n_target: int) -> List[int]:
    """
    Sample *n_target* evenly-spaced indices from [0, n_total-1].
    If n_total < n_target the last index is repeated to pad to n_target.
    """
    if n_total <= 0:
        return [0] * n_target
    if n_total >= n_target:
        return [int(round(i * (n_total - 1) / (n_target - 1))) for i in range(n_target)]
    indices = list(range(n_total))
    while len(indices) < n_target:
        indices.append(n_total - 1)
    return indices


def _crop_and_resize(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    target_size: int = 288,
    padding: int = 4,
) -> np.ndarray:
    """Crop a bounding-box region from *frame* and resize to *target_size*."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (target_size, target_size))
    return crop


def _preprocess_clip(frames: List[np.ndarray]) -> np.ndarray:
    """Convert a list of BGR uint8 frames to (T, 288, 288, 3) float32 RGB [0,1]."""
    processed = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (288, 288))
        processed.append(f.astype(np.float32) / 255.0)
    return np.stack(processed)  # (T, 288, 288, 3)


# ── Main embedder ──────────────────────────────────────────────────────────

class TrackletEmbedder:
    """
    Compute high-dimensional embeddings for video tracklets using VideoPrism LvT.

    Uses JAX/Flax for inference. Weights are downloaded from Hugging Face on
    first use (~991 MB) and cached locally.
    """

    def __init__(self, config: VideoPrismConfig):
        self.config = config
        self._flax_model = None
        self._params = None
        self._tokenizer = None
        self._forward = None
        self._forward_text_only = None
        self._dummy_text_ids = None
        self._dummy_text_paddings = None

    def setup(self):
        """Load VideoPrism model and tokenizer. Downloads weights if needed."""
        # Add VideoPrism repo to sys.path so it can be imported
        model_path = os.path.abspath(self.config.model_path)
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"VideoPrism model_path not found: '{model_path}'. "
                "Clone https://github.com/google-deepmind/videoprism and set model_path."
            )
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        from videoprism import models as vp  # type: ignore

        model_name = self.config.model_name
        logger.info(f"Loading VideoPrism model: {model_name}")

        self._flax_model = vp.get_model(model_name)
        logger.info("  Downloading/loading pretrained weights (first run: ~991 MB) …")
        self._params = vp.load_pretrained_weights(model_name)
        logger.info("  Weights loaded")

        self._tokenizer = vp.load_text_tokenizer("c4_en")

        # Pre-tokenize dummy empty string for video-only inference
        self._dummy_text_ids, self._dummy_text_paddings = vp.tokenize_texts(
            self._tokenizer, [""], max_length=64
        )

        # JIT-compile the forward function once
        flax_model = self._flax_model
        params = self._params

        @jax.jit
        def _forward(video, text_ids, text_paddings):
            video_emb, text_emb, _ = flax_model.apply(
                params, video, text_ids, text_paddings, train=False
            )
            return video_emb, text_emb

        self._forward = _forward

        @jax.jit
        def _forward_text_only(text_ids, text_paddings):
            _, text_emb, _ = flax_model.apply(
                params, None, text_ids, text_paddings, train=False
            )
            return text_emb

        self._forward_text_only = _forward_text_only
        logger.success(f"VideoPrism ({model_name}) ready")

    def embed_tracklets(
        self,
        video_path: str,
        tracklets_info: List[Dict],
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for a list of tracklets.

        Args:
            video_path: Path to the source video.
            tracklets_info: Each element must have keys:
                - "tracklet_id": str
                - "bboxes": list of (x1, y1, x2, y2) tuples ordered by frame
                - "frame_nums": list of frame numbers corresponding to bboxes

        Returns:
            Dict mapping tracklet_id → 1-D float32 embedding array of shape (768,).
        """
        if self._forward is None:
            raise RuntimeError("Call setup() before embed_tracklets()")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        embeddings: Dict[str, np.ndarray] = {}
        n_frames = self.config.num_frames
        batch_size = self.config.batch_size

        # Collect clips in batches, then run JAX inference once per batch
        for batch_start in range(0, len(tracklets_info), batch_size):
            batch = tracklets_info[batch_start: batch_start + batch_size]
            batch_clips: List[np.ndarray] = []
            batch_ids: List[str] = []

            for info in batch:
                tid = info["tracklet_id"]
                frame_nums: List[int] = info["frame_nums"]
                bboxes: List[Tuple[float, float, float, float]] = info["bboxes"]

                sample_idx = _sample_indices(len(frame_nums), n_frames)
                clip_frames: List[np.ndarray] = []

                for si in sample_idx:
                    fn = frame_nums[si]
                    bbox = bboxes[si]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((288, 288, 3), dtype=np.uint8)
                        clip_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
                    else:
                        crop = _crop_and_resize(frame, bbox)
                        clip_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)

                batch_clips.append(np.stack(clip_frames))  # (T, 288, 288, 3)
                batch_ids.append(tid)

            # Stack into (B, T, 288, 288, 3) and run forward pass
            video_batch = jnp.asarray(np.stack(batch_clips))  # (B, T, H, W, 3)
            b = video_batch.shape[0]
            text_ids = jnp.asarray(np.tile(self._dummy_text_ids, (b, 1)))
            text_pads = jnp.asarray(np.tile(self._dummy_text_paddings, (b, 1)))

            video_emb, _ = self._forward(video_batch, text_ids, text_pads)
            video_emb_np = np.asarray(video_emb)  # (B, 768)

            for i, tid in enumerate(batch_ids):
                embeddings[tid] = video_emb_np[i].astype(np.float32)

            done = min(batch_start + batch_size, len(tracklets_info))
            logger.info(f"  Embedded {done} / {len(tracklets_info)} tracklets")

        cap.release()
        return embeddings

    def embed_global_clips(
        self,
        clip_frame_arrays: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for pre-extracted global clip frame arrays.

        Args:
            clip_frame_arrays: Dict mapping clip_id → (T, 288, 288, 3) float32 RGB.
                The arrays are already in the correct format (no video read or
                cropping needed).

        Returns:
            Dict mapping clip_id → (768,) float32 embedding array.
        """
        if self._forward is None:
            raise RuntimeError("Call setup() before embed_global_clips()")

        embeddings: Dict[str, np.ndarray] = {}
        clip_ids = list(clip_frame_arrays.keys())
        batch_size = self.config.batch_size

        for batch_start in range(0, len(clip_ids), batch_size):
            batch_ids = clip_ids[batch_start: batch_start + batch_size]
            batch_clips = [clip_frame_arrays[cid] for cid in batch_ids]

            video_batch = jnp.asarray(np.stack(batch_clips))  # (B, T, H, W, 3)
            b = video_batch.shape[0]
            text_ids = jnp.asarray(np.tile(self._dummy_text_ids, (b, 1)))
            text_pads = jnp.asarray(np.tile(self._dummy_text_paddings, (b, 1)))

            video_emb, _ = self._forward(video_batch, text_ids, text_pads)
            video_emb_np = np.asarray(video_emb)  # (B, 768)

            for i, cid in enumerate(batch_ids):
                embeddings[cid] = video_emb_np[i].astype(np.float32)

            done = min(batch_start + batch_size, len(clip_ids))
            logger.info(f"  Embedded {done} / {len(clip_ids)} global clips")

        return embeddings

    def embed_global_clips_streaming(
        self,
        clip_infos: List[Dict],
        cap: cv2.VideoCapture,
        gc_processor: "GlobalClipProcessor",
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for global clips one batch at a time (memory-efficient).

        Unlike embed_global_clips(), this method does NOT require all frame arrays
        to be in memory simultaneously. It interleaves frame extraction and embedding
        one batch at a time, capping peak memory at batch_size × ~6 MB.

        Args:
            clip_infos: List of dicts with clip_id, sample_frame_nums.
            cap: Already-opened cv2.VideoCapture for the source video.
            gc_processor: GlobalClipProcessor instance for extract_frames_for_clip().

        Returns:
            Dict mapping clip_id → (768,) float32 embedding array.
        """
        if self._forward is None:
            raise RuntimeError("Call setup() before embed_global_clips_streaming()")

        import gc as _gc

        embeddings: Dict[str, np.ndarray] = {}
        batch_size = self.config.batch_size

        for batch_start in range(0, len(clip_infos), batch_size):
            batch = clip_infos[batch_start: batch_start + batch_size]
            batch_clips: List[np.ndarray] = []
            batch_ids: List[str] = []

            for clip in batch:
                arr = gc_processor.extract_frames_for_clip(cap, clip["sample_frame_nums"])
                batch_clips.append(arr)
                batch_ids.append(clip["clip_id"])

            video_batch = jnp.asarray(np.stack(batch_clips))
            b = video_batch.shape[0]
            text_ids = jnp.asarray(np.tile(self._dummy_text_ids, (b, 1)))
            text_pads = jnp.asarray(np.tile(self._dummy_text_paddings, (b, 1)))

            video_emb, _ = self._forward(video_batch, text_ids, text_pads)
            video_emb_np = np.asarray(video_emb)

            for i, cid in enumerate(batch_ids):
                embeddings[cid] = video_emb_np[i].astype(np.float32)

            del batch_clips, video_batch, video_emb, video_emb_np
            _gc.collect()

            done = min(batch_start + batch_size, len(clip_infos))
            logger.info(f"  Embedded {done} / {len(clip_infos)} global clips (streaming)")

        return embeddings

    def embed_text(self, query: str) -> np.ndarray:
        """
        Convert a text query to an embedding for similarity search.

        Args:
            query: Natural language description.

        Returns:
            1-D float32 embedding array of shape (768,).
        """
        if self._forward_text_only is None:
            raise RuntimeError("Call setup() before embed_text()")

        from videoprism import models as vp  # type: ignore

        text_ids, text_pads = vp.tokenize_texts(
            self._tokenizer, [query], max_length=64
        )
        text_emb = self._forward_text_only(
            jnp.asarray(text_ids),
            jnp.asarray(text_pads),
        )
        return np.asarray(text_emb[0]).astype(np.float32)
