"""
Checkpoint manager for TrackletViz indexing pipeline.

Provides atomic write + corruption-resistant checkpoint persistence so the
pipeline can resume from any step after a kill or crash.
"""

from __future__ import annotations

import os
import pickle
import shutil
from typing import Any

from loguru import logger


class CheckpointManager:
    """
    Persists pipeline intermediate results to disk with atomic writes.

    Layout::

        <cache_dir>/<video_id>/
            <name>.pkl   — pickled checkpoint data (atomic write via .tmp + rename)
            <name>       — sentinel file (touch) marking a step fully done
    """

    def __init__(self, cache_dir: str, video_id: str):
        self.base = os.path.join(cache_dir, video_id)
        os.makedirs(self.base, exist_ok=True)

    def clear(self) -> None:
        """Remove all checkpoints for this video_id."""
        if os.path.isdir(self.base):
            shutil.rmtree(self.base)
            os.makedirs(self.base, exist_ok=True)
            logger.info(f"  Cleared checkpoint directory: {self.base}")

    def _pkl_path(self, name: str) -> str:
        return os.path.join(self.base, f"{name}.pkl")

    def _sentinel_path(self, name: str) -> str:
        return os.path.join(self.base, name)

    def has(self, name: str) -> bool:
        """Return True if a non-corrupt checkpoint pickle exists for *name*."""
        p = self._pkl_path(name)
        if not os.path.isfile(p):
            return False
        try:
            with open(p, "rb") as f:
                pickle.load(f)
            return True
        except Exception:
            logger.warning(f"  Checkpoint '{name}' is corrupt or unreadable, ignoring")
            return False

    def save(self, name: str, obj: Any) -> None:
        """Atomically write *obj* to a checkpoint pickle."""
        p = self._pkl_path(name)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        logger.info(f"  Checkpoint saved: {name}")

    def load(self, name: str) -> Any:
        """Load and return the checkpoint data for *name*."""
        p = self._pkl_path(name)
        with open(p, "rb") as f:
            return pickle.load(f)

    def is_done(self, name: str) -> bool:
        """Return True if the sentinel file for *name* exists."""
        return os.path.isfile(self._sentinel_path(name))

    def mark_done(self, name: str) -> None:
        """Create the sentinel file for *name*."""
        open(self._sentinel_path(name), "w").close()
        logger.info(f"  Step marked done: {name}")

    def delete(self, name: str) -> None:
        """Remove checkpoint pickle and sentinel for *name* if they exist."""
        for p in [self._pkl_path(name), self._sentinel_path(name)]:
            if os.path.isfile(p):
                os.remove(p)
                logger.info(f"  Deleted checkpoint file: {p}")
