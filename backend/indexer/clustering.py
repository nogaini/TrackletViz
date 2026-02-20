"""
Dimensionality reduction and clustering for tracklet embeddings.

Pipeline:
  1. UMAP: reduce high-dim embeddings → 2D for scatter-plot visualization
  2. HDBSCAN: cluster on the original high-dim space (not UMAP output)
  3. Farthest Point Sampling: pick diverse representative tracklets per cluster
  4. Cluster statistics: member count, avg speed, class distribution
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import umap as umap_lib
import hdbscan as hdbscan_lib
from loguru import logger

from indexer.config import ClusteringConfig
from models.schemas import ClusterStatistics, TrackletMetadata


class TrackletClusterer:
    """
    Cluster tracklet embeddings and compute per-cluster statistics.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._umap_model = None
        self._hdbscan_model = None

    def fit(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
        """
        Run UMAP + HDBSCAN on the provided embeddings.

        Args:
            embeddings: Dict mapping tracklet_id → high-dim float32 array.

        Returns:
            (umap_coords, cluster_labels) where each is a Dict mapping
            tracklet_id → value.
        """
        if not embeddings:
            return {}, {}

        ids = list(embeddings.keys())
        matrix = np.stack([embeddings[tid] for tid in ids], axis=0)  # (N, D)

        logger.info(f"Running UMAP on {len(ids)} embeddings (dim={matrix.shape[1]})")
        u_cfg = self.config.umap
        reducer = umap_lib.UMAP(
            n_neighbors=u_cfg.n_neighbors,
            min_dist=u_cfg.min_dist,
            metric=u_cfg.metric,
            n_components=u_cfg.n_components,
            random_state=u_cfg.random_state,
        )
        coords_2d = reducer.fit_transform(matrix)  # (N, 2)
        self._umap_model = reducer

        logger.info("Running HDBSCAN clustering")
        h_cfg = self.config.hdbscan
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=h_cfg.min_cluster_size,
            min_samples=h_cfg.min_samples,
            metric=h_cfg.metric,
            cluster_selection_method=h_cfg.cluster_selection_method,
        )
        labels = clusterer.fit_predict(matrix)  # (N,) — noise = -1
        self._hdbscan_model = clusterer

        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = int(np.sum(unique >= 0))
        n_noise = int(np.sum(labels == -1))
        logger.info(
            f"HDBSCAN: {n_clusters} clusters found, {n_noise} noise points"
        )

        umap_coords: Dict[str, Tuple[float, float]] = {}
        cluster_labels: Dict[str, int] = {}
        for i, tid in enumerate(ids):
            umap_coords[tid] = (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
            cluster_labels[tid] = int(labels[i])

        return umap_coords, cluster_labels

    # ── Farthest Point Sampling ────────────────────────────────────────────

    @staticmethod
    def get_fps_representatives(
        embeddings: Dict[str, np.ndarray],
        cluster_labels: Dict[str, int],
        cluster_id: int,
        k: int,
    ) -> List[str]:
        """
        Select *k* diverse representatives from a cluster using FPS.

        Args:
            embeddings: All embeddings (may include other clusters).
            cluster_labels: Mapping tracklet_id → cluster_id.
            cluster_id: Target cluster to sample from.
            k: Number of representatives to select.

        Returns:
            List of up to *k* tracklet_id strings.
        """
        # Gather IDs belonging to this cluster
        ids = [tid for tid, cid in cluster_labels.items() if cid == cluster_id]
        if not ids:
            return []

        k = min(k, len(ids))
        if k == 1:
            return [ids[0]]

        matrix = np.stack([embeddings[tid] for tid in ids], axis=0)  # (M, D)
        n = len(ids)

        selected_indices: List[int] = [0]  # start at index 0
        # Track minimum distance from each point to any selected point
        min_dists = np.full(n, np.inf)

        for _ in range(k - 1):
            last = matrix[selected_indices[-1]]
            # Euclidean distance from last selected to all others
            dists = np.linalg.norm(matrix - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            # Zero out already-selected
            for si in selected_indices:
                min_dists[si] = -np.inf
            next_idx = int(np.argmax(min_dists))
            selected_indices.append(next_idx)

        return [ids[i] for i in selected_indices]

    # ── Cluster statistics ─────────────────────────────────────────────────

    def compute_cluster_stats(
        self,
        tracklets: List[TrackletMetadata],
        cluster_labels: Dict[str, int],
        embeddings: Dict[str, np.ndarray],
    ) -> List[ClusterStatistics]:
        """
        Compute per-cluster statistics and select FPS representatives.

        Args:
            tracklets: All TrackletMetadata objects (with speeds filled in).
            cluster_labels: Mapping tracklet_id → cluster label (-1 = noise).
            embeddings: Full embedding dict (needed for FPS).

        Returns:
            List of ClusterStatistics, one per unique cluster label.
        """
        k_reps = self.config.fps_representatives

        # Index tracklets by id
        tracklet_map: Dict[str, TrackletMetadata] = {t.tracklet_id: t for t in tracklets}

        # Group tracklet IDs by cluster
        cluster_members: Dict[int, List[str]] = defaultdict(list)
        for tid, cid in cluster_labels.items():
            cluster_members[cid].append(tid)

        stats: List[ClusterStatistics] = []
        for cluster_id in sorted(cluster_members.keys(), key=lambda c: (c < 0, c)):
            member_ids = cluster_members[cluster_id]
            members = [tracklet_map[tid] for tid in member_ids if tid in tracklet_map]

            if not members:
                continue

            avg_speed = float(np.mean([m.avg_speed for m in members]))

            # Class distribution as percentages
            class_counts: Dict[str, int] = defaultdict(int)
            for m in members:
                class_counts[m.class_name] += 1
            total = len(members)
            class_dist = {cls: round(cnt / total * 100.0, 1) for cls, cnt in class_counts.items()}

            # FPS representatives
            reps = self.get_fps_representatives(embeddings, cluster_labels, cluster_id, k_reps)

            stats.append(
                ClusterStatistics(
                    cluster_id=cluster_id,
                    member_count=total,
                    avg_speed=avg_speed,
                    class_distribution=class_dist,
                    representative_tracklet_ids=reps,
                )
            )

        return stats
