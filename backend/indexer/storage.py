"""
Qdrant vector database operations for TrackletViz.

Manages two collections:
  - "tracklets": stores embeddings + full TrackletMetadata payloads
  - "videos": stores VideoMetadata payloads (no vectors)

Both collections support efficient filtering by video_id, cluster_id,
and class_name via payload indices.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from indexer.config import QdrantConfig
from models.schemas import TrackletMetadata, VideoMetadata

_BATCH_SIZE = 100  # Qdrant upsert batch size


def _tracklet_to_payload(t: TrackletMetadata) -> dict:
    """Serialize a TrackletMetadata to a plain dict for Qdrant payload storage."""
    d = t.model_dump()
    # Convert tuple bbox_centers to lists (JSON serialisable)
    d["bbox_centers"] = [list(c) for c in d["bbox_centers"]]
    return d


def _video_to_payload(v: VideoMetadata) -> dict:
    return v.model_dump()


def _make_point_id(tracklet_id: str) -> str:
    """Generate a stable UUID string from a tracklet_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, tracklet_id))


class QdrantStorage:
    """
    Manages TrackletViz data in a Qdrant instance.

    All operations are synchronous (uses the default HTTP client).
    """

    def __init__(self, config: QdrantConfig):
        self.config = config
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self.config.host, port=self.config.port)
        return self._client

    # ── Collection setup ───────────────────────────────────────────────────

    def setup_collections(self):
        """Create collections and payload indices if they do not already exist."""
        existing = {c.name for c in self.client.get_collections().collections}

        tc = self.config.tracklets_collection
        if tc not in existing:
            self.client.create_collection(
                collection_name=tc,
                vectors_config=VectorParams(
                    size=self.config.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{tc}'")
        else:
            info = self.client.get_collection(tc)
            actual_dim = info.config.params.vectors.size
            if actual_dim != self.config.vector_dim:
                logger.warning(
                    f"Collection '{tc}' has dim={actual_dim}, "
                    f"expected {self.config.vector_dim}. Recreating."
                )
                self.client.delete_collection(tc)
                self.client.create_collection(
                    collection_name=tc,
                    vectors_config=VectorParams(
                        size=self.config.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Recreated collection '{tc}' with dim={self.config.vector_dim}")
            else:
                logger.info(f"Collection '{tc}' already exists (dim={actual_dim})")

        vc = self.config.videos_collection
        if vc not in existing:
            # Videos collection has no vectors (use dummy dim=1 with zero vector)
            self.client.create_collection(
                collection_name=vc,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{vc}'")
        else:
            logger.info(f"Collection '{vc}' already exists")

        # Create payload indices for efficient filtering
        for field in ("video_id", "cluster_id", "class_name"):
            try:
                self.client.create_payload_index(
                    collection_name=tc,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                    if field in ("video_id", "class_name")
                    else PayloadSchemaType.INTEGER,
                )
            except Exception:
                pass  # Index may already exist

        try:
            self.client.create_payload_index(
                collection_name=tc,
                field_name="start_world_time",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Index may already exist

        logger.success("Qdrant collections ready")

    # ── Upsert ─────────────────────────────────────────────────────────────

    def upsert_video(
        self,
        video_id: str,
        tracklets: List[TrackletMetadata],
        video_meta: VideoMetadata,
        embeddings: Dict[str, np.ndarray],
    ):
        """
        Store a complete indexed video result in Qdrant.

        Existing records for *video_id* are deleted first so re-indexing is
        idempotent.

        Args:
            video_id: Unique identifier for the video.
            tracklets: All TrackletMetadata objects for this video.
            video_meta: VideoMetadata for this video.
            embeddings: Dict mapping tracklet_id → embedding vector.
        """
        tc = self.config.tracklets_collection
        vc = self.config.videos_collection

        # Delete existing data for this video_id
        logger.info(f"Deleting existing data for video_id='{video_id}'")
        self._delete_by_video_id(tc, video_id)
        self._delete_by_video_id(vc, video_id)

        # Upsert tracklets in batches
        logger.info(f"Upserting {len(tracklets)} tracklets …")
        for batch_start in range(0, len(tracklets), _BATCH_SIZE):
            batch = tracklets[batch_start: batch_start + _BATCH_SIZE]
            points = []
            for t in batch:
                emb = embeddings.get(t.tracklet_id)
                if emb is None:
                    logger.warning(f"No embedding for {t.tracklet_id}, skipping")
                    continue
                points.append(
                    PointStruct(
                        id=_make_point_id(t.tracklet_id),
                        vector=emb.tolist(),
                        payload=_tracklet_to_payload(t),
                    )
                )
            if points:
                self.client.upsert(collection_name=tc, points=points)
        logger.info("Tracklets upserted")

        # Upsert video metadata (dummy zero-vector)
        video_point = PointStruct(
            id=_make_point_id(video_id),
            vector=[0.0],
            payload=_video_to_payload(video_meta),
        )
        self.client.upsert(collection_name=vc, points=[video_point])
        logger.success(f"Video '{video_id}' stored in Qdrant")

    def delete_video(self, video_id: str):
        """Remove all tracklet and video metadata for *video_id* from Qdrant."""
        logger.info(f"Deleting all data for video_id='{video_id}' …")
        self._delete_by_video_id(self.config.tracklets_collection, video_id)
        self._delete_by_video_id(self.config.videos_collection, video_id)
        logger.success(f"Video '{video_id}' removed from Qdrant")

    def _delete_by_video_id(self, collection: str, video_id: str):
        """Delete all points in *collection* whose payload.video_id matches."""
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
            )
        except Exception as exc:
            logger.warning(f"Delete by video_id failed (collection may be empty): {exc}")

    # ── Read operations ────────────────────────────────────────────────────

    def get_all_videos(self) -> List[dict]:
        """
        Return a list of lightweight video summaries from the videos collection.

        Returns fields: video_id, video_path, duration, total_tracklets.
        """
        vc = self.config.videos_collection
        results, _next = self.client.scroll(
            collection_name=vc,
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )

        summaries = []
        for point in results:
            p = point.payload or {}
            summaries.append(
                {
                    "video_id": p.get("video_id"),
                    "video_path": p.get("video_path"),
                    "duration": p.get("duration"),
                    "total_tracklets": p.get("total_tracklets"),
                    "tag": p.get("tag"),
                }
            )
        return summaries

    def get_video_metadata(self, video_id: str) -> Optional[dict]:
        """
        Retrieve full VideoMetadata payload for a specific video.

        Returns:
            Payload dict or None if not found.
        """
        vc = self.config.videos_collection
        results, _ = self.client.scroll(
            collection_name=vc,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if not results:
            return None
        return results[0].payload

    def get_tracklets_for_video(self, video_id: str) -> List[dict]:
        """
        Retrieve all tracklet payloads for a given video.

        Handles large collections by paginating with Qdrant scroll API.
        """
        tc = self.config.tracklets_collection
        all_payloads: List[dict] = []
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=tc,
                scroll_filter=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=500,
                offset=offset,
            )
            for point in results:
                if point.payload:
                    all_payloads.append(point.payload)

            if next_offset is None:
                break
            offset = next_offset

        return all_payloads

    def get_video_id_by_tag(self, tag: str) -> Optional[str]:
        """Look up the video_id for a video with the given tag. Returns None if not found."""
        vc = self.config.videos_collection
        results, _ = self.client.scroll(
            collection_name=vc,
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )
        for point in results:
            p = point.payload or {}
            if p.get("tag") == tag:
                return p.get("video_id")
        return None

    def search_by_text_embedding(
        self,
        video_id: str,
        query_vec: np.ndarray,
        limit: int = 20,
    ) -> List[dict]:
        """
        Search tracklets by embedding similarity within a specific video.

        Args:
            video_id: Scope the search to this video.
            query_vec: 1-D float32 query embedding (e.g. from text encoder).
            limit: Maximum results to return.

        Returns:
            List of dicts with "tracklet" (payload) and "score" (float) keys,
            sorted by descending similarity.
        """
        tc = self.config.tracklets_collection
        response = self.client.query_points(
            collection_name=tc,
            query=query_vec.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{"tracklet": hit.payload, "score": hit.score} for hit in response.points]
