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
from models.schemas import GlobalClipMetadata, TrackletMetadata, VideoMetadata

_BATCH_SIZE = 100     # Qdrant upsert batch size
_GC_BATCH_SIZE = 20  # Smaller batch for global clips (heavy payloads)


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
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout_seconds,
            )
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
        """Remove all tracklet, video metadata, and global clips for *video_id* from Qdrant."""
        logger.info(f"Deleting all data for video_id='{video_id}' …")
        self._delete_by_video_id(self.config.tracklets_collection, video_id)
        self._delete_by_video_id(self.config.videos_collection, video_id)
        gc_col = getattr(self.config, "global_clips_collection", "global_clips")
        self._delete_by_video_id(gc_col, video_id)
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

    def get_tracklets_for_video(
        self,
        video_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        include_thumbnails: bool = False,
        include_bboxes: bool = False,
    ) -> List[dict]:
        """
        Retrieve tracklet payloads for a given video.

        When *limit* is None, all tracklets are returned (full internal
        pagination via Qdrant scroll).  When *limit* is set, at most *limit*
        tracklets are returned starting from *offset* (0-based).

        Set ``include_thumbnails=False`` (default) to strip thumbnail data
        from the returned payloads.

        Set ``include_bboxes=False`` (default) to strip bounding_boxes and
        bbox_centers from the returned payloads (significantly reduces response
        size for videos with many tracklets).
        """
        tc = self.config.tracklets_collection

        def _strip(payload: dict) -> dict:
            if not include_thumbnails:
                payload.pop("thumbnail_base64", None)
            if not include_bboxes:
                payload.pop("bounding_boxes", None)
                payload.pop("bbox_centers", None)
            return payload

        if limit is not None:
            # Single Qdrant scroll page with the requested window.
            # Qdrant's scroll offset is a point-ID cursor, not an integer, so
            # we implement integer offset by skipping earlier pages ourselves.
            # For reasonable offsets (<500K) this is fast enough.
            qdrant_page = 500
            collected: List[dict] = []
            qdrant_cursor = None
            skipped = 0

            while True:
                fetch = min(qdrant_page, offset + limit - len(collected) + qdrant_page)
                results, qdrant_cursor = self.client.scroll(
                    collection_name=tc,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                    ),
                    with_payload=True,
                    with_vectors=False,
                    limit=fetch,
                    offset=qdrant_cursor,
                )

                for point in results:
                    if point.payload is None:
                        continue
                    if skipped < offset:
                        skipped += 1
                        continue
                    collected.append(_strip(point.payload))
                    if len(collected) >= limit:
                        return collected

                if qdrant_cursor is None:
                    break

            return collected

        # limit is None → return everything
        all_payloads: List[dict] = []
        qdrant_cursor = None

        while True:
            results, qdrant_cursor = self.client.scroll(
                collection_name=tc,
                scroll_filter=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=500,
                offset=qdrant_cursor,
            )
            for point in results:
                if point.payload:
                    all_payloads.append(_strip(point.payload))

            if qdrant_cursor is None:
                break

        return all_payloads

    def get_tracklet_thumbnail(self, tracklet_id: str) -> Optional[str]:
        """
        Return the ``thumbnail_base64`` string for a single tracklet, or None
        if the tracklet does not exist.
        """
        tc = self.config.tracklets_collection
        point_id = _make_point_id(tracklet_id)
        results = self.client.retrieve(
            collection_name=tc,
            ids=[point_id],
            with_payload=["thumbnail_base64"],
        )
        if not results:
            return None
        return (results[0].payload or {}).get("thumbnail_base64")

    def get_tracklets_by_ids(self, tracklet_ids: List[str]) -> List[dict]:
        """Fetch full tracklet payloads (incl. bboxes) for a list of tracklet IDs."""
        point_ids = [_make_point_id(tid) for tid in tracklet_ids]
        results = self.client.retrieve(
            collection_name=self.config.tracklets_collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload for p in results if p.payload]

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

    # ── Global Clips ───────────────────────────────────────────────────────

    def setup_global_clips_collection(self):
        """Create the global_clips collection and payload indices if needed."""
        gc_col = self.config.global_clips_collection
        existing = {c.name for c in self.client.get_collections().collections}

        if gc_col not in existing:
            self.client.create_collection(
                collection_name=gc_col,
                vectors_config=VectorParams(
                    size=self.config.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{gc_col}'")
        else:
            info = self.client.get_collection(gc_col)
            actual_dim = info.config.params.vectors.size
            if actual_dim != self.config.vector_dim:
                logger.warning(
                    f"Collection '{gc_col}' has dim={actual_dim}, "
                    f"expected {self.config.vector_dim}. Recreating."
                )
                self.client.delete_collection(gc_col)
                self.client.create_collection(
                    collection_name=gc_col,
                    vectors_config=VectorParams(
                        size=self.config.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.info(f"Collection '{gc_col}' already exists (dim={actual_dim})")

        for field, schema in [
            ("video_id", PayloadSchemaType.KEYWORD),
            ("cluster_id", PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=gc_col,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

        logger.success(f"Global clips collection '{gc_col}' ready")

    def upsert_global_clips(
        self,
        video_id: str,
        clips: List[GlobalClipMetadata],
        embeddings: Dict[str, np.ndarray],
    ):
        """
        Store global clip metadata and embeddings in Qdrant.

        Existing clips for *video_id* are deleted first so re-indexing is idempotent.
        """
        gc_col = self.config.global_clips_collection
        logger.info(f"Deleting existing global clips for video_id='{video_id}'")
        self._delete_by_video_id(gc_col, video_id)

        logger.info(f"Upserting {len(clips)} global clips …")
        for batch_start in range(0, len(clips), _GC_BATCH_SIZE):
            batch = clips[batch_start: batch_start + _GC_BATCH_SIZE]
            points = []
            for clip in batch:
                emb = embeddings.get(clip.clip_id)
                if emb is None:
                    logger.warning(f"No embedding for {clip.clip_id}, skipping")
                    continue
                points.append(
                    PointStruct(
                        id=_make_point_id(clip.clip_id),
                        vector=emb.tolist(),
                        payload=clip.model_dump(),
                    )
                )
            if points:
                self.client.upsert(collection_name=gc_col, points=points)

        logger.success(f"{len(clips)} global clips stored")

    def get_global_clips_for_video(
        self,
        video_id: str,
        include_flow: bool = False,
        include_median: bool = False,
    ) -> List[dict]:
        """
        Retrieve all global clips for a video, sorted by clip_index.

        Heavy fields (optical_flow_b64, median_frame_b64) are stripped by default.
        """
        gc_col = self.config.global_clips_collection
        all_clips: List[dict] = []
        cursor = None

        while True:
            results, cursor = self.client.scroll(
                collection_name=gc_col,
                scroll_filter=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=500,
                offset=cursor,
            )
            for point in results:
                if point.payload:
                    p = dict(point.payload)
                    if not include_flow:
                        p.pop("optical_flow_b64", None)
                    if not include_median:
                        p.pop("median_frame_b64", None)
                    all_clips.append(p)
            if cursor is None:
                break

        all_clips.sort(key=lambda c: c.get("clip_index", 0))
        return all_clips

    def get_global_clip_detail(self, clip_id: str) -> Optional[dict]:
        """Retrieve a single global clip with all fields including heavy ones."""
        gc_col = self.config.global_clips_collection
        point_id = _make_point_id(clip_id)
        results = self.client.retrieve(
            collection_name=gc_col,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        return results[0].payload

    def get_global_cluster_stats_for_video(self, video_id: str) -> List[dict]:
        """
        Derive global cluster statistics by grouping clips by cluster_id.

        Returns list of {cluster_id, member_count, representative_clip_ids}.
        """
        clips = self.get_global_clips_for_video(
            video_id, include_flow=False, include_median=False
        )
        from collections import defaultdict

        groups: Dict[int, List[str]] = defaultdict(list)
        for clip in clips:
            cid = clip.get("cluster_id", -1)
            groups[cid].append(clip.get("clip_id", ""))

        # representative_clip_ids are already stored in each clip payload
        # (set during indexing); collect them per cluster
        rep_map: Dict[int, List[str]] = defaultdict(list)
        for clip in clips:
            cid = clip.get("cluster_id", -1)
            if clip.get("is_representative", False):
                rep_map[cid].append(clip.get("clip_id", ""))

        # Fallback: if reps weren't flagged, just use first few per cluster
        stats = []
        for cid in sorted(groups.keys(), key=lambda c: (c < 0, c)):
            members = groups[cid]
            reps = rep_map.get(cid) or members[:5]
            stats.append(
                {
                    "cluster_id": cid,
                    "member_count": len(members),
                    "representative_clip_ids": reps,
                }
            )
        return stats

    def video_tracklets_indexed(self, video_id: str) -> bool:
        """Return True if this video_id has a record in the videos collection."""
        return self.get_video_metadata(video_id) is not None

    def video_clips_indexed(self, video_id: str) -> bool:
        """Return True if any global clips exist for this video_id."""
        gc_col = self.config.global_clips_collection
        try:
            results, _ = self.client.scroll(
                collection_name=gc_col,
                scroll_filter=Filter(
                    must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except Exception:
            return False

    def get_representative_clips(self, video_id: str) -> List[dict]:
        """Return payloads of all is_representative=True clips for a video."""
        gc_col = self.config.global_clips_collection
        rep_clips: List[dict] = []
        cursor = None
        while True:
            results, cursor = self.client.scroll(
                collection_name=gc_col,
                scroll_filter=Filter(must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id)),
                    FieldCondition(key="is_representative", match=MatchValue(value=True)),
                ]),
                with_payload=True,
                with_vectors=False,
                limit=200,
                offset=cursor,
            )
            for pt in results:
                if pt.payload:
                    rep_clips.append(dict(pt.payload))
            if cursor is None:
                break
        return rep_clips

    def patch_global_clip(self, clip_id: str, fields: dict) -> None:
        """Surgically update specific payload fields for one global clip."""
        self.client.set_payload(
            collection_name=self.config.global_clips_collection,
            payload=fields,
            points=[_make_point_id(clip_id)],
        )

    def patch_tracklet(self, tracklet_id: str, fields: dict) -> None:
        """Surgically update specific payload fields for one tracklet."""
        self.client.set_payload(
            collection_name=self.config.tracklets_collection,
            payload=fields,
            points=[_make_point_id(tracklet_id)],
        )

    def search_clips_by_text_embedding(
        self,
        video_id: str,
        query_vec: np.ndarray,
        limit: int = 20,
    ) -> List[dict]:
        """Search global clips by embedding similarity within a specific video."""
        gc_col = self.config.global_clips_collection
        response = self.client.query_points(
            collection_name=gc_col,
            query=query_vec.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{"clip": hit.payload, "score": hit.score} for hit in response.points]
