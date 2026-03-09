"""
API routes for global clip embeddings.

GET  /api/global-clips/{video_id}                   - list clips (lightweight)
GET  /api/global-clips/{video_id}/cluster-stats     - cluster summaries
GET  /api/global-clips/detail/{clip_id}             - single clip with all fields
POST /api/search/clips                              - text-similarity search over clips
"""

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


@router.get("/global-clips/{video_id}/cluster-stats")
def get_global_cluster_stats(video_id: str, request: Request):
    """Return cluster statistics for all global clip clusters in a video."""
    return request.app.state.storage.get_global_cluster_stats_for_video(video_id)


@router.get("/global-clips/{video_id}")
def get_global_clips(
    video_id: str,
    request: Request,
    include_flow: bool = Query(False),
    include_median: bool = Query(False),
):
    """
    Return all global clips for a video, ordered by clip_index.

    Heavy base64 fields (optical_flow_b64, median_frame_b64) are excluded by
    default; pass include_flow=true or include_median=true to include them.
    """
    return request.app.state.storage.get_global_clips_for_video(
        video_id,
        include_flow=include_flow,
        include_median=include_median,
    )


@router.get("/global-clips/detail/{clip_id}")
def get_global_clip_detail(clip_id: str, request: Request):
    """Return full detail for a single clip including all heavy fields."""
    clip = request.app.state.storage.get_global_clip_detail(clip_id)
    if clip is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Clip not found")
    return clip


class ClipSearchRequest(BaseModel):
    video_id: str
    query: str
    limit: int = 20


@router.post("/search/clips")
def search_clips(body: ClipSearchRequest, request: Request):
    """Convert text query to embedding and return similar global clips."""
    query_vec = request.app.state.embedder.embed_text(f"a video of {body.query}")
    return request.app.state.storage.search_clips_by_text_embedding(
        body.video_id, query_vec, body.limit
    )
