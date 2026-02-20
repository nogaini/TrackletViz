from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/tracklets/{video_id}")
def get_tracklets(video_id: str, request: Request):
    """Get all tracklets for a video (paginates internally via Qdrant scroll)."""
    return request.app.state.storage.get_tracklets_for_video(video_id)
