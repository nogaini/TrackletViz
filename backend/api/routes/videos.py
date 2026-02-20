from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/videos/")
def list_videos(request: Request):
    """List all indexed videos with lightweight summaries."""
    return request.app.state.storage.get_all_videos()


@router.get("/videos/{video_id}")
def get_video(video_id: str, request: Request):
    """Get full VideoMetadata for a specific video."""
    result = request.app.state.storage.get_video_metadata(video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return result
