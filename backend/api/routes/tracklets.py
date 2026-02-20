from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/tracklets/{video_id}")
def get_tracklets(
    video_id: str,
    request: Request,
    limit: int = 10000,
    offset: int = 0,
    include_thumbnails: bool = False,
):
    """
    Get tracklets for a video.

    Query parameters:
    - **limit**: max tracklets to return (default 10 000; set to 0 for all)
    - **offset**: number of tracklets to skip (default 0)
    - **include_thumbnails**: include base64 thumbnail in each tracklet
      (default false; thumbnails can be fetched individually via the
      /tracklets/{tracklet_id}/thumbnail endpoint)
    """
    storage = request.app.state.storage
    effective_limit = limit if limit > 0 else None
    return storage.get_tracklets_for_video(
        video_id,
        limit=effective_limit,
        offset=offset,
        include_thumbnails=include_thumbnails,
    )


@router.get("/tracklets/{tracklet_id}/thumbnail")
def get_tracklet_thumbnail(tracklet_id: str, request: Request):
    """
    Return the base64 JPEG thumbnail for a single tracklet.

    Response JSON: ``{"thumbnail_base64": "<base64 string>"}``
    """
    storage = request.app.state.storage
    thumb = storage.get_tracklet_thumbnail(tracklet_id)
    if thumb is None:
        raise HTTPException(status_code=404, detail="Tracklet not found")
    return JSONResponse({"thumbnail_base64": thumb})
