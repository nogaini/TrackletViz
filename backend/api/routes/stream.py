"""
Video streaming endpoint with HTTP range request support.

Enables seeking in the HTML5 <video> player by responding to
206 Partial Content requests.
"""

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_CHUNK = 1024 * 1024  # 1 MB read chunks


@router.get("/videos/{video_id}/stream")
def stream_video(
    video_id: str,
    request: Request,
    range: Optional[str] = Header(None),
) -> StreamingResponse:
    """Stream a video file, supporting HTTP range requests for seeking."""
    meta = request.app.state.storage.get_video_metadata(video_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Video not found")

    path: str = meta.get("video_path", "")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = os.path.getsize(path)

    if range is None:
        # No range requested — stream entire file
        def _iter_full():
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(_CHUNK)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _iter_full(),
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
        )

    # Parse "bytes=start-end"
    range_str = range.replace("bytes=", "")
    parts = range_str.split("-")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

    if start > end or start >= file_size:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")

    end = min(end, file_size - 1)
    content_length = end - start + 1

    def _iter_range():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = fh.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)

    return StreamingResponse(
        _iter_range(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
