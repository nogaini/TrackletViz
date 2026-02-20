from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class TextSearchRequest(BaseModel):
    video_id: str
    query: str
    limit: int = 20


@router.post("/search/text")
def search_text(body: TextSearchRequest, request: Request):
    """Convert text query to embedding and return similar tracklets."""
    query_vec = request.app.state.embedder.embed_text(f"a video of {body.query}")
    return request.app.state.storage.search_by_text_embedding(
        body.video_id, query_vec, body.limit
    )
