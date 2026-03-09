import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from indexer.config import load_config
from indexer.storage import QdrantStorage
from indexer.embeddings import TrackletEmbedder
from api.routes import videos, tracklets, search, stream, global_clips

CONFIG_PATH = os.environ.get("TRACKVIZ_CONFIG", "config/default.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(CONFIG_PATH)
    app.state.storage = QdrantStorage(cfg.qdrant)
    app.state.embedder = TrackletEmbedder(cfg.videoprism)
    app.state.embedder.setup()  # downloads weights ~991 MB on first run
    yield


app = FastAPI(title="TrackletViz API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/api")
app.include_router(tracklets.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
app.include_router(global_clips.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
