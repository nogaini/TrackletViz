"""
Configuration loading for TrackletViz indexing pipeline.

Reads default.yaml (or a user-specified path) and returns a typed AppConfig
object whose sub-configs are passed directly to each pipeline module.
"""

from dataclasses import dataclass, field
from typing import List
import yaml

from models.schemas import ProcessingConfig


# ── Sub-configs not already defined in schemas.py ─────────────────────────

@dataclass
class VideoPrismConfig:
    model_path: str = "../videoprism"
    model_name: str = "videoprism_lvt_public_v1_base"
    batch_size: int = 8
    device: str = "cuda"
    num_frames: int = 16


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    n_components: int = 2
    random_state: int = 42


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 5
    min_samples: int = 3
    metric: str = "cosine"
    cluster_selection_method: str = "eom"


@dataclass
class PreprocessConfig:
    l2_normalize: bool = False
    pre_umap_dims: int = 0          # 0 = disabled; >0 = reduce to this many dims before clustering
    pre_umap_n_neighbors: int = 15
    pre_umap_metric: str = "cosine"
    pre_umap_random_state: int = 42
    pre_umap_min_dist: float = 0.0


@dataclass
class ClusteringConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    fps_representatives: int = 10


@dataclass
class GlobalClipsConfig:
    clip_duration: float = 10.0
    num_frames: int = 16
    thumbnail_width: int = 320
    flow_width: int = 160        # target width for downscaled optical flow grid
    median_frame_width: int = 640  # target width for median frame JPEG
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    umap: UMAPConfig = field(default_factory=lambda: UMAPConfig(
        n_neighbors=5, min_dist=0.05, metric="cosine", n_components=2, random_state=42
    ))
    hdbscan: HDBSCANConfig = field(default_factory=lambda: HDBSCANConfig(
        min_cluster_size=3, min_samples=1, metric="euclidean", cluster_selection_method="eom"
    ))
    fps_representatives: int = 5


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    tracklets_collection: str = "tracklets"
    videos_collection: str = "videos"
    global_clips_collection: str = "global_clips"
    vector_dim: int = 1024
    timeout_seconds: int = 60


@dataclass
class MLLMConfig:
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_tokens: int = 60
    timeout_s: float = 30.0
    enabled: bool = True


@dataclass
class ThumbnailConfig:
    width: int = 128
    height: int = 128
    format: str = "jpeg"
    quality: int = 85
    padding: int = 8


# ── Top-level app config ───────────────────────────────────────────────────

@dataclass
class AppConfig:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    videoprism: VideoPrismConfig = field(default_factory=VideoPrismConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    thumbnails: ThumbnailConfig = field(default_factory=ThumbnailConfig)
    global_clips: GlobalClipsConfig = field(default_factory=GlobalClipsConfig)
    mllm: MLLMConfig = field(default_factory=MLLMConfig)
    cache_dir: str = ".indexer_cache"


# ── Loader ─────────────────────────────────────────────────────────────────

def load_config(path: str) -> AppConfig:
    """
    Load AppConfig from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Fully populated AppConfig with all sub-configs.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    p = raw.get("processing", {})
    processing = ProcessingConfig(
        yolo_model=p.get("yolo_model", "yolo11n.pt"),
        tracker=p.get("tracker", "botsort"),
        confidence_threshold=p.get("confidence_threshold", 0.3),
        target_classes=p.get("target_classes", [0, 1, 2, 3, 5, 7]),
        min_tracklet_frames=p.get("min_tracklet_frames", 16),
        device=p.get("device", "cuda"),
    )

    vp = raw.get("videoprism", {})
    videoprism = VideoPrismConfig(
        model_path=vp.get("model_path", "../videoprism"),
        model_name=vp.get("model_name", "videoprism_lvt_public_v1_base"),
        batch_size=vp.get("batch_size", 8),
        device=vp.get("device", "cuda"),
        num_frames=vp.get("num_frames", 16),
    )

    c = raw.get("clustering", {})
    u = c.get("umap", {})
    h = c.get("hdbscan", {})
    pp = c.get("preprocess", {})
    clustering = ClusteringConfig(
        preprocess=PreprocessConfig(
            l2_normalize=pp.get("l2_normalize", False),
            pre_umap_dims=pp.get("pre_umap_dims", 0),
            pre_umap_n_neighbors=pp.get("pre_umap_n_neighbors", 15),
            pre_umap_metric=pp.get("pre_umap_metric", "cosine"),
            pre_umap_random_state=pp.get("pre_umap_random_state", 42),
            pre_umap_min_dist=pp.get("pre_umap_min_dist", 0.0),
        ),
        umap=UMAPConfig(
            n_neighbors=u.get("n_neighbors", 15),
            min_dist=u.get("min_dist", 0.1),
            metric=u.get("metric", "cosine"),
            n_components=u.get("n_components", 2),
            random_state=u.get("random_state", 42),
        ),
        hdbscan=HDBSCANConfig(
            min_cluster_size=h.get("min_cluster_size", 5),
            min_samples=h.get("min_samples", 3),
            metric=h.get("metric", "euclidean"),
            cluster_selection_method=h.get("cluster_selection_method", "eom"),
        ),
        fps_representatives=c.get("fps_representatives", 5),
    )

    q = raw.get("qdrant", {})
    qdrant = QdrantConfig(
        host=q.get("host", "localhost"),
        port=q.get("port", 6333),
        tracklets_collection=q.get("tracklets_collection", "tracklets"),
        videos_collection=q.get("videos_collection", "videos"),
        global_clips_collection=q.get("global_clips_collection", "global_clips"),
        vector_dim=q.get("vector_dim", 1024),
        timeout_seconds=q.get("timeout_seconds", 60),
    )

    t = raw.get("thumbnails", {})
    thumbnails = ThumbnailConfig(
        width=t.get("width", 128),
        height=t.get("height", 128),
        format=t.get("format", "jpeg"),
        quality=t.get("quality", 85),
        padding=t.get("padding", 8),
    )

    gc = raw.get("global_clips", {})
    gcu = gc.get("umap", {})
    gch = gc.get("hdbscan", {})
    gcpp = gc.get("preprocess", {})
    global_clips_cfg = GlobalClipsConfig(
        clip_duration=gc.get("clip_duration", 10.0),
        num_frames=gc.get("num_frames", 16),
        thumbnail_width=gc.get("thumbnail_width", 320),
        flow_width=gc.get("flow_width", 160),
        median_frame_width=gc.get("median_frame_width", 640),
        preprocess=PreprocessConfig(
            l2_normalize=gcpp.get("l2_normalize", False),
            pre_umap_dims=gcpp.get("pre_umap_dims", 0),
            pre_umap_n_neighbors=gcpp.get("pre_umap_n_neighbors", 15),
            pre_umap_metric=gcpp.get("pre_umap_metric", "cosine"),
            pre_umap_random_state=gcpp.get("pre_umap_random_state", 42),
            pre_umap_min_dist=gcpp.get("pre_umap_min_dist", 0.0),
        ),
        umap=UMAPConfig(
            n_neighbors=gcu.get("n_neighbors", 5),
            min_dist=gcu.get("min_dist", 0.05),
            metric=gcu.get("metric", "cosine"),
            n_components=gcu.get("n_components", 2),
            random_state=gcu.get("random_state", 42),
        ),
        hdbscan=HDBSCANConfig(
            min_cluster_size=gch.get("min_cluster_size", 3),
            min_samples=gch.get("min_samples", 1),
            metric=gch.get("metric", "euclidean"),
            cluster_selection_method=gch.get("cluster_selection_method", "eom"),
        ),
        fps_representatives=gc.get("fps_representatives", 5),
    )

    m = raw.get("mllm", {})
    mllm_cfg = MLLMConfig(
        base_url=m.get("base_url", "http://localhost:8001/v1"),
        model=m.get("model", "Qwen/Qwen2-VL-2B-Instruct"),
        max_tokens=m.get("max_tokens", 60),
        timeout_s=m.get("timeout_s", 30.0),
        enabled=m.get("enabled", True),
    )

    cache_dir = raw.get("cache_dir", ".indexer_cache")

    return AppConfig(
        processing=processing,
        videoprism=videoprism,
        clustering=clustering,
        qdrant=qdrant,
        thumbnails=thumbnails,
        global_clips=global_clips_cfg,
        mllm=mllm_cfg,
        cache_dir=cache_dir,
    )
