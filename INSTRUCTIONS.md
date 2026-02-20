# TrackletViz — Setup & Usage Guide

## 1. Overview

TrackletViz processes surveillance videos to extract object tracklets, compute high-dimensional embeddings with VideoPrism, reduce them to 2D with UMAP, cluster them with HDBSCAN, and store everything in Qdrant. The interactive frontend lets you explore, filter, and search tracklets visually.

Three services must be running simultaneously:

| Service | What it does |
|---------|-------------|
| **Qdrant** | Vector database storing tracklets and embeddings |
| **Backend API** | FastAPI server serving tracklet data and text search |
| **Frontend** | React app at http://localhost:5173 |

---

## 2. Prerequisites

- **Docker** — to run Qdrant
- **Python 3.11** and **`uv`** — backend virtual environment
- **Node.js 20+** — frontend build tooling
- **CUDA-capable GPU** — required for VideoPrism embeddings (JAX + CUDA)
- **VideoPrism repository** — must be cloned manually (not available on PyPI)

---

## 3. One-Time Setup

### 3a. Clone and Install VideoPrism

VideoPrism is a Google DeepMind model not available on PyPI. Clone it and install it into the backend virtual environment:

```bash
git clone https://github.com/google-deepmind/videoprism
cd videoprism
/home/jobin/Projects/TrackletViz/backend/.venv/bin/pip install -e .
cd ..
```

After installation, set `videoprism.model_path` in `backend/config/default.yaml` to the path of the cloned repository (see Section 4).

### 3b. Start Qdrant (Docker)

Run Qdrant with a persistent storage volume. Run this from the project root:

```bash
docker run -d --name qdrant-trackviz \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

To stop and restart later:

```bash
docker stop qdrant-trackviz
docker start qdrant-trackviz
```

### 3c. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## 4. Configuration

The configuration file is `backend/config/default.yaml`. Key fields to review before first use:

```yaml
processing:
  yolo_model: yolo11n.pt         # YOLO variant: n (fastest) / s / m / l / x (most accurate)
  tracker: botsort               # BoxMOT tracker: botsort, bytetrack, ocsort, etc.
  confidence_threshold: 0.3      # Minimum detection confidence (0.0–1.0)
  target_classes: [0, 1, 2, 3, 5, 7]  # COCO IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
  min_tracklet_frames: 16        # Discard tracks shorter than this many frames
  device: cuda                   # 'cuda' or 'cpu'

videoprism:
  model_path: ../videoprism      # Path to cloned VideoPrism repo (relative to backend/, or absolute)
  model_name: videoprism_lvt_public_v1_base
  batch_size: 8                  # Tracklets per embedding batch (reduce if OOM)
  device: cuda

clustering:
  umap:
    n_neighbors: 15              # Higher = more global structure
    min_dist: 0.1
    metric: cosine
  hdbscan:
    min_cluster_size: 2          # Minimum points to form a cluster
    min_samples: 1

qdrant:
  host: localhost
  port: 6333                     # Change if Qdrant runs on a different port
```

---

## 5. Indexing a Video

The indexing pipeline runs 13 steps and stores results in Qdrant. Run it from the `backend/` directory:

```bash
cd backend
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python indexer/main.py --video /path/to/video.mp4
```

Optionally specify a non-default config:

```bash
.venv/bin/python indexer/main.py --video /path/to/video.mp4 --config config/default.yaml
```

To attach a wall-clock start time to the video, pass `--start-time` in `YYYYMMDDTHHMMSS` format:

```bash
.venv/bin/python indexer/main.py --video /path/to/video.mp4 --start-time 20240315T143022
```

When `--start-time` is provided, each tracklet's `start_world_time` and `end_world_time` fields are computed as `video_start_time + tracklet.start/end_timestamp` and stored in Qdrant (ISO format `YYYY-MM-DDTHH:MM:SS`). The `video_start_time` is also stored in the video metadata. When the flag is omitted these fields are `null` — fully backwards compatible with previously indexed videos.

To attach a human-readable label to the video (shown in the UI dropdown), pass `--tag`:

```bash
.venv/bin/python indexer/main.py --video /path/to/video.mp4 --tag "morning_lot"
```

The tag is stored alongside the video metadata and displayed in the header dropdown in place of the raw `video_id`. Tags do not need to be unique, but using unique tags makes deletion by tag unambiguous.

**The `XLA_PYTHON_CLIENT_PREALLOCATE=false` environment variable is required** to prevent JAX from pre-allocating all GPU memory, which would conflict with PyTorch.

### Pipeline Steps

| Step | Description | Notes |
|------|-------------|-------|
| 1 | Load configuration | |
| 2 | Generate `video_id` | SHA-256 of absolute path + modification time — re-indexing the same file replaces existing data |
| 3 | Extract background frame | Temporal median of sampled frames |
| 4 | Detection + tracking | YOLO + BoxMOT on every frame |
| 5 | Filter short tracklets | Drops tracks with fewer than `min_tracklet_frames` frames |
| 6 | Speed calculation | Displacement-based, outliers filtered |
| 7 | Build tracklet clips | Prepares frame ranges and bboxes for embedding |
| 8 | VideoPrism embeddings | Most time-consuming step; processes in batches on GPU |
| 9 | UMAP + HDBSCAN | Reduces to 2D; clusters in original high-dim space |
| 10 | Generate thumbnails | 128×128 JPEG crops of representative frames |
| 11 | Build metadata objects | Assembles all data per tracklet |
| 12 | Compute cluster stats | Speed averages, class distributions, FPS representatives |
| 13 | Store in Qdrant | Upserts all data; old data for this `video_id` is replaced |

When complete, the terminal will print the `video_id`, tracklet count, cluster count, and class distribution.

### Deleting an indexed video

To remove all data for a video from Qdrant, pass its `video_id` to `--delete-video`:

```bash
.venv/bin/python indexer/main.py --delete-video <video_id>
```

The `video_id` is printed at the end of every indexing run ("Indexing complete!" block).
This deletes all tracklet points and the video metadata record from both Qdrant collections.

If you indexed with `--tag`, you can also delete by tag without needing the `video_id`:

```bash
.venv/bin/python indexer/main.py --delete-tag "morning_lot"
```

If no video with that tag exists, the command exits with an error.

---

## 6. Starting the Backend API Server

Run from the `backend/` directory:

```bash
cd backend
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Note:** On first startup, the server loads the VideoPrism model into GPU memory (~991 MB). This takes 20–60 seconds. Wait for the log line confirming the server is ready before using the frontend.

To use a non-default config path:

```bash
export TRACKVIZ_CONFIG=config/default.yaml   # this is already the default
```

---

## 7. Starting the Frontend

```bash
cd frontend
npm run dev
```

The development server starts at **http://localhost:5173** and proxies all `/api` requests to the backend at `http://localhost:8000`.

---

## 8. Complete Startup Checklist

Run these in order:

1. **Qdrant** — `docker start qdrant-trackviz` (or the full `docker run` command if first time)
2. **Index video(s)** — run the indexing pipeline for each video you want to explore
3. **Backend API** — `cd backend && export XLA_PYTHON_CLIENT_PREALLOCATE=false && .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
4. **Frontend** — `cd frontend && npm run dev`
5. **Open** http://localhost:5173

---

## 9. Using the UI

### Header — Video Selector

The dropdown in the header lists all indexed videos. Videos indexed with `--tag` display their tag; others display their raw `video_id`. Select a video to load its tracklets and metadata. The scatter plot and all tabs update automatically.

### 2D Embeddings Panel (left side)

The main scatter plot renders all tracklets as points positioned by their UMAP 2D coordinates.

- **Color mode** — toggle between coloring by object class or by cluster ID using the buttons above the plot
- **Selection tools** — use the toolbar to switch between lasso selection, rectangle selection, and no-selection mode; click and drag to select a group of points
- **Tooltip** — hover over any point to see a thumbnail of that tracklet
- **Selection effect** — once a selection is made, all tabs on the right panel activate and show data for selected tracklets only; non-selected points are dimmed

### Tab 0: Video Player

- Shows an annotated timeline bar over the video duration
- Colored regions on the timeline indicate where selected tracklets appear in time
- Click a region to see a list of tracklet cards for that time window
- Click a card to enter **loop mode**: the video plays that tracklet's segment on repeat with a bounding box and track line overlaid

### Tab 1: Heatmap

- Visualizes the spatial density of selected tracklets on the video's background frame
- Regions where tracked objects spent more time appear warmer (red/orange); sparse regions are cooler (blue)
- Useful for identifying high-activity zones in the scene

### Tab 2: Track List

- Top section: canvas showing track lines for all filtered tracklets, colored by speed (cool → warm)
  - Click a track line to loop that tracklet in the video with bounding box overlay
  - Scroll to zoom, drag to pan
- Bottom section: filterable list of tracklet cards
  - **Class filter**: toggle badges to show/hide specific object classes
  - **Speed filter**: slider to set minimum average speed threshold
  - Each card shows a speed sparkline, average speed, class badge, and tracklet ID
  - Filters affect both the list and the canvas track lines

### Tab 3: Cluster Summaries

- Each cluster appears as a card with color-coded border matching the scatter plot
- Cards show: cluster ID (or "Noise" for cluster −1), member count, average speed, class distribution percentages, and thumbnail representatives
- **Hover a card** → highlights all points of that cluster in the scatter plot
- **Hover a representative thumbnail** → highlights that specific point in the scatter plot
- **Click a representative thumbnail** → opens a modal with a looping video player for that tracklet

### Tab 4: Text Search

- Enter a natural language query (e.g., "person running", "red car turning") and press Enter
- The backend converts the query to a VideoPrism text embedding and searches Qdrant for the most similar tracklet embeddings within the current video
- Results are shown as cards sorted by similarity score (highest first)
- Click a card to open a looping video player for that tracklet

---

## 10. API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/api/videos/` | List all indexed videos with summary info |
| `GET` | `/api/videos/{video_id}` | Full metadata for a specific video (background image, cluster stats, class distribution) |
| `GET` | `/api/tracklets/{video_id}` | All tracklets for a video including UMAP coords, bboxes, thumbnails |
| `GET` | `/api/videos/{video_id}/stream` | Range-aware video stream (supports seeking in HTML5 video element) |
| `POST` | `/api/search/text` | Text-to-tracklet search; body: `{"video_id": "...", "query": "...", "limit": 20}` |

---

## 11. Troubleshooting

**Qdrant connection error during indexing or API startup**
- Verify Qdrant is running: `docker ps | grep qdrant-trackviz`
- Start it if stopped: `docker start qdrant-trackviz`
- Check that port 6333 is not blocked: `curl http://localhost:6333/healthz`

**CUDA / JAX import errors or GPU OOM**
- Always set `export XLA_PYTHON_CLIENT_PREALLOCATE=false` before running the indexer or API server
- If OOM during embedding, reduce `videoprism.batch_size` in `default.yaml`
- The indexer clears the PyTorch GPU cache before JAX takes over (Step 8), but GPU memory must fit both model weights

**VideoPrism model not found**
- Confirm `videoprism.model_path` in `default.yaml` points to the cloned repository directory
- The path is relative to `backend/`; use an absolute path to avoid ambiguity
- Run `.venv/bin/python -c "import videoprism"` to verify the package is importable

**Frontend shows network errors / blank data**
- Confirm the backend is running on port 8000: `curl http://localhost:8000/health`
- The Vite dev server proxies `/api` to `http://localhost:8000` — both must be running
- Check the browser console for specific error messages

**"No videos found" in the header dropdown**
- The video must be indexed before the API can serve it
- Run the indexing pipeline (Section 5) and wait for the "Indexing complete!" message
- Confirm Qdrant received the data: `curl http://localhost:6333/collections`

**Re-indexing a video**
- The pipeline uses a hash of the file path + modification time as `video_id`
- Modifying and re-saving the file produces a new `video_id` (both versions will exist in Qdrant)
- To replace data for the exact same file, the pipeline automatically deletes old data for that `video_id` before upserting

---

## 12. Project Structure

```
TrackletViz/
├── backend/
│   ├── .venv/                  # Python 3.11 virtual environment (uv-managed)
│   ├── config/
│   │   └── default.yaml        # Main configuration file
│   ├── indexer/                # Standalone indexing pipeline
│   │   ├── main.py             # Entry point (13-step pipeline)
│   │   ├── config.py           # YAML config loading
│   │   ├── detector.py         # YOLO + BoxMOT integration
│   │   ├── trajectory.py       # Track history accumulation
│   │   ├── speed.py            # Displacement-based speed calculation
│   │   ├── embeddings.py       # VideoPrism embedding extraction
│   │   ├── clustering.py       # UMAP + HDBSCAN + FPS sampling
│   │   ├── storage.py          # Qdrant read/write operations
│   │   └── thumbnails.py       # JPEG thumbnail generation
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             # App with lifespan, CORS, routers
│   │   └── routes/
│   │       ├── videos.py       # GET /api/videos/ and /api/videos/{id}
│   │       ├── tracklets.py    # GET /api/tracklets/{video_id}
│   │       ├── search.py       # POST /api/search/text
│   │       └── stream.py       # GET /api/videos/{id}/stream
│   ├── models/
│   │   └── schemas.py          # Pydantic data models
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Root layout (Header + EmbeddingsPanel + RightPanel)
│   │   ├── components/
│   │   │   ├── Header/         # Video selector dropdown
│   │   │   ├── EmbeddingsPanel/# deck.gl scatter plot with selection tools
│   │   │   └── RightPanel/
│   │   │       └── tabs/
│   │   │           ├── VideoPlayerTab.tsx
│   │   │           ├── HeatmapTab.tsx
│   │   │           ├── TrackListTab.tsx
│   │   │           ├── ClusterSummariesTab.tsx
│   │   │           └── TextSearchTab.tsx
│   │   ├── hooks/              # React Query data fetching hooks
│   │   ├── lib/                # Shared utilities (colors, etc.)
│   │   ├── stores/             # Zustand global state
│   │   └── types/              # TypeScript type definitions
│   └── package.json
├── existing_code/              # Reference code (do not modify)
├── qdrant_storage/             # Qdrant persistent data (created by Docker volume)
├── CLAUDE.md                   # Project specification for AI assistance
└── INSTRUCTIONS.md             # This file
```
