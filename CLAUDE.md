# TrackletViz: Surveillance Video Tracklet Visualization System

## Project Overview

TrackletViz is a system for visualizing and summarizing object tracklets in surveillance video data by representing them as 2D feature vectors/embeddings. The system consists of two main components:

1. **Indexing Pipeline**: A standalone script that processes videos to extract tracklets, compute embeddings, perform clustering, and store everything in a Qdrant vector database.
2. **Interactive Frontend**: A web application for browsing, filtering, and analyzing tracklets with multiple visualization panels.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                                │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌────────┐ │
│  │  Video   │ → │ BoxMOT   │ → │ VideoPrism│ → │  UMAP + │ → │ Qdrant │ │
│  │  Input   │   │ Tracking │   │ Embeddings│   │ HDBSCAN │   │   DB   │ │
│  └──────────┘   └──────────┘   └───────────┘   └─────────┘   └────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTERACTIVE FRONTEND                             │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                          Header (Logo + Title + Video Selector)     ││
│  ├─────────────────────────────┬───────────────────────────────────────┤│
│  │                             │  Local: Tab 1: Video Player + Timeline││
│  │    2D Embeddings Panel      │  Local: Tab 2: Heatmap Overlay        ││
│  │    (WebGL-based)            │  Local: Tab 3: Track List + Filters   ││
│  │    - Lasso/Rect Selection   │  Local: Tab 4: Cluster Summaries      ││
│  │    - Color by Class/Cluster │  Local: Tab 5: Video Moment Retrieval ││
│  │    - Thumbnail Tooltips     │  Global: Video / Heatmap / Clusters / ││
│  │                             │         Search / Summarizations       ││
│  └─────────────────────────────┴───────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend / Indexing

- **Python 3.11+** (via `uv` virtual environment - never use system Python)
- **BoxMOT**: Object detection and tracking (https://github.com/mikel-brostrom/boxmot)
- **VideoPrism**: Video foundation model for embeddings (https://github.com/google-deepmind/videoprism)
- **UMAP**: Dimensionality reduction to 2D
- **HDBSCAN**: Density-based clustering
- **Qdrant**: Vector database for storage and retrieval
- **FastAPI**: Lightweight API for text-to-video retrieval

### Frontend

- **React 18** with TypeScript
- **Vite**: Build tool
- **TailwindCSS**: Styling
- **Deck.gl** or **regl-scatterplot**: WebGL-based scatter plot capable of rendering 100K+ points
- **Konva.js**: Canvas-based video overlays (reuse patterns from existing code)
- **React Query**: Data fetching and caching
- **Zustand**: State management

---

## Project Structure

```
TrackletViz/
├── backend/
│   ├── indexer/               # Standalone indexing script
│   │   ├── main.py            # Entry point for indexing
│   │   ├── config.py          # Configuration loading (YAML)
│   │   ├── detector.py        # YOLO + BoxMOT integration
│   │   ├── trajectory.py      # Trajectory extraction
│   │   ├── speed.py           # Speed calculation
│   │   ├── embeddings.py      # VideoPrism embedding extraction
│   │   ├── clustering.py      # UMAP + HDBSCAN + FPS sampling
│   │   ├── storage.py         # Qdrant database operations
│   │   └── thumbnails.py      # Thumbnail generation
│   ├── api/                   # FastAPI backend for retrieval
│   │   ├── main.py            # FastAPI app
│   │   └── routes/
│   │       ├── videos.py      # Video metadata endpoints
│   │       ├── tracklets.py   # Tracklet data endpoints
│   │       └── search.py      # Text-to-video retrieval endpoint
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── config/
│   │   └── default.yaml       # Default configuration
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header/
│   │   │   ├── EmbeddingsPanel/
│   │   │   ├── RightPanel/
│   │   │   │   └── tabs/
│   │   │   └── shared/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── stores/
│   │   └── types/
│   └── package.json
├── existing_code/             # Reference code (do not modify)
└── CLAUDE.md
```

---

## Phase 1: Environment Setup

### 1.1 Python Environment

Use `uv` to create and manage the virtual environment. Install dependencies including:

- PyTorch with CUDA support
- ultralytics and boxmot for detection/tracking
- qdrant-client for vector database
- umap-learn and hdbscan for clustering
- FastAPI and uvicorn for the API
- OpenCV, Pillow for image processing
- PyYAML, loguru for configuration and logging

Clone VideoPrism from https://github.com/google-deepmind/videoprism and install according to their instructions.

### 1.2 Frontend Environment

Initialize a Vite React TypeScript project. Install:

- TailwindCSS for styling
- React Query for data fetching
- deck.gl for WebGL scatter plot rendering
- react-konva for canvas-based overlays
- zustand for state management
- lucide-react for icons

### 1.3 Configuration File

Create a YAML configuration file with sections for:

- **Processing**: YOLO model variant, tracker type, confidence threshold, target COCO classes, minimum tracklet frames (16)
- **VideoPrism**: Model path, batch size, device
- **Clustering**: UMAP parameters (n_neighbors, min_dist, metric, n_components=2), HDBSCAN parameters (min_cluster_size=5, min_samples), FPS representatives count (5)
- **Qdrant**: Host, port, collection name, embedding dimension
- **Thumbnails**: Dimensions, format, quality

---

## Phase 2: Backend Indexing Pipeline

### 2.1 Data Models

Define Pydantic models for:

- **BoundingBox**: frame_num, timestamp, x1/y1/x2/y2, center_x/y, width/height, speed
- **TrackletMetadata**: tracklet_id, video_id, class_name, class_id, list of bounding boxes, bbox_centers (for track lines), start/end timestamps, duration, avg/max speed, point_count, thumbnail_base64, cluster_id, umap_x, umap_y
- **ClusterStatistics**: cluster_id, member_count, avg_speed, class_distribution (percentages), representative_tracklet_ids
- **VideoMetadata**: video_id, video_path, fps, width, height, duration, total_frames, background_image_base64, list of cluster statistics, total_tracklets, class_distribution

### 2.2 Video Processing and Tracking

**Adapt from `existing_code/detector.py`:**

- Reuse the VideoProcessor class structure with YOLO model initialization and BoxMOT tracker setup
- Reuse the frame-by-frame processing generator pattern
- Reuse the YOLO-to-BoxMOT format conversion method
- Modify to load configuration from the YAML config object instead of hardcoded values
- Add a new method to extract a representative background frame using temporal median across sampled frames

### 2.3 Trajectory Extraction

**Adapt from `existing_code/trajectory.py`:**

- Reuse the TrajectoryExtractor class for converting BoxMOT tracks to trajectory points
- Reuse the track history accumulation pattern (per track_id)
- Reuse the statistics computation methods
- Add a filtering method to return only tracklets with at least N frames (configurable, default 16)
- Add a method to get frame ranges for tracklets (for thumbnail and embedding extraction)

### 2.4 Speed Calculation

**Adapt from `existing_code/speed.py`:**

- Reuse the SpeedEstimator class directly with minimal modifications
- Reuse the displacement-based speed calculation logic
- Reuse the speed statistics computation
- Reuse the outlier filtering method

### 2.5 VideoPrism Embedding Extraction

**New module using https://github.com/google-deepmind/videoprism:**

- Create an embedder class that loads the VideoPrism model from the repository
- For each tracklet, extract cropped frames from the video using the bounding boxes
- Sample 16 frames evenly from the tracklet (pad by repeating last frame if fewer)
- Pass the cropped frame sequence through VideoPrism to get a high-dimensional embedding
- Also implement a text embedding method for the retrieval feature (VideoPrism supports text-video alignment)
- Handle batching for efficiency when processing many tracklets

### 2.6 Clustering and Dimensionality Reduction

**New module:**

- Apply UMAP to reduce high-dimensional embeddings to 2D coordinates for visualization
- Apply HDBSCAN clustering on the original high-dimensional embeddings (not the UMAP output)
- HDBSCAN will label noise points as -1; keep these points (do not discard)
- For each cluster (including noise), select representative tracklets using Farthest Point Sampling (FPS):
  - Start with an arbitrary point
  - Iteratively select the point farthest from all already-selected points
  - Continue until K representatives are selected (K=5 or cluster size, whichever is smaller)
- Compute cluster statistics: member count, average speed, class distribution percentages

### 2.7 Thumbnail Generation

**New module:**

- For each tracklet, select the middle frame as the representative
- Extract and crop the region using the bounding box (with small padding)
- Resize to thumbnail dimensions (e.g., 128x128)
- Encode as base64 JPEG for storage and transmission
- Also create a method to encode the background frame as base64

### 2.8 Qdrant Storage

**New module:**

- Create two collections: one for tracklets (with vectors), one for video metadata (metadata only)
- For the tracklets collection, configure vector parameters matching VideoPrism embedding dimension
- Create payload indices on video_id, cluster_id, and class_name for efficient filtering
- Implement upsert methods that delete existing data for a video_id before inserting new data
- Serialize bounding boxes and cluster statistics as nested payloads
- Implement retrieval methods: list videos, get video metadata, get tracklets for video
- Implement similarity search method for text-to-video retrieval

### 2.9 Main Indexing Script

**Orchestration script that runs the full pipeline:**

1. Load configuration from YAML file
2. Generate unique video_id from file path and modification time
3. Initialize video processor and run detection/tracking on all frames
4. Filter tracklets to keep only those with >= minimum frames
5. Calculate speeds for all tracklet points
6. Extract VideoPrism embeddings for each tracklet
7. Run UMAP dimensionality reduction and HDBSCAN clustering
8. Compute cluster statistics and select representative tracklets via FPS
9. Generate thumbnails for all tracklets and background image
10. Build metadata objects for all tracklets and the video
11. Store everything in Qdrant

Provide progress logging at each step.

---

## Phase 3: Backend API

### 3.1 FastAPI Application

Create a FastAPI app with:

- Lifespan handler to initialize Qdrant client and VideoPrism embedder on startup
- CORS middleware configured for frontend origins
- Health check endpoint

### 3.2 API Routes

**Videos routes:**

- GET /api/videos/ - List all indexed videos (video_id, duration, total_tracklets)
- GET /api/videos/{video_id} - Get full metadata for a specific video

**Tracklets routes:**

- GET /api/tracklets/{video_id} - Get all tracklets for a video (supports 100K+ tracklets)

**Search routes:**

- POST /api/search/text - Accept video_id, query text, and limit
  - Convert query to embedding using VideoPrism text encoder
  - Search Qdrant for similar tracklet embeddings within the specified video
  - Return tracklets sorted by similarity score

---

## Phase 4: Frontend Implementation

### 4.1 State Management

Use Zustand to manage global state including:

- List of available videos
- Currently selected video_id
- Video metadata and tracklets for selected video
- Set of selected tracklet IDs (from embeddings panel selection)
- Selection mode (lasso, rectangle, none)
- Color mode (class, cluster)
- Highlighted cluster ID and tracklet ID (for hover effects)
- Active tab index

### 4.2 Header Component

- Display logo and application title
- Dropdown selector populated with available video_ids from the API
- On selection change, fetch video metadata and tracklets, reset selection state

### 4.3 2D Embeddings Panel

**Main visualization of tracklet embeddings:**

Use deck.gl ScatterplotLayer for WebGL rendering (handles 100K+ points efficiently):

- Position each point using umap_x, umap_y coordinates
- Color points based on current color mode (class colors or cluster colors)
- Adjust point opacity based on selection state (dim non-selected when selection exists)
- Increase radius for selected/highlighted points

**Selection tools:**

- Provide icon buttons to toggle between lasso, rectangle, and no selection modes
- Implement lasso selection by tracking mouse path and testing point containment
- Implement rectangle selection by tracking drag bounds and testing point containment
- Update selectedTrackletIds in store when selection completes

**Tooltips:**

- On point hover, display a tooltip near the cursor showing the tracklet's thumbnail image
- Use the thumbnail_base64 field for the image source

**Color mode toggle:**

- Provide toggle buttons to switch between coloring by object class and by cluster_id
- Use distinct color palettes for each mode

**Interaction with other panels:**

- When highlightedClusterId changes (from Tab 4 hover), highlight all points in that cluster
- When highlightedTrackletId changes (from Tab 4 hover), highlight that specific point

### 4.4 Right Panel with Tabs

Container component with tab navigation for the five panels described below.

---

### 4.5 Tab 1: Video Player with Annotated Timeline

**Purpose:** Play video with timeline annotations showing where selected tracklets appear.

**Empty state:** If no tracklets selected, show placeholder message asking user to make a selection.

**Video player:**

- Standard HTML5 video player with the video source
- Overlay bounding boxes when playing a specific tracklet in loop mode

**Annotated timeline:**

- Display a timeline bar representing the video duration
- Color/highlight temporal regions where any selected tracklet is present
- Regions are determined by start/end timestamps of selected tracklets
- Merge overlapping regions for cleaner display

**Region interaction:**

- Clicking an annotated region opens a floating list of tracklet cards
- Each card shows the tracklet thumbnail and object class
- Clicking a card enters loop mode: video plays that tracklet's time segment on loop with bounding boxes overlaid

**Bounding box overlay:**

- When looping a tracklet, find the bounding box corresponding to current video time
- Draw the box on a canvas layer over the video
- Also draw the track line (path of bbox centers) on the overlay

### 4.6 Tab 2: Heatmap Overlay

**Purpose:** Visualize spatial distribution of selected tracklets as a heatmap on the background image.

**Empty state:** If no tracklets selected, show placeholder message.

**Heatmap generation:**

- Start with the video's background_image_base64
- Create an accumulator grid at lower resolution for performance
- For each bounding box of each selected tracklet, increment grid cells that the bbox covers
- Normalize by maximum value
- Apply a color gradient (cool to warm) based on intensity
- Blend with background image
- Render to canvas

**Display:**

- Show the composited heatmap image
- Regions with more bbox overlap appear more intense (warmer colors)

### 4.7 Tab 3: Filterable and Sortable Track List

**Purpose:** Browse individual tracks with filtering and video overlay.

**Empty state:** If no tracklets selected, show placeholder message.

**Layout:** Two rows (horizontal separation) - top for video overlay, bottom for track list.

**Top row - Video overlay with track lines:**

- **Adapt patterns from `existing_code/TrajectoryMap.jsx`:**
  - Reuse the Konva Stage/Layer structure for canvas rendering
  - Reuse the track line rendering logic with color modes
  - Reuse the zoom/pan handling with mouse wheel
  - Reuse the video playback integration with bounding box overlay
  - Reuse the tooltip display on track hover
- Display only tracks that are in the current selection AND pass current filters
- Color each point of each track line based on speed at that point (speed gradient)
- Clicking a track line plays that tracklet segment on loop with bounding boxes

**Bottom row - Track list with filters:**

- **Adapt patterns from `existing_code/TrackListPanel.jsx`:**
  - Reuse the SpeedSparkline component for visualizing speed over time
  - Reuse the class filter toggle badges with color indicators
  - Reuse the speed filter slider with range display
  - Reuse the track card layout with metadata display
  - Reuse the filter state management patterns
- Filter controls:
  - Class filter: toggleable badges for each object class
  - Speed filter: range slider to set minimum speed threshold
- Track cards show:
  - Speed histogram/sparkline over time
  - Average speed value
  - Object class badge
  - Tracklet ID
- Adjusting filters updates both the track list AND the visible tracks in the top row video overlay

### 4.8 Tab 4: Cluster Summaries

**Purpose:** View statistics and representatives for each cluster.

**Display:**

- Show each cluster as a card
- Card border/accent color matches the cluster color in the embeddings panel
- Card contents:
  - Cluster ID (or "Noise" for cluster_id = -1)
  - Member count
  - Average speed of tracklets in cluster
  - Object class distribution as percentages
  - Thumbnails of representative tracklets (from representative_tracklet_ids)

**Interactions:**

- Hovering over a cluster card sets highlightedClusterId in store, which highlights all points of that cluster in the embeddings panel
- Hovering over a representative thumbnail sets highlightedTrackletId in store, which highlights that specific point in the embeddings panel
- Clicking a representative thumbnail opens a modal with a video player that loops the tracklet segment with bounding boxes displayed

### 4.9 Tab 5: Video Moment Retrieval

**Purpose:** Search for tracklets using natural language queries.

**Interface:**

- Text input field for the query
- Submit on Enter or button click

**Search flow:**

1. Send query to backend POST /api/search/text endpoint
2. Backend converts text to embedding using VideoPrism
3. Backend searches Qdrant for similar tracklet embeddings
4. Backend returns tracklets sorted by similarity score

**Results display:**

- Show matching tracklets as cards with thumbnails
- Display similarity score on each card
- Sort by score (highest first)
- Clicking a card opens a modal with video player looping that tracklet with bounding boxes

### 4.10 Global Tab: Summarizations

**Purpose:** Provide aggregate spatial and temporal overviews of all activity in the video without requiring a manual tracklet selection.

This tab is only visible in Global View mode. It has two subtabs: **Spatial** and **Temporal**.

**Spatial subtab:**
- Divides the video into N time buckets (1–16, configurable) and renders a 128×72 occupancy heatmap per bucket
- Two accumulation modes: Centroid (bbox center only) and BBox (full bounding box area)
- Class filter toggles limit which object classes contribute to the grid
- Clicking a bucket sets `highlightedSpatialClipIds` in the Zustand store, which highlights matching clips in the global scatter plot and auto-switches the color mode to temporal
- Module-level caches (`bboxCache`, `gridCache`) persist across tab switches; batch-fetch bboxes in chunks of 20 via POST `/api/tracklets/batch`

**Temporal subtab:**
- SVG activity chart with per-class polylines; Y-axis switchable between Count and Speed (px/s)
- Configurable bucket duration (presets: 5 m, 10 m, 30 m, 1 h; custom: min 300 s)
- Keyframe storyboard below the chart: k representative clips per bucket (k = 1–5); uses existing `is_representative` and `start_time`/`end_time` fields from `/api/global-clips/{videoId}`
- Thumbnail click opens a loop modal with the video player looping the clip's time segment
- Class filter badges control chart lines and storyboard clips
- All state (bucket duration, metric, k, selected classes) persists across tab switches

---

## Phase 5: Integration and Testing

### 5.1 Infrastructure Setup

- Run Qdrant using Docker with persistent storage volume
- Configure ports for Qdrant (6333, 6334)

### 5.2 Testing Workflow

1. Index a test video using the indexing script
2. Verify data in Qdrant (correct tracklet count, embeddings, metadata)
3. Start the FastAPI backend
4. Start the frontend development server
5. Test the complete user flow

### 5.3 Testing Checklist

**Backend indexing:**

- Video processing completes without errors
- Tracklets correctly filtered by minimum frame count
- Embeddings have expected dimensionality from VideoPrism
- UMAP produces valid 2D coordinates
- HDBSCAN produces cluster labels (with -1 for noise)
- FPS correctly selects diverse representatives per cluster
- Thumbnails generated and properly encoded
- All data stored in Qdrant with correct structure

**API:**

- Video list endpoint returns all indexed videos
- Video metadata endpoint returns complete metadata
- Tracklets endpoint returns all tracklets for a video
- Search endpoint returns relevant results sorted by score

**Frontend:**

- Video selector populates correctly
- Scatter plot renders smoothly with 100K+ points
- Selection tools (lasso, rectangle) work correctly
- Color mode toggle updates point colors
- Tooltip shows thumbnail on hover
- All five tabs function as specified
- Cross-panel interactions work (hover highlighting, selection sync)

---

## Phase 6: Optimization and Polish

### 6.1 Performance Considerations

**Scatter plot:**

- Use deck.gl binary attributes for better performance with large datasets
- Consider view-dependent level of detail

**Data loading:**

- Lazy load thumbnails in scrollable lists
- Use virtualization for long track lists
- Consider pagination for API endpoints if needed

**Caching:**

- Cache video metadata and tracklet data in React Query
- Add HTTP caching headers for static data

### 6.2 UI Polish

- Add loading skeletons while data fetches
- Add smooth transitions between states
- Show progress indicators during long operations
- Implement error boundaries with retry options

### 6.3 Accessibility

- Keyboard navigation for selection and tabs
- Proper ARIA labels
- Sufficient color contrast

---

## Appendix: Existing Code Reference

### Files and Their Reuse

| File                               | What to Reuse                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `existing_code/detector.py`        | VideoProcessor class structure, YOLO/BoxMOT integration, frame processing generator, format conversion                                            |
| `existing_code/trajectory.py`      | TrajectoryExtractor class, track history accumulation, statistics methods                                                                         |
| `existing_code/speed.py`           | SpeedEstimator class (use with minimal changes), displacement calculation, statistics                                                             |
| `existing_code/summarizer.py`      | Reference for time-based aggregation patterns (not directly needed, but useful for understanding data flow)                                       |
| `existing_code/TrackListPanel.jsx` | SpeedSparkline component, class filter toggles, speed filter slider, track card layout, filter state management                                   |
| `existing_code/TrajectoryMap.jsx`  | Konva canvas structure, track line rendering, color modes, zoom/pan handling, video playback integration, bbox overlay animation, tooltip display |

### Color Utilities

Both `existing_code/TrackListPanel.jsx` and `existing_code/TrajectoryMap.jsx` import from a colors utility:

- `getClassColor(className)` - returns color for object class
- `speedToColor(speed, maxSpeed)` - returns color on gradient based on speed

Recreate these utilities in the frontend, adding:

- `getClusterColor(clusterId)` - returns color from categorical palette for cluster visualization

---

## Key Implementation Notes

### VideoPrism Integration

- Clone from https://github.com/google-deepmind/videoprism
- Follow their installation instructions for dependencies
- The model supports both video embedding and text embedding for cross-modal retrieval
- Use their provided inference APIs for extracting embeddings

### Handling Large Numbers of Tracklets

- The frontend must handle 100K+ points - SVG-based rendering will not work
- deck.gl with WebGL is designed for this scale
- Qdrant efficiently handles similarity search on large collections
- Use scroll/pagination in Qdrant when retrieving all tracklets

### Selection State Flow

1. User makes selection in embeddings panel (lasso/rectangle)
2. Selected tracklet IDs stored in Zustand
3. All tabs react to selection changes
4. Tabs display "make a selection" placeholder when selection is empty
5. Filters in Tab 3 work within the current selection

### Cross-Panel Highlighting

1. Hovering cluster card in Tab 4 → sets highlightedClusterId → embeddings panel highlights cluster points
2. Hovering representative thumbnail in Tab 4 → sets highlightedTrackletId → embeddings panel highlights that point
3. Clearing hover → clears highlight state → embeddings panel returns to normal display
