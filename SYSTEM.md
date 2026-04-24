# TrackletViz: An Interactive System for Surveillance Video Tracklet Exploration

---

## 1. System Overview

### 1.1 Problem and Motivation

Surveillance video streams produce vast quantities of temporal data in which objects of interest — people, vehicles, animals — appear, move, and disappear across thousands of frames. Analysts working with these streams face two compounding challenges: the sheer volume of data renders manual inspection infeasible, and the intrinsic high dimensionality of visual content makes it difficult to quickly identify behaviorally or semantically similar observations. Existing tools typically offer either low-level video playback or rigid query interfaces; they rarely provide holistic, exploratory views that let a user discover structure in tracking data without knowing what they are looking for in advance.

TrackletViz addresses these challenges by building an end-to-end pipeline that transforms raw surveillance video into a richly annotated, interactively explorable representation. The system extracts individual object tracklets — contiguous sequences of bounding boxes belonging to a single tracked entity — encodes each one as a high-dimensional semantic embedding using a video foundation model, clusters them in embedding space, and projects the entire collection into a navigable two-dimensional scatter plot. Alongside this tracklet-level view, the system provides a complementary scene-level view that divides the video into fixed-duration clips, embeds them independently, and exposes scene-change analysis tools including motion-flow visualizations and illumination-shift comparisons. Both views are served through a unified interactive application that supports spatial, temporal, and semantic querying without leaving the interface.

---

### 1.2 Indexing Pipeline

The offline indexing pipeline is the computational backbone of TrackletViz. It ingests a raw video file and produces all derivative data structures — embeddings, cluster assignments, thumbnails, and metadata — storing them in a vector database for real-time retrieval. The pipeline is organized into 21 sequential steps executed in two phases: the **local (tracklet-level)** phase and the **global (clip-level)** phase.

#### 1.2.1 Local Phase: Tracklet Extraction and Embedding

**Detection and Tracking.** The pipeline begins by running a multi-object detector and tracker on every frame of the video. The detector identifies bounding boxes for target object classes (persons, vehicles, cyclists, etc.), and the tracker associates boxes across frames to form persistent identities. The output is a set of raw trajectories — one per detected identity — represented as sequences of (frame, bounding box, timestamp) tuples.

**Tracklet Filtering.** Trajectories shorter than a minimum duration threshold (default: 16 frames, configurable) are discarded. These ultra-short tracks arise from detection noise and are unlikely to carry meaningful semantic content. This filtering step also ensures that subsequent embedding extraction, which relies on sampling a fixed number of frames per tracklet, is meaningful.

**Speed Estimation.** The system computes per-frame instantaneous speed for each tracklet by measuring Euclidean displacement between consecutive bounding box centers, normalized by the inter-frame time interval. Outlier speeds (exceeding twice the observed maximum) are clipped. Per-tracklet statistics (mean speed, maximum speed) are then derived.

**Embedding Extraction.** Each surviving tracklet is encoded into a 768-dimensional vector using a video foundation model (VideoPrism LvT). For each tracklet, 16 frames are sampled evenly across its temporal span; if the tracklet contains fewer than 16 frames, the last frame is repeated to pad the sequence. The crop region for each frame is the tracklet's bounding box at that moment, padded by a fixed margin and resized to a standard spatial resolution. These cropped frame sequences are passed through the model in batches, and the resulting embeddings capture rich semantic and appearance information about the tracked object.

**Dimensionality Reduction and Clustering.** The full collection of 768-dimensional embeddings is processed in two ways simultaneously:

- *UMAP* reduces the embeddings to two dimensions for visualization. This projection is computed globally so that relative distances in the scatter plot reflect semantic similarity. Configuration parameters (neighborhood count, minimum distance, distance metric) control the trade-off between local detail preservation and global structure.

- *HDBSCAN* density-based clustering is applied to the original high-dimensional embeddings (not the UMAP projections) to assign each tracklet a cluster label. Points that do not belong to any dense cluster receive a noise label (−1). Clustering in the original high-dimensional space avoids artifacts introduced by dimensionality reduction and yields semantically coherent groups.

**Representative Selection via Farthest Point Sampling.** For each cluster, the system selects a small set of representative tracklets using Farthest Point Sampling (FPS). Starting from an arbitrary point, FPS greedily adds the point that maximizes the minimum distance to all already-selected points. This deterministic algorithm ensures diversity: the representatives span the within-cluster semantic variation rather than clustering near the centroid.

**Thumbnail Generation and Metadata Assembly.** A 128×128 JPEG thumbnail is generated for each tracklet by cropping the video frame at the tracklet's temporal midpoint to the bounding box region. The video's background frame is computed as the per-pixel temporal median across a set of sampled frames, serving as a static reference for spatial overlays. Per-tracklet and per-cluster metadata is assembled into structured records and written to the vector database alongside the raw embeddings.

#### 1.2.2 Global Phase: Scene-Level Clip Analysis

**Clip Segmentation.** The video is divided into non-overlapping, fixed-duration clips (default: 10 seconds each). Each clip is assigned a unique identifier and associated with the frame range and timestamps it spans.

**Clip Embedding.** Sixteen frames are sampled evenly from each clip's frame range and passed through the same video foundation model used for tracklets. Because clips capture full-scene content rather than individual object crops, these embeddings represent scene-level appearance and activity. Embedding extraction for clips is performed in a streaming fashion — one batch of clips is loaded, embedded, and freed before the next batch is processed — to control peak memory usage on long videos.

**Clip Clustering.** The same UMAP and HDBSCAN pipeline applied to tracklets is independently applied to clip embeddings, with parameters tuned for scene-level granularity. This yields a separate two-dimensional scatter plot of clips and a separate set of cluster assignments.

**Auxiliary Feature Extraction.** For each representative clip, three additional features are computed:

- *Median Frame*: The per-pixel median of frames sampled across the clip, providing a static composite that suppresses transient objects and reveals the persistent scene structure.
- *Optical Flow*: Dense Farneback flow is computed between consecutive sampled frame pairs and averaged, then downsampled to a coarse grid. The resulting velocity field (horizontal and vertical components per grid cell) encodes aggregate motion patterns within the clip.
- *Thumbnail*: A full-scene frame from the clip's midpoint, resized for display.

**Tracklet–Clip Association.** Each clip is linked to all tracklets whose temporal span overlaps with the clip's time interval. This cross-index enables the interface to highlight relevant tracklets when a clip is selected.

#### 1.2.3 Storage Schema

All derived data is stored in a vector database organized into three collections:

- **Tracklets**: Each record stores the 768-dimensional embedding as its searchable vector and carries a rich payload including cluster assignment, 2D UMAP coordinates, bounding box sequence, speed statistics, thumbnail, and class label. Payload indices on video ID, cluster ID, and class name enable efficient server-side filtering without scanning the full collection.

- **Videos**: Each record stores the full video-level metadata — frame dimensions, duration, background image, cluster statistics, class distribution — without a searchable vector. This collection acts as a key-value store for session initialization.

- **Global Clips**: Each record stores the clip's 768-dimensional embedding alongside clip-level metadata, thumbnail, median frame, and encoded optical flow. Payload indices on video ID and cluster ID support filtering.

This schema supports both similarity-based retrieval (text-to-video search via embedding comparison) and structured retrieval (fetching all tracklets for a video, paginated).

#### 1.2.4 Checkpoint-Based Resumption

The indexing pipeline maintains an atomic checkpoint after each major step. Intermediate results (trajectories, embeddings, clip descriptors) are serialized to disk and protected by atomic write-then-rename operations. On re-run, completed steps are detected and skipped, and the pipeline resumes from the last successful checkpoint. A pre-run database scan also detects whether a video has already been fully indexed, preventing unnecessary recomputation.

---

### 1.3 Query API

A lightweight REST API serves all retrieval operations to the frontend. Key endpoints include:

- **Video listing**: Returns lightweight summaries (ID, path, duration, tracklet count) for all indexed videos.
- **Video metadata**: Returns the full video record including cluster statistics for session initialization.
- **Tracklet retrieval**: Fetches all tracklets for a video, supporting configurable pagination and optional stripping of heavyweight fields (bounding boxes, thumbnails) for lightweight initial loads.
- **Batch tracklet fetch**: Retrieves full bounding-box data for a specified set of tracklet IDs on demand.
- **Text search (tracklets)**: Accepts a natural-language query, encodes it to a 768-dimensional text embedding using the foundation model's text encoder, and returns the top-K most similar tracklets within a specified video using cosine similarity.
- **Global clip retrieval and search**: Mirrors the tracklet endpoints for the clip collection, with optional inclusion of heavy fields (median frames, optical flow).
- **Video streaming**: Range-aware HTTP byte-serving of the original video file, enabling seeking in the browser video player.

---

### 1.4 Optimizations

TrackletViz incorporates a suite of optimizations spanning the indexing pipeline, storage layer, and client-server data flow.

**JIT-Compiled Inference.** Embedding extraction is accelerated by ahead-of-time JIT compilation of the video model's forward pass. The compiled function is cached after the first call, yielding substantially faster inference on subsequent batches.

**Streaming Clip Embedding.** Rather than materializing all clip frame arrays simultaneously, the pipeline processes one batch at a time: load frames → embed → free → repeat. This bounds peak memory consumption to a single batch regardless of video length, enabling the same pipeline to run on hour-long videos as on 10-second clips.

**Qdrant Payload Indices.** Filtering queries (e.g., "all tracklets for video X with cluster Y") are accelerated by payload indices maintained by the vector database. These pre-built indices avoid full-collection scans, keeping API response times low even on collections with millions of records.

**Stripped API Responses.** Tracklet records contain several kilobytes of bounding boxes and base64-encoded thumbnails. The default tracklet listing API omits these heavyweight fields, returning only the metadata needed for the scatter plot. Heavyweight fields are fetched on demand — either per-tracklet when a user opens a video segment, or in batches when a tab requires spatial overlay data.

**Progressive Bounding Box Loading.** Tabs requiring bounding-box data (heatmap, track lines, video overlay) fetch bounding boxes in small chunks of 20 tracklets at a time, updating the display incrementally. This allows the interface to become responsive immediately after a selection is made, without waiting for all data to arrive.

**WebGL Scatter Plot Rendering.** The 2D embedding scatter plot is rendered using hardware-accelerated WebGL rather than SVG or canvas 2D. This allows smooth interaction — zooming, panning, selection, hover highlighting — with datasets of 100,000+ points without frame-rate degradation. Point attributes (color, radius, opacity) are expressed as per-vertex data updated via GPU buffer uploads.

**Web Worker Heatmap Computation.** Building the spatial heatmap requires iterating over potentially thousands of bounding boxes and accumulating their footprints into a grid. This computation is offloaded to a background worker thread to prevent it from blocking the main UI thread. The resulting grid is transferred back to the main thread via a zero-copy buffer transfer.

**Virtualized Track List.** The per-tracklet list in the Track List panel renders only the cards currently visible in the viewport using row virtualization. Scrolling through thousands of tracks thus does not create thousands of DOM nodes simultaneously.

**Module-Level Thumbnail Cache.** Thumbnails fetched on demand are stored in a persistent in-memory cache. In-flight requests are deduplicated so that concurrent components requesting the same thumbnail issue only a single network request. This cache survives component re-mounts and eliminates redundant fetches across tab switches.

**Binary Search for Bounding Box Lookup.** During video playback with bounding-box overlay, the system must rapidly find the bounding box matching the current video timestamp. A binary search over the tracklet's sorted frame array provides O(log N) lookup, ensuring smooth overlay rendering at video frame rate.

---

## 2. Usage Workflow

TrackletViz is organized around two complementary analytical perspectives: the **Local View**, which focuses on individual object tracklets, and the **Global View**, which examines the video at the scene level through fixed-duration clips. Both views share a common two-panel layout: a left panel showing a two-dimensional embedding scatter plot, and a right panel with tabbed analysis tools. A top header allows video selection and switching between views.

### 2.1 Session Initialization

Upon opening the application, the user selects an indexed video from the dropdown in the header. This action triggers background fetching of the full metadata record (cluster statistics, background image, class distribution) and all tracklet or clip records for that video. The scatter plot in the left panel immediately populates with points representing every tracklet (or clip), laid out according to their two-dimensional UMAP projections. No selection is required at this stage; the scatter plot is navigable from the moment data arrives.

---

### 2.2 Local View: Exploring Individual Tracklets

The Local View is the default perspective, with the scatter plot showing one point per tracklet.

#### 2.2.1 Navigating the Embedding Scatter Plot

Users pan by dragging and zoom by scrolling. The zoom is centered on the cursor position, preserving spatial context. Points in the scatter plot can be colored by three modes, toggled via buttons at the top of the panel:

- **Class color**: Each point is colored by the object class of the tracked entity (person, car, truck, etc.), using a fixed categorical palette consistent across sessions.
- **Cluster color**: Each point is colored by its HDBSCAN cluster assignment. Points in the noise cluster (cluster −1) are rendered gray.
- **Temporal color**: Each point is colored by the tracklet's start timestamp using a perceptual color gradient (violet for early, yellow for late), revealing temporal patterns in the data.

A legend at the bottom of the scatter plot lists the color assignments for the active mode. Clicking a legend entry activates **Legend Focus**: points not matching the selected class or cluster are made invisible, and lasso/rectangle selections operate only on visible points. Multiple entries can be focused simultaneously.

#### 2.2.2 Selecting Tracklets

The panel toolbar offers two selection tools:

- **Lasso selection**: The user draws a freehand polygon around a region of the scatter plot. Points inside the polygon's interior (tested using a ray-casting algorithm) are added to the selection.
- **Rectangle selection**: The user drags a rectangular marquee. Points within the bounding box are added to the selection.

All selected tracklet IDs are stored in the application state and simultaneously consumed by all five tabs in the right panel. Buttons at the bottom of the scatter plot offer "Select All" (selects every visible point) and "Clear" (empties the selection). Hovering over any point displays a tooltip with the tracklet's thumbnail image.

#### 2.2.3 Tab 1 — Video Player with Annotated Timeline

The Video Player tab displays the original video alongside a timeline canvas spanning the full video duration. Each selected tracklet's time span is rendered as a colored segment on the timeline, with overlapping spans from different tracklets merged into single regions. This provides an immediate visual summary of where selected tracklets appear temporally.

Clicking a timeline region opens a popover listing the tracklets active in that interval, each shown as a thumbnail card. Clicking a card enters **loop mode**: the video jumps to the tracklet's temporal span, plays it on repeat, and overlays the tracklet's current bounding box and full trajectory (path of bounding box centers) on a transparent canvas layer above the video. A red pulsing indicator and a stop button signal that loop mode is active.

#### 2.2.4 Tab 2 — Spatial Heatmap

The Heatmap tab renders a spatial density map on the video's static background image. A grid of 128×72 cells spans the image; each selected tracklet's bounding boxes are rasterized into the grid, incrementing every cell the box covers. The grid is normalized and colored using a blue-to-red gradient proportional to local density. This reveals which spatial regions of the scene are most traversed by the selected object set.

Heatmap updates are computed in a background worker thread and updated progressively as bounding boxes are fetched in chunks, so the display remains responsive even for large selections.

#### 2.2.5 Tab 3 — Track List with Spatial Overlay

The Track List tab is divided into two rows. The top row displays a zoomable, pannable canvas showing the video background overlaid with the trajectory lines of all selected tracklets that pass the current filters. Each line segment is colored by the instantaneous speed at that point, using a blue–green–yellow–red gradient. Clicking a track line enters loop mode for that tracklet.

The bottom row presents a scrollable, virtualized list of tracklet cards. Each card shows:

- The object class badge with its categorical color.
- A speed sparkline — a miniature line chart of speed over time, colored by speed level.
- Numeric statistics: average speed, duration, point count.

Two filter controls narrow the visible tracks:

- **Class filter**: Toggle badges for each object class; deselecting a class hides its tracks from both the canvas and the list.
- **Speed filter**: A range slider sets a minimum average-speed threshold; tracks below the threshold are hidden.

Filters are applied client-side and take effect immediately without additional network requests.

#### 2.2.6 Tab 4 — Cluster Summaries

The Cluster Summaries tab lists all HDBSCAN clusters identified in the video, including the noise cluster. Each cluster is displayed as a card with:

- A left border accent colored with the cluster's color.
- Aggregate statistics: member count, average speed of member tracklets, class distribution as percentages.
- Thumbnail images of the cluster's FPS-selected representative tracklets.

Hovering over a cluster card triggers cross-panel highlighting: all points in the corresponding cluster in the scatter plot are raised in prominence while others are dimmed. Hovering over a representative thumbnail highlights that specific point in the scatter plot. Clicking any representative thumbnail opens a modal dialog with a video player looping that tracklet with bounding-box overlay.

#### 2.2.7 Tab 5 — Natural Language Search

The Text Search tab provides a semantic retrieval interface. The user types a natural language description (e.g., "a person running near the entrance") and submits it. The backend encodes the query into a 768-dimensional text embedding using the video foundation model's text encoder and performs cosine similarity search against all tracklet embeddings in the video. The top-K results are returned and displayed as a grid of thumbnail cards, each annotated with the similarity score. Clicking a card opens the loop modal.

---

### 2.3 Global View: Scene-Level Analysis

Switching to the Global View changes the scatter plot to show one point per clip and loads a different set of right-panel tabs. The scatter plot navigation, color modes (cluster and temporal), and legend focus work identically to the Local View.

#### 2.3.1 Clip Selection

Individual clips are selected by clicking their scatter plot point. Two selection modes are available:

- **Standard selection**: Multiple clips accumulate in the selection set via lasso or rectangle tools, and the selection-dependent tabs (Global Video, Global Heatmap) display all selected clips.
- **Two-point selection**: The user activates this mode and clicks two representative clips in sequence. The pair is stored as a special comparison state consumed exclusively by the Global Heatmap tab.

#### 2.3.2 Global Video Player

The Global Video tab mirrors the Local Video Player tab, but the timeline segments represent selected clips rather than tracklets. Clicking a clip on the timeline or a clip card enters loop mode for that clip's time segment, with the same bounding-box overlay mechanics (using the tracklet associations stored with each clip).

#### 2.3.3 Global Heatmap — Scene Change Analysis

The Global Heatmap tab compares two clips selected in two-point mode using three sub-tabs:

- **State Change**: Decodes the median frame (the per-pixel temporal median) of each selected clip and computes per-pixel luminance variance between the two frames. High-variance pixels are colored warm (red), low-variance pixels cool (blue), revealing locations where the scene appearance has changed between the two clips.
- **Activity Shift**: Decodes the optical flow field (stored as a two-channel float32 grid) for each clip and visualizes the velocity vectors using hue-saturation encoding — hue represents flow direction, saturation represents magnitude. Differences between the two clips' flow fields reveal how patterns of motion have shifted over time.
- **Illumination Shift**: Compares color histogram distributions between the two clip median frames, highlighting regions with different color or brightness characteristics.

All sub-tab analyses are computed client-side by decoding the pre-stored auxiliary features retrieved from the server on demand.

#### 2.3.4 Global Cluster Summaries

The Global Cluster Summaries tab lists clusters from the clip-level HDBSCAN assignment. Cards display clip count, aggregate statistics, and representative clip thumbnails. Hover interactions mirror the Local View: hovering a cluster highlights it in the scatter plot; hovering a clip thumbnail highlights that specific point. Clicking a representative thumbnail opens a clip loop modal.

#### 2.3.5 Global Text Search

The Global Search tab accepts natural language queries and retrieves semantically similar clips by embedding the query and performing cosine similarity search against clip embeddings. Results are displayed as cards with clip index, time range, and similarity score.

#### 2.3.6 Global Summarizations — Activity Summaries

The Summarizations tab provides two subtabs that compute aggregate views over the entire video without requiring a manual clip selection.

##### Spatial Subtab

The Spatial subtab divides the video duration into N equal-length time buckets (N configurable from 1 to 16) and computes a 128×72 spatial occupancy heatmap for each bucket. For every bounding box recorded in the bucket's time window, either the centroid cell or all cells covered by the full bbox are incremented (user-selectable). The resulting grid is normalized and rendered with a blue-to-red color gradient: cool colors indicate sparse regions, warm colors indicate high-activity zones.

Class filter badges let the user include or exclude specific object classes from the grid computation. Clicking a bucket highlights the corresponding global clips in the scatter plot (setting `highlightedSpatialClipIds` in application state) and auto-switches the scatter plot color mode to temporal so clip timing is immediately visible.

Bounding box data is fetched once per video via `POST /api/tracklets/batch` in chunks of 20, then cached in module-level maps that survive tab switching. Grid results are additionally cached per configuration key (video ID × bucket count × mode × class set) to avoid redundant computation.

##### Temporal Subtab

The Temporal subtab renders a per-class SVG activity chart showing how object activity evolves over configurable time buckets. Two metrics are supported: **Count** (number of distinct objects active in each bucket) and **Speed** (average speed in px/s across all bboxes in the bucket). The bucket duration is set using preset buttons (5 min, 10 min, 30 min, 1 h) or a custom numeric input with a 300 s floor.

Below the activity chart, a **keyframe storyboard** displays k representative clip thumbnails per time bucket (k from 1 to 5). Representative clips are identified using the `is_representative` flag stored with each clip at indexing time. Clicking a storyboard thumbnail opens a loop modal that plays and loops the clip's time segment with the same video player mechanics used throughout the application.

Class filter badges apply to both the activity chart and the storyboard. All subtab configuration (bucket duration, metric, k, selected classes) is preserved in component-level state and survives tab-switching.

---

### 2.4 Cross-Panel Coherence

Throughout both views, the application maintains strict cross-panel coherence:

- Any change to the selection (lasso, rectangle, clear, select-all) is immediately reflected across all currently active tabs without requiring explicit refresh.
- Hover events on cluster cards, representative thumbnails, or list items propagate to the scatter plot via shared application state, providing immediate visual feedback about the correspondence between summary-level representations and their raw scatter plot positions.
- Legend focus acts as a global visibility filter shared between the scatter plot and all tabs, ensuring that the user's focus context is consistent everywhere.
- Switching between videos resets all transient selection and highlight state, preventing stale cross-video associations.
