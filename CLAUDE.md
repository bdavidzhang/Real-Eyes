# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT-SLAM 2.0 is a real-time dense feed-forward monocular SLAM system that uses the VGGT model for depth/pose prediction and optimizes on the SL(4) manifold via GTSAM. It supports loop closure detection, open-set 3D object detection, and interactive 3D visualization.
k

## Setup & Installation

```bash
conda create -n vggt-slam python=3.11
conda activate vggt-slam
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs pip requirements, clones four third-party repos into `third_party/` (Salad, VGGT_SPARK, Perception Encoder, SAM3), and installs the main package in editable mode.

**First-run model downloads** (happen automatically, require network/disk space):
- VGGT-1B weights from HuggingFace (~4 GB) → cached in `~/.cache/torch/hub/checkpoints/model.pt`
- DINO-Salad checkpoint (~350 MB) → cached in `~/.cache/torch/hub/checkpoints/dino_salad.ckpt`
- DINOv2 backbone (~350 MB) → pulled via `torch.hub`
- PE-Core-L14-336 CLIP model (if `--run_os`) → downloaded from HuggingFace

## Running

```bash
# Basic run with visualization (Viser UI on http://localhost:8080)
python main.py --image_folder /path/to/images --max_loops 1 --vis_map

# With open-set object detection (requires Perception Encoder + SAM3)
python main.py --image_folder /path/to/images --max_loops 1 --vis_map --run_os

# Quick test with bundled data
unzip office_loop.zip
python main.py --image_folder office_loop --max_loops 1 --vis_map

# Log poses and dense point clouds to disk
python main.py --image_folder office_loop --max_loops 1 --log_results --log_path poses.txt

# Offline visualization of saved results
python visualize_results.py
```

Key arguments: `--submap_size` (frames per submap, default 16), `--min_disparity` (keyframe threshold, default 50), `--conf_threshold` (filter low-confidence points %, default 25), `--lc_thres` (loop closure similarity threshold, default 0.95), `--vis_voxel_size` (downsample for visualization).

**Limitations:** `--max_loops` only supports 0 or 1; `--overlapping_window_size` only supports 1.

### Custom Data Collection

```bash
mkdir /path/to/img_folder
ffmpeg -i /path/to/video.MOV -vf "fps=10" /path/to/img_folder/frame_%04d.jpg
```

Use horizontal videos to avoid cropping. Images are sorted by the numeric value in their filename.

## Evaluation

```bash
# TUM RGB-D benchmark
./evals/eval_tum.sh 32       # argument is submap_size
python evals/process_logs_tum.py --submap_size 32

# 7-Scenes benchmark
./evals/eval_7scenes.sh 32
```

Set `abs_dir` in the eval shell scripts to your MASt3R-SLAM dataset download location.

## Cloud Deployment (Modal)

Two Modal deployment scripts exist for different use cases:

### modal_app.py — Batch/Offline SLAM

```bash
# Run SLAM on a remote A100 GPU — uploads images, runs SLAM, downloads results
modal run modal_app.py --image-folder ./office_loop --submap-size 16 --max-loops 1

# Pre-cache model weights (optional, idempotent)
modal run modal_app.py::app.download_models
```

**How it works:**
1. **Local entrypoint** (`main()`): uploads local image folder to a Modal Volume (`vggt-slam-data`), calls `download_models.remote()` to ensure weights are cached in `vggt-slam-models` volume
2. **Spawns `run_slam` non-blocking** on an A100-80GB GPU, then immediately waits on `modal.Queue` (`vggt-slam-url`) for the Viser tunnel URL — opens it in the browser the moment the server starts
3. **`run_slam()`** on the remote container: initializes `Solver`, loads VGGT-1B, loops over images in submap batches (mirrors `main.py`), saves poses + dense point clouds to the data volume, then returns a summary dict
4. **Local side** downloads `poses.txt` and dense log files from the volume to `./modal_results/`

**Key Modal constructs:**
- `modal.Volume` — `vggt-slam-models` persists model weights across runs; `vggt-slam-data` holds input images and results
- `modal.Queue` — passes the live Viser URL from the remote container back to the local entrypoint
- `modal.forward(8080)` — creates a public HTTPS tunnel to the Viser port inside the container

### modal_streaming.py — Persistent Streaming Server

```bash
# Development (auto-reload on code changes)
modal serve modal_streaming.py

# Production deployment (stable URL, always-on)
modal deploy modal_streaming.py

# Pre-cache model weights
modal run modal_streaming.py::app.download_models
```

**How it works:**
1. `@modal.web_server(port=5000)` gives a stable `*.modal.run` URL; `min_containers=1` keeps the container warm
2. On startup, `run_streaming_server()` calls `start_server()` in a daemon thread (the decorator expects the function to return after launching the server)
3. The **frontend is built inside the Docker image** at build time (`npm install && npx vite build`) and served as static files — no separate Vite dev server needed in production
4. Configuration via environment variables (`SUBMAP_SIZE`, `MIN_DISPARITY`, `CONF_THRESHOLD`, `VIS_STRIDE`) set in Modal dashboard or secrets
5. Uses `modal.Secret` for `huggingface-secret` and `gemini-secret`

**vs modal_app.py:** streaming is a persistent always-on server for live camera input; modal_app.py is a one-shot batch job for processing a folder of images.

## Streaming / Web Server Mode

The `server/` directory implements real-time browser-based SLAM streaming.

### server/app.py — Flask + SocketIO Server

**HTTP routes:**
- `GET /health` — returns GPU status
- `POST /reset` — soft reset (clears SLAM data, keeps models loaded); emits `slam_reset` to clients
- `POST /api/plan` — LLM-powered tracking plan: sends a natural language prompt to Gemini (`gemini-1.5-flash`) and returns a structured JSON of objects to detect + waypoint/pathfinding justification; falls back to keyword extraction on error

**SocketIO events (server → client):**
- `slam_update` — streamed after each submap is processed; contains points, colors, camera poses, detections, beacons
- `global_map` — full map state on demand
- `slam_reset`, `slam_stopped`, `beacon_queued`, `detection_preview` — status/data events

**SocketIO events (client → server):**
- `frame` — send a base64-encoded JPEG frame; auto-starts SLAM on first frame
- `stop_slam` — stop the processing loop
- `set_detection_queries` — set active CLIP queries for object detection; immediately re-runs on existing submaps
- `get_detection_preview` — return keyframe + SAM3 mask overlay for a specific submap/frame/query
- `place_beacon` / `clear_beacons` — place a named marker at a frame's 3D camera position
- `get_global_map` — request the full current map state

**VideoFeeder** class: reads from a video file and pushes frames into `frame_queue` for offline testing. Supports FPS throttling (`--video-fps`) or fast-forward (`--fast`).

**SSL:** local mode loads `server/webserver/server.cert` + `server.key` for HTTPS (required for phone camera access via WebRTC). Modal deployment skips SSL since the tunnel provides HTTPS.

```bash
# Run locally with live camera
python -m server.app --port 5000

# Run with a video file for testing
python -m server.app --video /path/to/video.mp4 --video-fps 2 --submap-size 8
```

### server/streaming_slam.py — StreamingSLAM

Wraps `Solver` for frame-by-frame streaming (no Viser, `skip_viewer=True`).

**Initialization:** loads VGGT-1B + `ObjectDetector` (PE-Core CLIP + SAM3); sets CLIP model on `Solver` via `solver.set_clip_model()` so `run_predictions()` can compute per-frame semantic embeddings.

**Processing loop** (`process_loop()`):
1. Reads base64 JPEG frames from `frame_queue`
2. Checks optical flow disparity — skips frames without enough motion
3. Saves keyframes to a `tempfile.mkdtemp()` directory
4. When `submap_size + 1` keyframes accumulate, calls `process_submap()`

**Submap processing** (`process_submap()`):
1. `solver.run_predictions()` → VGGT inference
2. `solver.add_points()` → adds to map
3. `solver.graph.optimize()` → GTSAM pose graph optimization
4. `extract_stream_data()` → collects all points/colors/camera poses, recenters around scene mean, resolves pending beacons
5. `_detect_after_submap_update()` → CLIP+SAM3 on latest submap if queries active
6. Pushes result dict to `result_queue`; keeps last 1 frame as overlap window

**Object detection** (cache-aware):
- CLIP embeddings fetched via `submap.get_all_semantic_vectors()`
- SAM3 segmentation runs only on frames with CLIP similarity above threshold
- Cache key: `(submap_id, frame_idx, query)` — avoids reprocessing on query changes
- 3D bboxes computed via `ObjectDetector.compute_3d_bbox()` using masked point cloud
- `_dedup_and_store()` deduplicates overlapping 3D boxes across submaps

**Soft reset** (`soft_reset()`): clears all SLAM state and temp files without reloading VGGT or CLIP models; re-attaches CLIP model to solver after `solver.reset()`.

## Architecture

**Pipeline flow** (orchestrated in `main.py`):
1. **Keyframe selection** — `FrameTracker` (frame_overlap.py) uses Lucas-Kanade optical flow to skip frames without enough motion
2. **VGGT inference** — `Solver.run_predictions()` calls the VGGT model to predict dense depth, confidence maps, camera poses, and intrinsics for each submap batch
3. **Loop closure detection** — `ImageRetrieval` (loop_closure.py) uses DINO-Salad descriptors to find matching frames; loop closure is rejected if VGGT's `image_match_ratio < 0.85`
4. **Scale estimation** — `estimate_scale_pairwise()` (scale_solver.py) computes scale between submaps via median depth ratios of overlapping points
5. **Submap construction** — `Submap` (submap.py) bundles frames, poses, 3D points, colors, and confidences; stored in `GraphMap` (map.py)
6. **Pose graph optimization** — `PoseGraph` (graph.py) uses GTSAM SL(4) manifold optimization with intra-submap, inter-submap, and loop closure constraints
7. **Visualization** — `Viewer` (viewer.py) renders interactive 3D point clouds and camera frustums via Viser web UI on port 8080

**Key class relationships:**
- `Solver` (solver.py) is the central coordinator — owns the `GraphMap`, `PoseGraph`, `ImageRetrieval`, `FrameTracker`, and `Viewer`
- `Solver.reset()` resets SLAM state without reloading models (used by streaming mode)
- `GraphMap` holds a dict of `Submap` objects keyed by submap ID; tracks non-LC submap IDs separately
- `PoseGraph` wraps GTSAM; node IDs are `submap_id + frame_index` (where `submap_id` is already offset so IDs are globally unique)
- `Submap` stores per-frame data (images, points, poses as 4x4 matrices); loop-closure submaps have `is_lc_submap=True`
- `ObjectDetector` (object_detector.py) wraps PE-Core CLIP + SAM3 for open-set 3D bounding box detection

**Third-party dependencies** (in `third_party/`):
- `vggt/` — VGGT_SPARK fork: monocular dense depth/pose model
- `salad/` — DINO-Salad: image descriptor model for loop closure
- `perception_models/` — Facebook Perception Encoder CLIP (optional, for open-set detection)
- `sam3/` — SAM3 segmentation model (optional, for open-set 3D object detection)

## Conventions

- Poses are 4x4 homography matrices normalized to SL(4) (det=1) via `normalize_to_sl4()`
- Point clouds: numpy arrays shaped `(S, H, W, 3)` in world frame; colors as uint8 `[0-255]`
- Confidence maps: per-pixel float in `[0, 1]`, thresholded by percentile (`conf_threshold` arg = % of lowest-confidence points to filter)
- Images: torch tensors `(B, 3, H, W)` normalized to `[0, 1]` for model input
- `slam_utils.py` contains shared utilities (image sorting, camera decomposition, geometry, timing via `Accumulator` context manager)
- Pose output format (TUM): `timestamp tx ty tz qx qy qz qw` written by `GraphMap.write_poses_to_file()`
