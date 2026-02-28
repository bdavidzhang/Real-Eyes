# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT-SLAM 2.0 is a real-time dense feed-forward monocular SLAM system that uses the VGGT model for depth/pose prediction and optimizes on the SL(4) manifold via GTSAM. It supports loop closure detection, open-set 3D object detection, and interactive 3D visualization.

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

```bash
# Run SLAM on a remote A100 GPU — uploads images, runs SLAM, downloads results
modal run modal_app.py --image-folder ./office_loop --submap-size 16 --max-loops 1

# Pre-cache model weights (optional, idempotent)
modal run modal_app.py::app.download_models
```

`modal_app.py` mirrors `main.py` logic in headless mode. Results are saved to `./modal_results/`. The live Viser map URL is printed and opened in the browser once the container starts.

## Streaming / Web Server Mode

`server/streaming_slam.py` wraps the `Solver` for real-time frame-by-frame processing:
- `StreamingSLAM` reads frames from `frame_queue`, runs SLAM, pushes JSON results to `result_queue`
- Instantiated with `skip_viewer=True` (no Viser) — visualization streams to web frontend
- `server/app.py` is the web server; `server/webserver/` contains the frontend

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
