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

## Running

```bash
# Basic run with visualization
python main.py --image_folder /path/to/images --max_loops 1 --vis_map

# With open-set object detection (requires Perception Encoder + SAM3)
python main.py --image_folder /path/to/images --max_loops 1 --vis_map --run_os

# Quick test with bundled data
unzip office_loop.zip
python main.py --image_folder office_loop --max_loops 1 --vis_map
```

Key arguments: `--submap_size` (frames per submap, default 16), `--min_disparity` (keyframe threshold, default 50), `--conf_threshold` (filter low-confidence points %, default 25), `--lc_thres` (loop closure similarity threshold, default 0.95), `--vis_voxel_size` (downsample for visualization).

## Evaluation

```bash
# TUM RGB-D benchmark (requires evo_ape tool and dataset)
./evals/eval_tum.sh 32  # argument is submap_size
python evals/process_logs_tum.py --submap_size 32
```

## Architecture

**Pipeline flow** (orchestrated in `main.py`):
1. **Keyframe selection** — `FrameTracker` (frame_overlap.py) uses Lucas-Kanade optical flow to skip frames without enough motion
2. **VGGT inference** — `Solver.run_predictions()` calls the VGGT model to predict dense depth, confidence maps, camera poses, and intrinsics for each submap batch
3. **Loop closure detection** — `ImageRetrieval` (loop_closure.py) uses DINO-Salad descriptors to find matching frames across submaps
4. **Scale estimation** — `estimate_scale_pairwise()` (scale_solver.py) computes scale between submaps via median depth ratios of overlapping points
5. **Submap construction** — `Submap` (submap.py) bundles frames, poses, 3D points, colors, and confidences; stored in `GraphMap` (map.py)
6. **Pose graph optimization** — `PoseGraph` (graph.py) uses GTSAM SL(4) manifold optimization with intra-submap, inter-submap, and loop closure constraints
7. **Visualization** — `Viewer` (viewer.py) renders interactive 3D point clouds and camera frustums via Viser web UI

**Key class relationships:**
- `Solver` (solver.py) is the central coordinator — owns the `GraphMap`, `PoseGraph`, `ImageRetrieval`, `FrameTracker`, and `Viewer`
- `GraphMap` holds a list of `Submap` objects and provides retrieval queries
- `PoseGraph` wraps GTSAM; node IDs encode `submap_id * submap_size + frame_index`
- `Submap` stores per-frame data (images, points, poses as 4x4 SL(4)-normalized homographies)

**Third-party dependencies** (in `third_party/`):
- `vggt/` — VGGT_SPARK fork: monocular dense depth/pose model
- `salad/` — DINO-Salad: image descriptor model for loop closure
- `perception_models/` — Facebook Perception Encoder CLIP (optional, for open-set detection)
- `sam3/` — SAM3 segmentation model (optional, for open-set 3D object detection)

## Conventions

- Poses are 4x4 homography matrices normalized to SL(4) (det=1) via `normalize_to_sl4()`
- Point clouds: numpy arrays shaped `(S, H, W, 3)` in world frame; colors as uint8 `[0-255]`
- Confidence maps: per-pixel float in `[0, 1]`, thresholded by percentile
- Images: torch tensors `(B, 3, H, W)` for model input
- `slam_utils.py` contains shared utilities (image sorting, camera decomposition, geometry, timing via `Accumulator` context manager)
