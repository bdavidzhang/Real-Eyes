# Copilot Instructions — VGGT-SLAM

## Architecture

**Central coordinator:** `Solver` ([vggt_slam/solver.py](../vggt_slam/solver.py)) owns `GraphMap`, `PoseGraph`, `ImageRetrieval`, `FrameTracker`, and `Viewer`. All SLAM logic flows through it.

**Pipeline:** keyframe selection (`FrameTracker`) → VGGT batch inference → loop closure (DINO-Salad) → scale estimation → `Submap` construction → GTSAM SL(4) pose graph optimization → Viser visualization.

**Streaming variant:** `StreamingSLAM` ([server/streaming_slam.py](../server/streaming_slam.py)) wraps `Solver` with `skip_viewer=True`; reads base64 frames from `frame_queue`, pushes JSON to `result_queue`. All viewer calls are guarded by `if self.viewer is None: return`.

## Data Conventions

| Data | Shape / dtype |
|------|--------------|
| Model input images | `(B, 3, H, W)` torch `float` `[0,1]`, cast to `bfloat16` |
| `world_points` | `(S, H, W, 3)` numpy `float32`, world frame |
| `colors` | `(S, H, W, 3)` numpy `uint8` `[0,255]` |
| `depth_conf` | `(S, H, W)` numpy `float32` `[0,1]` |
| Poses (cam-to-world) | `(S, 4, 4)` numpy `float32`, SL(4) homography |
| `extrinsic` (world-to-cam) | `(S, 3, 4)` numpy `float32` |
| `proj_mats` (K padded) | `(S, 4, 4)` numpy `float32` |

`conf_threshold` arg is a **percentile** value, not a fraction. Retrieval vectors use **L2 distance** (SALAD); semantic vectors use **cosine similarity** (CLIP).

**GTSAM node IDs:** `node_id = submap_id + frame_index_within_submap`. `submap_id` offsets are chosen so all IDs are globally unique. Use `X(node_id)` symbols.

## Code Style

- `snake_case` throughout; no abstract base classes; duck-typed component interfaces.
- Type hints are minimal. Short single-line docstrings on key public methods only.
- `DEBUG = False` module-level flag in [solver.py](../vggt_slam/solver.py) gates matplotlib/open3d debug plots.
- Time sections with `Accumulator` context manager from [slam_utils.py](../vggt_slam/slam_utils.py): `with vggt_timer: ...`.
- `pred` dict returned by `Solver.run_predictions()` has keys: `images`, `extrinsic`, `intrinsic`, `depth`, `depth_conf`, `detected_loops`. LC data uses `_lc`-suffixed keys.

## Build and Test

```bash
# Setup (first time)
conda create -n vggt-slam python=3.11 && conda activate vggt-slam
chmod +x setup.sh && ./setup.sh

# Run (requires ~5 GB model downloads on first run)
python main.py --image_folder office_loop --max_loops 1 --vis_map

# Evaluation
./evals/eval_tum.sh 32
python evals/process_logs_tum.py --submap_size 32

# Cloud (Modal, headless A100)
modal run modal_app.py --image-folder ./office_loop --submap-size 16 --max-loops 1
```

No unit test suite — validate by running against `office_loop/` sample data.

## Known Limitations & Gotchas

- `--max_loops` supports only `0` or `1`; `--overlapping_window_size` supports only `1`.
- `get_projection_matrix` in [graph.py](../vggt_slam/graph.py) has typo `return projection_matri` — **method is broken**, avoid calling it.
- `normalize_to_sl4()` calls are commented out in `PoseGraph`; GTSAM matrices may not be strictly SL(4).
- LC submaps (`is_lc_submap=True`) are excluded from pose output and visualization; iterate `non_lc_submap_ids` for display.
- Detection cache in `StreamingSLAM` is keyed by `(submap_id, frame_idx, query)` and guarded by `_detection_lock`.

## Key File Map

| File | Purpose |
|------|---------|
| [vggt_slam/solver.py](../vggt_slam/solver.py) | Main SLAM logic, VGGT inference, submap assembly |
| [vggt_slam/graph.py](../vggt_slam/graph.py) | GTSAM PoseGraph on SL(4) manifold |
| [vggt_slam/map.py](../vggt_slam/map.py) | `GraphMap`: submap registry, pose I/O (TUM/KITTI) |
| [vggt_slam/submap.py](../vggt_slam/submap.py) | Per-submap data bundle |
| [vggt_slam/scale_solver.py](../vggt_slam/scale_solver.py) | Median-depth-ratio scale estimation |
| [vggt_slam/slam_utils.py](../vggt_slam/slam_utils.py) | Shared geometry, `Accumulator` timer, image sorting |
| [server/streaming_slam.py](../server/streaming_slam.py) | Headless streaming wrapper with CLIP+SAM3 cache |
| [main.py](../main.py) | CLI entry point, top-level orchestration loop |
