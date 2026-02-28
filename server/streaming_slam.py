"""
StreamingSLAM — wraps the VGGT-SLAM 2.0 Solver for real-time streaming.

Uses VGGT-SLAM 2.0 APIs exclusively:
  - submap.get_points_in_world_frame(graph)
  - submap.get_all_poses_world(graph)
  - submap.get_points_colors()
  - submap.get_all_semantic_vectors()
  - submap.get_points_in_mask(frame_idx, mask, graph)
  - solver.run_predictions(image_names, model, max_loops)
"""

import cv2
import numpy as np
import torch
import base64
import threading
import tempfile
import os
import time
import queue
from PIL import Image

from vggt_slam.solver import Solver
from vggt_slam.object_detector import ObjectDetector
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import compute_image_embeddings


class StreamingSLAM:
    def __init__(self,
                 submap_size=8,
                 min_disparity=30.0,
                 conf_threshold=25.0,
                 vis_stride=4,
                 lc_thres=0.80):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Object detector (PE-Core CLIP + SAM3)
        print("Loading ObjectDetector (PE-Core CLIP + SAM3)...")
        self.object_detector = ObjectDetector(device=self.device)
        print("ObjectDetector loaded!")

        # Solver — skip viser viewer since we stream to the web frontend
        self.solver = Solver(
            init_conf_threshold=conf_threshold,
            lc_thres=lc_thres,
            skip_viewer=True,
        )

        # Store CLIP model on solver so run_predictions() can use it
        self.solver.set_clip_model(
            self.object_detector.clip_model,
            self.object_detector.clip_preprocess,
        )

        # VGGT model
        print("Loading VGGT model...")
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device)
        print("VGGT model loaded!")

        # Configuration
        self.submap_size = submap_size
        self.overlapping_window_size = 1
        self.min_disparity = min_disparity
        self.max_loops = 1
        self.vis_stride = vis_stride

        # State
        self.frame_count = 0
        self.image_names_subset = []
        self.temp_dir = tempfile.mkdtemp()
        self.is_running = False

        # Beacon state
        self.pending_beacons = []
        self.resolved_beacons = []
        self.latest_scene_center = np.zeros(3)

        # Detection state
        self.active_queries = []
        self.accumulated_detections = []
        self._detection_lock = threading.Lock()
        self._sam_cache = {}

        # Detection thresholds
        self.detection_clip_thresholds = {"default": 0.15}
        self.detection_sam_thresholds = {"default": 0.80}

        # External queues (set by app.py)
        self.frame_queue = None
        self.result_queue = None

        print(f"Temp directory: {self.temp_dir}")

    def start(self):
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.process_loop, daemon=True).start()
            print("SLAM processing loop started")

    def stop(self):
        self.is_running = False
        self.image_names_subset.clear()
        print("SLAM processing loop stopped")

    # ------------------------------------------------------------------
    # Frame processing loop
    # ------------------------------------------------------------------

    def process_loop(self):
        """Main processing loop — reads frames, checks disparity, triggers submap processing."""
        while self.is_running:
            try:
                frame_data = self.frame_queue.get(timeout=1)

                # Decode image
                img_bytes = base64.b64decode(frame_data['image'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Failed to decode frame")
                    continue

                # Check disparity via optical flow
                enough_disparity = self.solver.flow_tracker.compute_disparity(
                    frame, self.min_disparity
                )

                if enough_disparity:
                    frame_path = self._save_frame(frame)
                    self.image_names_subset.append(frame_path)
                    self.frame_count += 1
                    print(f"Keyframe {self.frame_count} added (subset size: {len(self.image_names_subset)})")

                # Process submap when batch is full
                if len(self.image_names_subset) == self.submap_size + self.overlapping_window_size:
                    print(f"Processing submap with {len(self.image_names_subset)} frames...")
                    self.process_submap()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()

    def process_submap(self):
        """Process a submap batch — run VGGT, optimize graph, extract data, detect objects."""
        try:
            # Guard: skip if any saved images are missing (e.g. after a reset mid-batch)
            valid_names = [p for p in self.image_names_subset if os.path.exists(p)]
            if not valid_names:
                print("process_submap: all image files missing, skipping")
                self.image_names_subset = []
                return
            self.image_names_subset = valid_names

            # 1. Run predictions (uses stored clip model via solver)
            predictions = self.solver.run_predictions(
                self.image_names_subset,
                self.model,
                self.max_loops,
            )

            # 2. Add points to map
            self.solver.add_points(predictions)

            # 3. Optimize graph
            self.solver.graph.optimize()

            # 4. Check for loop closures
            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if loop_closure_detected:
                print("Loop closure detected!")

            # 5. Extract data for streaming
            stream_data = self.extract_stream_data()

            # 6. Run object detection if queries are active
            with self._detection_lock:
                has_queries = len(self.active_queries) > 0
            if has_queries:
                t0 = time.time()
                self._detect_after_submap_update()
                with self._detection_lock:
                    n_det = len(self.accumulated_detections)
                print(f"Detection: {n_det} detections in {(time.time()-t0)*1000:.0f}ms")

            # Add detections to stream data
            with self._detection_lock:
                stream_data['detections'] = list(self.accumulated_detections)
                stream_data['active_queries'] = list(self.active_queries)

            # 7. Send to result queue
            if stream_data and self.result_queue is not None and not self.result_queue.full():
                self.result_queue.put(stream_data)

            # 8. Keep overlapping frames
            self.image_names_subset = self.image_names_subset[-self.overlapping_window_size:]

            print(f"Submap processed. Total submaps: {self.solver.map.get_num_submaps()}, "
                  f"Loop closures: {self.solver.graph.get_num_loops()}")

        except Exception as e:
            print(f"Submap processing error: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def extract_stream_data(self):
        """Extract visualization data from all submaps using VGGT-SLAM 2.0 APIs."""
        try:
            num_submaps = self.solver.map.get_num_submaps()
            if num_submaps == 0:
                return self._empty_data()

            all_points = []
            all_colors = []
            all_cam_positions = []
            all_cam_rotations = []

            stride = self.vis_stride

            for submap in self.solver.map.get_submaps():
                # Get points via VGGT-SLAM 2.0 API (requires graph)
                points_world = submap.get_points_in_world_frame(self.solver.graph)
                colors = submap.get_points_colors()

                # Get camera poses via VGGT-SLAM 2.0 API (requires graph)
                cam_poses_world = submap.get_all_poses_world(self.solver.graph)

                if points_world is not None and len(points_world) > 0:
                    # Apply stride via numpy slicing (VGGT-SLAM 2.0 doesn't have stride param)
                    if stride > 1:
                        points_world = points_world[::stride]
                        colors = colors[::stride]

                    all_points.append(points_world)
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    all_colors.append(colors)

                # Extract camera poses
                for cam_pose in cam_poses_world:
                    position = cam_pose[:3, 3]
                    rotation = cam_pose[:3, :3]
                    all_cam_positions.append(position.tolist())
                    all_cam_rotations.append(rotation.tolist())

            if len(all_points) > 0:
                all_points = np.vstack(all_points)
                all_colors = np.vstack(all_colors)

                # Compute scene center and recenter
                scene_center = np.mean(all_points, axis=0)
                self.latest_scene_center = scene_center
                all_points = all_points - scene_center

                # Recenter camera positions
                all_cam_positions = np.array(all_cam_positions)
                all_cam_positions = all_cam_positions - scene_center
            else:
                all_points = np.zeros((0, 3))
                all_colors = np.zeros((0, 3))
                all_cam_positions = np.array([])
                all_cam_rotations = []
                scene_center = np.zeros(3)

            # Resolve pending beacons
            self._resolve_beacons(all_cam_positions)

            n_points = len(all_points)
            n_cameras = len(all_cam_positions) if isinstance(all_cam_positions, np.ndarray) and all_cam_positions.ndim == 2 else 0

            # Binary encode points/colors — ~10-100x faster than .tolist() and 3x smaller
            # Positions: float32 (N×3 flat) → base64
            # Colors:    uint8   (N×3 flat) → base64  (saves 4x vs float32)
            pts_f32 = all_points.astype(np.float32)
            cols_u8 = (all_colors * 255).clip(0, 255).astype(np.uint8)
            points_b64 = base64.b64encode(pts_f32.tobytes()).decode('ascii')
            colors_b64 = base64.b64encode(cols_u8.tobytes()).decode('ascii')

            cam_pos_list = all_cam_positions.tolist() if n_cameras > 0 else []

            return {
                'frame_id': self.frame_count,
                'num_submaps': num_submaps,
                'num_loops': self.solver.graph.get_num_loops(),
                'points_b64': points_b64,
                'colors_b64': colors_b64,
                'points': [],
                'colors': [],
                'camera_positions': cam_pos_list,
                'camera_rotations': all_cam_rotations,
                'scene_center': scene_center.tolist(),
                'n_points': n_points,
                'n_cameras': n_cameras,
                'detections': [],
                'active_queries': list(self.active_queries),
                'resolved_beacons': self.resolved_beacons,
            }

        except Exception as e:
            print(f"Extract data error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_data()

    def _resolve_beacons(self, all_cam_positions):
        """Resolve pending beacons to 3D positions using camera positions."""
        if not isinstance(all_cam_positions, np.ndarray) or all_cam_positions.ndim != 2:
            return
        if len(all_cam_positions) == 0 or len(self.pending_beacons) == 0:
            return

        n_cameras = len(all_cam_positions)
        still_pending = []
        for beacon in self.pending_beacons:
            cam_idx = min(beacon['frame_number'] - 1, n_cameras - 1)
            cam_idx = max(cam_idx, 0)
            if cam_idx < n_cameras:
                pos = all_cam_positions[cam_idx]
                resolved = {
                    'beacon_id': beacon['beacon_id'],
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'z': float(pos[2]),
                }
                self.resolved_beacons.append(resolved)
                print(f"Beacon {beacon['beacon_id']} resolved at camera {cam_idx}")
            else:
                still_pending.append(beacon)
        self.pending_beacons = still_pending

    def _empty_data(self):
        return {
            'frame_id': self.frame_count,
            'num_submaps': 0,
            'num_loops': 0,
            'points_b64': '',
            'colors_b64': '',
            'points': [],
            'colors': [],
            'camera_positions': [],
            'camera_rotations': [],
            'scene_center': [0, 0, 0],
            'n_points': 0,
            'n_cameras': 0,
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

    # ------------------------------------------------------------------
    # Object detection (cache-aware)
    # ------------------------------------------------------------------

    def set_detection_queries(self, queries):
        """Set detection queries. Cache-aware: handles add/remove of queries."""
        with self._detection_lock:
            old_queries = set(self.active_queries)
            self.active_queries = [q.strip() for q in queries if q.strip()]
            new_queries = set(self.active_queries)

        if len(self.active_queries) == 0:
            self._sam_cache.clear()
            with self._detection_lock:
                self.accumulated_detections = []
            return

        # Purge cache for removed queries
        removed = old_queries - new_queries
        if removed:
            keys_to_delete = [k for k in self._sam_cache if k[2] in removed]
            for k in keys_to_delete:
                del self._sam_cache[k]

        # Run CLIP+SAM on all existing submaps for new queries
        added = new_queries - old_queries
        if added and self.solver.map.get_num_submaps() > 0:
            t0 = time.time()
            added_list = list(added)
            for submap in self.solver.map.get_submaps():
                try:
                    new_keys = self._run_clip_sam_on_submap(submap, added_list)
                    if new_keys:
                        self._compute_bboxes_for_keys(new_keys)
                except Exception as e:
                    print(f"  Detection error on submap {submap.get_id()}: {e}")
            print(f"New queries {added_list}: SAM on all submaps in {(time.time()-t0)*1000:.0f}ms")

        self._dedup_and_store()

    def _run_clip_sam_on_submap(self, submap, queries):
        """Run CLIP matching + SAM on unprocessed (submap, frame, query) combos.

        Uses submap.get_all_semantic_vectors() for CLIP embeddings (VGGT-SLAM 2.0 API).
        Returns list of cache keys that got new mask entries.
        """
        od = self.object_detector
        submap_id = submap.get_id()

        # VGGT-SLAM 2.0 API: get_all_semantic_vectors()
        clip_embs = submap.get_all_semantic_vectors()
        if clip_embs is None or len(clip_embs) == 0:
            return []

        # Convert to tensor if numpy
        if isinstance(clip_embs, np.ndarray):
            clip_embs = torch.from_numpy(clip_embs)

        new_mask_keys = []

        for query in queries:
            query = query.strip()
            if not query:
                continue

            ct = self.detection_clip_thresholds
            clip_thresh = ct.get(query, ct.get("default", 0.2))
            st = self.detection_sam_thresholds
            sam_thresh = st.get(query, st.get("default", 0.0))

            text_emb = od.encode_text_vector(query)
            sims = clip_embs @ text_emb  # (S,)

            last_orig = submap.get_last_non_loop_frame_index()
            if last_orig is None or last_orig < 0:
                last_orig = sims.shape[0] - 1

            for frame_idx in range(last_orig + 1):
                cache_key = (submap_id, frame_idx, query)
                if cache_key in self._sam_cache:
                    continue

                sim_val = sims[frame_idx].item()
                if sim_val < clip_thresh:
                    self._sam_cache[cache_key] = []
                    continue

                try:
                    frame_tensor = submap.get_frame_at_index(frame_idx)
                    frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)

                    seg_results = od.segment_all(frame_pil, query)
                    passed = []
                    for mask_2d, box_2d, seg_score in seg_results:
                        if seg_score >= sam_thresh:
                            passed.append({
                                'mask_2d': mask_2d,
                                'box_2d': box_2d,
                                'seg_score': float(seg_score),
                                'bbox_3d': None,
                            })
                    self._sam_cache[cache_key] = passed
                    if passed:
                        new_mask_keys.append(cache_key)
                except Exception as e:
                    print(f"  SAM error submap {submap_id} frame {frame_idx} query '{query}': {e}")
                    self._sam_cache[cache_key] = []

        return new_mask_keys

    def _compute_bboxes_for_keys(self, cache_keys):
        """Compute 3D bounding boxes for cached mask entries."""
        od = self.object_detector
        scene_center = self.latest_scene_center
        for key in cache_keys:
            submap_id, frame_idx, _query = key
            masks = self._sam_cache.get(key, [])
            if not masks:
                continue
            submap = self.solver.map.get_submap(submap_id)
            if submap is None:
                continue
            for entry in masks:
                entry['bbox_3d'] = od.compute_3d_bbox(
                    submap, frame_idx, entry['mask_2d'],
                    self.solver.graph, scene_center
                )

    def _recompute_all_bboxes(self):
        """Recompute ALL 3D bboxes from cached SAM masks (after graph optimization)."""
        od = self.object_detector
        scene_center = self.latest_scene_center
        for key, masks in self._sam_cache.items():
            if not masks:
                continue
            submap_id, frame_idx, _query = key
            submap = self.solver.map.get_submap(submap_id)
            if submap is None:
                continue
            for entry in masks:
                entry['bbox_3d'] = od.compute_3d_bbox(
                    submap, frame_idx, entry['mask_2d'],
                    self.solver.graph, scene_center
                )

    def _dedup_and_store(self):
        """Build detection list from cache, dedup, store."""
        raw = []
        for (submap_id, frame_idx, query), masks in self._sam_cache.items():
            for entry in masks:
                bbox = entry.get('bbox_3d')
                if bbox is None:
                    continue
                raw.append({
                    "success": True,
                    "query": query,
                    "bounding_box": bbox,
                    "confidence": entry['seg_score'],
                    "keyframe_image": None,
                    "mask_image": None,
                    "matched_submap": int(submap_id),
                    "matched_frame": int(frame_idx),
                    "query_time_ms": 0,
                    "error": None,
                })
        deduped = ObjectDetector.deduplicate_detections(raw)
        with self._detection_lock:
            self.accumulated_detections = deduped

    def _detect_after_submap_update(self):
        """Run detection after a new submap is added."""
        with self._detection_lock:
            queries = list(self.active_queries)
        if not queries:
            return

        latest_submap = self.solver.map.get_latest_submap()
        if latest_submap is not None:
            try:
                self._run_clip_sam_on_submap(latest_submap, queries)
            except Exception as e:
                print(f"  CLIP+SAM error on latest submap: {e}")

        self._recompute_all_bboxes()
        self._dedup_and_store()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def soft_reset(self):
        """Reset SLAM state without reloading models."""
        print("Performing soft reset...")
        was_running = self.is_running
        self.stop()

        self.frame_count = 0
        self.image_names_subset.clear()
        self.pending_beacons.clear()
        self.resolved_beacons.clear()
        self.latest_scene_center = np.zeros(3)

        with self._detection_lock:
            self.accumulated_detections = []
        self._sam_cache = {}

        self.solver.reset()
        # Re-set clip model after reset
        self.solver.set_clip_model(
            self.object_detector.clip_model,
            self.object_detector.clip_preprocess,
        )

        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.temp_dir = tempfile.mkdtemp()

        print(f"Soft reset complete. New temp dir: {self.temp_dir}")

        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_frame(self, frame):
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
