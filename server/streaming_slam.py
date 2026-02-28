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
                 lc_thres=0.95):
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

        # Cached last stream data (avoid redundant extract_stream_data calls)
        self._last_stream_data: dict | None = None

        # Incremental extraction cache
        self._submap_cache: dict[int, dict] = {}
        self._scene_center: np.ndarray = np.zeros(3)

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

            # 5. Extract data for streaming — incremental on normal updates, full on loop closure
            if loop_closure_detected:
                stream_data = self.extract_stream_data_full()
            else:
                latest = self.solver.map.get_latest_submap()
                stream_data = self.extract_stream_data_incremental(latest.get_id())
            self._last_stream_data = stream_data

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

    def _extract_one_submap(self, submap) -> dict:
        """Extract points, colors, and camera poses from a single submap."""
        pts = submap.get_points_in_world_frame(self.solver.graph)
        cols = submap.get_points_colors()
        poses = submap.get_all_poses_world(self.solver.graph)

        if self.vis_stride > 1 and pts is not None and len(pts) > 0:
            pts = pts[::self.vis_stride]
            cols = cols[::self.vis_stride]

        if cols is not None and len(cols) > 0 and cols.max() > 1.0:
            cols = cols / 255.0

        cam_positions = [p[:3, 3].tolist() for p in poses]
        cam_rotations = [p[:3, :3].tolist() for p in poses]

        return {
            'submap_id': submap.get_id(),
            'points': pts,
            'colors': cols,
            'cam_positions': cam_positions,
            'cam_rotations': cam_rotations,
        }

    def extract_stream_data_full(self) -> dict:
        """Re-extract all submaps. Called on loop closure or get_global_map."""
        try:
            num_submaps = self.solver.map.get_num_submaps()
            if num_submaps == 0:
                self._submap_cache.clear()
                return self._empty_data()

            self._submap_cache.clear()
            for submap in self.solver.map.get_submaps():
                entry = self._extract_one_submap(submap)
                self._submap_cache[entry['submap_id']] = entry

            return self._build_full_payload()

        except Exception as e:
            print(f"Extract data error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_data()

    def extract_stream_data_incremental(self, new_submap_id: int) -> dict:
        """Extract only the new submap. O(1) submap extraction."""
        try:
            submap = self.solver.map.get_submap(new_submap_id)
            entry = self._extract_one_submap(submap)
            self._submap_cache[new_submap_id] = entry
            return self._build_incremental_payload(entry)

        except Exception as e:
            print(f"Incremental extract error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_data()

    def extract_stream_data(self):
        """Backward-compatible alias — full extraction."""
        return self.extract_stream_data_full()

    def _build_full_payload(self) -> dict:
        """Build a full payload from all cached submap entries."""
        all_pts, all_cols, all_cam_pos, all_cam_rot = [], [], [], []
        for e in self._submap_cache.values():
            if e['points'] is not None and len(e['points']) > 0:
                all_pts.append(e['points'])
                all_cols.append(e['colors'])
            all_cam_pos.extend(e['cam_positions'])
            all_cam_rot.extend(e['cam_rotations'])

        if all_pts:
            pts = np.vstack(all_pts)
            cols = np.vstack(all_cols)
            center = np.mean(pts, axis=0)
            self._scene_center = center
            self.latest_scene_center = center
            pts = pts - center
        else:
            pts = np.zeros((0, 3))
            cols = np.zeros((0, 3))
            center = np.zeros(3)

        cam_arr = np.array(all_cam_pos) - center if all_cam_pos else np.array([])

        # Resolve pending beacons
        self._resolve_beacons(cam_arr)

        n_points = len(pts)
        n_cameras = len(cam_arr) if isinstance(cam_arr, np.ndarray) and cam_arr.ndim == 2 else 0

        pts_f32 = pts.astype(np.float32)
        cols_u8 = (cols * 255).clip(0, 255).astype(np.uint8)
        points_b64 = base64.b64encode(pts_f32.tobytes()).decode('ascii')
        colors_b64 = base64.b64encode(cols_u8.tobytes()).decode('ascii')

        return {
            'type': 'full',
            'frame_id': self.frame_count,
            'num_submaps': len(self._submap_cache),
            'num_loops': self.solver.graph.get_num_loops(),
            'points_b64': points_b64,
            'colors_b64': colors_b64,
            'points': [],
            'colors': [],
            'camera_positions': cam_arr.tolist() if n_cameras > 0 else [],
            'camera_rotations': all_cam_rot,
            'scene_center': center.tolist(),
            'n_points': n_points,
            'n_cameras': n_cameras,
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

    def _build_incremental_payload(self, entry: dict) -> dict:
        """Build an incremental payload for a single new submap."""
        pts, cols = entry['points'], entry['colors']
        if pts is not None and len(pts) > 0:
            # Use the current scene center for recentering (computed on last full extraction)
            pts_f32 = (pts - self._scene_center).astype(np.float32)
            cols_u8 = (cols * 255).clip(0, 255).astype(np.uint8)
            # Recenter camera positions too
            cam_pos = (np.array(entry['cam_positions']) - self._scene_center).tolist()
        else:
            pts_f32 = np.zeros((0, 3), dtype=np.float32)
            cols_u8 = np.zeros((0, 3), dtype=np.uint8)
            cam_pos = entry['cam_positions']

        points_b64 = base64.b64encode(pts_f32.tobytes()).decode('ascii')
        colors_b64 = base64.b64encode(cols_u8.tobytes()).decode('ascii')

        return {
            'type': 'incremental',
            'submap_id': entry['submap_id'],
            'frame_id': self.frame_count,
            'num_submaps': self.solver.map.get_num_submaps(),
            'num_loops': self.solver.graph.get_num_loops(),
            'points_b64': points_b64,
            'colors_b64': colors_b64,
            'points': [],
            'colors': [],
            'camera_positions': cam_pos,
            'camera_rotations': entry['cam_rotations'],
            'scene_center': self._scene_center.tolist(),
            'n_points': len(pts_f32),
            'n_cameras': len(entry['cam_positions']),
            'detections': [],
            'active_queries': list(self.active_queries),
            'resolved_beacons': self.resolved_beacons,
        }

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

    def run_detection_progressive(self, queries):
        """Generator: run CLIP+SAM submap-by-submap, yield partial detections."""
        with self._detection_lock:
            old_queries = set(self.active_queries)
            self.active_queries = [q.strip() for q in queries if q.strip()]
            new_queries = set(self.active_queries)

        if not self.active_queries:
            self._sam_cache.clear()
            with self._detection_lock:
                self.accumulated_detections = []
            yield {'detections': [], 'is_final': True}
            return

        # Purge cache for removed queries
        removed = old_queries - new_queries
        if removed:
            for k in [k for k in self._sam_cache if k[2] in removed]:
                del self._sam_cache[k]

        added = list(new_queries - old_queries)
        all_submaps = list(self.solver.map.get_submaps())

        if not added or not all_submaps:
            self._dedup_and_store()
            with self._detection_lock:
                yield {'detections': list(self.accumulated_detections), 'is_final': True}
            return

        for i, submap in enumerate(all_submaps):
            try:
                new_keys = self._run_clip_sam_on_submap(submap, added)
                if new_keys:
                    self._compute_bboxes_for_keys(new_keys)
            except Exception as e:
                print(f"Detection error submap {submap.get_id()}: {e}")

            self._dedup_and_store()
            with self._detection_lock:
                yield {
                    'detections': list(self.accumulated_detections),
                    'is_final': i == len(all_submaps) - 1,
                }

    # Maximum number of frames per submap to run SAM on, chosen by CLIP similarity rank.
    # Only the top-K most semantically matching frames are segmented, so SAM focuses
    # on clear, well-composed views rather than every frame that barely clears the threshold.
    SAM_TOP_K_FRAMES = 3

    def _run_clip_sam_on_submap(self, submap, queries):
        """Run CLIP matching + SAM on unprocessed (submap, frame, query) combos.

        Uses submap.get_all_semantic_vectors() for CLIP embeddings (VGGT-SLAM 2.0 API).
        Only the top SAM_TOP_K_FRAMES frames by CLIP similarity are segmented, which
        focuses SAM on the clearest, most semantically relevant views of the object.
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

            # Mark all frames as processed (empty) first, then overwrite for top-K
            candidate_frames = []
            for frame_idx in range(last_orig + 1):
                cache_key = (submap_id, frame_idx, query)
                if cache_key in self._sam_cache:
                    continue
                sim_val = sims[frame_idx].item()
                if sim_val < clip_thresh:
                    self._sam_cache[cache_key] = []
                else:
                    candidate_frames.append((sim_val, frame_idx))

            # Sort candidates by CLIP similarity descending; only run SAM on top-K
            candidate_frames.sort(key=lambda x: x[0], reverse=True)
            top_frames = candidate_frames[:self.SAM_TOP_K_FRAMES]
            skipped_frames = candidate_frames[self.SAM_TOP_K_FRAMES:]

            # Mark frames outside top-K as empty so they aren't reconsidered
            for _, frame_idx in skipped_frames:
                self._sam_cache[(submap_id, frame_idx, query)] = []

            for sim_val, frame_idx in top_frames:
                cache_key = (submap_id, frame_idx, query)
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
                                'clip_score': float(sim_val),
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
        """Build detection list from cache, dedup, store.

        Confidence used for ranking is a combined CLIP × SAM score so that the
        deduplication prefers frames that are both semantically clear (high CLIP)
        and well-segmented (high SAM), not just whichever had the biggest blob.
        """
        raw = []
        for (submap_id, frame_idx, query), masks in self._sam_cache.items():
            for entry in masks:
                bbox = entry.get('bbox_3d')
                if bbox is None:
                    continue
                clip_score = entry.get('clip_score', 1.0)
                seg_score = entry['seg_score']
                combined = clip_score * seg_score
                raw.append({
                    "success": True,
                    "query": query,
                    "bounding_box": bbox,
                    "confidence": combined,
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
    # Debug detection — full pipeline with rich per-frame diagnostics
    # ------------------------------------------------------------------

    def debug_detect_full(self, queries, clip_thresholds=None, sam_thresholds=None, top_k=None):
        """Run the full detection pipeline and return rich per-frame diagnostics.

        Does NOT touch _sam_cache or accumulated_detections — purely diagnostic.
        Respects top_k frame selection and combined CLIP×SAM ranking exactly as
        production does, so the debug page reflects what production would pick.

        Returns a dict matching the DebugDetectResponse type expected by the frontend.
        """
        import time as _time
        t0 = _time.time()

        od = self.object_detector
        clip_thresh_map = clip_thresholds or {}
        sam_thresh_map = sam_thresholds or {}
        effective_top_k = top_k if (top_k is not None and top_k > 0) else self.SAM_TOP_K_FRAMES

        all_frames_diag = []
        # Maps (submap_id, frame_idx, query) -> mask diag list for dedup
        key_to_masks = {}

        all_submaps = list(self.solver.map.get_submaps())
        if not all_submaps:
            return {
                'queries': queries, 'clip_thresholds': clip_thresh_map,
                'sam_thresholds': sam_thresh_map, 'top_k': effective_top_k,
                'frames': [], 'raw_detection_count': 0,
                'deduped_detection_count': 0, 'detections': [],
                'total_frames_scanned': 0, 'query_time_ms': 0,
            }

        for submap in all_submaps:
            submap_id = submap.get_id()
            clip_embs = submap.get_all_semantic_vectors()
            if clip_embs is None or len(clip_embs) == 0:
                continue
            if isinstance(clip_embs, np.ndarray):
                clip_embs = torch.from_numpy(clip_embs)

            last_orig = submap.get_last_non_loop_frame_index()
            if last_orig is None or last_orig < 0:
                last_orig = clip_embs.shape[0] - 1

            for query in queries:
                query = query.strip()
                if not query:
                    continue

                clip_thresh = clip_thresh_map.get(query, clip_thresh_map.get('default', 0.2))
                sam_thresh = sam_thresh_map.get(query, sam_thresh_map.get('default', 0.3))

                text_emb = od.encode_text_vector(query)
                sims = clip_embs @ text_emb  # (S,)

                # Rank all frames by CLIP similarity
                scored = []
                for frame_idx in range(last_orig + 1):
                    scored.append((sims[frame_idx].item(), frame_idx))
                scored.sort(key=lambda x: x[0], reverse=True)

                candidates_above = [(s, fi) for s, fi in scored if s >= clip_thresh]
                top_k_set = {fi for _, fi in candidates_above[:effective_top_k]}
                clip_rank_map = {fi: rank + 1 for rank, (_, fi) in enumerate(scored)}

                for frame_idx in range(last_orig + 1):
                    sim_val = sims[frame_idx].item()
                    above = sim_val >= clip_thresh
                    in_top_k = frame_idx in top_k_set
                    rank = clip_rank_map.get(frame_idx, frame_idx + 1)

                    # Thumbnail for all frames (cheap)
                    thumbnail = None
                    resolution = None
                    try:
                        frame_tensor = submap.get_frame_at_index(frame_idx)
                        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        h, w = frame_np.shape[:2]
                        resolution = f"{w}×{h}"
                        thumb_h = 120
                        thumb_w = int(w * thumb_h / h)
                        thumb = cv2.resize(frame_np, (thumb_w, thumb_h))
                        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode('.jpg', thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        thumbnail = base64.b64encode(buf.tobytes()).decode('ascii')
                    except Exception:
                        pass

                    sam_masks_diag = []
                    sam_error = None
                    sam_skipped = above and not in_top_k

                    if in_top_k and above:
                        try:
                            frame_tensor = submap.get_frame_at_index(frame_idx)
                            frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            frame_pil = Image.fromarray(frame_np)
                            seg_results = od.segment_all(frame_pil, query)

                            for mask_2d, box_2d, seg_score in seg_results:
                                above_sam = float(seg_score) >= sam_thresh
                                combined = sim_val * float(seg_score)
                                bbox_3d = None
                                has_3d_box = False
                                if above_sam:
                                    try:
                                        bbox_3d = od.compute_3d_bbox(
                                            submap, frame_idx, mask_2d,
                                            self.solver.graph, self.latest_scene_center
                                        )
                                        has_3d_box = bbox_3d is not None
                                    except Exception:
                                        pass

                                mask_image = None
                                try:
                                    mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, mask_2d)
                                except Exception:
                                    pass

                                mask_entry = {
                                    'score': float(seg_score),
                                    'clip_score': float(sim_val),
                                    'combined_score': combined,
                                    'box_2d': [float(v) for v in box_2d],
                                    'mask_image': mask_image,
                                    'above_sam_threshold': above_sam,
                                    'sam_threshold_used': sam_thresh,
                                    'has_3d_box': has_3d_box,
                                    'bbox_3d': bbox_3d,
                                    'dedup_kept': None,
                                }
                                sam_masks_diag.append(mask_entry)
                                if above_sam and has_3d_box:
                                    key_to_masks.setdefault((submap_id, frame_idx, query), []).append(mask_entry)
                        except Exception as e:
                            sam_error = str(e)

                    all_frames_diag.append({
                        'submap_id': submap_id,
                        'frame_idx': frame_idx,
                        'query': query,
                        'clip_similarity': float(sim_val),
                        'clip_rank': rank,
                        'above_threshold': above,
                        'in_top_k': in_top_k,
                        'sam_skipped': sam_skipped,
                        'clip_threshold_used': clip_thresh,
                        'sam_threshold_used': sam_thresh,
                        'top_k_used': effective_top_k,
                        'thumbnail': thumbnail,
                        'resolution': resolution,
                        'sam_masks': sam_masks_diag,
                        'sam_error': sam_error,
                        'detections_before_dedup': [],
                    })

        # Build raw detections and deduplicate using combined score
        raw_detections = []
        for (submap_id, frame_idx, query), masks in key_to_masks.items():
            for entry in masks:
                raw_detections.append({
                    'success': True,
                    'query': query,
                    'bounding_box': entry['bbox_3d'],
                    'confidence': entry['combined_score'],
                    'matched_submap': int(submap_id),
                    'matched_frame': int(frame_idx),
                    'clip_score': entry['clip_score'],
                    'sam_score': entry['score'],
                })

        deduped = ObjectDetector.deduplicate_detections(raw_detections)
        kept_keys = {(d['matched_submap'], d['matched_frame'], d['query']) for d in deduped}

        # Mark dedup_kept on mask entries
        for (submap_id, frame_idx, query), masks in key_to_masks.items():
            kept = (submap_id, frame_idx, query) in kept_keys
            for m in masks:
                if m['has_3d_box']:
                    m['dedup_kept'] = kept

        # Populate detections_before_dedup on frame diag entries
        raw_by_key = {}
        for r in raw_detections:
            raw_by_key.setdefault((r['matched_submap'], r['matched_frame'], r['query']), []).append(r)
        for fd in all_frames_diag:
            k = (fd['submap_id'], fd['frame_idx'], fd['query'])
            fd['detections_before_dedup'] = raw_by_key.get(k, [])

        elapsed_ms = int((_time.time() - t0) * 1000)
        return {
            'queries': queries,
            'clip_thresholds': clip_thresh_map,
            'sam_thresholds': sam_thresh_map,
            'top_k': effective_top_k,
            'frames': all_frames_diag,
            'raw_detection_count': len(raw_detections),
            'deduped_detection_count': len(deduped),
            'detections': deduped,
            'total_frames_scanned': len(all_frames_diag),
            'query_time_ms': elapsed_ms,
        }

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
        self._last_stream_data = None
        self._submap_cache.clear()
        self._scene_center = np.zeros(3)

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
