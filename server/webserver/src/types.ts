/**
 * Type definitions for VGGT-SLAM data structures
 */

export interface SLAMUpdate {
  frame_id: number;
  num_submaps: number;
  num_loops: number;
  points: number[][];
  colors: number[][];
  camera_positions: number[][];
  camera_rotations: number[][][];
  scene_center: number[];
  n_points: number;
  n_cameras: number;
  resolved_beacons?: ResolvedBeacon[];
  detections?: DetectionResult[];
  active_queries?: string[];
}

export interface ResolvedBeacon {
  beacon_id: number;
  x: number;
  y: number;
  z: number;
}

export interface SLAMStats {
  frames: number;
  submaps: number;
  loops: number;
  points: number;
  cameras: number;
  fps: number;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ConnectionStatus {
  state: ConnectionState;
  message: string;
  latency?: number;
}

export interface ControlsConfig {
  showCameras: boolean;
  showPoints: boolean;
  showGrid: boolean;
  showAxes: boolean;
  pointSize: number;
  cameraSize: number;
  followCamera: boolean;
  flipY: boolean;
  showDetectionBoxes: boolean;
  showDetectionLabels: boolean;
}

/**
 * 3D oriented bounding box from object detection
 */
export interface BoundingBox3D {
  center: number[];
  extent: number[];
  rotation: number[][];
  corners: number[][];
}

/**
 * Result from a single object detection query (VGGT-SLAM 2.0 pipeline)
 */
export interface DetectionResult {
  success: boolean;
  query: string;
  bounding_box?: BoundingBox3D;
  confidence?: number;
  keyframe_image?: string;   // base64 JPEG of matched keyframe
  mask_image?: string;       // base64 PNG of SAM 3 mask overlay
  matched_submap?: number;
  matched_frame?: number;
  query_time_ms?: number;
  error?: string;
}

/**
 * Response from batch query endpoint
 */
export interface BatchDetectionResponse {
  success: boolean;
  results: DetectionResult[];
  total_queries: number;
  error?: string;
}

/**
 * On-demand preview data returned for a specific detection click
 */
export interface DetectionPreview {
  query: string;
  submap_id: number;
  frame_idx: number;
  keyframe_image?: string;   // base64 JPEG
  mask_image?: string;       // base64 PNG
  error?: string;
}
