/**
 * Type definitions for Open Reality data structures
 */

export interface SLAMUpdate {
  frame_id: number;
  num_submaps: number;
  num_loops: number;
  /** Binary-encoded point positions: base64(float32 N×3 flat). Preferred over `points`. */
  points_b64?: string;
  /** Binary-encoded point colors: base64(uint8 N×3 flat, values 0-255). Preferred over `colors`. */
  colors_b64?: string;
  /** Legacy JSON arrays — empty when binary fields are present. */
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
  /** Update type: 'full' replaces entire scene, 'incremental' adds a single submap. */
  type?: 'full' | 'incremental';
  /** Submap ID for incremental updates. */
  submap_id?: number;
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
 * Result from a single object detection query (Open Reality pipeline)
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
 * Partial detection results streamed progressively during submap-by-submap scanning
 */
export interface DetectionPartialResult {
  detections: DetectionResult[];
  active_queries: string[];
  is_final: boolean;
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

// ------------------------------------------------------------------
// Spatial Agent types
// ------------------------------------------------------------------

export interface AgentThought {
  id: string;
  timestamp: number;
  type: 'observation' | 'thinking' | 'chat_response' | 'error' | 'action';
  content: string;
  keyframe_b64?: string;
  subagent?: string;
  phase?: string;
  confidence?: number;
  attachments?: AgentAttachment[];
}

export interface AgentAction {
  id: string;
  timestamp: number;
  action: string;          // mission_created | mission_completed | queries_updated
  details: string;
  mission_id?: number;
  queries?: string[];
}

export interface AgentFinding {
  id: string;
  timestamp: number;
  query: string;
  description: string;
  confidence: number;
  position?: number[];
  mission_id?: number;
  evidence?: Record<string, unknown>;
}

export interface MissionState {
  id: number;
  category: string;
  goal: string;
  queries: string[];
  found: string[];
  status: 'active' | 'recovering' | 'completed' | 'stalled';
  confidence: number;
  findings_count: number;
  last_progress_ts?: number;
  stall_reason?: string | null;
  stall_count?: number;
}

export interface AgentTaskState {
  id: string;
  timestamp: number;
  task_type: 'orchestrator' | 'subagent' | 'mission' | 'tool_batch';
  name: string;
  status: 'started' | 'heartbeat' | 'succeeded' | 'failed' | 'timed_out' | 'stalled' | 'resumed';
  mission_id?: number;
}

export interface AgentTaskEvent extends AgentTaskState {
  latency_ms?: number;
  details?: string;
  error?: string;
}

export interface AgentState {
  enabled: boolean;
  scene_description: string;
  room_type: string;
  missions: MissionState[];
  active_queries: string[];
  discovered_objects: string[];
  current_goal: string | null;
  submaps_processed: number;
  coverage_estimate: number;
  health?: 'ok' | 'degraded' | 'disabled';
  degraded_mode?: boolean;
  runtime_v2_enabled?: boolean;
  active_tasks?: AgentTaskState[];
  pending_jobs?: AgentJobEvent[];
  running_jobs?: AgentJobEvent[];
  last_job_errors?: string[];
  orchestrator_busy?: boolean;
}

export interface AgentAttachment {
  kind: 'keyframe' | 'mask' | 'other';
  image_b64?: string;
  label?: string;
}

export interface AgentUICommand {
  id: string;
  name:
    | 'focus_detection'
    | 'set_detection_queries'
    | 'show_waypoint'
    | 'show_path'
    | 'show_toast'
    | 'open_detection_preview';
  args: Record<string, unknown>;
  mission_id?: number;
  ttl_ms?: number;
}

export interface AgentUIResult {
  id: string;
  status: 'ok' | 'error' | 'ignored' | 'timeout';
  result?: Record<string, unknown>;
  error?: string;
}

export interface AgentToolEvent {
  id: string;
  tool: string;
  status: 'started' | 'succeeded' | 'failed';
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  latency_ms?: number;
}

export interface AgentJobEvent {
  id?: string;
  job_id: string;
  job_name: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'timed_out' | 'canceled' | 'not_found';
  mission_id?: number;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  created_at?: number;
  started_at?: number;
  finished_at?: number;
}
