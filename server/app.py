"""
Flask + python-socketio ASGI streaming server for VGGT-SLAM 2.0.

Handles:
  - WebSocket frame streaming from phone/browser
  - SLAM update broadcasting to viewer clients
  - Object detection queries (CLIP + SAM3)
  - Session-scoped spatial agents with shared SLAM core
  - Video file testing mode
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
import queue
import threading
import time
import uuid
import base64
import argparse
import asyncio
import mimetypes
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
import json
import re

from flask import Flask, jsonify, request, send_file
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
from PIL import Image
import socketio as socketio_pkg

from server.llm import OpenRouterClient
from server.streaming_slam import StreamingSLAM
from vggt_slam.object_detector import ObjectDetector

# ------------------------------
# Flask + python-socketio Setup
# ------------------------------
app = Flask(__name__)
_cors_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "*").strip()
if _cors_raw == "*":
    _cors_origins: str | list[str] = "*"
else:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
CORS(app, origins=_cors_origins if _cors_origins != "*" else "*")

sio = socketio_pkg.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=_cors_origins,
    max_http_buffer_size=10_000_000,
    ping_timeout=120,
    ping_interval=25,
)
asgi_application = socketio_pkg.ASGIApp(sio, WsgiToAsgi(app))

# Queues for streaming pipeline
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=10)

# Global SLAM processor (initialized in initialize() or start_server())
slam_processor: Optional[StreamingSLAM] = None
client_connected = threading.Event()

# Track connected socket IDs so we only stop SLAM when the last client leaves
_connected_sids: set[str] = set()
_sids_lock = threading.Lock()

# Background streaming task handle — started once when the first client connects
_stream_task: Optional[asyncio.Task] = None

# Event loop for scheduling cross-thread async tasks
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Single-threaded executor for blocking GPU ops (segment_all / detection pipeline)
_gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Demo mode (pre-recorded local videos)
_DEMO_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}

_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_video_rotation(frame, angle):
    """Rotate a frame to correct for orientation metadata (e.g. iPhone portrait MOV)."""
    code = _ROTATION_MAP.get(int(angle) % 360)
    if code is not None:
        return cv2.rotate(frame, code)
    return frame
_demo_lock = threading.Lock()
_demo_video_feeder = None
_demo_active_video_id = None
_demo_active_video_path = None
_demo_started_at = None
_demo_target_fps = None
_demo_thumbnail_cache = {}
_demo_catalog_cache = None
# Agent executor (model calls for multiple sessions)
_agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# Per-query detection task registry: (sid, query) -> asyncio.Task
_query_tasks: dict[tuple[str, str], asyncio.Task] = {}

# Query update lock (lazy-initialized under async context)
_query_update_lock: Optional[asyncio.Lock] = None

# OpenRouter clients for HTTP APIs
_plan_client: Optional[OpenRouterClient] = None
_assistant_client: Optional[OpenRouterClient] = None

# OpenRouter key cached during initialize()
_openrouter_api_key: str = ""

# Input guardrails (no-auth deployment still needs abuse protection)
MAX_QUERY_COUNT = int(os.environ.get("MAX_QUERY_COUNT", "16"))
MAX_QUERY_LEN = int(os.environ.get("MAX_QUERY_LEN", "120"))
MAX_FRAME_B64_LEN = int(os.environ.get("MAX_FRAME_B64_LEN", "10000000"))
FRAME_RATE_WINDOW_S = float(os.environ.get("FRAME_RATE_WINDOW_S", "1.0"))
FRAME_RATE_LIMIT = int(os.environ.get("FRAME_RATE_LIMIT", "45"))

_frame_rate_state: dict[str, list[float]] = {}


@dataclass
class SessionState:
    sid: str
    manual_queries: set[str] = field(default_factory=set)
    agent_queries: set[str] = field(default_factory=set)
    agent: Any = None
    connected_at: float = field(default_factory=time.time)
    ui_results: list[dict[str, Any]] = field(default_factory=list)


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


# ------------------------------
# Helpers: Query/session management
# ------------------------------
def _ensure_query_lock() -> asyncio.Lock:
    global _query_update_lock
    if _query_update_lock is None:
        _query_update_lock = asyncio.Lock()
    return _query_update_lock


def _normalize_query_list(queries: list[Any]) -> list[str]:
    norm: list[str] = []
    for item in queries:
        q = str(item).strip().lower()
        if len(q) > MAX_QUERY_LEN:
            q = q[:MAX_QUERY_LEN]
        if q and q not in norm:
            norm.append(q)
        if len(norm) >= MAX_QUERY_COUNT:
            break
    return norm


def _session_active_queries(sid: str) -> list[str]:
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return []
        merged = sorted(state.manual_queries | state.agent_queries)
    return merged


def _filter_detections_by_queries(detections: list[dict[str, Any]], queries: list[str]) -> list[dict[str, Any]]:
    if not queries:
        return []
    query_set = set(queries)
    return [
        det for det in detections
        if str(det.get("query", "")).strip().lower() in query_set
    ]


def _allow_frame_for_sid(sid: str) -> bool:
    now = time.time()
    with _sessions_lock:
        ts = _frame_rate_state.setdefault(sid, [])
        ts.append(now)
        cutoff = now - FRAME_RATE_WINDOW_S
        while ts and ts[0] < cutoff:
            ts.pop(0)
        return len(ts) <= FRAME_RATE_LIMIT


def _is_sid_connected(sid: str) -> bool:
    with _sids_lock:
        return sid in _connected_sids


def _emit_to_sid_threadsafe(sid: str, event: str, data: dict[str, Any]):
    loop = _event_loop
    if loop is None or loop.is_closed():
        return
    if not _is_sid_connected(sid):
        return

    def _on_done(fut: concurrent.futures.Future):
        try:
            fut.result()
        except Exception as emit_err:
            print(f"Emit error sid={sid} event={event}: {emit_err}")

    try:
        fut = asyncio.run_coroutine_threadsafe(sio.emit(event, data, to=sid), loop)
        fut.add_done_callback(_on_done)
    except Exception as e:
        print(f"Emit error sid={sid} event={event}: {e}")


def _build_agent_state_payload(sid: str) -> dict[str, Any]:
    with _sessions_lock:
        state = _sessions.get(sid)

    if state is None:
        return {"enabled": False, "active_queries": []}

    active_queries = sorted(state.manual_queries | state.agent_queries)

    if state.agent is None:
        return {
            "enabled": False,
            "scene_description": "",
            "room_type": "unknown",
            "missions": [],
            "active_queries": active_queries,
            "discovered_objects": [],
            "current_goal": None,
            "submaps_processed": 0,
            "coverage_estimate": 0.0,
            "health": "disabled",
            "degraded_mode": False,
            "active_tasks": [],
            "pending_jobs": [],
            "running_jobs": [],
            "last_job_errors": [],
            "orchestrator_busy": False,
        }

    payload = state.agent.get_state()
    payload["active_queries"] = active_queries
    return payload


def _collect_global_query_union() -> list[str]:
    with _sessions_lock:
        query_set: set[str] = set()
        for state in _sessions.values():
            query_set.update(state.manual_queries)
            query_set.update(state.agent_queries)
    return sorted(query_set)


def _agent_persist_query(sid: str, query: str):
    """Called from agent tools when they scan for a new object.

    Adds the query to manual_queries so it persists as a chip + bounding boxes
    until the user explicitly removes it via the UI. Also spawns a detection worker
    if one isn't already running for this query.
    """
    query = query.strip().lower()
    if not query or _event_loop is None or _event_loop.is_closed():
        return
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return
        if query in state.manual_queries:
            return  # Already visible, nothing to do
        state.manual_queries.add(query)

    _emit_to_sid_threadsafe(sid, "agent_ui_command", {
        "id": str(uuid.uuid4())[:12],
        "name": "add_detection_query",
        "args": {"query": query},
        "mission_id": None,
        "ttl_ms": None,
        "timestamp": time.time(),
    })
    asyncio.run_coroutine_threadsafe(
        _spawn_query_worker_if_needed(query, sid),
        _event_loop,
    )


async def _spawn_query_worker_if_needed(query: str, sid: str):
    existing = _query_tasks.get((sid, query))
    if existing and not existing.done():
        return
    task = asyncio.create_task(_run_single_query_detection(query, sid))
    _query_tasks[(sid, query)] = task


def _on_agent_queries_changed(sid: str, queries: list[str]):
    """Called from agent threads; spawns/cancels per-query detection workers."""
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return
        old = set(state.agent_queries)
        state.agent_queries = set(_normalize_query_list(queries))
        new = set(state.agent_queries)

    if _event_loop is None or _event_loop.is_closed():
        return

    removed = old - new
    added = new - old

    if not removed and not added:
        return

    asyncio.run_coroutine_threadsafe(
        _apply_agent_query_diff(sid, added, removed),
        _event_loop,
    )


async def _apply_agent_query_diff(sid: str, added: set[str], removed: set[str]):
    """Cancel tasks for removed agent queries and spawn tasks for new ones."""
    if slam_processor is None:
        return

    print(
        f"[agent_query_diff] sid={sid[:8]} "
        f"+{sorted(added)} -{sorted(removed)} "
        f"slam_active={slam_processor.active_queries}"
    )

    for q in removed:
        task = _query_tasks.pop((sid, q), None)
        if task and not task.done():
            task.cancel()
            print(f"  [agent_query_diff] cancelled task for '{q}'")

        with _sessions_lock:
            state = _sessions.get(sid)
            pinned = state is not None and q in state.manual_queries

        if pinned:
            # Query was added by an agent tool and is visible in the UI — keep it
            print(f"  [agent_query_diff] keeping '{q}' — pinned in manual_queries")
        else:
            slam_processor.remove_query(q)
            print(f"  [agent_query_diff] removed '{q}' — slam active_queries now: {slam_processor.active_queries}")
            await sio.emit(
                "agent_ui_command",
                {
                    "id": str(uuid.uuid4())[:12],
                    "name": "remove_detection_query",
                    "args": {"query": q},
                    "mission_id": None,
                    "ttl_ms": None,
                    "timestamp": time.time(),
                },
                to=sid,
            )

    if removed:
        active = _session_active_queries(sid)
        accumulated = list(slam_processor.accumulated_detections)
        filtered = _filter_detections_by_queries(accumulated, active)
        print(f"  [agent_query_diff] immediate update: active={active} filtered={len(filtered)}")
        await sio.emit(
            "detection_partial",
            {
                "detections": filtered,
                "active_queries": active,
                "is_final": True,
            },
            to=sid,
        )

    for q in sorted(added):
        print(f"  [agent_query_diff] spawning task for '{q}'")
        # Tell the frontend to add a chip for this agent query immediately
        await sio.emit(
            "agent_ui_command",
            {
                "id": str(uuid.uuid4())[:12],
                "name": "add_detection_query",
                "args": {"query": q},
                "mission_id": None,
                "ttl_ms": None,
                "timestamp": time.time(),
            },
            to=sid,
        )
        task = asyncio.create_task(_run_single_query_detection(q, sid))
        _query_tasks[(sid, q)] = task


def _create_session_agent(sid: str):
    if not _openrouter_api_key:
        return None

    from server.spatial_agent import SpatialAgent

    return SpatialAgent(
        streaming_slam=slam_processor,
        emit_fn=lambda event, data: _emit_to_sid_threadsafe(sid, event, data),
        openrouter_api_key=_openrouter_api_key,
        session_id=sid,
        on_queries_changed=_on_agent_queries_changed,
        on_query_persisted=lambda q: _agent_persist_query(sid, q),
    )


def _ensure_session(sid: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is not None:
            return state

        state = SessionState(sid=sid)
        if slam_processor is not None:
            state.agent = _create_session_agent(sid)
        _sessions[sid] = state
        return state


async def _run_single_query_detection(query: str, trigger_sid: str):
    """Run detection for a single query, streaming partials to the triggering session."""
    if slam_processor is None:
        return

    print(f"[query_worker:{query}] started sid={trigger_sid[:8]}")

    loop = asyncio.get_event_loop()
    partial_q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for partial in slam_processor.add_query_progressive(query):
                loop.call_soon_threadsafe(partial_q.put_nowait, partial)
        except Exception as e:
            loop.call_soon_threadsafe(
                partial_q.put_nowait,
                {"detections": [], "is_final": True, "error": str(e)},
            )

    _gpu_executor.submit(run)

    partial_count = 0
    try:
        while True:
            partial = await partial_q.get()
            partial_count += 1
            active = _session_active_queries(trigger_sid)
            all_dets = partial.get("detections", [])
            filtered = _filter_detections_by_queries(all_dets, active)
            is_final = bool(partial.get("is_final", False))

            print(
                f"[query_worker:{query}] partial #{partial_count} "
                f"total_dets={len(all_dets)} filtered={len(filtered)} "
                f"active={active} is_final={is_final}"
                + (f" error={partial['error']}" if partial.get("error") else "")
            )

            if _is_sid_connected(trigger_sid):
                await sio.emit(
                    "detection_partial",
                    {
                        "detections": filtered,
                        "active_queries": active,
                        "is_final": is_final,
                    },
                    to=trigger_sid,
                )
            if is_final:
                break
    except asyncio.CancelledError:
        print(f"[query_worker:{query}] cancelled after {partial_count} partials")
        raise

    print(f"[query_worker:{query}] done — {partial_count} partials emitted")
    _query_tasks.pop((trigger_sid, query), None)


async def _refresh_global_detection_queries(trigger_sid: Optional[str], emit_progress: bool):
    """Recompute shared detection cache from session query union.

    Compatibility behavior:
      - Emits detection_partial only to the triggering session (when requested).
      - Global detector still runs on union of all session queries.
    """
    if slam_processor is None:
        return

    lock = _ensure_query_lock()
    async with lock:
        global_queries = _collect_global_query_union()

        loop = asyncio.get_event_loop()
        partial_q: asyncio.Queue = asyncio.Queue()

        def run_gen():
            try:
                for partial in slam_processor.run_detection_progressive(global_queries):
                    loop.call_soon_threadsafe(partial_q.put_nowait, partial)
            except Exception as e:
                loop.call_soon_threadsafe(
                    partial_q.put_nowait,
                    {"detections": [], "is_final": True, "error": str(e)},
                )

        _gpu_executor.submit(run_gen)

        while True:
            partial = await partial_q.get()
            if emit_progress and trigger_sid is not None:
                active = _session_active_queries(trigger_sid)
                filtered = _filter_detections_by_queries(partial.get("detections", []), active)
                await sio.emit(
                    "detection_partial",
                    {
                        "detections": filtered,
                        "active_queries": active,
                        "is_final": bool(partial.get("is_final", False)),
                    },
                    to=trigger_sid,
                )
            if partial.get("is_final", False):
                break

        if trigger_sid is not None:
            await sio.emit("agent_state", _build_agent_state_payload(trigger_sid), to=trigger_sid)


# ------------------------------
# Video File Feeder (testing mode)
# ------------------------------
class VideoFeeder:
    """Reads frames from a video file and pushes them into frame_queue."""

    def __init__(self, video_path, fast=False, target_fps=2.0):
        self.video_path = video_path
        self.fast = fast
        self.target_fps = target_fps
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._thread.start()
        print(
            f"VideoFeeder started: {self.video_path} "
            f"(target_fps={self.target_fps}, fast={self.fast})"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _feed_loop(self):
        if _is_lfs_pointer(self.video_path):
            print(f"Failed to open video: {self.video_path}"
                  " (file is a Git LFS pointer, not actual video content)")
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return

        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        skip = max(1, int(round(video_fps / self.target_fps)))
        effective_fps = video_fps / skip
        delay = 0.0 if self.fast else 1.0 / effective_fps

        frames_to_feed = total_frames // skip
        print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS")
        print(
            f"  Feeding every {skip} frame(s) -> ~{effective_fps:.1f} effective FPS "
            f"(~{frames_to_feed} frames, delay={delay * 1000:.0f}ms)"
        )

        raw_idx = 0
        fed_count = 0
        t0 = time.time()

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if (raw_idx - 1) % skip != 0:
                continue

            if rotation:
                frame = _apply_video_rotation(frame, rotation)

            ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
            data = {"image": b64, "timestamp": time.time()}

            try:
                frame_queue.put(data, timeout=10)
            except queue.Full:
                print(f"frame_queue full, dropping frame {fed_count}")
                continue

            if fed_count == 0 and slam_processor is not None and not slam_processor.is_running:
                print("Auto-starting SLAM processing (video mode)...")
                slam_processor.start()

            fed_count += 1
            if fed_count % 50 == 0:
                elapsed = time.time() - t0
                print(f"Fed {fed_count}/{frames_to_feed} frames ({elapsed:.1f}s elapsed)")

            if delay > 0:
                time.sleep(delay)

        cap.release()
        elapsed = time.time() - t0
        print(f"VideoFeeder finished: {fed_count} frames fed in {elapsed:.1f}s")


def _get_demo_video_dir():
    return os.environ.get(
        'DEMO_VIDEO_DIR',
        os.path.join(os.path.dirname(__file__), 'demo_videos'),
    )


def _clear_queues():
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Exception:
            break
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break


def _stop_demo_feeder():
    global _demo_video_feeder, _demo_active_video_id, _demo_active_video_path
    global _demo_started_at, _demo_target_fps
    with _demo_lock:
        if _demo_video_feeder is not None:
            print("Stopping active demo feeder...")
            _demo_video_feeder.stop()
            _demo_video_feeder = None
            _demo_active_video_id = None
            _demo_active_video_path = None
            _demo_started_at = None
            _demo_target_fps = None


def _is_lfs_pointer(path):
    """Return True if the file is a Git LFS pointer (not actual video content)."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "r") as f:
            return "git-lfs" in f.read(50)
    except Exception:
        return False


def _is_supported_video_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in _DEMO_VIDEO_EXTENSIONS


def _safe_demo_path(video_id):
    demo_dir = os.path.abspath(_get_demo_video_dir())
    if not video_id:
        raise ValueError('video_id is required')

    normalized_id = os.path.normpath(str(video_id).replace('\\', '/')).lstrip('/')
    full_path = os.path.abspath(os.path.join(demo_dir, normalized_id))

    if os.path.commonpath([demo_dir, full_path]) != demo_dir:
        raise ValueError('Invalid video_id path')
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f'Video not found: {video_id}')
    if not _is_supported_video_file(full_path):
        raise ValueError('Unsupported video format')
    return full_path


def _collect_demo_video_files():
    demo_dir = os.path.abspath(_get_demo_video_dir())
    if not os.path.isdir(demo_dir):
        return []
    candidates = []
    for root, _, files in os.walk(demo_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if not _is_supported_video_file(full_path):
                continue
            if _is_lfs_pointer(full_path):
                print(f"Skipping Git LFS pointer: {full_path}")
                continue
            candidates.append(full_path)
    candidates.sort()
    return candidates


def _build_thumbnail_data_url(video_path):
    cache_key = (video_path, os.path.getmtime(video_path))
    cached = _demo_thumbnail_cache.get(cache_key)
    if cached:
        return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    thumbnail = None
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            continue
        if rotation:
            frame = _apply_video_rotation(frame, rotation)
        h, w = frame.shape[:2]
        if w > 320:
            scale = 320.0 / float(w)
            frame = cv2.resize(frame, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
            thumbnail = f'data:image/jpeg;base64,{b64}'
            break
    cap.release()

    if thumbnail:
        _demo_thumbnail_cache[cache_key] = thumbnail
    return thumbnail


def _build_demo_catalog(force_refresh=False):
    global _demo_catalog_cache
    if _demo_catalog_cache is not None and not force_refresh:
        return _demo_catalog_cache

    demo_dir = os.path.abspath(_get_demo_video_dir())
    videos = []
    for video_path in _collect_demo_video_files():
        rel_path = os.path.relpath(video_path, demo_dir).replace(os.sep, '/')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration_sec = float(frame_count / fps) if fps > 0 else 0.0
        videos.append({
            'video_id': rel_path,
            'name': os.path.splitext(os.path.basename(video_path))[0],
            'filename': os.path.basename(video_path),
            'mime_type': mimetypes.guess_type(video_path)[0] or 'video/mp4',
            'thumbnail': _build_thumbnail_data_url(video_path),
            'fps': round(float(fps), 3) if fps > 0 else None,
            'duration_sec': round(duration_sec, 2) if duration_sec > 0 else None,
            'width': width or None,
            'height': height or None,
        })
    _demo_catalog_cache = videos
    return videos


# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "gpu": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Soft reset: clear SLAM data, keep models loaded."""
    _stop_demo_feeder()
    _clear_queues()
    if slam_processor is None:
        return jsonify({"status": "no_processor"}), 503

    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Exception:
            break

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break

    # Cancel all per-query detection tasks
    for key in list(_query_tasks.keys()):
        task = _query_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    agents_to_reset = []
    with _sessions_lock:
        _frame_rate_state.clear()
        for state in _sessions.values():
            state.manual_queries.clear()
            state.agent_queries.clear()
            if state.agent is not None:
                agents_to_reset.append(state.agent)

    for agent in agents_to_reset:
        try:
            agent.reset()
        except Exception:
            pass

    slam_processor.soft_reset()
    slam_processor.set_detection_queries([])

    if _event_loop is not None and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(sio.emit("slam_reset", {"status": "reset"}), _event_loop)

    return jsonify({"status": "reset_complete", "message": "SLAM and session state cleared"})


@app.route('/api/demo/videos', methods=['GET'])
def list_demo_videos():
    """Return local demo videos with base64 thumbnails for selection UI."""
    refresh = request.args.get('refresh', '').lower() in {'1', 'true', 'yes'}
    videos = _build_demo_catalog(force_refresh=refresh)
    return jsonify({
        'videos': videos,
        'demo_dir': _get_demo_video_dir(),
        'active_video_id': _demo_active_video_id,
    })


@app.route('/api/demo/status', methods=['GET'])
def demo_status():
    """Return current demo feeder state for sender-side preview sync."""
    return jsonify({
        'running': _demo_video_feeder is not None,
        'video_id': _demo_active_video_id,
        'started_at': _demo_started_at,
        'target_fps': _demo_target_fps,
    })


@app.route('/api/demo/video', methods=['GET'])
def get_demo_video():
    """Serve a selected demo video file for browser-side preview playback."""
    video_id = request.args.get('video_id')
    try:
        video_path = _safe_demo_path(video_id)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400
    return send_file(video_path, conditional=True)


@app.route('/api/demo/start', methods=['POST'])
def start_demo():
    """Start feeding a selected local demo video into frame_queue."""
    global _demo_video_feeder, _demo_active_video_id, _demo_active_video_path
    global _demo_started_at, _demo_target_fps
    data = request.get_json() or {}
    video_id = data.get('video_id')
    target_fps = float(data.get('fps', 10.0))

    if slam_processor is None:
        return jsonify({'error': 'SLAM processor not initialized'}), 503

    try:
        video_path = _safe_demo_path(video_id)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400

    target_fps = max(0.5, min(30.0, target_fps))

    with _demo_lock:
        if _demo_video_feeder is not None:
            _demo_video_feeder.stop()
            _demo_video_feeder = None

        _clear_queues()
        slam_processor.soft_reset()

        _demo_video_feeder = VideoFeeder(video_path, fast=False, target_fps=target_fps)
        _demo_video_feeder.start()
        _demo_active_video_id = video_id
        _demo_active_video_path = video_path
        _demo_started_at = time.time()
        _demo_target_fps = target_fps

    return jsonify({
        'status': 'demo_started',
        'video_id': video_id,
        'fps': target_fps,
    })


@app.route('/api/demo/stop', methods=['POST'])
def stop_demo():
    """Stop active demo video feeder."""
    _stop_demo_feeder()
    return jsonify({'status': 'demo_stopped'})
# Lazy-initialized OpenRouter client for the /api/plan route
_plan_client = None

_PLAN_STOPWORDS = {
    "i",
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "my",
    "me",
    "we",
    "is",
    "are",
    "was",
    "want",
    "need",
    "track",
    "find",
    "locate",
    "detect",
    "identify",
    "watch",
    "follow",
    "using",
    "with",
    "this",
    "that",
    "please",
    "help",
    "looking",
    "look",
    "search",
    "explore",
    "navigate",
    "moving",
    "move",
    "walk",
    "walking",
    "around",
    "inside",
    "outside",
    "toward",
    "towards",
    "front",
    "back",
    "left",
    "right",
    "from",
    "into",
    "through",
    "near",
    "scene",
    "room",
    "area",
    "object",
    "objects",
    "item",
    "items",
    "stuff",
    "things",
    "video",
    "camera",
    "demo",
    "live",
}

_GENERIC_OBJECT_WORDS = {
    "object",
    "objects",
    "item",
    "items",
    "thing",
    "things",
    "target",
    "targets",
}

_PLAN_OBJECT_HINTS: list[tuple[set[str], list[str]]] = [
    ({"office", "workspace", "desk"}, ["chair", "table", "laptop", "monitor", "keyboard", "bottle"]),
    ({"classroom", "school"}, ["chair", "desk", "backpack", "laptop", "whiteboard"]),
    ({"kitchen"}, ["refrigerator", "microwave", "sink", "cup", "bottle"]),
    ({"living", "livingroom", "sofa"}, ["couch", "coffee table", "tv", "lamp", "remote"]),
    ({"hallway", "corridor"}, ["door", "sign", "chair", "trash can"]),
    ({"garage", "workshop"}, ["toolbox", "drill", "ladder", "box"]),
    ({"crime", "evidence"}, ["phone", "wallet", "bag", "bottle"]),
    ({"disaster", "rescue"}, ["person", "backpack", "bottle", "helmet"]),
]

_PLAN_DEFAULT_OBJECTS = [
    "person",
    "chair",
    "table",
    "bottle",
    "backpack",
    "door",
]


def _clean_plan_object_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 36:
        cleaned = cleaned[:36].strip()
    return cleaned


def _extract_prompt_object_candidates(prompt: str) -> list[str]:
    text = str(prompt or "").lower()
    words = re.findall(r"[a-z][a-z0-9\-]{1,}", text)
    word_set = set(words)
    candidates: list[str] = []

    def add(name: str):
        obj = _clean_plan_object_name(name)
        if not obj or obj in candidates:
            return
        if obj in _PLAN_STOPWORDS or obj in _GENERIC_OBJECT_WORDS:
            return
        candidates.append(obj)

    # Domain hints improve relevance for common scene types.
    for keys, objs in _PLAN_OBJECT_HINTS:
        if keys & word_set:
            for obj in objs:
                add(obj)

    # Lightweight keyword extraction from prompt text.
    for word in words:
        if word in _PLAN_STOPWORDS or word in _GENERIC_OBJECT_WORDS:
            continue
        if len(word) < 3 or len(word) > 24:
            continue
        add(word)

    return candidates


def _finalize_plan_objects(
    llm_objects: Any,
    prompt: str,
    min_count: int = 5,
    max_count: int = 8,
) -> list[str]:
    merged: list[str] = []

    def add(value: Any):
        obj = _clean_plan_object_name(value)
        if not obj:
            return
        if obj in _PLAN_STOPWORDS or obj in _GENERIC_OBJECT_WORDS:
            return
        if obj not in merged:
            merged.append(obj)

    if isinstance(llm_objects, str):
        for part in re.split(r"[,;]\s*", llm_objects):
            add(part)
    elif isinstance(llm_objects, list):
        for item in llm_objects:
            add(item)

    for candidate in _extract_prompt_object_candidates(prompt):
        add(candidate)

    for fallback in _PLAN_DEFAULT_OBJECTS:
        add(fallback)

    if len(merged) < min_count:
        for fallback in ["phone", "bag", "laptop", "cup"]:
            add(fallback)
            if len(merged) >= min_count:
                break

    return merged[:max_count]

def _get_plan_client() -> OpenRouterClient:
    global _plan_client
    if _plan_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        _plan_client = OpenRouterClient(
            api_key=api_key,
            primary_model=os.environ.get(
                "PLAN_MODEL", "google/gemini-3-flash-preview"
            ),
            fallback_models=[
                os.environ.get("PLAN_FALLBACK_MODEL", "openai/gpt-4o-mini")
            ],
            timeout=15.0,
            app_name="Real-Eyes Plan API",
            max_retries=2,
        )
    return _plan_client


def _plan_response_from_result(result):
    return jsonify(
        {
            "objects": result.get("objects", []),
            "waypoints": {
                "enabled": True,
                "justification": result.get(
                    "waypoints_justification", "Waypoints mark key locations."
                ),
            },
            "pathfinding": {
                "enabled": True,
                "justification": result.get(
                    "pathfinding_justification",
                    "Pathfinding visualizes your traversed route.",
                ),
            },
        }
    )


@app.route("/api/plan", methods=["POST"])
def generate_plan():
    """Generate a tracking plan from a natural language prompt via OpenRouter."""
    data = request.get_json() or {}
    prompt = str(data.get("prompt", "")).strip()

    try:
        client = _get_plan_client()
        system_prompt = (
            "You extract concrete visible physical objects from user scenarios for "
            "3D spatial tracking. Output strict JSON only."
        )
        user_prompt = (
            f'Given this scenario: "{prompt}"\n'
            "Return JSON with exact keys: "
            '{"objects": ["obj1", "obj2", "obj3", "obj4", "obj5"], '
            '"waypoints_justification": "1-2 sentences", '
            '"pathfinding_justification": "1-2 sentences", '
            '"agent_intro": "1-2 sentence first-person statement (starting with I will...) describing what you will do as the spatial intelligence agent for this mission"}. '
            "Objects must be concrete physical items trackable in 3D space. "
            "Return 5-8 unique objects, prioritized by relevance."
        )
        result, _ = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=400,
        )
        finalized_objects = _finalize_plan_objects(result.get("objects", []), prompt)

        return jsonify(
            {
                "objects": finalized_objects,
                "waypoints": {
                    "enabled": True,
                    "justification": result.get(
                        "waypoints_justification", "Waypoints mark key locations."
                    ),
                },
                "pathfinding": {
                    "enabled": True,
                    "justification": result.get(
                        "pathfinding_justification",
                        "Pathfinding visualizes your traversed route.",
                    ),
                },
                "agent_intro": result.get(
                    "agent_intro",
                    "I will scan the scene and lock onto all high-value targets in the environment.",
                ),
            }
        )
    except Exception as e:
        print(f"Plan generation error: {e}; falling back to keyword extraction")
        objects = _finalize_plan_objects([], prompt)
        return jsonify(
            {
                "objects": objects,
                "waypoints": {
                    "enabled": True,
                    "justification": "Waypoints help mark key locations.",
                },
                "pathfinding": {
                    "enabled": True,
                    "justification": "Pathfinding visualizes your traversed route.",
                },
                "agent_intro": "I will scan the scene and lock onto all high-value targets in the environment.",
            }
        )


def _get_assistant_client() -> OpenRouterClient:
    global _assistant_client
    if _assistant_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        _assistant_client = OpenRouterClient(
            api_key=api_key,
            primary_model=os.environ.get(
                "ASSISTANT_MODEL", "google/gemini-3-flash-preview"
            ),
            fallback_models=[
                os.environ.get("ASSISTANT_FALLBACK_MODEL", "openai/gpt-4o-mini")
            ],
            timeout=15.0,
            app_name="Real-Eyes Summary Assistant",
            max_retries=2,
        )
    return _assistant_client


@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    """Server-side chat endpoint for summary/dashboard assistants."""
    data = request.get_json() or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    history = data.get("history")
    if not isinstance(history, list):
        history = []

    context = data.get("context") if isinstance(data.get("context"), dict) else {}
    snapshot = context.get("snapshot") if isinstance(context, dict) else None
    detections = context.get("detections") if isinstance(context, dict) else None
    images_b64 = context.get("images_b64") if isinstance(context, dict) else None

    context_parts: list[str] = []
    if isinstance(snapshot, dict):
        n_points = snapshot.get("n_points", 0)
        n_cameras = snapshot.get("n_cameras", 0)
        num_submaps = snapshot.get("num_submaps", 0)
        context_parts.append(
            f"Scan stats: {n_points} points, {n_cameras} camera frames, {num_submaps} submaps."
        )

    if isinstance(detections, list) and detections:
        names = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            q = str(det.get("query", "")).strip().lower()
            if q and q not in names:
                names.append(q)
        if names:
            context_parts.append("Detected objects: " + ", ".join(names[:20]))

    system_prompt = (
        "You are a helpful assistant for a live 3D SLAM mapping system. "
        "Use the provided scan context, be precise, and stay concise (2-5 sentences)."
    )
    user_prompt = user_message + "\n\nContext:\n" + "\n".join(context_parts)

    try:
        client = _get_assistant_client()
        response = client.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history[-10:],
            images_b64=images_b64 if isinstance(images_b64, list) else None,
            temperature=0.4,
            max_tokens=512,
        )
        return jsonify(
            {
                "reply": response.content,
                "model": response.model,
                "degraded": response.degraded,
            }
        )
    except Exception as e:
        print(f"Assistant chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------
# SocketIO Events
# ------------------------------
@sio.on("connect")
async def handle_connect(sid, environ, auth):
    global _stream_task, _event_loop
    if slam_processor is None:
        return

    _event_loop = asyncio.get_running_loop()

    with _sids_lock:
        _connected_sids.add(sid)

    _ensure_session(sid)

    print(f"Client connected ({len(_connected_sids)} total)")
    client_connected.set()

    if _stream_task is None or _stream_task.done():
        _stream_task = asyncio.ensure_future(stream_results())

    await sio.emit("connected", {"status": "ready"}, to=sid)
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("disconnect")
async def handle_disconnect(sid):
    global _event_loop
    if slam_processor is None:
        return

    with _sids_lock:
        _connected_sids.discard(sid)
        remaining = len(_connected_sids)

    with _sessions_lock:
        state = _sessions.pop(sid, None)
        _frame_rate_state.pop(sid, None)
    if state is not None and state.agent is not None:
        try:
            state.agent.shutdown()
        except Exception:
            pass

    # Cancel all per-query detection tasks for this session
    for key in [k for k in _query_tasks if k[0] == sid]:
        task = _query_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()

    print(f"Client disconnected ({remaining} remaining)")

    await _refresh_global_detection_queries(trigger_sid=None, emit_progress=False)

    if remaining == 0:
        _event_loop = None
        client_connected.clear()
        slam_processor.stop()
        _stop_demo_feeder()
        _clear_queues()

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Exception:
                break
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break


@sio.on("frame")
async def handle_frame(sid, data):
    if slam_processor is None:
        return
    if not isinstance(data, dict):
        return
    img_b64 = data.get("image")
    if not isinstance(img_b64, str) or not img_b64:
        return
    if len(img_b64) > MAX_FRAME_B64_LEN:
        await sio.emit("error", {"error": "frame_too_large"}, to=sid)
        return
    if not _allow_frame_for_sid(sid):
        return
    if not frame_queue.full():
        frame_queue.put(data)
    if not slam_processor.is_running:
        print("Auto-starting SLAM processing...")
        slam_processor.start()


@sio.on("stop_slam")
async def handle_stop(sid, data=None):
    if slam_processor is None:
        return
    _stop_demo_feeder()
    slam_processor.stop()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_gpu_executor, slam_processor.finalize_detection_state)
    except Exception as e:
        print(f"Final detection reconciliation error: {e}")
    await sio.emit('slam_stopped', {'status': 'stopped'}, to=sid)


@sio.on("set_detection_queries")
async def handle_set_detection_queries(sid, data):
    if slam_processor is None:
        return
    if not isinstance(data, dict):
        data = {}
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = _normalize_query_list(raw_queries)
    state = _ensure_session(sid)

    with _sessions_lock:
        old = set(state.manual_queries)
    new = set(queries)

    removed = old - new
    added = new - old

    print(
        f"[set_detection_queries] sid={sid[:8]} "
        f"old={sorted(old)} new={sorted(new)} "
        f"+{sorted(added)} -{sorted(removed)} "
        f"active_tasks={[k[1] for k in _query_tasks if k[0] == sid]}"
    )

    # Cancel tasks and remove detections for removed queries
    for q in removed:
        task = _query_tasks.pop((sid, q), None)
        if task and not task.done():
            task.cancel()
            print(f"  [set_detection_queries] cancelled task for '{q}'")
        slam_processor.remove_query(q)
        print(f"  [set_detection_queries] removed query '{q}' — slam active_queries now: {slam_processor.active_queries}")

    with _sessions_lock:
        state.manual_queries = new

    # Emit immediate update if anything was removed
    if removed:
        active = _session_active_queries(sid)
        accumulated = list(slam_processor.accumulated_detections)
        filtered = _filter_detections_by_queries(accumulated, active)
        print(
            f"  [set_detection_queries] immediate update: "
            f"accumulated={len(accumulated)} filtered={len(filtered)} "
            f"active_queries={active}"
        )
        await sio.emit(
            "detection_partial",
            {
                "detections": filtered,
                "active_queries": active,
                "is_final": True,
            },
            to=sid,
        )

    # Spawn tasks for newly added queries
    for q in sorted(added):
        print(f"  [set_detection_queries] spawning detection task for '{q}'")
        task = asyncio.create_task(_run_single_query_detection(q, sid))
        _query_tasks[(sid, q)] = task


@sio.on("get_detection_preview")
async def handle_get_detection_preview(sid, data):
    if slam_processor is None:
        return

    if not isinstance(data, dict):
        data = {}
    try:
        submap_id = int(data.get("submap_id"))
        frame_idx = int(data.get("frame_idx"))
    except Exception:
        await sio.emit("detection_preview", {"error": "Invalid submap/frame"}, to=sid)
        return
    query = str(data.get("query", "")).strip()[:MAX_QUERY_LEN]

    try:
        submap = slam_processor.solver.map.get_submap(submap_id)
        if submap is None:
            await sio.emit(
                "detection_preview",
                {"error": f"Submap {submap_id} not found"},
                to=sid,
            )
            return

        frame_tensor = submap.get_frame_at_index(frame_idx)
        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)

        keyframe_image = ObjectDetector.image_to_base64(frame_np)

        mask_image = None
        if query:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _gpu_executor,
                slam_processor.object_detector.segment_all,
                frame_pil,
                query,
            )
            if results:
                best_mask, _, _ = max(results, key=lambda r: r[2])
                mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, best_mask)

        await sio.emit(
            "detection_preview",
            {
                "query": query,
                "submap_id": submap_id,
                "frame_idx": frame_idx,
                "keyframe_image": keyframe_image,
                "mask_image": mask_image,
            },
            to=sid,
        )

    except Exception as e:
        print(f"Error generating preview: {e}")
        await sio.emit("detection_preview", {"error": str(e)}, to=sid)


@sio.on("place_beacon")
async def handle_place_beacon(sid, data):
    if slam_processor is None:
        return

    beacon_id = data.get("beacon_id")
    frame_number = data.get("frame_number", 0)
    slam_processor.pending_beacons.append({"beacon_id": beacon_id, "frame_number": frame_number})
    print(f"Beacon {beacon_id} queued at frame {frame_number}")
    await sio.emit("beacon_queued", {"beacon_id": beacon_id}, to=sid)


@sio.on("clear_beacons")
async def handle_clear_beacons(sid, data=None):
    if slam_processor is None:
        return
    slam_processor.pending_beacons.clear()
    slam_processor.resolved_beacons.clear()
    print("All beacons cleared")


@sio.on("debug_detect")
async def handle_debug_detect(sid, data):
    if slam_processor is None:
        await sio.emit("debug_detect_results", {"error": "SLAM not initialized"}, to=sid)
        return

    queries = data.get("queries", [])
    clip_thresholds = data.get("clip_thresholds", {})
    sam_thresholds = data.get("sam_thresholds", {})
    top_k = data.get("top_k", None)

    if not queries:
        await sio.emit("debug_detect_results", {"error": "No queries provided"}, to=sid)
        return

    print(f"Debug detect: {queries} (CLIP={clip_thresholds}, SAM={sam_thresholds}, top_k={top_k})")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _gpu_executor,
            lambda: slam_processor.debug_detect_full(queries, clip_thresholds, sam_thresholds, top_k),
        )
        await sio.emit("debug_detect_results", result, to=sid)
    except Exception as e:
        print(f"Debug detect error: {e}")
        await sio.emit("debug_detect_results", {"error": str(e)}, to=sid)


@sio.on("get_global_map")
async def handle_get_global_map(sid, data=None):
    if slam_processor is None:
        return

    print("Client requested global map")
    try:
        if slam_processor.solver.map.get_num_submaps() > 0:
            if slam_processor._last_stream_data and slam_processor._last_stream_data.get("type") == "full":
                stream_data = dict(slam_processor._last_stream_data)
            else:
                stream_data = slam_processor.extract_stream_data_full()

            with slam_processor._detection_lock:
                all_detections = list(slam_processor.accumulated_detections)

            active_queries = _session_active_queries(sid)
            stream_data["active_queries"] = active_queries
            stream_data["detections"] = _filter_detections_by_queries(all_detections, active_queries)

            if stream_data and stream_data.get("n_points", 0) > 0:
                await sio.emit("global_map", stream_data, to=sid)
            else:
                empty = slam_processor._empty_data()
                empty["active_queries"] = active_queries
                empty["detections"] = []
                await sio.emit("global_map", empty, to=sid)
        else:
            empty = slam_processor._empty_data()
            empty["active_queries"] = _session_active_queries(sid)
            empty["detections"] = []
            await sio.emit("global_map", empty, to=sid)
    except Exception as e:
        print(f"Error fetching global map: {e}")


# ------------------------------
# Spatial Agent SocketIO Events
# ------------------------------
@sio.on("agent_chat")
async def handle_agent_chat(sid, data):
    state = _ensure_session(sid)
    if state.agent is None:
        return

    message = str(data.get("message", "")).strip()
    if not message:
        return

    await sio.emit(
        "agent_action",
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "action": "request_received",
            "details": "Chat request received; executing.",
        },
        to=sid,
    )
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(_agent_executor, state.agent.handle_user_message, message),
            timeout=float(os.environ.get("SPATIAL_CHAT_TIMEOUT_S", "25")),
        )
    except asyncio.TimeoutError:
        await sio.emit(
            "agent_task_event",
            {
                "id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "task_type": "orchestrator",
                "name": "chat_request",
                "status": "timed_out",
                "error": "chat request timed out",
            },
            to=sid,
        )
        await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("agent_set_goal")
async def handle_agent_set_goal(sid, data):
    state = _ensure_session(sid)
    if state.agent is None:
        return

    goal = str(data.get("goal", "")).strip()
    initial_queries = [str(q) for q in data.get("initial_queries", []) if q]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _agent_executor,
        state.agent.set_initial_context,
        goal,
        initial_queries,
    )


@sio.on("agent_toggle")
async def handle_agent_toggle(sid, data):
    state = _ensure_session(sid)
    if state.agent is None:
        await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)
        return

    enabled = bool(data.get("enabled", True))
    state.agent.enabled = enabled
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("get_agent_state")
async def handle_get_agent_state(sid, data=None):
    _ensure_session(sid)
    await sio.emit("agent_state", _build_agent_state_payload(sid), to=sid)


@sio.on("agent_ui_result")
async def handle_agent_ui_result(sid, data):
    if not isinstance(data, dict):
        return
    cmd_id = str(data.get("id", "")).strip()
    status = str(data.get("status", "")).strip().lower()
    if not cmd_id or status not in {"ok", "error", "ignored", "timeout"}:
        return

    state = _ensure_session(sid)
    result = {
        "id": cmd_id,
        "status": status,
        "result": data.get("result"),
        "error": data.get("error"),
        "timestamp": time.time(),
    }
    with _sessions_lock:
        state.ui_results.append(result)
        if len(state.ui_results) > 128:
            state.ui_results = state.ui_results[-128:]


# ------------------------------
# Background Streaming Task
# ------------------------------
async def _broadcast_slam_update(result: dict[str, Any]):
    with _sids_lock:
        target_sids = list(_connected_sids)

    detections = result.get("detections", [])

    for sid in target_sids:
        active_queries = _session_active_queries(sid)
        payload = dict(result)
        payload["active_queries"] = active_queries
        payload["detections"] = _filter_detections_by_queries(detections, active_queries)
        await sio.emit("slam_update", payload, to=sid)


def _resolve_agent_submap_id(result: dict[str, Any]) -> Optional[int]:
    if slam_processor is None:
        return None

    graph_map = slam_processor.solver.map
    raw_submap_id = result.get("submap_id")
    if raw_submap_id is not None:
        try:
            submap_id = int(raw_submap_id)
            graph_map.get_submap(submap_id)
            return submap_id
        except Exception:
            pass

    candidates: list[Optional[int]] = []
    try:
        candidates.append(graph_map.get_largest_key(ignore_loop_closure_submaps=True))
    except Exception:
        candidates.append(None)
    try:
        candidates.append(graph_map.get_largest_key())
    except Exception:
        candidates.append(None)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            graph_map.get_submap(candidate)
            return int(candidate)
        except Exception:
            continue

    return None


def _schedule_agent_cycles_for_result(result: dict[str, Any]):
    with _sessions_lock:
        states = list(_sessions.values())

    if not states:
        return

    submap_id = _resolve_agent_submap_id(result)
    if submap_id is None:
        return

    detections_all = result.get("detections", [])
    loop = asyncio.get_running_loop()

    for state in states:
        if state.agent is None or not state.agent.enabled:
            continue
        active = sorted(state.manual_queries | state.agent_queries)
        filtered = _filter_detections_by_queries(detections_all, active)
        loop.run_in_executor(_agent_executor, state.agent.on_submap_processed, submap_id, filtered)


async def stream_results():
    """Stream results to connected clients."""
    while True:
        try:
            if client_connected.is_set():
                try:
                    result = result_queue.get_nowait()
                    await _broadcast_slam_update(result)
                    _schedule_agent_cycles_for_result(result)
                    print(
                        f"Sent update: {result['n_points']} points, "
                        f"{result['n_cameras']} cameras, "
                        f"{result['num_submaps']} submaps"
                    )
                except queue.Empty:
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Stream emit error: {e}")
            await asyncio.sleep(0.1)


# ------------------------------
# Server Startup
# ------------------------------
def initialize(
    submap_size=8,
    min_disparity=30.0,
    conf_threshold=25.0,
    vis_stride=4,
    serve_static_dir=None,
):
    """Initialize SLAM processor and static routes."""
    global slam_processor, _openrouter_api_key

    if serve_static_dir:
        from flask import send_from_directory

        @app.route("/")
        def serve_index():
            return send_from_directory(serve_static_dir, "index.html")

        @app.route("/<path:path>")
        def serve_static(path):
            return send_from_directory(serve_static_dir, path)

    slam_processor = StreamingSLAM(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
    )
    slam_processor.frame_queue = frame_queue
    slam_processor.result_queue = result_queue

    # Session-scoped agent architecture keeps this None to avoid global per-submap callback.
    slam_processor.spatial_agent = None

    _openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if _openrouter_api_key:
        print("Spatial Agent runtime enabled (OPENROUTER_API_KEY found)")
    else:
        print("Spatial Agent runtime disabled (no OPENROUTER_API_KEY)")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print("VGGT-SLAM 2.0 Streaming Server — initialized")
    print(f"GPU: {gpu}  |  submap_size={submap_size}  |  vis_stride={vis_stride}")
    if serve_static_dir:
        print(f"Serving frontend from: {serve_static_dir}")
    print("=" * 60)


def start_server(
    port=5000,
    submap_size=8,
    min_disparity=30.0,
    conf_threshold=25.0,
    vis_stride=4,
    video=None,
    fast=False,
    video_fps=2.0,
    serve_static_dir=None,
):
    """Start the streaming SLAM server locally using uvicorn."""
    initialize(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
        serve_static_dir=serve_static_dir,
    )

    video_feeder = None
    if video:
        video_feeder = VideoFeeder(video, fast=fast, target_fps=video_fps)
        video_feeder.start()

    ssl_certfile = None
    ssl_keyfile = None
    if not serve_static_dir:
        cert_path = os.path.join(os.path.dirname(__file__), "webserver", "server.cert")
        key_path = os.path.join(os.path.dirname(__file__), "webserver", "server.key")
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_certfile = cert_path
            ssl_keyfile = key_path
            print(f"SSL enabled: {cert_path}")
        else:
            print(
                "Warning: SSL certs not found. HTTPS disabled. "
                "Phone camera streaming requires HTTPS."
            )

    print("=" * 60)
    print("VGGT-SLAM 2.0 Streaming Server")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if video:
        print(f"Video input: {video} ({video_fps} fps, {'fast' if fast else 'real-time'})")
    else:
        print("Input: live WebSocket feed")
    print(f"Submap size: {submap_size}")
    print(f"Temp directory: {slam_processor.temp_dir}")
    if serve_static_dir:
        print(f"Serving frontend from: {serve_static_dir}")
    proto = "https" if ssl_certfile else "http"
    print(f"Server: {proto}://0.0.0.0:{port}")
    print("=" * 60)

    import uvicorn
    try:
        import uvloop
        loop = 'uvloop'
    except ImportError:
        loop = 'asyncio'

    uvicorn.run(
        asgi_application,
        host="0.0.0.0",
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        loop=loop,
    )


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT-SLAM 2.0 Streaming Server")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file for offline testing")
    parser.add_argument("--fast", action="store_true", help="Feed video frames as fast as possible")
    parser.add_argument("--video-fps", type=float, default=2.0, help="Effective FPS to extract from video")
    parser.add_argument("--submap-size", type=int, default=8, help="Frames per submap")
    parser.add_argument(
        "--min-disparity", type=float, default=30.0, help="Minimum disparity for keyframe selection"
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=25.0, help="Confidence threshold percentage"
    )
    parser.add_argument("--vis-stride", type=int, default=4, help="Visualization stride")
    args = parser.parse_args()

    if args.video and not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}")
        raise SystemExit(1)

    start_server(
        port=args.port,
        submap_size=args.submap_size,
        min_disparity=args.min_disparity,
        conf_threshold=args.conf_threshold,
        vis_stride=args.vis_stride,
        video=args.video,
        fast=args.fast,
        video_fps=args.video_fps,
    )
