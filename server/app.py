"""
Flask + python-socketio ASGI streaming server for VGGT-SLAM 2.0.

Handles:
  - WebSocket frame streaming from phone/browser
  - SLAM update broadcasting to viewer clients
  - Object detection queries (CLIP + SAM3)
  - Beacon placement and resolution
  - Video file testing mode
"""

import os
import ssl
import queue
import threading
import time
import base64
import argparse
import asyncio

import cv2
import numpy as np
import torch
import json
import re

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import socketio as socketio_pkg
from asgiref.wsgi import WsgiToAsgi

from server.streaming_slam import StreamingSLAM
from vggt_slam.object_detector import ObjectDetector

# ------------------------------
# Flask + python-socketio Setup
# ------------------------------
app = Flask(__name__)
CORS(app)

sio = socketio_pkg.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    max_http_buffer_size=10_000_000,
    ping_timeout=120,
    ping_interval=25,
)
asgi_application = socketio_pkg.ASGIApp(sio, WsgiToAsgi(app))

# Queues for streaming pipeline
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=10)

# Global SLAM processor (initialized in initialize() or start_server())
slam_processor = None
client_connected = threading.Event()

# Track connected socket IDs so we only stop SLAM when the last client leaves
_connected_sids: set = set()
_sids_lock = threading.Lock()

# Background streaming task handle — started once when the first client connects
_stream_task = None

# The running asyncio event loop, captured when the first async handler fires.
# Used by sync Flask routes to schedule coroutines thread-safely.
_event_loop: asyncio.AbstractEventLoop | None = None

# Single-threaded executor for blocking GPU ops (segment_all) so they don't
# freeze the asyncio event loop while running SAM3 inference
import concurrent.futures
_gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


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
        print(f"VideoFeeder started: {self.video_path} "
              f"(target_fps={self.target_fps}, fast={self.fast})")

    def stop(self):
        self._stop_event.set()

    def _feed_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        skip = max(1, int(round(video_fps / self.target_fps)))
        effective_fps = video_fps / skip
        delay = 0.0 if self.fast else 1.0 / effective_fps

        frames_to_feed = total_frames // skip
        print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS")
        print(f"  Feeding every {skip} frame(s) -> ~{effective_fps:.1f} effective FPS "
              f"(~{frames_to_feed} frames, delay={delay*1000:.0f}ms)")

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

            ok, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
            data = {'image': b64, 'timestamp': time.time()}

            try:
                frame_queue.put(data, timeout=10)
            except queue.Full:
                print(f"frame_queue full, dropping frame {fed_count}")
                continue

            # Auto-start SLAM on first frame
            if fed_count == 0 and slam_processor is not None:
                if not slam_processor.is_running:
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


# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'gpu': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Soft reset: clear SLAM data, keep models loaded."""
    # Clear queues
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

    slam_processor.soft_reset()
    # Schedule the broadcast on the ASGI event loop (this route runs in a thread)
    if _event_loop is not None:
        asyncio.run_coroutine_threadsafe(
            sio.emit('slam_reset', {'status': 'reset'}),
            _event_loop,
        )

    return jsonify({
        'status': 'reset_complete',
        'message': 'SLAM data cleared, model still loaded',
    })


@app.route('/api/plan', methods=['POST'])
def generate_plan():
    """Generate a tracking plan from a natural language prompt using Claude."""
    data = request.get_json() or {}
    prompt = data.get('prompt', '')

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f'Given this scenario: "{prompt}"\n'
            'Return JSON with this exact format:\n'
            '{"objects": ["obj1", "obj2"], '
            '"waypoints_justification": "1-2 sentences", '
            '"pathfinding_justification": "1-2 sentences"}\n'
            'Objects should be concrete, visible, physical items trackable in 3D space. '
            'Return only the JSON, no extra text.'
        )
        content = response.text
        match = re.search(r'\{[\s\S]*\}', content)
        result = json.loads(match.group())
        return jsonify({
            'objects': result.get('objects', []),
            'waypoints': {
                'enabled': True,
                'justification': result.get('waypoints_justification', 'Waypoints mark key locations.'),
            },
            'pathfinding': {
                'enabled': True,
                'justification': result.get('pathfinding_justification', 'Pathfinding visualizes your traversed route.'),
            },
        })
    except Exception as e:
        print(f'Plan generation error: {e}; falling back to keyword extraction')
        stopwords = {
            'i', 'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'and',
            'or', 'my', 'me', 'we', 'is', 'are', 'was', 'want', 'need',
            'track', 'find', 'locate', 'using', 'with', 'this', 'that',
        }
        words = [w.strip('.,!?') for w in prompt.lower().split()]
        objects = list(dict.fromkeys([w for w in words if w and w not in stopwords]))[:5]
        return jsonify({
            'objects': objects or ['object'],
            'waypoints': {'enabled': True, 'justification': 'Waypoints help mark key locations.'},
            'pathfinding': {'enabled': True, 'justification': 'Pathfinding visualizes your traversed route.'},
        })


# ------------------------------
# SocketIO Events
# ------------------------------
@sio.on('connect')
async def handle_connect(sid, environ, auth):
    global _stream_task, _event_loop
    if slam_processor is None:
        return
    # Capture the running event loop so sync Flask routes can schedule coroutines
    _event_loop = asyncio.get_event_loop()
    with _sids_lock:
        _connected_sids.add(sid)
    print(f"Client connected ({len(_connected_sids)} total)")
    client_connected.set()
    # Start the result-streaming loop once (it runs forever; survives reconnects)
    if _stream_task is None or _stream_task.done():
        _stream_task = asyncio.ensure_future(stream_results())
    await sio.emit('connected', {'status': 'ready'}, to=sid)


@sio.on('disconnect')
async def handle_disconnect(sid):
    if slam_processor is None:
        return
    with _sids_lock:
        _connected_sids.discard(sid)
        remaining = len(_connected_sids)
    print(f"Client disconnected ({remaining} remaining)")
    # Only stop SLAM and clear queues when the LAST client leaves
    if remaining == 0:
        client_connected.clear()
        slam_processor.stop()
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


@sio.on('frame')
async def handle_frame(sid, data):
    if slam_processor is None:
        return
    if not frame_queue.full():
        frame_queue.put(data)
    # Auto-start SLAM on first frame
    if not slam_processor.is_running:
        print("Auto-starting SLAM processing...")
        slam_processor.start()


@sio.on('stop_slam')
async def handle_stop(sid, data=None):
    if slam_processor is None:
        return
    slam_processor.stop()
    await sio.emit('slam_stopped', {'status': 'stopped'}, to=sid)


@sio.on('set_detection_queries')
async def handle_set_detection_queries(sid, data):
    if slam_processor is None:
        return
    """Set active object detection queries — streams partial results progressively."""
    queries = data.get('queries', [])
    print(f"Detection queries received: {queries}")

    loop = asyncio.get_event_loop()
    partial_q: asyncio.Queue = asyncio.Queue()

    def run_gen():
        try:
            for partial in slam_processor.run_detection_progressive(queries):
                loop.call_soon_threadsafe(partial_q.put_nowait, partial)
        except Exception as e:
            print(f"Progressive detection error: {e}")
            loop.call_soon_threadsafe(
                partial_q.put_nowait,
                {'detections': [], 'is_final': True, 'error': str(e)},
            )

    _gpu_executor.submit(run_gen)

    while True:
        partial = await partial_q.get()
        await sio.emit('detection_partial', {
            'detections': partial['detections'],
            'active_queries': list(slam_processor.active_queries),
            'is_final': partial['is_final'],
        }, to=sid)
        if partial['is_final']:
            break


@sio.on('get_detection_preview')
async def handle_get_detection_preview(sid, data):
    if slam_processor is None:
        return
    """Generate and return keyframe + SAM3 mask preview."""
    submap_id = data.get('submap_id')
    frame_idx = data.get('frame_idx')
    query = data.get('query', '')

    try:
        submap = slam_processor.solver.map.get_submap(submap_id)
        if submap is None:
            await sio.emit('detection_preview', {'error': f'Submap {submap_id} not found'}, to=sid)
            return

        frame_tensor = submap.get_frame_at_index(frame_idx)
        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)

        keyframe_image = ObjectDetector.image_to_base64(frame_np)

        mask_image = None
        if query:
            # Run SAM3 inference in a thread-pool executor so it doesn't
            # freeze the asyncio event loop (which would block slam_update too)
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

        await sio.emit('detection_preview', {
            'query': query,
            'submap_id': submap_id,
            'frame_idx': frame_idx,
            'keyframe_image': keyframe_image,
            'mask_image': mask_image,
        }, to=sid)

    except Exception as e:
        print(f"Error generating preview: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('detection_preview', {'error': str(e)}, to=sid)


@sio.on('place_beacon')
async def handle_place_beacon(sid, data):
    if slam_processor is None:
        return
    """Queue a beacon request."""
    beacon_id = data.get('beacon_id')
    frame_number = data.get('frame_number', 0)
    slam_processor.pending_beacons.append({
        'beacon_id': beacon_id,
        'frame_number': frame_number,
    })
    print(f"Beacon {beacon_id} queued at frame {frame_number}")
    await sio.emit('beacon_queued', {'beacon_id': beacon_id}, to=sid)


@sio.on('clear_beacons')
async def handle_clear_beacons(sid, data=None):
    if slam_processor is None:
        return
    """Clear all pending and resolved beacons."""
    slam_processor.pending_beacons.clear()
    slam_processor.resolved_beacons.clear()
    print("All beacons cleared")


@sio.on('debug_detect')
async def handle_debug_detect(sid, data):
    if slam_processor is None:
        await sio.emit('debug_detect_results', {'error': 'SLAM not initialized'}, to=sid)
        return
    """Run full detection pipeline and stream back rich per-frame diagnostics."""
    queries = data.get('queries', [])
    clip_thresholds = data.get('clip_thresholds', {})
    sam_thresholds = data.get('sam_thresholds', {})
    top_k = data.get('top_k', None)

    if not queries:
        await sio.emit('debug_detect_results', {'error': 'No queries provided'}, to=sid)
        return

    print(f"Debug detect: {queries} (CLIP={clip_thresholds}, SAM={sam_thresholds}, top_k={top_k})")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _gpu_executor,
            lambda: slam_processor.debug_detect_full(
                queries, clip_thresholds, sam_thresholds, top_k
            )
        )
        await sio.emit('debug_detect_results', result, to=sid)
    except Exception as e:
        print(f"Debug detect error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('debug_detect_results', {'error': str(e)}, to=sid)


@sio.on('get_global_map')
async def handle_get_global_map(sid, data=None):
    if slam_processor is None:
        return
    """Return the current global map state."""
    print("Client requested global map")
    try:
        if slam_processor.solver.map.get_num_submaps() > 0:
            # Use cached full payload if available, otherwise re-extract
            if slam_processor._last_stream_data and slam_processor._last_stream_data.get('type') == 'full':
                stream_data = dict(slam_processor._last_stream_data)
            else:
                stream_data = slam_processor.extract_stream_data_full()
            with slam_processor._detection_lock:
                stream_data['detections'] = list(slam_processor.accumulated_detections)
                stream_data['active_queries'] = list(slam_processor.active_queries)

            if stream_data and stream_data['n_points'] > 0:
                print(f"Sending global map: {stream_data['n_points']} points, "
                      f"{stream_data['n_cameras']} cameras")
                await sio.emit('global_map', stream_data, to=sid)
            else:
                await sio.emit('global_map', slam_processor._empty_data(), to=sid)
        else:
            await sio.emit('global_map', slam_processor._empty_data(), to=sid)
    except Exception as e:
        print(f"Error fetching global map: {e}")
        import traceback
        traceback.print_exc()


# ------------------------------
# Background Streaming Task
# ------------------------------
async def stream_results():
    """Stream results to connected clients."""
    while True:
        try:
            if client_connected.is_set():
                try:
                    result = result_queue.get_nowait()
                    await sio.emit('slam_update', result)
                    print(f"Sent update: {result['n_points']} points, "
                          f"{result['n_cameras']} cameras, "
                          f"{result['num_submaps']} submaps")
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
    """Initialize SLAM processor and static routes.

    Called from Modal's ASGI entrypoint or from start_server() for local dev.
    Does NOT start a server — the ASGI framework handles all serving.
    """
    global slam_processor

    if serve_static_dir:
        from flask import send_from_directory

        @app.route('/')
        def serve_index():
            return send_from_directory(serve_static_dir, 'index.html')

        @app.route('/<path:path>')
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
    # Note: stream_results() is started lazily on first client connect (handle_connect)
    # because asyncio.ensure_future() requires a running event loop.

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
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
    """Start the streaming SLAM server locally using uvicorn.

    Args:
        serve_static_dir: If set, serve built frontend files from this directory.
            When None, the frontend is expected to run separately (e.g. via ``npm run dev``).
    """
    initialize(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
        serve_static_dir=serve_static_dir,
    )

    # Start video feeder if provided
    video_feeder = None
    if video:
        video_feeder = VideoFeeder(video, fast=fast, target_fps=video_fps)
        video_feeder.start()

    # SSL for HTTPS (required for phone camera access)
    ssl_certfile = None
    ssl_keyfile = None
    if not serve_static_dir:
        cert_path = os.path.join(os.path.dirname(__file__), 'webserver', 'server.cert')
        key_path = os.path.join(os.path.dirname(__file__), 'webserver', 'server.key')
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_certfile = cert_path
            ssl_keyfile = key_path
            print(f"SSL enabled: {cert_path}")
        else:
            print("Warning: SSL certs not found. HTTPS disabled. "
                  "Phone camera streaming requires HTTPS.")

    print("=" * 60)
    print("VGGT-SLAM 2.0 Streaming Server")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if video:
        print(f"Video input: {video} ({video_fps} fps, "
              f"{'fast' if fast else 'real-time'})")
    else:
        print("Input: live WebSocket feed")
    print(f"Submap size: {submap_size}")
    print(f"Temp directory: {slam_processor.temp_dir}")
    if serve_static_dir:
        print(f"Serving frontend from: {serve_static_dir}")
    proto = 'https' if ssl_certfile else 'http'
    print(f"Server: {proto}://0.0.0.0:{port}")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        asgi_application,
        host='0.0.0.0',
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGT-SLAM 2.0 Streaming Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a video file for offline testing')
    parser.add_argument('--fast', action='store_true',
                        help='Feed video frames as fast as possible (no FPS throttle)')
    parser.add_argument('--video-fps', type=float, default=2.0,
                        help='Effective FPS to extract from video (default: 2)')
    parser.add_argument('--submap-size', type=int, default=8,
                        help='Frames per submap (default: 8)')
    parser.add_argument('--min-disparity', type=float, default=30.0,
                        help='Minimum disparity for keyframe selection (default: 30)')
    parser.add_argument('--conf-threshold', type=float, default=25.0,
                        help='Confidence threshold percentage (default: 25)')
    parser.add_argument('--vis-stride', type=int, default=4,
                        help='Stride for point cloud visualization (default: 4)')
    args = parser.parse_args()

    # Validate video path
    if args.video and not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}")
        exit(1)

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
