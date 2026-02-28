"""
Flask + SocketIO streaming server for VGGT-SLAM 2.0.

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

import cv2
import numpy as np
import torch
import json
import re

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from PIL import Image

from server.streaming_slam import StreamingSLAM
from vggt_slam.object_detector import ObjectDetector

# ------------------------------
# Flask + SocketIO Setup
# ------------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=10_000_000,
    ping_timeout=120,
    ping_interval=25,
)

# Queues for streaming pipeline
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=10)

# Global SLAM processor (initialized in __main__)
slam_processor = None
client_connected = threading.Event()


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
    socketio.emit('slam_reset', {'status': 'reset'})

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
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    client_connected.set()
    emit('connected', {'status': 'ready'})


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
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


@socketio.on('frame')
def handle_frame(data):
    if not frame_queue.full():
        frame_queue.put(data)
    # Auto-start SLAM on first frame
    if not slam_processor.is_running:
        print("Auto-starting SLAM processing...")
        slam_processor.start()


@socketio.on('stop_slam')
def handle_stop():
    slam_processor.stop()
    emit('slam_stopped', {'status': 'stopped'})


@socketio.on('set_detection_queries')
def handle_set_detection_queries(data):
    """Set active object detection queries."""
    queries = data.get('queries', [])
    print(f"Detection queries received: {queries}")
    slam_processor.set_detection_queries(queries)

    # If we already have submaps, send an immediate update
    if slam_processor.solver.map.get_num_submaps() > 0:
        try:
            stream_data = slam_processor.extract_stream_data()
            with slam_processor._detection_lock:
                stream_data['detections'] = list(slam_processor.accumulated_detections)
                stream_data['active_queries'] = list(slam_processor.active_queries)
            emit('slam_update', stream_data)
        except Exception as e:
            print(f"  Error sending detection update: {e}")


@socketio.on('get_detection_preview')
def handle_get_detection_preview(data):
    """Generate and return keyframe + SAM3 mask preview."""
    submap_id = data.get('submap_id')
    frame_idx = data.get('frame_idx')
    query = data.get('query', '')

    try:
        submap = slam_processor.solver.map.get_submap(submap_id)
        if submap is None:
            emit('detection_preview', {'error': f'Submap {submap_id} not found'})
            return

        frame_tensor = submap.get_frame_at_index(frame_idx)
        frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)

        keyframe_image = ObjectDetector.image_to_base64(frame_np)

        mask_image = None
        if query:
            results = slam_processor.object_detector.segment_all(frame_pil, query)
            if results:
                best_mask, _, _ = max(results, key=lambda r: r[2])
                mask_image = ObjectDetector.mask_overlay_to_base64(frame_np, best_mask)

        emit('detection_preview', {
            'query': query,
            'submap_id': submap_id,
            'frame_idx': frame_idx,
            'keyframe_image': keyframe_image,
            'mask_image': mask_image,
        })

    except Exception as e:
        print(f"Error generating preview: {e}")
        import traceback
        traceback.print_exc()
        emit('detection_preview', {'error': str(e)})


@socketio.on('place_beacon')
def handle_place_beacon(data):
    """Queue a beacon request."""
    beacon_id = data.get('beacon_id')
    frame_number = data.get('frame_number', 0)
    slam_processor.pending_beacons.append({
        'beacon_id': beacon_id,
        'frame_number': frame_number,
    })
    print(f"Beacon {beacon_id} queued at frame {frame_number}")
    emit('beacon_queued', {'beacon_id': beacon_id})


@socketio.on('clear_beacons')
def handle_clear_beacons():
    """Clear all pending and resolved beacons."""
    slam_processor.pending_beacons.clear()
    slam_processor.resolved_beacons.clear()
    print("All beacons cleared")


@socketio.on('get_global_map')
def handle_get_global_map():
    """Return the current global map state."""
    print("Client requested global map")
    try:
        if slam_processor.solver.map.get_num_submaps() > 0:
            stream_data = slam_processor.extract_stream_data()
            with slam_processor._detection_lock:
                stream_data['detections'] = list(slam_processor.accumulated_detections)
                stream_data['active_queries'] = list(slam_processor.active_queries)

            if stream_data and stream_data['n_points'] > 0:
                print(f"Sending global map: {stream_data['n_points']} points, "
                      f"{stream_data['n_cameras']} cameras")
                emit('global_map', stream_data)
            else:
                emit('global_map', slam_processor._empty_data())
        else:
            emit('global_map', slam_processor._empty_data())
    except Exception as e:
        print(f"Error fetching global map: {e}")
        import traceback
        traceback.print_exc()


# ------------------------------
# Background Streaming Thread
# ------------------------------
def stream_results():
    """Stream results to connected clients."""
    while True:
        try:
            if client_connected.is_set():
                result = result_queue.get(timeout=0.5)
                socketio.emit('slam_update', result)
                print(f"Sent update: {result['n_points']} points, "
                      f"{result['n_cameras']} cameras, "
                      f"{result['num_submaps']} submaps")
            else:
                socketio.sleep(0.1)
        except queue.Empty:
            socketio.sleep(0.1)
        except Exception as e:
            print(f"Stream emit error: {e}")
            socketio.sleep(0.1)


threading.Thread(target=stream_results, daemon=True).start()


# ------------------------------
# Server Startup
# ------------------------------
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
    """Start the streaming SLAM server.

    Args:
        serve_static_dir: If set, serve built frontend files from this directory
            (used by Modal where there's no Vite dev server). When None, the
            frontend is expected to run separately (e.g. via ``npm run dev``).
    """
    global slam_processor

    # Serve built frontend if directory provided (Modal deployment)
    if serve_static_dir:
        from flask import send_from_directory

        @app.route('/')
        def serve_index():
            return send_from_directory(serve_static_dir, 'index.html')

        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory(serve_static_dir, path)

    # Initialize SLAM processor
    slam_processor = StreamingSLAM(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
    )
    slam_processor.frame_queue = frame_queue
    slam_processor.result_queue = result_queue

    # Start video feeder if provided
    video_feeder = None
    if video:
        video_feeder = VideoFeeder(video, fast=fast, target_fps=video_fps)
        video_feeder.start()

    # SSL context for HTTPS (required for phone camera access)
    # Skip when serving static (Modal tunnel provides HTTPS)
    ssl_context = None
    if not serve_static_dir:
        cert_path = os.path.join(os.path.dirname(__file__), 'webserver', 'server.cert')
        key_path = os.path.join(os.path.dirname(__file__), 'webserver', 'server.key')
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
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
    print(f"Server: {'https' if ssl_context else 'http'}://0.0.0.0:{port}")
    print("=" * 60)

    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context,
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
