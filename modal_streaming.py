"""
Modal deployment for VGGT-SLAM streaming server.

Runs the ASGI streaming server on a cloud GPU with a stable public URL.
Uses @modal.asgi_app() with allow_concurrent_inputs so WebSocket and HTTP
connections are handled concurrently (no blocking).

Usage:
    modal deploy modal_streaming.py

    # Pre-download model weights (optional)
    modal run modal_streaming.py::download_models
"""

import modal
import os

app = modal.App("vggt-slam-streaming")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "git",
        "cmake",
        "build-essential",
        "curl",
    )
    # Install Node.js for frontend build
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
        " && apt-get install -y nodejs"
    )
    .pip_install(
        # Core ML
        "torch==2.3.1",
        "torchvision==0.18.1",
        # Vision / 3D
        "numpy",
        "Pillow",
        "open3d",
        "opencv-python",
        "trimesh",
        # Model loading
        "huggingface_hub",
        "einops",
        "safetensors",
        # Optimization / SLAM
        "gtsam-develop",
        "scipy",
        # Training utilities (transitive deps)
        "pytorch_metric_learning",
        "pytorch-lightning",
        # Visualization (unused in streaming mode but imported by solver)
        "viser==0.2.23",
        "matplotlib",
        "gradio",
        # Streaming server — ASGI for concurrent WebSocket + HTTP
        "flask",
        "flask-cors",
        "python-socketio",
        "asgiref",
        "uvicorn",
        "google-generativeai",
        "openai",
        "anthropic",
        # Misc
        "termcolor",
        "tqdm",
        "omegaconf",
        "requests",
        "lz4",
        "ftfy",
        "regex",
        "uvloop",
    )
    # Third-party repos
    .run_commands(
        "git clone https://github.com/Dominic101/salad.git /root/third_party/salad"
        " && pip install -e /root/third_party/salad",
    )
    .run_commands(
        "git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /root/third_party/vggt"
        " && pip install -e /root/third_party/vggt",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/perception_models.git"
        " /root/third_party/perception_models"
        " && pip install -e /root/third_party/perception_models --no-deps",
    )
    .run_commands(
        # sam3's training-data import chain pulls in decord + pycocotools which we
        # don't need for inference. Stub out decord (no Python 3.11 wheel exists),
        # install pycocotools, then install sam3.
        "python -c \""
        "import site, os; p=site.getsitepackages()[0]+'/decord';"
        "os.makedirs(p,exist_ok=True);"
        "open(p+'/__init__.py','w').write('cpu=None\\nVideoReader=None')"
        "\""
        " && pip install pycocotools"
        " && git clone https://github.com/facebookresearch/sam3.git /root/third_party/sam3"
        " && pip install -e /root/third_party/sam3",
    )
    # VGGT-SLAM package
    .add_local_dir("vggt_slam", remote_path="/root/project/vggt_slam", copy=True)
    .add_local_file("setup.py", remote_path="/root/project/setup.py", copy=True)
    .run_commands("cd /root/project && pip install -e .")
    # Server code
    .add_local_dir("server", remote_path="/root/project/server", copy=True)
    # Build frontend inside the image
    .run_commands(
        "cd /root/project/server/webserver && npm install && node node_modules/vite/bin/vite.js build"
    )
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)
CACHE_PATH = "/root/.cache/torch/hub"

demo_video_vol = modal.Volume.from_name("vggt-slam-demo-videos", create_if_missing=True)
DEMO_VIDEO_PATH = "/root/demo_videos"


# ---------------------------------------------------------------------------
# Model download helper (CPU-only)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={CACHE_PATH: model_cache},
    timeout=1800,
)
def download_models():
    """Download and cache all model weights. Idempotent."""
    import torch

    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    vggt_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(vggt_path):
        print("Downloading VGGT-1B...")
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            vggt_path,
        )
    else:
        print("VGGT-1B: already cached")

    salad_path = os.path.join(ckpt_dir, "dino_salad.ckpt")
    if not os.path.exists(salad_path):
        print("Downloading dino_salad...")
        torch.hub.download_url_to_file(
            "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            salad_path,
        )
    else:
        print("dino_salad: already cached")

    dinov2_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.exists(dinov2_repo):
        print("Downloading DINOv2 backbone...")
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    else:
        print("DINOv2: already cached")

    model_cache.commit()
    print("All model weights cached.")


# ---------------------------------------------------------------------------
# Streaming server — ASGI with concurrent WebSocket + HTTP
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_PATH: model_cache, DEMO_VIDEO_PATH: demo_video_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("gemini-secret"),
        modal.Secret.from_name("openrouter-secret"),
    ],
    timeout=86400,
    min_containers=1,
    max_containers=1,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def web():
    import sys
    import asyncio
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    sys.path.insert(0, "/root/project")

    os.environ.setdefault("DEMO_VIDEO_DIR", DEMO_VIDEO_PATH)

    from server.app import asgi_application, initialize

    submap_size = int(os.environ.get("SUBMAP_SIZE", "8"))
    min_disparity = float(os.environ.get("MIN_DISPARITY", "30.0"))
    conf_threshold = float(os.environ.get("CONF_THRESHOLD", "25.0"))
    vis_stride = int(os.environ.get("VIS_STRIDE", "4"))
    frontend_dist = "/root/project/server/webserver/dist"

    initialize(
        submap_size=submap_size,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        vis_stride=vis_stride,
        serve_static_dir=frontend_dist,
    )

    return asgi_application


# ---------------------------------------------------------------------------
# Upload demo videos to Modal volume
# ---------------------------------------------------------------------------
_DEMO_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}


@app.function(
    image=modal.Image.debian_slim(),
    volumes={DEMO_VIDEO_PATH: demo_video_vol},
    timeout=1200,
)
def _save_demo_video(filename: str, data: bytes):
    dest = os.path.join(DEMO_VIDEO_PATH, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    demo_video_vol.commit()
    print(f"  Saved {filename} ({len(data) / 1024 / 1024:.1f} MB)")


@app.local_entrypoint()
def upload_demo_videos():
    """Upload local demo videos to Modal volume (all in parallel).

    Videos are Git LFS tracked, so run `git lfs pull` first to materialise
    actual video content (otherwise only the tiny pointer files exist locally).

    Usage:
        git lfs pull
        modal run modal_streaming.py
    """
    demo_dir = os.path.join(os.path.dirname(__file__), "server", "demo_videos")
    if not os.path.isdir(demo_dir):
        print(f"Demo video directory not found: {demo_dir}")
        return

    videos = sorted(
        f
        for f in os.listdir(demo_dir)
        if os.path.splitext(f)[1].lower() in _DEMO_VIDEO_EXTENSIONS
    )
    if not videos:
        print("No video files found in server/demo_videos/")
        return

    to_upload = []
    skipped = 0
    for filename in videos:
        path = os.path.join(demo_dir, filename)
        size = os.path.getsize(path)
        if size < 1024:
            with open(path, "r") as f:
                head = f.read(50)
            if "git-lfs" in head:
                print(f"  SKIP {filename} (Git LFS pointer — run `git lfs pull` first)")
                skipped += 1
                continue

        print(f"  Queuing {filename} ({size / 1024 / 1024:.1f} MB) ...")
        with open(path, "rb") as f:
            data = f.read()
        to_upload.append((filename, data))

    if to_upload:
        print(f"\nUploading {len(to_upload)} videos in parallel...")
        for result in _save_demo_video.starmap(to_upload):
            pass

    print(f"\nDone: {len(to_upload)} uploaded, {skipped} skipped (LFS pointers).")
    if skipped:
        print("Run `git lfs pull` to download actual video files, then re-run this.")
