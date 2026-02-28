"""
Modal deployment for VGGT-SLAM streaming server.

Runs the Flask+SocketIO streaming server on a cloud GPU and provides a public
HTTPS URL. Users open the URL in a browser to stream camera frames and view
the live 3D SLAM map.

Usage:
    # Launch the streaming server on a remote A100
    modal run modal_streaming.py

    # With custom parameters
    modal run modal_streaming.py --submap-size 8 --min-disparity 30

    # Pre-download model weights (optional, happens automatically on first run)
    modal run modal_streaming.py::app.download_models
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
        # Visualization (unused in streaming mode but imported)
        "viser==0.2.23",
        "matplotlib",
        "gradio",
        # Streaming server
        "flask",
        "flask-socketio",
        "flask-cors",
        "google-generativeai",
        # Misc
        "termcolor",
        "tqdm",
        "omegaconf",
        "requests",
        "lz4",
        "ftfy",
        "regex",
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
        " && pip install -e /root/third_party/perception_models",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /root/third_party/sam3"
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
        "cd /root/project/server/webserver && npm install && npx vite build"
    )
)

# ---------------------------------------------------------------------------
# Persistent volume for model weights
# ---------------------------------------------------------------------------
model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)
CACHE_PATH = "/root/.cache/torch/hub"


# ---------------------------------------------------------------------------
# Model download (CPU-only)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("gemini-secret")],
    timeout=1800,
)
def download_models():
    """Download and cache all model weights. Idempotent."""
    import torch

    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # VGGT-1B
    vggt_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(vggt_path):
        print("Downloading VGGT-1B model weights...")
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            vggt_path,
        )
    else:
        print("VGGT-1B: already cached")

    # DINO-Salad
    salad_path = os.path.join(ckpt_dir, "dino_salad.ckpt")
    if not os.path.exists(salad_path):
        print("Downloading dino_salad checkpoint...")
        torch.hub.download_url_to_file(
            "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            salad_path,
        )
    else:
        print("dino_salad: already cached")

    # DINOv2 backbone
    dinov2_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.exists(dinov2_repo):
        print("Downloading DINOv2 backbone...")
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    else:
        print("DINOv2: already cached")

    # PE-Core (Perception Encoder) — downloaded automatically by ObjectDetector
    # but we trigger it here to pre-cache
    try:
        from huggingface_hub import hf_hub_download
        pe_path = os.path.join(ckpt_dir, "pe_core_l14_336.pt")
        if not os.path.exists(pe_path):
            print("Downloading PE-Core CLIP model...")
            hf_hub_download(
                repo_id="facebook/PE-Core-L14-336",
                filename="open_clip_pytorch_model.bin",
                local_dir=ckpt_dir,
            )
        else:
            print("PE-Core: already cached")
    except Exception as e:
        print(f"PE-Core download skipped: {e}")

    model_cache.commit()
    print("All model weights cached.")


# ---------------------------------------------------------------------------
# Streaming server (runs on GPU)
# ---------------------------------------------------------------------------
SERVER_PORT = 5000

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("gemini-secret")],
    timeout=7200,
    min_containers=1,
)
@modal.web_server(port=SERVER_PORT, startup_timeout=600)
def run_streaming_server():
    """Start the streaming SLAM server.

    Uses @modal.web_server for a stable URL that auto-wakes the container.
    Configure via environment variables (set in Modal secrets or dashboard):
        SUBMAP_SIZE, MIN_DISPARITY, CONF_THRESHOLD, VIS_STRIDE
    """
    import sys
    import threading
    sys.path.insert(0, "/root/project")

    from server.app import start_server

    submap_size = int(os.environ.get("SUBMAP_SIZE", "8"))
    min_disparity = float(os.environ.get("MIN_DISPARITY", "30.0"))
    conf_threshold = float(os.environ.get("CONF_THRESHOLD", "25.0"))
    vis_stride = int(os.environ.get("VIS_STRIDE", "4"))

    # start_server() blocks (socketio.run), so run it in a daemon thread.
    # @modal.web_server expects this function to return after starting the server.
    threading.Thread(
        target=start_server,
        kwargs=dict(
            port=SERVER_PORT,
            submap_size=submap_size,
            min_disparity=min_disparity,
            conf_threshold=conf_threshold,
            vis_stride=vis_stride,
            serve_static_dir="/root/project/server/webserver/dist",
        ),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Local entrypoint — pre-cache models only
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Download model weights. The server itself is started via `modal serve` or `modal deploy`."""
    print("Ensuring model weights are cached...")
    download_models.remote()
    print("\nModel weights cached. To start the streaming server:")
    print("  Development:  modal serve modal_streaming.py")
    print("  Production:   modal deploy modal_streaming.py")
    print("\nThe stable URL will be printed in the terminal (*.modal.run).")
