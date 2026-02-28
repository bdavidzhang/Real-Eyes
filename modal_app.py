"""
Modal deployment script for VGGT-SLAM.

Usage:
    # Run the full pipeline (upload images → run SLAM on GPU → download results)
    modal run modal_app.py --image-folder ./office_loop --submap-size 16 --max-loops 1

    # Pre-download model weights (optional, happens automatically on first run)
    modal run modal_app.py::app.download_models
"""

import modal
import os

app = modal.App("vggt-slam")

# Queue used to send the live Viser URL from the remote container back to the local entrypoint
url_queue = modal.Queue.from_name("vggt-slam-url", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image: system deps + pip deps + third-party repos + vggt_slam
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "git",
        "cmake",
        "build-essential",
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
        # Visualization (Viewer starts but is unused in headless mode)
        "viser==0.2.23",
        "matplotlib",
        "gradio",
        # Misc
        "termcolor",
        "tqdm",
        "omegaconf",
        "requests",
        "lz4",
        "ftfy",
        "regex",
    )
    # Third-party repos (same as setup.sh)
    .run_commands(
        "git clone https://github.com/Dominic101/salad.git /root/third_party/salad"
        " && pip install -e /root/third_party/salad",
    )
    .run_commands(
        "git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /root/third_party/vggt"
        " && pip install -e /root/third_party/vggt",
    )
    # VGGT-SLAM package (copy=True needed because run_commands follows)
    .add_local_dir("vggt_slam", remote_path="/root/project/vggt_slam", copy=True)
    .add_local_file("setup.py", remote_path="/root/project/setup.py", copy=True)
    .run_commands("cd /root/project && pip install -e .")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
# Model weights cache (VGGT-1B, dino_salad, DINOv2) — persists across runs
model_cache = modal.Volume.from_name("vggt-slam-models", create_if_missing=True)
# Data I/O — input images uploaded here, results saved here
data_vol = modal.Volume.from_name("vggt-slam-data", create_if_missing=True)

CACHE_PATH = "/root/.cache/torch/hub"
DATA_PATH = "/root/data"


# ---------------------------------------------------------------------------
# Model download (CPU-only, no GPU needed)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={CACHE_PATH: model_cache},
    timeout=1800,
)
def download_models():
    """Download and cache all model weights. Idempotent — skips existing files."""
    import torch

    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. VGGT-1B (~4GB)
    vggt_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(vggt_path):
        print("Downloading VGGT-1B model weights...")
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            vggt_path,
        )
    else:
        print("VGGT-1B: already cached")

    # 2. DINO-Salad checkpoint (~350MB)
    salad_path = os.path.join(ckpt_dir, "dino_salad.ckpt")
    if not os.path.exists(salad_path):
        print("Downloading dino_salad checkpoint...")
        torch.hub.download_url_to_file(
            "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            salad_path,
        )
    else:
        print("dino_salad: already cached")

    # 3. DINOv2 backbone (repo clone + weights, ~350MB)
    dinov2_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.exists(dinov2_repo):
        print("Downloading DINOv2 backbone...")
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    else:
        print("DINOv2: already cached")

    model_cache.commit()
    print("All model weights cached.")


# ---------------------------------------------------------------------------
# Remote SLAM pipeline (runs on GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        CACHE_PATH: model_cache,
        DATA_PATH: data_vol,
    },
    timeout=3600,
)
def run_slam(
    image_folder: str,
    submap_size: int = 16,
    max_loops: int = 1,
    min_disparity: float = 50.0,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    log_results: bool = True,
    skip_dense_log: bool = False,
) -> dict:
    """Run the full VGGT-SLAM pipeline. Mirrors main.py logic in headless mode."""
    import glob
    import time

    import cv2
    import torch
    from tqdm import tqdm

    import vggt_slam.slam_utils as utils
    from vggt_slam.solver import Solver
    from vggt.models.vggt import VGGT

    device = "cuda"

    # Expose Viser port and push the public URL to the local entrypoint via queue
    _tunnel_ctx = modal.forward(8080)
    tunnel = _tunnel_ctx.__enter__()
    url_queue.put(tunnel.url)

    # --- Initialise solver (Viewer starts on :8080) ---
    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
    )

    # --- Load VGGT model ---
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)

    # No open-set detection in headless mode
    clip_model, clip_preprocess = None, None
    
    # --- Load images ---
    abs_image_folder = os.path.join(DATA_PATH, image_folder)
    print(f"Loading images from {abs_image_folder}...")
    image_names = [
        f
        for f in glob.glob(os.path.join(abs_image_folder, "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, 1)
    print(f"Found {len(image_names)} images")

    if len(image_names) == 0:
        return {"error": f"No images found in {abs_image_folder}"}

    # --- Main processing loop (mirrors main.py) ---
    image_names_subset = []
    image_count = 0
    count = 0
    overlapping_window_size = 1
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()

    for image_name in tqdm(image_names):
        with keyframe_time:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(
                img, min_disparity, False
            )
            if enough_disparity:
                image_names_subset.append(image_name)
                image_count += 1

        if (
            len(image_names_subset) == submap_size + overlapping_window_size
            or image_name == image_names[-1]
        ):
            if len(image_names_subset) == 0:
                continue
            count += 1
            print(f"Processing submap {count} ({len(image_names_subset)} frames)...")

            predictions = solver.run_predictions(
                image_names_subset, model, max_loops, clip_model, clip_preprocess
            )
            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if loop_closure_detected:
                solver.update_all_submap_vis()
            else:
                solver.update_latest_submap_vis()

            image_names_subset = image_names_subset[-overlapping_window_size:]

    total_time = time.time() - total_time_start

    # --- Save results ---
    results_dir = os.path.join(DATA_PATH, "results", image_folder)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "poses.txt")

    if log_results:
        solver.map.write_poses_to_file(log_path, solver.graph, kitti_format=False)
        if not skip_dense_log:
            solver.map.save_framewise_pointclouds(
                solver.graph, log_path.replace(".txt", "_logs")
            )

    data_vol.commit()

    # --- Summary ---
    summary = {
        "frames_processed": image_count,
        "submaps": count,
        "loop_closures": solver.graph.get_num_loops(),
        "total_time_s": round(total_time, 2),
        "avg_fps": round(image_count / total_time, 2) if total_time > 0 else 0,
        "vggt_time_s": round(solver.vggt_timer.total_time, 2),
        "backend_time_s": round(backend_time.total_time, 2),
        "results_path": f"results/{image_folder}/",
    }
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint: upload → run → download
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    image_folder: str,
    submap_size: int = 16,
    max_loops: int = 1,
    min_disparity: float = 50.0,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    skip_dense_log: bool = False,
    output_dir: str = "./modal_results",
):
    from pathlib import Path

    local_folder = Path(image_folder)
    if not local_folder.is_dir():
        print(f"Error: {image_folder} is not a directory")
        return

    remote_name = local_folder.name

    # Step 1: Ensure model weights are cached (CPU-only, fast if already cached)
    print("Ensuring model weights are cached...")
    download_models.remote()

    # Step 2: Upload images to data volume
    print(f"Uploading images from {local_folder}...")
    image_files = sorted(
        f for f in local_folder.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )
    print(f"  {len(image_files)} files to upload")

    with data_vol.batch_upload(force=True) as batch:
        for img_file in image_files:
            batch.put_file(str(img_file), f"{remote_name}/{img_file.name}")
    print("  Upload complete.")

    # Step 3: Run SLAM pipeline on remote GPU
    print(f"\nStarting SLAM on remote A100 (submap_size={submap_size}, max_loops={max_loops})...")

    # Drain any stale URL from a previous run
    while url_queue.len() > 0:
        url_queue.get()

    # Spawn non-blocking so we can grab the Viser URL the moment it's ready
    slam_call = run_slam.spawn(
        image_folder=remote_name,
        submap_size=submap_size,
        max_loops=max_loops,
        min_disparity=min_disparity,
        conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        log_results=True,
        skip_dense_log=skip_dense_log,
    )

    print("Waiting for Viser server to start...")
    viser_url = url_queue.get(block=True, timeout=300)
    import webbrowser
    print(f"\n{'='*60}")
    print(f"  LIVE MAP -> {viser_url}")
    print(f"{'='*60}\n")
    webbrowser.open(viser_url)

    # Now wait for SLAM to finish
    result = slam_call.get()

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nPipeline complete:")
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Submaps: {result['submaps']}")
    print(f"  Loop closures: {result['loop_closures']}")
    print(f"  Total time: {result['total_time_s']}s ({result['avg_fps']} FPS)")
    print(f"  VGGT inference: {result['vggt_time_s']}s")
    print(f"  Backend optimization: {result['backend_time_s']}s")

    # Step 4: Download results from volume
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading results to {output_path}/...")
    results_prefix = f"results/{remote_name}"

    # Download poses.txt
    try:
        data = b"".join(data_vol.read_file(f"{results_prefix}/poses.txt"))
        (output_path / "poses.txt").write_bytes(data)
        print("  Downloaded poses.txt")
    except Exception as e:
        print(f"  Could not download poses.txt: {e}")

    # Download dense point cloud logs
    if not skip_dense_log:
        logs_dir = output_path / "poses_logs"
        logs_dir.mkdir(exist_ok=True)
        try:
            for entry in data_vol.listdir(f"{results_prefix}/poses_logs"):
                file_data = b"".join(data_vol.read_file(entry.path))
                local_file = logs_dir / Path(entry.path).name
                local_file.write_bytes(file_data)
            print(f"  Downloaded dense logs to {logs_dir}/")
        except Exception as e:
            print(f"  Could not download dense logs: {e}")

    print(f"\nDone. Results saved to {output_path}/")
