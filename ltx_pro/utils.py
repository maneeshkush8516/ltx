"""
Utility functions for the LTX-2 pipeline.

Provides tensor/image conversion helpers, ComfyUI node output accessor,
video display and saving, model file validation, image upload, custom node
loading, system package installation, and model download via aria2c.

All heavy imports (torch, numpy, cv2, PIL, comfy, IPython, google.colab)
are guarded inside function bodies so this module passes py_compile without
those packages installed.
"""

import os
import json
import subprocess
from typing import Any, Optional, Sequence, Mapping, Union
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image

__all__ = [
    "get_value_at_index",
    "pil_to_tensor",
    "tensor_to_pil",
    "load_image_tensor",
    "get_last_frame_tensor",
    "display_video",
    "save_video_from_components",
    "save_metadata_sidecar",
    "validate_model_files",
    "upload_image",
    "import_custom_nodes",
    "install_apt_packages",
    "model_download",
]


def get_value_at_index(obj, index):
    """Get a value from a ComfyUI node output by index.

    ComfyUI nodes return either a tuple/list (indexed by position) or a dict
    with a 'result' key. This helper handles both patterns.

    Args:
        obj: A sequence (tuple/list) or mapping (dict) from a node call.
        index: Integer index to retrieve.

    Returns:
        The value at the given index.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def pil_to_tensor(img):
    """Convert a PIL Image to a ComfyUI NHWC float tensor.

    Args:
        img: A PIL Image instance.

    Returns:
        A torch.Tensor of shape (1, H, W, 3) with values in [0, 1].
    """
    import numpy as np
    import torch as _torch

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return _torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(t):
    """Convert a ComfyUI NHWC tensor to a PIL Image.

    Args:
        t: A torch.Tensor of shape (1, H, W, 3) or (H, W, 3).

    Returns:
        A PIL Image in RGB mode.
    """
    import numpy as np
    from PIL import Image as _Image

    if t.ndim == 4:
        t = t[0]
    return _Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def load_image_tensor(path):
    """Load an image file as a ComfyUI NHWC tensor.

    Args:
        path: Filesystem path to the image file.

    Returns:
        A torch.Tensor of shape (1, H, W, 3) or None if path is missing/invalid.
    """
    from PIL import Image as _Image

    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(_Image.open(path).convert("RGB"))


def get_last_frame_tensor(video_path):
    """Extract last frame of a video as NHWC float tensor shape (1, H, W, 3).

    Uses OpenCV to seek to the final frame of the video file.

    Args:
        video_path: Filesystem path to the video file.

    Returns:
        A torch.Tensor of shape (1, H, W, 3) with values in [0, 1],
        or None if the video cannot be read.
    """
    import cv2
    import torch as _torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _torch.from_numpy(frame).float().unsqueeze(0) / 255.0


def display_video(path):
    """Display a video inline in a Jupyter/Colab notebook.

    Reads the video file, base64-encodes it, and renders an HTML5 video tag
    via IPython.display.

    Args:
        path: Filesystem path to the video file (mp4).
    """
    from base64 import b64encode
    from IPython.display import display, HTML

    if not path or not os.path.exists(path):
        print(f"   Not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=800 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        '</video>'
    ))


def save_video_from_components(video_obj, prefix="LTX-2-PRO"):
    """Save a ComfyUI video object and return the output path.

    Args:
        video_obj: A ComfyUI video object with get_dimensions() and save_to() methods.
        prefix: Filename prefix for the output video.

    Returns:
        The filesystem path where the video was saved.
    """
    import folder_paths
    from comfy_api.latest import Types

    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path


def save_metadata_sidecar(output_path, meta):
    """Write a .json sidecar file next to the generated video.

    Args:
        output_path: Path to the video file (the sidecar shares its basename).
        meta: Dictionary of metadata to serialize as JSON.

    Returns:
        The path to the written sidecar file.
    """
    sidecar = os.path.splitext(output_path)[0] + "_meta.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"   Metadata: {sidecar}")
    except Exception as e:
        print(f"   Could not save metadata: {e}")
    return sidecar


def validate_model_files(model_dict):
    """Check required model files exist in ComfyUI folder_paths.

    Args:
        model_dict: Dictionary mapping label to filename, e.g.
            {"unet": "model.gguf", "clip1": "clip.safetensors"}

    Returns:
        True only if all files are found in their expected directories.
    """
    import folder_paths

    folder_map = {
        "unet": "unet",
        "clip1": "text_encoders",
        "clip2": "text_encoders",
        "vae_vid": "vae",
        "vae_aud": "vae",
        "upscaler": "latent_upscale_models",
    }
    ok = True
    for label, filename in model_dict.items():
        fk = folder_map.get(label, "loras")
        found = any(
            os.path.exists(os.path.join(base, filename))
            for base in folder_paths.get_folder_paths(fk)
        )
        status = "OK" if found else "MISSING"
        print(f"   [{status}] [{label:9s}] {filename}")
        if not found:
            ok = False
    return ok


def upload_image(save_dir="/content/ComfyUI/input"):
    """Upload an image via Google Colab's file upload widget.

    Args:
        save_dir: Directory where the uploaded file will be saved.

    Returns:
        The path to the saved file, or None if no file was uploaded.
    """
    from google.colab import files

    os.makedirs(save_dir, exist_ok=True)
    uploaded = files.upload()
    for fname, data in uploaded.items():
        path = os.path.join(save_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        print(f"   Saved: {path}")
        return path
    return None


def import_custom_nodes():
    """Load all built-in and external custom nodes in a Jupyter/Colab-safe way.

    Uses nest_asyncio to handle the event loop in Jupyter environments where
    an event loop is already running.
    """
    import asyncio
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def _load():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print(f"   Some nodes failed: {[str(n) for n in failed]}")

    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())


def install_apt_packages():
    """Install system packages (aria2, ffmpeg) via apt-get.

    Runs apt-get install quietly and reports success or failure.
    """
    packages = ["aria2", "ffmpeg"]
    try:
        subprocess.run(["apt-get", "-y", "install", "-qq"] + packages,
                       check=True, capture_output=True)
        print("apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"apt error: {e.stderr.decode().strip() or 'unknown'}")


def model_download(url, dest_dir, filename=None, silent=True):
    """Download a model file using aria2c with skip-if-cached logic.

    Uses aria2c for fast parallel downloads with 16 connections. Skips the
    download if the file already exists and is larger than 1 MB.

    Args:
        url: URL to download from.
        dest_dir: Local directory to save the file.
        filename: Override filename (derived from URL if None).
        silent: If True, suppress aria2c console output.

    Returns:
        The local filename on success, or False on failure.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  cached: {filename}")
        return filename
    cmd = ["aria2c", "--console-log-level=error",
           "-c", "-x", "16", "-s", "16", "-k", "1M",
           "-d", dest_dir, "-o", filename]
    if silent:
        cmd += ["--summary-interval=0", "--quiet"]
        print(f"  downloading {filename}...", end=" ", flush=True)
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  Failed: {result.stderr.strip()}")
        return False
    if silent:
        print("done.")
    return filename
