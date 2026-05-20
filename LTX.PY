# @title {"single-column":true}
# @markdown # 💥Prepare Environment & Download LTX-Video Models

!pip install torch torchvision torchaudio

%cd /content
from IPython.display import clear_output
clear_output()
!pip install -q torchsde einops diffusers accelerate
!pip install av spandrel albumentations onnx opencv-python onnxruntime
!git clone  --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt
clear_output()


%cd /content/ComfyUI/custom_nodes

import subprocess

!git clone --branch kj_1.2.6 https://github.com/Isi-dev/ComfyUI_KJNodes
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git

%cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
!pip install -r requirements.txt
%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
!pip install -r requirements.txt

def install_apt_packages():
    packages = ['aria2', 'ffmpeg']

    try:
        # Run apt install silently (using -qq)
        subprocess.run(
            ['apt-get', '-y', 'install', '-qq'] + packages,
            check=True,
            capture_output=True
        )
        print("✓ apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing apt packages: {e.stderr.decode().strip() or 'Unknown error'}")


print("Installing apt packages...")
install_apt_packages()

clear_output()


%cd /content/ComfyUI

# !apt -y install -qq aria2 ffmpeg

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
from pathlib import Path
import random
from typing import Sequence, Mapping, Any, Union
import torch

import numpy as np
import cv2
from PIL import Image
import gc
import imageio
import shutil
from google.colab import files
from IPython.display import display, HTML, Image as IPImage
sys.path.insert(0, '/content/ComfyUI')


from nodes import NODE_CLASS_MAPPINGS
from nodes import LoraLoaderModelOnly


def download_with_aria2c(link, folder="/content/ComfyUI/models/loras"):
    import os

    filename = link.split("/")[-1]
    command = f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {link} -d {folder} -o {filename}"

    print("Executing download command:")
    print(command)

    os.makedirs(folder, exist_ok=True)
    get_ipython().system(command)

    return filename



def download_civitai_model(civitai_link, civitai_token, folder="/content/ComfyUI/models/loras"):
    import os
    import time

    os.makedirs(folder, exist_ok=True)

    try:
        model_id = civitai_link.split("/models/")[1].split("?")[0]
    except IndexError:
        raise ValueError("Invalid Civitai URL format. Please use a link like: https://civitai.com/api/download/models/1523247?...")

    civitai_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
    if civitai_token:
        civitai_url += f"&token={civitai_token}"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.safetensors"

    full_path = os.path.join(folder, filename)

    download_command = f"wget --max-redirect=10 --show-progress \"{civitai_url}\" -O \"{full_path}\""
    print("Downloading from Civitai...")

    os.system(download_command)

    local_path = os.path.join(folder, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"LoRA downloaded successfully: {local_path}")
    else:
        print(f"❌ LoRA download failed or file is empty: {local_path}")

    return filename

def download_lora(link, folder="/content/ComfyUI/models/loras", civitai_token=None):
    """
    Download a model file, automatically detecting if it's a Civitai link or huggingface download.
    """
    if "civitai.com" in link.lower():
        if not civitai_token:
            raise ValueError("Civitai token is required for Civitai downloads")
        return download_civitai_model(link, civitai_token, folder)
    else:
        return download_with_aria2c(link, folder)

def model_download(url: str, dest_dir: str, filename: str = None, silent: bool = True) -> bool:
    """
    Colab-optimized download with aria2c
    """
    try:
        # Create destination directory
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

        # Set filename if not specified
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]  # Remove URL parameters

        # Build command
        cmd = [
            'aria2c',
            '--console-log-level=error',
            '-c', '-x', '16', '-s', '16', '-k', '1M',
            '-d', dest_dir,
            '-o', filename,
            url
        ]

        # Add silent flags if requested
        if silent:
            cmd.extend(['--summary-interval=0', '--quiet'])
            print(f"Downloading {filename}...", end=' ', flush=True)

        # Run download
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if silent:
            print("Done!")
        else:
            print(f"Downloaded {filename} to {dest_dir}")
        return filename

    except subprocess.CalledProcessError as e:
        error = e.stderr.strip() or "Unknown error"
        print(f"\nError downloading {filename}: {error}")
        return False
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


import asyncio
import nest_asyncio

def import_custom_nodes() -> None:
    """Load all built-in and external custom nodes in a Jupyter/Colab-safe way."""

    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def loader():
        # Load built-in extra nodes
        import_failed = await init_builtin_extra_nodes()

        # Load external custom nodes
        await init_external_custom_nodes()

        if import_failed:
            print("WARNING: Some comfy_extras nodes failed to import:")
            for node in import_failed:
                print(" -", node)

    try:
        # Try running normally (works in scripts)
        asyncio.run(loader())
    except RuntimeError:
        # Already inside an event loop → Jupyter/Colab
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(loader())


# ------------------------------------------------------------------------------
# Core Model Downloads
# ------------------------------------------------------------------------------

ltx_model = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf" # @param {"type":"string"}
dit_model=model_download(ltx_model, "/content/ComfyUI/models/unet")

text_encoder_link = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" # @param {"type":"string"}
text_encoder_model = model_download(text_encoder_link, "/content/ComfyUI/models/text_encoders")

text_encoder2_link = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors" # @param {"type":"string"}
text_encoder2_model = model_download(text_encoder2_link, "/content/ComfyUI/models/text_encoders")

vae_link = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_video_vae_bf16.safetensors" # @param {"type":"string"}
vae_model = model_download(vae_link, "/content/ComfyUI/models/vae")

vae_audio_link = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_audio_vae_bf16.safetensors" # @param {"type":"string"}
vae_audio_model = model_download(vae_audio_link, "/content/ComfyUI/models/vae")

upscaler_link = "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" # @param {"type":"string"}
upscaler_model = model_download(upscaler_link, "/content/ComfyUI/models/latent_upscale_models")


# ------------------------------------------------------------------------------
# LoRA Batch Download Section
# ------------------------------------------------------------------------------

# Dictionary of LoRAs to download.
# Note: URLs have been corrected to point to the valid Lightricks repositories.
LORA_URLS = {
    "Canny": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Detailer": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Pose": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly In": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly Left": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly Out": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly Right": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib Down": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Jib Up": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Static": "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors"
}

LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(LORA_DIR, exist_ok=True)

print(f"\n--- Starting Batch LoRA Download ({len(LORA_URLS)} models) ---")

for lora_name, lora_url in LORA_URLS.items():
    print(f"⬇️ Processing: {lora_name}")
    # Using the existing model_download function which handles aria2c and error checking
    result = model_download(lora_url, LORA_DIR)

    if result:
        print(f"   ✅ {lora_name} ready.")
    else:
        print(f"   ❌ Failed to download {lora_name}.")

print("--- LoRA Download Complete ---")


# ------------------------------------------------------------------------------
# Helper Functions for Input/Output
# ------------------------------------------------------------------------------

def upload_file():
    """Handle file upload (image or video) and return paths."""
    os.makedirs('/content/ComfyUI/input', exist_ok=True)
    uploaded = files.upload()

    paths = []
    for filename in uploaded.keys():
        src_path = f'/content/ComfyUI/{filename}'
        dest_path = f'/content/ComfyUI/input/{filename}'
        shutil.move(src_path, dest_path)
        paths.append(dest_path)
        print(f"File saved to: {dest_path}")

    return paths[0] if paths else None

def upload_fileInt():
    """Handle file upload (image or video) and return paths."""
    os.makedirs('/content/ComfyUI/output', exist_ok=True)
    uploaded = files.upload()

    paths = []
    for filename in uploaded.keys():
        src_path = f'/content/ComfyUI/{filename}'
        dest_path = f'/content/ComfyUI/output/{filename}'
        shutil.move(src_path, dest_path)
        paths.append(dest_path)
        print(f"File saved to: {dest_path}")

    return paths[0] if paths else None

def extract_frames(video_path, max_frames=None):
    """Extract frames from video and return as a batch tensor."""
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        success, frame = vidcap.read()
        if not success or (max_frames and len(frames) >= max_frames):
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).float() / 255.0
        frames.append(frame)

    if not frames:
        return None, fps

    # Stack frames into a batch tensor: (N, H, W, 3)
    batch = torch.stack(frames, dim=0)
    # print(f"Extracted {len(frames)} frames (shape: {batch.shape})")
    return batch, fps


def select_every_n_frame_tensor(
    frames_tensor: torch.Tensor,
    fps: float,
    n: int,
    skip_first: int = 0,
    max_output_frames: int = 0
):
    if frames_tensor is None or frames_tensor.ndim != 4:
        raise ValueError("frames_tensor must be a 4D tensor of shape (N, H, W, C)")
    if n < 1:
        raise ValueError("n must be >= 1")

    total_frames = frames_tensor.shape[0]

    if skip_first >= total_frames:
        print("No frames available after skipping.")
        return None, 0.0

    frames_to_use = frames_tensor[skip_first:]

    # Select every nth frame
    selected_frames = frames_to_use[::n]

    # Cap output if needed
    if max_output_frames > 0 and selected_frames.shape[0] > max_output_frames:
        selected_frames = selected_frames[:max_output_frames]

    adjusted_fps = fps / n

    if max_output_frames:
        print(f"Frame cap: {max_output_frames} -> Final output: {selected_frames.shape[0]} frames")
    # print(f"Adjusted FPS: {adjusted_fps:.2f}  -> Final output: {selected_frames.shape[0]} frames")

    return selected_frames, adjusted_fps

def is_video_file(path):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    return os.path.splitext(path)[1].lower() in video_extensions

def get_last_frame_tensor(video_path: str) -> torch.Tensor | None:
    """Return the last frame of a video as a tensor of shape (1, H, W, C)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Jump directly to the last frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    # Convert BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float tensor in range [0,1], shape (1,H,W,C)
    tensor = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    return tensor


def swapT(pa, f, s):
    if pa == f:
        pa = s
    return pa


def image_width_height(image):
    if image.ndim == 4:
        _, height, width, _ = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return width, height

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    for obj in list(globals().values()):
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            del obj
    gc.collect()


def upload_image():
    """Handle image upload in Colab and store in /content/ComfyUI/input/"""
    from google.colab import files
    import os
    import shutil

    os.makedirs('/content/ComfyUI/input', exist_ok=True)

    uploaded = files.upload()

    # Move each uploaded file to ComfyUI input directory
    for filename in uploaded.keys():
        src_path = f'/content/ComfyUI/{filename}'
        dest_path = f'/content/ComfyUI/input/{filename}'

        shutil.move(src_path, dest_path)
        # print(f"Image saved to: {dest_path}")
        return dest_path

    return None


output_path =""

clear_output()
print("✅ Environment Setup & All Models Downloaded Successfully!")


# ╔══════════════════════════════════════════════════════════════╗
# ║   LTX-2 LD T2V — Google Colab Runner (Fixed)               ║
# ║   Translated from: LD - T2V.json  (LoRa Daddy workflow)    ║
# ║   Engine:  LTX2_TI2V_Distilled notebook pipeline           ║
# ╚══════════════════════════════════════════════════════════════╝
#
# HOW TO USE:
#   Run Cell 1 once to install.
#   Run Cell 2 once to download models.
#   Run Cell 3 every session to load imports/helpers.
#   Edit Cell 4 config, then run Cell 5 to generate.
#
# WHY GGUF instead of DEV fp8?
#   The LD JSON workflow targets the DEV model which requires a
#   separate distill LoRA and ~30 GB VRAM. For Colab (T4/L4/A100)
#   the GGUF Q4_K_M distilled checkpoint is the correct choice —
#   it is what the reference notebook uses and runs within 15 GB.

# ═══════════════════════════════════════════════════════════════
# CELL 1 — INSTALL  (run once per Colab session)
# ═══════════════════════════════════════════════════════════════

# @title Cell 1 — Install Environment
# !pip install torch torchvision torchaudio

# %cd /content
# from IPython.display import clear_output
# clear_output()

# !pip install -q torchsde einops diffusers accelerate nest_asyncio
# !pip install av spandrel albumentations onnx opencv-python onnxruntime imageio imageio-ffmpeg

# # Clone pinned ComfyUI (same branch as reference notebook)
# !git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
# !pip install -r /content/ComfyUI/requirements.txt -q
# clear_output()

# # Custom nodes
# %cd /content/ComfyUI/custom_nodes
# !git clone --branch kj_1.2.6        https://github.com/Isi-dev/ComfyUI_KJNodes
# !git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git
# !git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
# !git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git
# !git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# %cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
# !pip install -r requirements.txt -q
# %cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
# !pip install -r requirements.txt -q

# !apt-get -y install -qq aria2 ffmpeg
# clear_output()
# print("✅ Environment ready")

# ═══════════════════════════════════════════════════════════════
# CELL 2 — MODEL DOWNLOADS  (run once)
# ═══════════════════════════════════════════════════════════════

# @title Cell 2 — Download Models
# @markdown Downloads all required models via aria2c (fast parallel download).
# @markdown **Pick ONE Gemma text encoder based on your GPU** (see comments).

# import os, subprocess
# from pathlib import Path

# def dl(url, dest, filename=None):
#     Path(dest).mkdir(parents=True, exist_ok=True)
#     fn = filename or url.split("/")[-1].split("?")[0]
#     cmd = ["aria2c","--console-log-level=error","-c","-x","16","-s","16",
#            "-k","1M","--summary-interval=0","--quiet",url,"-d",dest,"-o",fn]
#     subprocess.run(cmd, check=True)
#     print(f"  ✓ {fn}")
#     return fn

# BASE  = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
# CORG  = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
# LTX   = "https://huggingface.co/Lightricks"

# print("Downloading models...")

# # ── UNet: GGUF Q4_K_M distilled (works on T4 / L4 / A100) ──────
# dl(f"{BASE}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
#    "/content/ComfyUI/models/unet")

# # ── Text encoders ───────────────────────────────────────────────
# # Gemma fp4  → RTX 5000 Blackwell ONLY
# dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
#    "/content/ComfyUI/models/text_encoders")
# # Gemma fp8  → RTX 3000 / 4000 / A100 / T4  (comment out one)
# # dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
# #    "/content/ComfyUI/models/text_encoders")

# # Embeddings connector — distilled version (matches GGUF model)
# dl(f"{BASE}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
#    "/content/ComfyUI/models/text_encoders")

# # ── VAEs ────────────────────────────────────────────────────────
# dl(f"{BASE}/VAE/LTX2_video_vae_bf16.safetensors",
#    "/content/ComfyUI/models/vae")
# dl(f"{BASE}/VAE/LTX2_audio_vae_bf16.safetensors",
#    "/content/ComfyUI/models/vae")

# # ── Spatial upscaler ────────────────────────────────────────────
# dl(f"{LTX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
#    "/content/ComfyUI/models/latent_upscale_models")

# # ── LoRAs (optional but recommended) ────────────────────────────
# # IC Detailer (from LD workflow LTX2MasterLoaderLD stack slot 1)
# dl(f"{LTX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
#    "/content/ComfyUI/models/loras")
# # Camera control LoRAs (optional — add the ones you want)
# # dl(f"{LTX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
# #    "/content/ComfyUI/models/loras")

# print("\n✅ All models downloaded")

# ═══════════════════════════════════════════════════════════════
# CELL 3 — IMPORTS & HELPERS  (run every session)
# ═══════════════════════════════════════════════════════════════

# @title Cell 3 — Imports & Helpers
import os, sys, gc, time, json, shutil, warnings, subprocess
import asyncio
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from pathlib import Path
from IPython.display import display, HTML, clear_output
from base64 import b64encode

# ── ComfyUI core ────────────────────────────────────────────────
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# ── Async loader (Jupyter/Colab safe) ──────────────────────────
def import_custom_nodes() -> None:
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _load():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print("   Some nodes failed:", [str(n) for n in failed])
    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())

# ── Utility helpers ─────────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def cleanup_memory(verbose: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if verbose:
        _print_vram()

def _print_vram() -> None:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pct   = used / total * 100 if total > 0 else 0
        bar   = "█" * int(20 * used / total) + "░" * (20 - int(20 * used / total))
        print(f"   💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%)")

def display_video(video_path: str) -> None:
    if not video_path or not os.path.exists(video_path):
        print(f"   ⚠️ Video not found: {video_path}")
        return
    data = b64encode(open(video_path, "rb").read()).decode()
    display(HTML(
        '<video width=800 controls autoplay loop>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        '</video>'
    ))

def save_video_from_components(video_obj, prefix="LTX-2") -> str:
    """Save a ComfyUI video object and return the output path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, filename, counter, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext  = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{filename}_{counter:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path

def validate_model_files(model_dict: dict) -> bool:
    """
    Check required model files exist in ComfyUI folder_paths.
    model_dict: { label: filename }
    """
    folder_map = {
        "unet"    : "unet",
        "clip1"   : "text_encoders",
        "clip2"   : "text_encoders",
        "vae_vid" : "vae",
        "vae_aud" : "vae",
        "upscaler": "latent_upscale_models",
    }
    ok = True
    for label, filename in model_dict.items():
        folder_key = folder_map.get(label, "loras")
        found = False
        try:
            for base in folder_paths.get_folder_paths(folder_key):
                if os.path.exists(os.path.join(base, filename)):
                    found = True
                    break
        except Exception:
            pass
        icon = "✅" if found else "❌"
        print(f"   {icon} [{label:9s}] {filename}")
        if not found:
            ok = False
    return ok

# ── LoRA stack — mirrors LTX2MasterLoaderLD node [326] ─────────
# 10-slot stack. Slot 1 = IC Detailer @ 0.4 (active in LD workflow).
# Set "on": True and fill "lora" for any slot to activate it.
LD_LORA_STACK = [
    {"on": True,  "lora": "ltx-2-19b-ic-lora-detailer.safetensors", "strength": 0.4},
    {"on": False, "lora": "None", "strength": 1.0},  # add camera LoRA here
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
    {"on": False, "lora": "None", "strength": 1.0},
]

print("✅ Imports & helpers ready")

# ═══════════════════════════════════════════════════════════════
# CELL 4 — CONFIGURATION  (edit before running Cell 6)
# ═══════════════════════════════════════════════════════════════

# @title Cell 4 — Configuration

# ── Resolution & length ─────────────────────────────────────────
# LD workflow: WIDTH=1280, HEIGHT=1280, FRAMES=480
# Colab T4 safe: 768×512 @ 121 frames. Scale up for A100.
WIDTH  = 768    # @param {type:"integer"}
HEIGHT = 512    # @param {type:"integer"}
FRAMES = 121    # @param {type:"integer"}
FPS    = 25     # @param {type:"integer"}

# ── Seed — mirrors "Shared seed" node [284] (increment mode) ───
SEED                = 47    # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}

# ── Prompt ─────────────────────────────────────────────────────
# LD workflow uses LTX2PromptArchitect (local LLM). Enter your
# expanded cinematic prompt directly here instead.
POSITIVE_PROMPT = (   # @param {type:"string"}
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "people walking, bokeh streetlights, moody atmosphere, volumetric god rays, "
    "ultra detailed, professional cinematography, shallow depth of field"
)
NEGATIVE_PROMPT = (   # @param {type:"string"}
    "blurry, distorted, low quality, watermark, text overlay, bad anatomy, "
    "deformed, grainy, overexposed, underexposed, flickering, motion artifacts, "
    "desaturated, flat lighting, AI artifacts, cartoonish, out of focus"
)

# ── Model filenames ─────────────────────────────────────────────
# GGUF distilled — correct for Colab (NOT the DEV fp8 which needs ~30 GB)
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"                         # @param {type:"string"}
# Gemma: pick ONE matching your GPU
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"                     # @param {type:"string"}
# Distilled embeddings connector — must match GGUF (not _dev_ version)
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"  # @param {type:"string"}
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"                          # @param {type:"string"}
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"                          # @param {type:"string"}
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"               # @param {type:"string"}

# ── Pass 1 — HIGH / "New Subgraph" (node baf97382) ─────────────
# Uses ManualSigmas + euler (proven schedule from reference notebook).
# LD JSON used LTXVScheduler but that is tuned for the DEV model.
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # @param {type:"string"}
PASS1_SAMPLER = "euler"   # @param {type:"string"}
PASS1_CFG     = 1.0       # @param {type:"number"}

# ── Pass 2 — LOW subgraph (node 75444468) ──────────────────────
# ManualSigmas from LD JSON: "0.509375, 0.425, 0.221875, 0.22, 0.0"
# Reference notebook: "0.909375, 0.725, 0.421875, 0.0" with gradient_estimation
# Using notebook values — they are validated working with GGUF.
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"  # @param {type:"string"}
PASS2_SAMPLER = "gradient_estimation"               # @param {type:"string"}
PASS2_CFG     = 1.0                                 # @param {type:"number"}
PASS2_SEED    = 0    # @param {type:"integer"}  # fixed (node [114] seed=12368 in LD)

# ── Tiled VAE — node [265] LTXVSpatioTemporalTiledVAEDecode ────
# Requires ComfyUI-LTXVideo. Falls back to VAEDecode if missing.
USE_TILED_VAE          = True   # @param {type:"boolean"}
TILED_SPATIAL_TILES    = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP  = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN     = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX   = False  # @param {type:"boolean"}

# ── Output — mirrors VHS_VideoCombine node [140] ────────────────
OUTPUT_PREFIX = "LTX-2"  # @param {type:"string"}

print("✅ Configuration set")
print(f"   Resolution : {WIDTH}×{HEIGHT}  |  Frames : {FRAMES}  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   UNet       : {UNET_MODEL}")
print(f"   Seed       : {SEED}  (auto-increment: {AUTO_INCREMENT_SEED})")
print(f"   Pass 1     : {PASS1_SAMPLER}  |  {PASS1_SIGMAS}")
print(f"   Pass 2     : {PASS2_SAMPLER}  |  {PASS2_SIGMAS}")

# ═══════════════════════════════════════════════════════════════
# CELL 5 — GENERATION FUNCTION  (run once per session)
# ═══════════════════════════════════════════════════════════════

# @title Cell 5 — Define generate_ld_t2v()

def generate_ld_t2v(
    positive_prompt:       str   = POSITIVE_PROMPT,
    negative_prompt:       str   = NEGATIVE_PROMPT,
    width:                 int   = WIDTH,
    height:                int   = HEIGHT,
    frames:                int   = FRAMES,
    fps:                   int   = FPS,
    seed:                  int   = SEED,
    pass1_sigmas:          str   = PASS1_SIGMAS,
    pass1_sampler:         str   = PASS1_SAMPLER,
    pass1_cfg:             float = PASS1_CFG,
    pass2_sigmas:          str   = PASS2_SIGMAS,
    pass2_sampler:         str   = PASS2_SAMPLER,
    pass2_cfg:             float = PASS2_CFG,
    pass2_seed:            int   = PASS2_SEED,
    use_tiled_vae:         bool  = USE_TILED_VAE,
    tiled_spatial_tiles:   int   = TILED_SPATIAL_TILES,
    tiled_spatial_overlap: int   = TILED_SPATIAL_OVERLAP,
    tiled_temporal_len:    int   = TILED_TEMPORAL_LEN,
    tiled_temporal_overlap:int   = TILED_TEMPORAL_OVERLAP,
    tiled_last_frame_fix:  bool  = TILED_LAST_FRAME_FIX,
    output_prefix:         str   = OUTPUT_PREFIX,
) -> Optional[str]:
    """
    Two-pass LTX-2 Text-to-Video pipeline.
    Translated from LD - T2V.json (LoRa Daddy ComfyUI workflow).
    Uses the GGUF distilled model which works on Colab T4/L4/A100.

    Node graph (LD JSON → Python):
    ────────────────────────────────────────────────────────────
    [255] UNETLoader          → UnetLoaderGGUF (GGUF)
    [326] LTX2MasterLoaderLD  → LoraLoaderModelOnly × LD_LORA_STACK
    [190] DualCLIPLoader      → DualCLIPLoader (gemma + connector)
    [121] CLIPTextEncode      → positive conditioning
    [110] CLIPTextEncode      → negative conditioning
    [107] LTXVConditioning    → apply frame_rate metadata
    [184] VAELoader           → video VAE (bf16)
    [196] VAELoaderKJ         → audio VAE (bf16)
    [189] LatentUpscaleModel  → spatial upscaler x2
    [256] EmptyImage          → full-res blank image
    [164] ResizeImageMaskNode → ×0.5 → half-res
    [163] GetImageSize        → half_w, half_h
    [108] EmptyLTXVLatentVid  → half-res video latent
    [199] LTXVEmptyLatentAudio→ audio latent
    [109] LTXVConcatAVLatent  → combined AV latent
    ── PASS 1 (New Subgraph) ──────────────────────────────────
    ManualSigmas(pass1_sigmas)
    KSamplerSelect(euler)
    RandomNoise(seed)
    CFGGuider(cfg=1)
    SamplerCustomAdvanced → p1_av_output[0]
    ── PASS 2 (LOW subgraph) ──────────────────────────────────
    LTXVSeparateAVLatent  → video_lat, audio_lat
    LTXVCropGuides        → cropped conditioning + latent
    LTXVLatentUpsampler   → 2× upsample
    LTXVConcatAVLatent    → upsampled AV latent
    ManualSigmas(pass2_sigmas)
    KSamplerSelect(gradient_estimation)
    RandomNoise(pass2_seed=0)
    CFGGuider(cfg=1)
    SamplerCustomAdvanced → p2_denoised[1]
    [del unet + PurgeVRAM]
    ── DECODE ─────────────────────────────────────────────────
    [125] LTXVSeparateAVLatent
    [265] LTXVSpatioTemporalTiledVAEDecode  (or VAEDecode)
    [201] LTXVAudioVAEDecode
          CreateVideo → save → display
    """
    t0 = time.time()
    import_custom_nodes()
    clear_output()

    print("🎬 LTX-2 LD T2V — Starting Generation")
    print(f"   Resolution : {width}×{height}  |  Frames : {frames}  |  Seed : {seed}")
    print(f"   Prompt     : {positive_prompt[:120]}...")
    _print_vram()

    # ── Pre-flight check ────────────────────────────────────────
    print("\n🔍 Model file check...")
    all_present = validate_model_files({
        "unet"    : UNET_MODEL,
        "clip1"   : CLIP_NAME1,
        "clip2"   : CLIP_NAME2,
        "vae_vid" : VAE_VIDEO_MODEL,
        "vae_aud" : VAE_AUDIO_MODEL,
        "upscaler": UPSCALER_MODEL,
    })
    if not all_present:
        raise FileNotFoundError(
            "One or more model files are missing — run Cell 2 first."
        )

    with torch.inference_mode():

        # ═══════════════════════════════════════════════════════
        # 1. MODEL LOADING
        # ═══════════════════════════════════════════════════════

        # ── UNet: UnetLoaderGGUF ────────────────────────────────
        # Node [255] in LD JSON → maps to UNETLoader,
        # but GGUF model requires UnetLoaderGGUF (from ComfyUI_GGUF).
        # Reference notebook: NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        print("\n📦 Loading UNet (GGUF distilled Q4_K_M)...")
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet = get_value_at_index(
            unetloadergguf.load_unet(unet_name=UNET_MODEL), 0
        )

        # ── LoRA stack — LTX2MasterLoaderLD [326] logic ─────────
        # Apply each active slot via LoraLoaderModelOnly.
        # No separate distill LoRA needed — GGUF has it baked in.
        active_loras = [
            s for s in LD_LORA_STACK
            if s.get("on") and s.get("lora") not in (None, "None", "")
        ]
        if active_loras:
            print(f"   🧬 Applying {len(active_loras)} LoRA(s) from stack...")
            for slot in active_loras:
                name     = slot["lora"]
                strength = slot.get("strength", 1.0)
                try:
                    ll   = LoraLoaderModelOnly()
                    unet = ll.load_lora_model_only(unet, name, strength)[0]
                    print(f"      ✓ {name} @ {strength}")
                except Exception as e:
                    print(f"      ⚠️ LoRA '{name}' failed ({e}) — skipping")
        else:
            print("   ℹ️ No active LoRAs in LD_LORA_STACK")

        # ── DualCLIPLoader [190] ────────────────────────────────
        print("   📝 Loading CLIP encoders...")
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_model = get_value_at_index(
            dualcliploader.load_clip(
                clip_name1=CLIP_NAME1,
                clip_name2=CLIP_NAME2,
                type="ltxv",
                device="default"
            ), 0
        )

        # ── VAEs ────────────────────────────────────────────────
        # [184] VAELoader (video VAE, bf16)
        # [196] VAELoaderKJ (audio VAE, bf16, main_device)
        vaeloader   = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        vae_video   = get_value_at_index(vaeloader.load_vae(vae_name=VAE_VIDEO_MODEL), 0)
        vae_audio   = get_value_at_index(
            vaeloaderkj.load_vae(vae_name=VAE_AUDIO_MODEL,
                                 device="main_device", weight_dtype="fp16"), 0
        )

        # ── Spatial upscaler [189] ──────────────────────────────
        uml           = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        upscale_model = get_value_at_index(
            uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0
        )

        # ═══════════════════════════════════════════════════════
        # 2. TEXT ENCODING  (Prompt group nodes [121], [110], [107])
        # ═══════════════════════════════════════════════════════
        print("\n📝 Encoding prompts...")

        # CLIPTextEncode positive [121]
        cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos = cte.encode(text=positive_prompt, clip=clip_model)
        cond_neg = cte.encode(text=negative_prompt, clip=clip_model)
        del clip_model
        cleanup_memory()

        # ConditioningZeroOut — applied to positive for negative branch
        # (matches reference notebook pattern)
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        cond_zero = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cond_pos, 0)
        )

        # LTXVConditioning [107] — injects frame_rate into conditioning
        # outputs [0]=positive, [1]=negative
        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=float(fps),
            positive=get_value_at_index(cond_pos, 0),
            negative=get_value_at_index(cond_zero, 0)
        )

        # ═══════════════════════════════════════════════════════
        # 3. LATENT PREPARATION  (Prepare LTX Latent group)
        # ═══════════════════════════════════════════════════════
        # [256] EmptyImage(W,H) → [164] ResizeImageMaskNode(×0.5 area)
        # → [163] GetImageSize → [108] EmptyLTXVLatentVideo(half)
        # + [199] LTXVEmptyLatentAudio → [109] LTXVConcatAVLatent
        print("\n🗂️ Preparing latents...")

        # EmptyImage at target resolution [256]
        ei       = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_img = ei.generate(width=width, height=height, batch_size=1, color=0)

        # ResizeImageMaskNode ×0.5 area [164]
        rimn     = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_img = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_img, 0),
            scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5}
        )

        # GetImageSize [163] → actual half-res pixel dims
        gis     = NODE_CLASS_MAPPINGS["GetImageSize"]()
        half_sz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_img, 0))
        half_w  = get_value_at_index(half_sz, 0)
        half_h  = get_value_at_index(half_sz, 1)
        print(f"   Latent dims : {half_w}×{half_h}  (half of {width}×{height})")

        # EmptyLTXVLatentVideo at half resolution [108]
        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1
        )

        # LTXVEmptyLatentAudio [199]
        elalat  = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1,
            audio_vae=vae_audio
        )

        # LTXVConcatAVLatent [109] → combined AV latent
        catav   = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat1 = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(vid_lat, 0),
            audio_latent=get_value_at_index(aud_lat, 0)
        )
        combined_latent = get_value_at_index(av_lat1, 0)

        # ═══════════════════════════════════════════════════════
        # 4. PASS 1 — HIGH / "New Subgraph" (baf97382)
        #    Nodes: ManualSigmas, KSamplerSelect(euler),
        #           RandomNoise(seed), CFGGuider, SamplerCustomAdvanced
        #    Output: p1_av_output[0]  (raw AV latent → Pass 2)
        # ═══════════════════════════════════════════════════════
        print(f"\n🚀 Pass 1 — {pass1_sampler} | sigmas: {pass1_sigmas[:55]}...")
        _print_vram()

        # ManualSigmas — replaces LTXVScheduler for GGUF distilled
        manualsigmas   = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        sig_p1         = manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas)

        # KSamplerSelect — euler
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        sampler_p1     = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)

        # RandomNoise — user seed
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        noise_p1    = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)

        # CFGGuider — cfg=1 (distilled model does not need high CFG)
        cfgguider  = NODE_CLASS_MAPPINGS["CFGGuider"]()
        guider_p1  = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg,
            model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1)
        )

        # SamplerCustomAdvanced — Pass 1
        sca  = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        out1 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise_p1, 0),
            guider=get_value_at_index(guider_p1, 0),
            sampler=get_value_at_index(sampler_p1, 0),
            sigmas=get_value_at_index(sig_p1, 0),
            latent_image=combined_latent
        )
        p1_av_output = get_value_at_index(out1, 0)  # raw output AV latent

        del guider_p1
        cleanup_memory()
        print("   ✓ Pass 1 done")

        # ═══════════════════════════════════════════════════════
        # 5. PASS 2 — LOW subgraph (75444468)
        #    Step A: LTXVSeparateAVLatent + LTXVCropGuides
        #    Step B: LTXVLatentUpsampler 2×
        #    Step C: ManualSigmas(pass2_sigmas) + gradient_estimation
        # ═══════════════════════════════════════════════════════
        print(f"\n🔧 Pass 2 — {pass2_sampler} | sigmas: {pass2_sigmas}")
        _print_vram()

        # LTXVSeparateAVLatent [116]: split P1 AV → video + audio
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1         = ltxvseparateavlatent.EXECUTE_NORMALIZED(av_latent=p1_av_output)
        vid_lat_p1 = get_value_at_index(s1, 0)
        aud_lat_p1 = get_value_at_index(s1, 1)

        # LTXVCropGuides — trims conditioning to match upscaled latent dims
        # (matches reference notebook: ltxvcropguides.EXECUTE_NORMALIZED)
        ltxvcropguides = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cropped = ltxvcropguides.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vid_lat_p1
        )
        # cropped[0]=positive, [1]=negative, [2]=cropped_video_lat

        # CFGGuider for Pass 2 with cropped conditioning
        guider_p2 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg,
            model=unet,
            positive=get_value_at_index(cropped, 0),
            negative=get_value_at_index(cropped, 1)
        )

        # LTXVLatentUpsampler [118]: 2× spatial upsample
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = ltxvlatentupsampler.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model,
            vae=vae_video
        )
        del upscale_model
        cleanup_memory()

        # LTXVConcatAVLatent [117]: upscaled video + audio
        av_lat2 = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(upsampled, 0),
            audio_latent=aud_lat_p1
        )

        # ManualSigmas — Pass 2 refinement schedule
        sig_p2    = manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas)

        # KSamplerSelect — gradient_estimation
        sampler_p2 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)

        # RandomNoise — fixed seed for reproducible refinement [114]
        noise_p2   = randomnoise.EXECUTE_NORMALIZED(noise_seed=pass2_seed)

        # SamplerCustomAdvanced — Pass 2
        out2 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise_p2, 0),
            guider=get_value_at_index(guider_p2, 0),
            sampler=get_value_at_index(sampler_p2, 0),
            sigmas=get_value_at_index(sig_p2, 0),
            latent_image=get_value_at_index(av_lat2, 0)
        )
        p2_denoised = get_value_at_index(out2, 1)  # denoised_output

        # Free UNet + guider (mirrors reference: del unetloadergguf_146)
        del guider_p2, unet
        cleanup_memory()
        print("   ✓ Pass 2 done")

        # ═══════════════════════════════════════════════════════
        # 6. DECODE  (Decode group)
        # ═══════════════════════════════════════════════════════
        # [125] LTXVSeparateAVLatent
        # [265] LTXVSpatioTemporalTiledVAEDecode (or VAEDecode)
        # [201] LTXVAudioVAEDecode
        #        CreateVideo → display
        print("\n🎞️ Decoding video & audio...")
        _print_vram()

        # LTXVSeparateAVLatent [125] — final split
        s2          = ltxvseparateavlatent.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        # ── Video decode ─────────────────────────────────────────
        decoded_frames = None

        if use_tiled_vae:
            # LTXVSpatioTemporalTiledVAEDecode [265]
            # Requires ComfyUI-LTXVideo by Lightricks.
            try:
                tiled_dec = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded_frames = get_value_at_index(
                    tiled_dec.EXECUTE_NORMALIZED(
                        vae=vae_video,
                        latents=vid_lat_fin,
                        spatial_tiles=tiled_spatial_tiles,
                        spatial_overlap=tiled_spatial_overlap,
                        temporal_tile_length=tiled_temporal_len,
                        temporal_overlap=tiled_temporal_overlap,
                        last_frame_fix=tiled_last_frame_fix,
                        working_device="auto",
                        working_dtype="auto"
                    ), 0
                )
                print("   ✓ Tiled VAE decode")
            except (KeyError, Exception) as e:
                print(f"   ⚠️ Tiled VAE unavailable ({type(e).__name__}: {e})")
                print("      Falling back to standard VAEDecode")
                use_tiled_vae = False

        if not use_tiled_vae:
            # Standard VAEDecode — always in core ComfyUI
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_video), 0
            )
            print("   ✓ Standard VAE decode")

        del vae_video
        cleanup_memory()

        # ── Audio decode [201] ─────────────────────────────────
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio_out = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=aud_lat_fin,
            audio_vae=vae_audio
        )
        del vae_audio
        cleanup_memory()

        # ── CreateVideo + save ─────────────────────────────────
        # Matches VHS_VideoCombine [140]: h264-mp4, fps=25, crf=19
        print("\n💾 Saving video...")
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        vid_obj     = createvideo.EXECUTE_NORMALIZED(
            fps=fps,
            images=decoded_frames,
            audio=get_value_at_index(audio_out, 0)
        )
        output_path = save_video_from_components(
            get_value_at_index(vid_obj, 0), prefix=output_prefix
        )

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n✅ Done in {mins}m {secs}s")
    print(f"   📁 {output_path}")
    _print_vram()
    display_video(output_path)
    return output_path


print("✅ generate_ld_t2v() is ready — run Cell 6 to generate")

# ═══════════════════════════════════════════════════════════════
# CELL 6 — RUN  (re-run this cell for each new video)
# ═══════════════════════════════════════════════════════════════

# @title Cell 6 — Run Generation
# @markdown Re-run this cell for each new video.
# @markdown Seed auto-increments after each successful generation.

_current_seed = SEED

try:
    output = generate_ld_t2v(
        positive_prompt        = POSITIVE_PROMPT,
        negative_prompt        = NEGATIVE_PROMPT,
        width                  = WIDTH,
        height                 = HEIGHT,
        frames                 = FRAMES,
        fps                    = FPS,
        seed                   = _current_seed,
        pass1_sigmas           = PASS1_SIGMAS,
        pass1_sampler          = PASS1_SAMPLER,
        pass1_cfg              = PASS1_CFG,
        pass2_sigmas           = PASS2_SIGMAS,
        pass2_sampler          = PASS2_SAMPLER,
        pass2_cfg              = PASS2_CFG,
        pass2_seed             = PASS2_SEED,
        use_tiled_vae          = USE_TILED_VAE,
        tiled_spatial_tiles    = TILED_SPATIAL_TILES,
        tiled_spatial_overlap  = TILED_SPATIAL_OVERLAP,
        tiled_temporal_len     = TILED_TEMPORAL_LEN,
        tiled_temporal_overlap = TILED_TEMPORAL_OVERLAP,
        tiled_last_frame_fix   = TILED_LAST_FRAME_FIX,
        output_prefix          = OUTPUT_PREFIX,
    )

    # Mirrors "Shared seed" node [284] increment mode
    if AUTO_INCREMENT_SEED:
        SEED = _current_seed + 1
        print(f"🔢 Next seed: {SEED}")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted — partial output may be in /content/ComfyUI/output/")

except FileNotFoundError as e:
    print(f"\n❌ Missing models: {e}")
    print("   Run Cell 2 to download, then retry Cell 6.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n❌ CUDA Out of Memory")
    print(f"   Current settings: {WIDTH}×{HEIGHT}, {FRAMES} frames")
    print("   T4  (15 GB) → try: 768×512,  121 frames")
    print("   L4  (24 GB) → try: 1024×576, 161 frames")
    print("   A100(40 GB) → try: 1280×720, 241 frames")

except Exception as e:
    import traceback
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n💡 Quick fixes:")
    print("   'UnetLoaderGGUF not found'          → Cell 1: clone ComfyUI_GGUF")
    print("   'LTXVSpatioTemporalTiledVAEDecode'  → Cell 1: clone ComfyUI-LTXVideo")
    print("   'DualCLIPLoader' error              → check CLIP_NAME1/CLIP_NAME2")
    print("   Deformed output in Pass 1           → change SEED and re-run Cell 6")
    print("   Persistent deformation (3× same)    → change POSITIVE_PROMPT")