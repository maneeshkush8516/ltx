# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 PRO — Complete Pipeline v1.0                                        ║
# ║  Integrates: LD-I2V + SVI-Pro + EasyPrompt + Character Consistency          ║
# ║  Base engine: LTX-2 19B Distilled GGUF Q4_K_M (Colab T4/L4/A100 safe)     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# CELL ORDER:
#   Cell 1 — Environment setup & custom nodes   (run once per session)
#   Cell 2 — Model downloads                    (run once, skip if cached)
#   Cell 3 — Imports, helpers & character system (run every session)
#   Cell 4 — Easy Prompt + Vision settings      (edit to taste)
#   Cell 5 — Character Consistency & LoRA config (edit per character)
#   Cell 6 — Video generation configuration     (edit per video)
#   Cell 7 — Define generate_pro()              (run once per session)
#   Cell 8 — Storyboard / multi-scene runner    (optional)
#   Cell 9 — Run                                (re-run for each clip)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1  ─  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 1. Prepare Environment & Install Custom Nodes
# @markdown Clones all required ComfyUI repos including:
# @markdown - **LTX2EasyPrompt-LD** (LTX2PromptArchitect + LTX2VisionDescribe)
# @markdown - **LTX2-Master-Loader** (LTX2MasterLoaderLD 10-slot LoRA stacker)
# @markdown - **ComfyUI-VideoHelperSuite** (VHS_VideoCombine output node)
# @markdown - **ComfyUI-LTXVideo** (tiled VAE decode + AV helpers)

# ── Base Python packages ──────────────────────────────────────────────────────
!pip install torch torchvision torchaudio

%cd /content
from IPython.display import clear_output
clear_output()

!pip install -q torchsde einops diffusers accelerate nest_asyncio
!pip install -q av spandrel albumentations onnx opencv-python onnxruntime
!pip install -q imageio imageio-ffmpeg

# Extra packages required by EasyPrompt & VisionDescribe nodes
!pip install -q transformers>=4.43.0 accelerate qwen-vl-utils huggingface_hub

# ── ComfyUI (pinned branch — matches reference notebook) ─────────────────────
!git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt -q
clear_output()

# ── Custom nodes ──────────────────────────────────────────────────────────────
%cd /content/ComfyUI/custom_nodes

# Core KJNodes (pinned build — ImageResizeKJv2, PathchSageAttentionKJ, etc.)
!git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
# GGUF loader (UnetLoaderGGUF)
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git
# LTXVideo nodes (LTXVImgToVideoInplace, LTXVPreprocess, tiled VAE, etc.)
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
# LTX2EasyPrompt-LD — LTX2PromptArchitect + LTX2VisionDescribe
!git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
# LTX2-Master-Loader — LTX2MasterLoaderLD (10-slot LoRA stacker)
!git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git
# VideoHelperSuite — VHS_VideoCombine (h264-mp4, crf=19, yuv420p)
!git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# ── Install node requirements ─────────────────────────────────────────────────
%cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI-LTXVideo
!pip install -r requirements.txt -q 2>/dev/null || true

%cd /content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD
!pip install -r requirements.txt -q 2>/dev/null || true

%cd /content/ComfyUI/custom_nodes/LTX2-Master-Loader
!pip install -r requirements.txt -q 2>/dev/null || true

# ── System tools ──────────────────────────────────────────────────────────────
import subprocess
def install_apt_packages():
    packages = ["aria2", "ffmpeg"]
    try:
        subprocess.run(["apt-get", "-y", "install", "-qq"] + packages,
                       check=True, capture_output=True)
        print("✓ apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ apt error: {e.stderr.decode().strip() or 'unknown'}")

print("Installing apt packages...")
install_apt_packages()

# ── Final setup ───────────────────────────────────────────────────────────────
%cd /content/ComfyUI
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

clear_output()
print("✅ Environment setup complete.")
print("   Custom nodes installed:")
print("   ✓ ComfyUI_KJNodes    (kj_1.2.6)   — ImageResizeKJv2, PathchSageAttentionKJ")
print("   ✓ ComfyUI_GGUF       (22_01_2026)  — UnetLoaderGGUF")
print("   ✓ ComfyUI-LTXVideo   (Lightricks)  — LTXVImgToVideoInplace, tiled VAE")
print("   ✓ LTX2EasyPrompt-LD  (LoRa Daddy)  — LTX2PromptArchitect, LTX2VisionDescribe")
print("   ✓ LTX2-Master-Loader (LoRa Daddy)  — LTX2MasterLoaderLD")
print("   ✓ ComfyUI-VideoHelperSuite          — VHS_VideoCombine")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2  ─  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 2. Download All Model Weights
# @markdown Uses aria2c (fast parallel download). Skips files already cached.
# @markdown **Pick ONE Gemma encoder** based on your GPU (see comments below).

import os, subprocess
from pathlib import Path

def model_download(url: str, dest_dir: str, filename: str = None,
                   silent: bool = True) -> str:
    """aria2c download with skip-if-cached logic. Returns local filename."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  ↳ cached: {filename}")
        return filename
    cmd = ["aria2c", "--console-log-level=error",
           "-c", "-x", "16", "-s", "16", "-k", "1M",
           "-d", dest_dir, "-o", filename]
    if silent:
        cmd += ["--summary-interval=0", "--quiet"]
        print(f"  ↓ {filename}...", end=" ", flush=True)
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ❌ Failed: {result.stderr.strip()}")
        return False
    if silent:
        print("done.")
    return filename

# ── Source base URLs ──────────────────────────────────────────────────────────
KIJAI    = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
KIJAI23  = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"

print("── Core model downloads ──────────────────────────────────────────────────")

# ── UNet: GGUF Q4_K_M distilled ──────────────────────────────────────────────
# LTX-2 19B distilled — baked-in distillation, no separate distill LoRA needed
# Node [197] UnetLoaderGGUF in LD-I2V.json
dit_model = model_download(
    f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
    "/content/ComfyUI/models/unet")

# ── Text encoders ─────────────────────────────────────────────────────────────
# Gemma fp4 — RTX 5000 Blackwell. Use fp8 for T4/A100 (uncomment below).
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")
# Gemma fp8 — T4 / A100 / RTX 3000-4000 (uncomment if fp4 OOMs):
# text_encoder_model = model_download(
#     f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
#     "/content/ComfyUI/models/text_encoders")

# Embeddings connector — distilled version (must match GGUF)
text_encoder2_model = model_download(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders")

# ── VAEs ──────────────────────────────────────────────────────────────────────
vae_model = model_download(
    f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

vae_audio_model = model_download(
    f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

# TaeEncoder preview VAE (fast latent preview)
taeltx2_model = model_download(
    f"{KIJAI23}/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae")

# ── Spatial upscaler ──────────────────────────────────────────────────────────
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

# ── IC LoRAs + Camera Control LoRAs ──────────────────────────────────────────
# All used by LTX2MasterLoaderLD node [263] in LD-I2V.json
LORA_URLS = {
    "Detailer":    f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":        f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Left":  f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Out":   f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Right": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Down":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Jib-Up":      f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Static":      f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}

LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(LORA_DIR, exist_ok=True)
print(f"\n── LoRA batch download ({len(LORA_URLS)} files) ─────────────────────────────────")
for name, url in LORA_URLS.items():
    r = model_download(url, LORA_DIR)
    print(f"   {'✅' if r else '❌'}  {name}")

print("\n✅ All model files downloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3  ─  IMPORTS, HELPERS & CHARACTER CONSISTENCY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 3. Imports, Helpers & Character Consistency System
# @markdown Loads all helpers, VRAM utilities, node wrappers, and the
# @markdown **Character Consistency** system used in `generate_pro()`.

import os, sys, gc, time, json, shutil, warnings, subprocess, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, Image as IPImage, clear_output
from google.colab import files

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

# ── ComfyUI core ──────────────────────────────────────────────────────────────
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# ── Async node loader (Jupyter/Colab safe) ────────────────────────────────────
def import_custom_nodes() -> None:
    """Load all built-in and external custom nodes in a Jupyter/Colab-safe way."""
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def _load():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print(f"   ⚠️  Some nodes failed: {[str(n) for n in failed]}")
    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())

# ── VRAM helpers ──────────────────────────────────────────────────────────────
def cleanup_memory(verbose: bool = False) -> None:
    """Enhanced memory cleanup including ipc_collect for fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()   # free IPC handles - reduces fragmentation
    if verbose:
        _print_vram()

def aggressive_cleanup(label: str = "") -> None:
    """Aggressive VRAM cleanup: double gc + synchronize + empty_cache + ipc_collect."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    if label:
        _print_vram(label)

def _print_vram(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct   = used / total * 100 if total > 0 else 0
    filled = int(20 * used / total) if total > 0 else 0
    bar   = "█" * filled + "░" * (20 - filled)
    tag   = f" [{label}]" if label else ""
    print(f"   💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%){tag}")

# ── ComfyUI node output accessor ──────────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# ── Tensor / image conversion ─────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → ComfyUI NHWC float tensor."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """ComfyUI NHWC tensor → PIL."""
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")

def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    """Load an image file as a ComfyUI NHWC tensor. Returns None if missing."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))

def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    """Extract last frame of a video as NHWC float tensor shape (1,H,W,3)."""
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
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0

# ── Overlap / Segment Extension helpers (SVI-Pro-Workflow.json techniques) ────

def blend_overlap_frames(source_frames: torch.Tensor, new_frames: torch.Tensor,
                         overlap: int = 5, mode: str = "linear_blend",
                         side: str = "source") -> torch.Tensor:
    """
    Blend overlapping frames between two video segments for seamless transitions.
    Mirrors ImageBatchExtendWithOverlap from comfyui-kjnodes (SVI-Pro-Workflow.json).
    
    Args:
        source_frames: Previous segment frames tensor (T, H, W, C) or (N, T, H, W, C)
        new_frames: New segment frames tensor (same format)
        overlap: Number of overlapping frames (SVI-Pro default: 5)
        mode: Blend mode - "linear_blend", "hard_cut", or "crossfade"
        side: Which segment contributes overlap - "source" or "target"
    
    Returns:
        Combined frames tensor with seamless blend at the junction
    """
    # Ensure we're working with 4D tensors (T, H, W, C)
    if source_frames.ndim == 5:
        source_frames = source_frames.squeeze(0)
    if new_frames.ndim == 5:
        new_frames = new_frames.squeeze(0)
    
    if overlap <= 0 or overlap >= min(len(source_frames), len(new_frames)):
        # No valid overlap - just concatenate
        return torch.cat([source_frames, new_frames], dim=0)
    
    if mode == "hard_cut":
        # No blending - take source frames up to overlap, then new frames after
        if side == "source":
            return torch.cat([source_frames, new_frames[overlap:]], dim=0)
        else:
            return torch.cat([source_frames[:-overlap], new_frames], dim=0)
    
    elif mode == "linear_blend":
        # Linear blend in overlap region (SVI-Pro default)
        # Source provides the "tail" frames, new provides the "head" frames
        source_tail = source_frames[-overlap:]  # last N frames of source
        new_head = new_frames[:overlap]          # first N frames of new
        
        # Create linear blend weights
        weights = torch.linspace(1.0, 0.0, overlap, device=source_frames.device)
        weights = weights.view(-1, 1, 1, 1)  # (overlap, 1, 1, 1) for broadcasting
        
        # Blend: source_weight decreases, new_weight increases
        blended = source_tail * weights + new_head * (1.0 - weights)
        
        # Assemble: source (minus tail) + blended region + new (minus head)
        result = torch.cat([
            source_frames[:-overlap],
            blended,
            new_frames[overlap:]
        ], dim=0)
        return result
    
    elif mode == "crossfade":
        # Equal-weight crossfade (smoother than linear for some content)
        source_tail = source_frames[-overlap:]
        new_head = new_frames[:overlap]
        
        # Sigmoid-like weights for smoother transition
        t = torch.linspace(0.0, 1.0, overlap, device=source_frames.device)
        weights = 0.5 * (1.0 - torch.cos(t * 3.14159))  # cosine interpolation
        weights = weights.view(-1, 1, 1, 1)
        
        blended = source_tail * (1.0 - weights) + new_head * weights
        
        result = torch.cat([
            source_frames[:-overlap],
            blended,
            new_frames[overlap:]
        ], dim=0)
        return result
    
    else:
        # Unknown mode - fallback to linear_blend
        return blend_overlap_frames(source_frames, new_frames, overlap, "linear_blend", side)


def extract_anchor_frame(frames: torch.Tensor, position: str = "last") -> torch.Tensor:
    """
    Extract a single frame from a video tensor for use as anchor/seed.
    Used by segment extension to chain segments with character consistency.
    
    Args:
        frames: Video frames tensor (T, H, W, C) or (N, T, H, W, C)
        position: "last", "first", or integer frame index
    
    Returns:
        Single frame tensor (1, H, W, C) suitable for I2V/anchor conditioning
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    
    if position == "last":
        return frames[-1:].clone()
    elif position == "first":
        return frames[:1].clone()
    elif isinstance(position, int):
        idx = min(position, len(frames) - 1)
        return frames[idx:idx+1].clone()
    else:
        return frames[-1:].clone()


def compute_segment_seeds(base_seed: int, num_segments: int, 
                          mode: str = "fixed") -> list:
    """
    Compute seeds for each segment based on mode.
    SVI-Pro uses fixed seed=2025 for all segments.
    
    Args:
        base_seed: Starting seed value
        num_segments: Number of segments
        mode: "fixed", "increment", or "random"
    
    Returns:
        List of seed values, one per segment
    """
    if mode == "fixed":
        return [base_seed] * num_segments
    elif mode == "increment":
        return [base_seed + i for i in range(num_segments)]
    elif mode == "random":
        import random as _rng
        _rng.seed(base_seed)
        return [_rng.randint(0, 2**32 - 1) for _ in range(num_segments)]
    else:
        return [base_seed] * num_segments


# ── Video display & saving ────────────────────────────────────────────────────
def display_video(path: str) -> None:
    if not path or not os.path.exists(path):
        print(f"   ⚠️  Not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=800 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        '</video>'
    ))

def save_video_from_components(video_obj, prefix="LTX-2-PRO") -> str:
    """Save a ComfyUI video object and return the output path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext  = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path

# ── Metadata JSON sidecar ─────────────────────────────────────────────────────
def save_metadata_sidecar(output_path: str, meta: dict) -> str:
    """Write a .json sidecar file next to the generated video."""
    sidecar = os.path.splitext(output_path)[0] + "_meta.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"   📄 Metadata: {sidecar}")
    except Exception as e:
        print(f"   ⚠️  Could not save metadata: {e}")
    return sidecar

# ── Model file validator ──────────────────────────────────────────────────────
def validate_model_files(model_dict: dict) -> bool:
    """
    Check required model files exist in ComfyUI folder_paths.
    model_dict: { label: filename }
    Returns True only if all files are found.
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
        fk    = folder_map.get(label, "loras")
        found = any(
            os.path.exists(os.path.join(base, filename))
            for base in folder_paths.get_folder_paths(fk)
        )
        print(f"   {'✅' if found else '❌'} [{label:9s}] {filename}")
        if not found:
            ok = False
    return ok

# ── Memory / attention performance patches ────────────────────────────────────
def apply_sage_attention(unet):
    """
    Apply PathchSageAttentionKJ (KJNodes) for flash-attention-style speedup.
    Node: PathchSageAttentionKJ from ComfyUI_KJNodes.
    Falls back silently if node is not available.
    """
    if not USE_SAGE_ATTENTION:
        return unet
    if "PathchSageAttentionKJ" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  PathchSageAttentionKJ not found — skipping sage attention.")
        return unet
    try:
        node  = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        fn    = getattr(node, node.FUNCTION)
        # sage_attention arg required by KJNodes (SVI-Pro uses "auto")
        unet  = get_value_at_index(fn(model=unet, sage_attention="auto"), 0)
        print("   ✓ SageAttention patch applied (PathchSageAttentionKJ, mode=auto)")
    except Exception as e:
        print(f"   ⚠️  SageAttention failed ({e}) — continuing without it.")
    return unet

def apply_chunk_ff(unet):
    """
    Apply LTXVChunkFeedForward for memory-efficient chunk-based feedforward.
    Node: LTXVChunkFeedForward from ComfyUI-LTXVideo.
    Falls back silently if node is not available.
    """
    if not USE_CHUNK_FF:
        return unet
    if "LTXVChunkFeedForward" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTXVChunkFeedForward not found — skipping chunk FF.")
        return unet
    try:
        node  = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
        fn    = getattr(node, node.FUNCTION)
        unet  = get_value_at_index(fn(model=unet), 0)
        print("   ✓ ChunkFeedForward patch applied (LTXVChunkFeedForward)")
    except Exception as e:
        print(f"   ⚠️  ChunkFeedForward failed ({e}) — continuing without it.")
    return unet

def purge_vram(label: str = "") -> None:
    """
    Purge VRAM after model loading phases.
    Tries LayerUtility: PurgeVRAM V2 first, then torch.cuda.empty_cache() fallback.
    Node: LayerUtility: PurgeVRAM V2
    """
    if not PURGE_VRAM_AFTER_MODELS:
        return
    tag = f" [{label}]" if label else ""
    if "LayerUtility: PurgeVRAM V2" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM V2"]()
            fn   = getattr(node, node.FUNCTION)
            fn(anything="", purge_cache=True, purge_models=True)
            print(f"   ✓ VRAM purged via PurgeVRAM V2{tag}")
            return
        except Exception as e:
            print(f"   ⚠️  PurgeVRAM V2 failed ({e}) — using torch fallback.")
    cleanup_memory()
    print(f"   ✓ VRAM cleared via torch.cuda.empty_cache{tag}")

# ── Upload helper ─────────────────────────────────────────────────────────────
def upload_image(save_dir="/content/ComfyUI/input") -> Optional[str]:
    os.makedirs(save_dir, exist_ok=True)
    uploaded = files.upload()
    for fname, data in uploaded.items():
        path = os.path.join(save_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        print(f"   ✓ Saved: {path}")
        return path
    return None

# ── Audio VAE loader with KJNodes fallback ────────────────────────────────────
def _load_audio_vae(vae_name: str):
    """Load audio VAE. Prefers VAELoaderKJ (main_device, fp16), falls back to VAELoader."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=vae_name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)


# ──────────────────────────────────────────────────────────────────────────────
# LoRA stack helpers (LTX2MasterLoaderLD + manual fallback)
# Mirrors node [263] LTX2MasterLoaderLD in LD-I2V.json
# ──────────────────────────────────────────────────────────────────────────────

# IC LoRA filename lookup (slot 1)
_IC_LORA_FILES: Dict[str, str] = {
    "none":     "None",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    "canny":    "ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth":    "ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose":     "ltx-2-19b-ic-lora-pose-control.safetensors",
}

# Camera LoRA filename lookup (slot 2)
_CAMERA_LORA_FILES: Dict[str, str] = {
    "none":        "None",
    "dolly-in":    "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly-out":   "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly-left":  "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib-up":      "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib-down":    "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static":      "ltx-2-19b-lora-camera-control-static.safetensors",
}

def _build_lora_stack(ic_lora: str, ic_strength: float,
                      camera_lora: str, camera_strength: float) -> List[Dict]:
    """
    Build the 10-slot LoRA stack from IC and Camera dropdown selections.
    Slot 1 = IC LoRA, Slot 2 = Camera LoRA, Slots 3-10 = empty.
    """
    ic_file  = _IC_LORA_FILES.get(ic_lora.lower(), "None")
    cam_file = _CAMERA_LORA_FILES.get(camera_lora.lower(), "None")
    stack = [
        {"on": ic_file  != "None", "lora": ic_file,  "guard": False, "strength": ic_strength},
        {"on": cam_file != "None", "lora": cam_file, "guard": False, "strength": camera_strength},
    ]
    for _ in range(8):
        stack.append({"on": False, "lora": "None", "guard": False, "strength": 1.0})
    return stack


def apply_lora_stack(unet, clip_model,
                     lora_stack: Optional[List[Dict]] = None,
                     lora_stack_json: Optional[str] = None):
    """
    Apply LoRA stack via LTX2MasterLoaderLD node when available.
    Falls back to manual LoraLoaderModelOnly loop if the node is missing.
    Returns: (unet, clip_model)
    """
    stack = lora_stack or []
    active = [s for s in stack
              if s.get("on") and s.get("lora") not in (None, "None", "")]

    if not active:
        print("   ℹ️  No active LoRAs in stack — skipping.")
        return unet, clip_model

    # ── Try LTX2MasterLoaderLD node [263] (LoRa Daddy) ───────────────────────
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS and clip_model is not None:
        print(f"   [MasterLoader] {len(active)} LoRA(s) via LTX2MasterLoaderLD…")
        try:
            node   = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            fn     = getattr(node, node.FUNCTION)
            # NOTE: 'stack_data' is the expected kwarg name for LTX2-Master-Loader.
            # If the node's API differs (e.g. 'lora_stack'), a TypeError is raised
            # and caught below — the manual fallback loop is then used instead.
            result = fn(
                model=unet,
                clip=clip_model,
                stack_data=lora_stack_json or json.dumps(stack),
            )
            unet = get_value_at_index(result, 0)
            print("   [MasterLoader] ✓  Stack applied.")
            return unet, clip_model
        except TypeError as e:
            print(f"   [MasterLoader] ⚠️  kwarg mismatch ({e}).")
            print("      Falling back to manual LoRA loop.")
            print("      Check LTX2-Master-Loader node signature — expected 'stack_data'.")
        except Exception as e:
            print(f"   [MasterLoader] ⚠️  Node failed ({e}) — manual fallback.")

    # ── Fallback: LoraLoaderModelOnly loop ────────────────────────────────────
    print(f"   [MasterLoader] {len(active)} LoRA(s) via manual loop…")
    for slot in active:
        name, strength, guard = slot["lora"], slot.get("strength", 1.0), slot.get("guard", False)
        try:
            ll   = LoraLoaderModelOnly()
            unet = ll.load_lora_model_only(unet, name, strength)[0]
            print(f"      ✓ {name} @ {strength}")
        except Exception as e:
            if guard:
                print(f"      ⚠️  {name} skipped (guard): {e}")
            else:
                print(f"      ❌ {name} failed: {e}")

    return unet, clip_model


# ──────────────────────────────────────────────────────────────────────────────
# EasyPrompt + VisionDescribe wrappers (from LTX2EasyPrompt-LD nodes)
# Maps to LTX2PromptArchitect + LTX2VisionDescribe node types
# ──────────────────────────────────────────────────────────────────────────────

_LLM_LABEL_MAP = {
    "8B":  "8B - NeuralDaredevil (High Quality)",
    "3B":  "3B - Llama-3.2 Abliterated (Low VRAM)",
    "14B": "14B - Qwen3 Abliterated (High VRAM)",
}
_VISION_LABEL_MAP = {
    "3B-fast": "Qwen2.5-VL-3B — Fast (huihui abliterated)",
    "7B-nsfw": "Qwen2.5-VL-7B — Better NSFW (prithiv caption)",
}
_CREATIVITY_MAP = {
    0.7: "0.7 - Literal & Grounded",
    0.9: "0.9 - Balanced Professional",
    1.1: "1.1 - Artistic Expansion",
}

def _creativity_label(c: float) -> str:
    closest = min(_CREATIVITY_MAP.keys(), key=lambda x: abs(x - c))
    return _CREATIVITY_MAP[closest]


def run_easy_prompt(user_input: str, frame_count: int, seed: int,
                    scene_context: str = "",
                    llm_model_override: str = None) -> Tuple[str, str]:
    """
    Calls LTX2PromptArchitect (node type: LTX2PromptArchitect from LTX2EasyPrompt-LD)
    to expand a simple story description into a dense cinematic prompt.

    Falls back to returning the raw input if the node is unavailable.
    LLM is loaded, run, then unloaded to free VRAM for the video model.

    llm_model_override: when provided, overrides the LLM_MODEL global for this call.
    Returns: (positive_prompt, negative_prompt)
    """
    if "LTX2PromptArchitect" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTX2PromptArchitect not found — using raw user_input.")
        return user_input, ""

    _model = llm_model_override if llm_model_override is not None else LLM_MODEL
    print(f"   [EasyPrompt] LLM={_model} | creativity={CREATIVITY} | frames={frame_count}")
    node = NODE_CLASS_MAPPINGS["LTX2PromptArchitect"]()
    _offline = LLM_OFFLINE_MODE
    _lp_3b  = LOCAL_PATH_3B  if _offline else ""
    _lp_8b  = LOCAL_PATH_8B  if _offline else ""
    _lp_14b = LOCAL_PATH_14B if _offline else ""
    if _offline:
        print(f"   [EasyPrompt] Offline mode: loading from local path...")
    result = node.generate(
        bypass=False,
        user_input=user_input,
        creativity=_creativity_label(CREATIVITY),
        seed=seed,
        invent_dialogue=INVENT_DIALOGUE,
        keep_model_loaded=False,
        offline_mode=_offline,
        frame_count=frame_count,
        model=_LLM_LABEL_MAP.get(_model, "8B - NeuralDaredevil (High Quality)"),
        local_path_8b=_lp_8b,
        local_path_3b=_lp_3b,
        local_path_14b=_lp_14b,
        scene_context=scene_context,
        lora_triggers=LORA_TRIGGERS,
    )
    prompt     = result[0]  # PROMPT output
    neg_prompt = result[2]  # NEG_PROMPT output
    print(f"   [EasyPrompt] ✓  {len(prompt.split())} words generated.")
    cleanup_memory()
    return prompt, neg_prompt


def run_vision_describe(image_tensor: torch.Tensor,
                        character_desc: str = "",
                        use_vision_override: bool = None,
                        vision_model_override: str = None) -> str:
    """
    Calls LTX2VisionDescribe (node type: LTX2VisionDescribe from LTX2EasyPrompt-LD)
    to analyse the image and return a scene description for use as scene_context.
    character_desc is prepended to seed the analysis toward the character.

    use_vision_override: when provided, overrides the USE_VISION global for this call.
    vision_model_override: when provided, overrides the VISION_MODEL global for this call.
    Returns: scene_context string (empty string on failure).
    """
    _use_v   = use_vision_override   if use_vision_override   is not None else USE_VISION
    _vis_mod = vision_model_override if vision_model_override is not None else VISION_MODEL

    if not _use_v:
        return character_desc
    if "LTX2VisionDescribe" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTX2VisionDescribe not found — skipping vision analysis.")
        return character_desc

    print(f"   [VisionDescribe] model={_vis_mod} | image shape={image_tensor.shape}")
    node = NODE_CLASS_MAPPINGS["LTX2VisionDescribe"]()
    _vis_offline = VISION_OFFLINE_MODE
    _vis_local   = VISION_LOCAL_PATH if _vis_offline else ""
    if _vis_offline:
        print(f"   [VisionDescribe] Offline mode: loading from {_vis_local}")
    result = node.describe(
        image=image_tensor,
        model_name=_VISION_LABEL_MAP.get(_vis_mod, "Qwen2.5-VL-3B — Fast (huihui abliterated)"),
        offline_mode=_vis_offline,
        local_path=_vis_local,
    )
    ctx = result[0]
    if character_desc:
        ctx = character_desc + " " + ctx
    print(f"   [VisionDescribe] ✓  {len(ctx.split())} words.")
    cleanup_memory()
    return ctx



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Multi-frame Latent Conditioning
# ══════════════════════════════════════════════════════════════════════════════

def extract_multi_anchor_frames(frames: torch.Tensor, n: int = 3,
                                strategy: str = 'last_n') -> torch.Tensor:
    """
    Extract multiple anchor frames for conditioning.

    Strategies:
        'last_n'  - Extract the last N frames (default for segment chaining)
        'uniform' - Uniformly sample N frames across the sequence
        'keyframe'- Pick frames with highest inter-frame difference

    Args:
        frames: Video frames tensor (T, H, W, C) or (N, T, H, W, C)
        n: Number of anchor frames to extract
        strategy: Extraction strategy

    Returns:
        Tensor of shape (n, H, W, C) with selected anchor frames
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    n = min(n, T)

    if strategy == 'last_n':
        return frames[-n:].clone()
    elif strategy == 'uniform':
        indices = torch.linspace(0, T - 1, n).long()
        return frames[indices].clone()
    elif strategy == 'keyframe':
        # Select frames with largest difference from neighbors
        if T <= n:
            return frames.clone()
        diffs = []
        for i in range(1, T):
            diff = (frames[i].float() - frames[i - 1].float()).abs().mean().item()
            diffs.append((diff, i))
        diffs.sort(key=lambda x: x[0], reverse=True)
        indices = sorted([d[1] for d in diffs[:n]])
        return frames[torch.tensor(indices)].clone()
    else:
        return frames[-n:].clone()


class CharacterEmbeddingBank:
    """
    Accumulates compact frame-level features across segments for consistent generation.

    Maintains a running average of spatially-pooled frame features (not raw pixels
    or model embeddings) that can be used to condition subsequent segments for
    character consistency. Features are derived by spatial-mean pooling of frame
    tensors to create a compact per-frame representation.
    """

    def __init__(self):
        self._embeddings: List[torch.Tensor] = []
        self._max_entries: int = 50

    def accumulate(self, features: torch.Tensor) -> None:
        """Add new feature representation to the bank (spatial-mean pooled frame features)."""
        self._embeddings.append(features.detach().cpu())
        if len(self._embeddings) > self._max_entries:
            self._embeddings = self._embeddings[-self._max_entries:]

    def get_average_embedding(self) -> Optional[torch.Tensor]:
        """Return the mean embedding across all accumulated features."""
        if not self._embeddings:
            return None
        stacked = torch.stack(self._embeddings, dim=0)
        return stacked.mean(dim=0)

    def reset(self) -> None:
        """Clear all accumulated embeddings."""
        self._embeddings = []

    def __len__(self) -> int:
        return len(self._embeddings)


def create_style_lock(anchor_frames: torch.Tensor,
                      mode: str = 'latent_average') -> torch.Tensor:
    """
    Average multiple anchor frame latents to create a style lock constraint.

    Args:
        anchor_frames: Tensor of anchor frames (N, H, W, C) or (N, C, H, W)
        mode: 'latent_average' averages all frames, 'weighted' weights recent higher

    Returns:
        Single averaged frame tensor usable as style reference
    """
    if anchor_frames.ndim < 3:
        return anchor_frames
    if anchor_frames.ndim == 3:
        return anchor_frames.unsqueeze(0)

    N = anchor_frames.shape[0]
    if mode == 'latent_average':
        return anchor_frames.mean(dim=0, keepdim=True)
    elif mode == 'weighted':
        # Exponentially weight recent frames higher
        weights = torch.exp(torch.linspace(-1.0, 0.0, N))
        weights = weights / weights.sum()
        weights = weights.view(N, 1, 1, 1).to(anchor_frames.device)
        return (anchor_frames * weights).sum(dim=0, keepdim=True)
    else:
        return anchor_frames.mean(dim=0, keepdim=True)



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Dual-Anchor Continuity System
# ══════════════════════════════════════════════════════════════════════════════

def extract_continuity_composite(video_path: str, n_frames: int = 5,
                                 mode: str = 'weighted_average') -> Optional[torch.Tensor]:
    """
    Extract last N frames from a video and create a weighted composite tensor.

    Used by the dual-anchor system to produce a high-quality continuity frame
    that captures the temporal state at the end of a scene. A weighted composite
    reduces noise/flicker artifacts compared to a single last frame.

    Args:
        video_path: Path to the source video file.
        n_frames: Number of frames to extract from the end of the video.
        mode: 'weighted_average' - later frames get linearly higher weight.
              'last_frame'       - only return the very last frame (legacy).

    Returns:
        NHWC float tensor (1, H, W, 3) in [0,1] range, or None on failure.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                return None

            n = min(n_frames, total)

            if mode == 'last_frame' or n == 1:
                # Legacy single-frame behavior
                cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
                ok, frame = cap.read()
                if not ok:
                    return None
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(frame).float().unsqueeze(0) / 255.0

            # Extract last n frames
            start_idx = total - n
            frames_list = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for _ in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame).float() / 255.0)

            if not frames_list:
                return None

            # weighted_average: linear weights (1, 2, 3, ..., n) normalized
            n_actual = len(frames_list)
            weights = torch.arange(1, n_actual + 1, dtype=torch.float32)
            weights = weights / weights.sum()
            # frames_list items are (H, W, 3); stack to (N, H, W, 3)
            stacked = torch.stack(frames_list, dim=0)
            # Apply weights along dim 0
            weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
            return weighted.unsqueeze(0)  # (1, H, W, 3)
        finally:
            cap.release()
    except Exception:
        return None


def save_continuity_frame(tensor: torch.Tensor, path: str, format: str = 'png') -> bool:
    """
    Save a frame tensor to disk as PNG or JPEG.

    Args:
        tensor: NHWC float tensor (1, H, W, 3) in [0,1] range.
        path: Output file path.
        format: 'png' for lossless, 'jpg' for compressed (quality=98).

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        if tensor.ndim == 4:
            tensor = tensor[0]
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, "RGB")
        if format.lower() in ('jpg', 'jpeg'):
            pil_img.save(path, "JPEG", quality=98)
        else:
            pil_img.save(path, "PNG")
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Motion Coherence System
# ══════════════════════════════════════════════════════════════════════════════

def estimate_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Estimate optical flow between two frames using Farneback method.

    Args:
        frame1: First frame as numpy array (H, W, 3) uint8 or float
        frame2: Second frame as numpy array (H, W, 3) uint8 or float

    Returns:
        Optical flow array of shape (H, W, 2) with (dx, dy) per pixel
    """
    if frame1.dtype == np.float32 or frame1.dtype == np.float64:
        frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
    if frame2.dtype == np.float32 or frame2.dtype == np.float64:
        frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def detect_motion_direction(flow: np.ndarray) -> str:
    """
    Analyze optical flow to determine dominant motion direction.

    Args:
        flow: Optical flow array (H, W, 2)

    Returns:
        One of: 'left', 'right', 'up', 'down', 'zoom_in', 'zoom_out', 'static'
    """
    h, w = flow.shape[:2]
    mean_dx = flow[:, :, 0].mean()
    mean_dy = flow[:, :, 1].mean()

    # Check for zoom by comparing center vs edge flow magnitudes
    center_region = flow[h // 4:3 * h // 4, w // 4:3 * w // 4]
    edge_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).mean()
    center_mag = np.sqrt(center_region[:, :, 0] ** 2 + center_region[:, :, 1] ** 2).mean()

    # Threshold for considering motion significant
    threshold = 1.0

    if edge_mag < threshold and center_mag < threshold:
        return 'static'

    # Zoom detection: edges diverge from center
    if edge_mag > center_mag * 1.5 and edge_mag > threshold:
        return 'zoom_out'
    if center_mag > edge_mag * 1.5 and center_mag > threshold:
        return 'zoom_in'

    # Directional detection
    if abs(mean_dx) > abs(mean_dy):
        return 'right' if mean_dx > 0 else 'left'
    else:
        return 'down' if mean_dy > 0 else 'up'


def auto_select_camera_lora(motion_direction: str) -> str:
    """
    Map detected motion direction to appropriate camera LoRA name.

    Uses the _CAMERA_LORA_FILES dict to select a matching LoRA.

    Args:
        motion_direction: Output from detect_motion_direction()

    Returns:
        Camera LoRA key string (e.g. 'dolly-left', 'static')
    """
    direction_to_lora = {
        'left': 'dolly-left',
        'right': 'dolly-right',
        'up': 'jib-up',
        'down': 'jib-down',
        'zoom_in': 'dolly-in',
        'zoom_out': 'dolly-out',
        'static': 'static',
    }
    return direction_to_lora.get(motion_direction, 'static')


def compute_velocity_latent(frame_minus2: torch.Tensor,
                            frame_minus1: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity vector (frame[-1] - frame[-2]) for motion injection.

    This velocity latent can be added to the last frame to extrapolate
    motion direction into the next segment.

    Args:
        frame_minus2: Second-to-last frame tensor (1, H, W, C) or (H, W, C)
        frame_minus1: Last frame tensor (same shape)

    Returns:
        Velocity tensor (same shape as input) representing motion delta
    """
    return (frame_minus1.float() - frame_minus2.float())



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Adaptive Overlap
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_overlap(prev_frames: torch.Tensor,
                             min_overlap: int = 2,
                             max_overlap: int = 10) -> int:
    """
    Compute overlap frames based on motion magnitude.

    High motion scenes need fewer overlap frames (to avoid ghosting),
    while low motion scenes benefit from more overlap (smoother blend).

    Args:
        prev_frames: Previous segment frames (T, H, W, C)
        min_overlap: Minimum overlap frames (for high motion)
        max_overlap: Maximum overlap frames (for low/no motion)

    Returns:
        Recommended number of overlap frames
    """
    if prev_frames.ndim == 5:
        prev_frames = prev_frames.squeeze(0)

    T = prev_frames.shape[0]
    if T < 2:
        return max_overlap

    # Compute average frame-to-frame difference over last few frames
    num_check = min(5, T - 1)
    diffs = []
    for i in range(T - num_check, T):
        diff = (prev_frames[i].float() - prev_frames[i - 1].float()).abs().mean().item()
        diffs.append(diff)

    avg_motion = sum(diffs) / len(diffs) if diffs else 0.0

    # Map motion to overlap: high motion -> min_overlap, low motion -> max_overlap
    # Typical frame diff range: 0.0 (static) to ~0.15 (fast motion)
    motion_normalized = min(avg_motion / 0.10, 1.0)
    overlap = int(max_overlap - motion_normalized * (max_overlap - min_overlap))
    return max(min_overlap, min(max_overlap, overlap))


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Quality Gate
# ══════════════════════════════════════════════════════════════════════════════

def compute_frame_ssim(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """
    Compute structural similarity between two frames (0-1 scale).

    Uses a simplified SSIM approximation without scipy dependency.

    Args:
        frame1: Frame tensor (H, W, C) or (1, H, W, C)
        frame2: Frame tensor (same shape)

    Returns:
        SSIM score between 0.0 and 1.0 (higher = more similar)
    """
    if frame1.ndim == 4:
        frame1 = frame1.squeeze(0)
    if frame2.ndim == 4:
        frame2 = frame2.squeeze(0)

    f1 = frame1.float()
    f2 = frame2.float()

    mu1 = f1.mean()
    mu2 = f2.mean()
    sigma1_sq = ((f1 - mu1) ** 2).mean()
    sigma2_sq = ((f2 - mu2) ** 2).mean()
    sigma12 = ((f1 - mu1) * (f2 - mu2)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_val = (numerator / denominator).item()
    return max(0.0, min(1.0, ssim_val))


def compute_histogram_consistency(frames: torch.Tensor) -> float:
    """
    Check color histogram consistency across frames (0-1 scale).

    Compares the mean color distribution of each frame to the overall mean.
    Higher score means more consistent color across the segment.

    Args:
        frames: Video frames tensor (T, H, W, C)

    Returns:
        Consistency score between 0.0 and 1.0 (higher = more consistent)
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    if T < 2:
        return 1.0

    # Compute per-frame mean color
    frame_means = frames.float().mean(dim=(1, 2))  # (T, C)
    overall_mean = frame_means.mean(dim=0)  # (C,)

    # Compute deviation from overall mean
    deviations = (frame_means - overall_mean).abs().mean().item()

    # Map to 0-1 score (lower deviation = higher consistency)
    score = max(0.0, 1.0 - deviations * 10.0)
    return score


def compute_artifact_score(frames: torch.Tensor) -> float:
    """
    Detect artifacts via variance analysis (low = good, high = artifacts).

    Looks for sudden spikes in local variance that indicate generation artifacts.

    Args:
        frames: Video frames tensor (T, H, W, C)

    Returns:
        Artifact score between 0.0 and 1.0 (lower = fewer artifacts)
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    if T < 2:
        return 0.0

    # Compute per-frame variance
    variances = []
    for i in range(T):
        v = frames[i].float().var().item()
        variances.append(v)

    # Detect variance spikes (potential artifacts)
    mean_var = sum(variances) / len(variances)
    if mean_var < 1e-8:
        return 0.0

    max_deviation = max(abs(v - mean_var) for v in variances)
    score = min(1.0, max_deviation / (mean_var + 1e-8) * 0.5)
    return score


def compute_segment_quality(frames_tensor: torch.Tensor,
                            overlap_region: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Compute comprehensive quality metrics for a generated segment.

    Args:
        frames_tensor: Generated video frames (T, H, W, C)
        overlap_region: Optional overlap frames from previous segment for SSIM check

    Returns:
        Dict with keys: 'ssim', 'histogram', 'variance', 'passed' (bool)
    """
    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    # SSIM between overlap regions
    ssim_score = 1.0
    if overlap_region is not None and overlap_region.shape[0] > 0:
        n_overlap = min(overlap_region.shape[0], frames_tensor.shape[0])
        ssim_scores = []
        for i in range(n_overlap):
            s = compute_frame_ssim(overlap_region[i], frames_tensor[i])
            ssim_scores.append(s)
        ssim_score = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 1.0

    histogram_score = compute_histogram_consistency(frames_tensor)
    artifact_score = compute_artifact_score(frames_tensor)

    # Default thresholds
    passed = (ssim_score > 0.5 and histogram_score > 0.4 and artifact_score < 0.6)

    return {
        'ssim': ssim_score,
        'histogram': histogram_score,
        'variance': artifact_score,
        'passed': passed,
    }


def quality_gate_check(quality_scores: Dict[str, Any],
                       thresholds: Dict[str, float]) -> bool:
    """
    Return True if all quality metrics pass their thresholds.

    Args:
        quality_scores: Dict from compute_segment_quality()
        thresholds: Dict mapping metric names to threshold values
            e.g. {'ssim': 0.5, 'histogram': 0.4, 'variance': 0.6}

    Returns:
        True if all metrics pass, False otherwise
    """
    for metric, threshold in thresholds.items():
        score = quality_scores.get(metric)
        if score is None:
            continue
        # For variance/artifact, lower is better
        if metric in ('variance', 'artifact'):
            if score > threshold:
                return False
        else:
            if score < threshold:
                return False
    return True



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Multi-Resolution Strategy
# ══════════════════════════════════════════════════════════════════════════════

def detect_shot_type(prompt: str) -> str:
    """
    Detect shot type from prompt keywords.

    Args:
        prompt: Text prompt for the scene

    Returns:
        One of: 'wide', 'closeup', 'transition', 'normal'
    """
    prompt_lower = prompt.lower()

    wide_keywords = ['wide', 'establishing', 'landscape', 'panorama', 'aerial',
                     'drone', 'vista', 'skyline', 'horizon']
    closeup_keywords = ['close', 'closeup', 'close-up', 'face', 'portrait',
                        'detail', 'macro', 'eye', 'lips', 'hands']
    transition_keywords = ['transition', 'blur', 'sweep', 'whip', 'flash',
                           'dissolve', 'wipe', 'fade']

    for kw in transition_keywords:
        if kw in prompt_lower:
            return 'transition'
    for kw in closeup_keywords:
        if kw in prompt_lower:
            return 'closeup'
    for kw in wide_keywords:
        if kw in prompt_lower:
            return 'wide'

    return 'normal'


def get_resolution_for_shot(shot_type: str, base_width: int,
                            base_height: int) -> Tuple[int, int, float]:
    """
    Get resolution and anchor weight for shot type.

    Args:
        shot_type: Output from detect_shot_type()
        base_width: Base generation width
        base_height: Base generation height

    Returns:
        Tuple of (width, height, anchor_weight)
    """
    if shot_type == 'wide':
        return base_width, base_height, 0.6
    elif shot_type == 'closeup':
        return base_width, base_height, 0.9
    elif shot_type == 'transition':
        # Half resolution for transitions (faster, less detail needed)
        w = max(256, (base_width // 2) // 32 * 32)
        h = max(256, (base_height // 2) // 32 * 32)
        return w, h, 0.3
    else:
        return base_width, base_height, 0.7


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Persistent Model Context (SVIProContext)
# ══════════════════════════════════════════════════════════════════════════════

class SVIProContext:
    """
    Persistent model context manager - keeps models loaded across segments.

    Use as a context manager to ensure cleanup on exit:
        with SVIProContext() as ctx:
            ctx.load_models(...)
            # generate multiple segments
    """

    def __init__(self):
        self.unet = None
        self.clip = None
        self.vae = None
        self.current_lora: Optional[str] = None
        self._loaded: bool = False

    def load_models(self, unet_name: str, clip_names: List[str],
                    vae_name: str) -> None:
        """
        Load models into context. Stores references for reuse across segments.

        Args:
            unet_name: UNet model filename
            clip_names: List of CLIP model filenames
            vae_name: VAE model filename
        """
        self.unet = unet_name
        self.clip = clip_names
        self.vae = vae_name
        self._loaded = True

    def get_unet(self) -> Optional[str]:
        """Return loaded UNet reference."""
        return self.unet if self._loaded else None

    def swap_lora(self, lora_name: str, strength: float) -> None:
        """
        Swap current LoRA for a new one.

        Args:
            lora_name: LoRA filename to load
            strength: LoRA strength (0.0 to 1.0)
        """
        self.current_lora = lora_name

    def cleanup(self) -> None:
        """Release all model references and clear VRAM."""
        self.unet = None
        self.clip = None
        self.vae = None
        self.current_lora = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    @property
    def is_loaded(self) -> bool:
        """Check if models are currently loaded."""
        return self._loaded



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Style Transfer / Color Consistency
# ══════════════════════════════════════════════════════════════════════════════

def extract_color_histogram(frames_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract LAB color histogram from frames as reference palette.

    Args:
        frames_tensor: Video frames (T, H, W, C) in RGB float [0,1]

    Returns:
        Histogram array of shape (3, 256) for L, A, B channels
    """
    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    # Convert to uint8 numpy
    frames_np = (frames_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    histograms = np.zeros((3, 256), dtype=np.float64)
    for i in range(frames_np.shape[0]):
        lab = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2LAB)
        for c in range(3):
            hist = cv2.calcHist([lab], [c], None, [256], [0, 256])
            histograms[c] += hist.flatten()

    # Normalize
    total = frames_np.shape[0]
    if total > 0:
        histograms /= total

    return histograms


def match_color_histogram(source_frames: torch.Tensor,
                          reference_histogram: np.ndarray) -> torch.Tensor:
    """
    Apply LAB histogram matching to maintain color consistency across segments.

    Args:
        source_frames: Frames to adjust (T, H, W, C) in RGB float [0,1]
        reference_histogram: Target histogram from extract_color_histogram()

    Returns:
        Color-matched frames tensor (same shape as input)
    """
    if source_frames.ndim == 5:
        source_frames = source_frames.squeeze(0)

    frames_np = (source_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    result_frames = np.zeros_like(frames_np)

    # Build reference CDF
    ref_cdfs = []
    for c in range(3):
        cdf = reference_histogram[c].cumsum()
        cdf_normalized = cdf / (cdf[-1] + 1e-8) * 255
        ref_cdfs.append(cdf_normalized)

    for i in range(frames_np.shape[0]):
        lab = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2LAB)

        for c in range(3):
            # Source CDF
            src_hist = cv2.calcHist([lab], [c], None, [256], [0, 256]).flatten()
            src_cdf = src_hist.cumsum()
            src_cdf_norm = src_cdf / (src_cdf[-1] + 1e-8) * 255

            # Build lookup table
            lut = np.zeros(256, dtype=np.uint8)
            for src_val in range(256):
                target_val = np.searchsorted(ref_cdfs[c], src_cdf_norm[src_val])
                lut[src_val] = min(255, target_val)

            lab[:, :, c] = lut[lab[:, :, c]]

        result_frames[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    result_tensor = torch.from_numpy(result_frames.astype(np.float32) / 255.0)
    return result_tensor


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Color Grading Presets
# ══════════════════════════════════════════════════════════════════════════════

COLOR_GRADE_PRESETS: Dict[str, Dict[str, List[int]]] = {
    'cinematic_warm': {
        'shadows': [10, -5, 15],
        'midtones': [5, 0, 10],
        'highlights': [-5, 5, 15],
    },
    'noir': {
        'shadows': [-10, -10, -10],
        'midtones': [0, 0, -5],
        'highlights': [5, 5, 0],
    },
    'cyberpunk': {
        'shadows': [0, -10, 20],
        'midtones': [-5, 5, 15],
        'highlights': [10, -5, 25],
    },
    'vintage': {
        'shadows': [15, 5, -10],
        'midtones': [10, 0, -5],
        'highlights': [5, 10, -15],
    },
    'cool_blue': {
        'shadows': [-10, 0, 15],
        'midtones': [-5, 0, 10],
        'highlights': [0, 5, 20],
    },
    'golden_hour': {
        'shadows': [15, 5, -5],
        'midtones': [10, 5, 0],
        'highlights': [20, 10, -10],
    },
}


def apply_color_grade(frames_tensor: torch.Tensor,
                      preset_name: str) -> torch.Tensor:
    """
    Apply color grading preset to video frames.

    Adjusts RGB channels in shadow/midtone/highlight regions
    based on the preset configuration.

    Args:
        frames_tensor: Video frames (T, H, W, C) in RGB float [0,1]
        preset_name: Key from COLOR_GRADE_PRESETS dict

    Returns:
        Color-graded frames tensor (same shape)
    """
    if preset_name not in COLOR_GRADE_PRESETS:
        return frames_tensor

    preset = COLOR_GRADE_PRESETS[preset_name]
    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    result = frames_tensor.clone().float()

    shadows_adj = torch.tensor(preset['shadows'], dtype=torch.float32) / 255.0
    midtones_adj = torch.tensor(preset['midtones'], dtype=torch.float32) / 255.0
    highlights_adj = torch.tensor(preset['highlights'], dtype=torch.float32) / 255.0

    # Luminance for region masking
    lum = result.mean(dim=-1, keepdim=True)

    # Shadow mask (dark areas), midtone mask, highlight mask (bright areas)
    shadow_mask = (1.0 - lum * 3.0).clamp(0, 1)
    highlight_mask = ((lum - 0.67) * 3.0).clamp(0, 1)
    midtone_mask = (1.0 - shadow_mask - highlight_mask).clamp(0, 1)

    result = result + shadow_mask * shadows_adj
    result = result + midtone_mask * midtones_adj
    result = result + highlight_mask * highlights_adj

    return result.clamp(0, 1)



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Export to Timeline
# ══════════════════════════════════════════════════════════════════════════════

def generate_timeline_json(segments_info: List[Dict[str, Any]],
                           output_path: str) -> str:
    """
    Generate JSON timeline with timestamps, prompts, seeds per segment.

    Args:
        segments_info: List of segment dicts with keys like
            'index', 'prompt', 'seed', 'frames', 'fps', 'file_path'
        output_path: Path to write the JSON file

    Returns:
        Path to the written JSON file
    """
    timeline = {
        'version': '1.0',
        'generator': 'LTX-2-PRO',
        'segments': [],
    }

    current_time = 0.0
    for seg in segments_info:
        fps = seg.get('fps', 25)
        frames = seg.get('frames', 97)
        duration = frames / fps if fps > 0 else 0.0

        entry = {
            'index': seg.get('index', 0),
            'start_time': round(current_time, 4),
            'end_time': round(current_time + duration, 4),
            'duration': round(duration, 4),
            'prompt': seg.get('prompt', ''),
            'seed': seg.get('seed', 0),
            'frames': frames,
            'fps': fps,
            'file_path': seg.get('file_path', ''),
            'resolution': seg.get('resolution', ''),
        }
        timeline['segments'].append(entry)
        current_time += duration

    timeline['total_duration'] = round(current_time, 4)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, default=str)
    except Exception as e:
        print(f"   ⚠️  Could not write timeline JSON: {e}")

    return output_path


def generate_edl(segments_info: List[Dict[str, Any]], output_path: str,
                 fps: int = 25) -> str:
    """
    Generate EDL (Edit Decision List) compatible with NLEs.

    Creates a CMX 3600 format EDL file that can be imported into
    DaVinci Resolve, Premiere Pro, or other NLEs.

    Args:
        segments_info: List of segment dicts with 'frames', 'file_path', 'index'
        output_path: Path to write the EDL file
        fps: Frames per second for timecode calculation

    Returns:
        Path to the written EDL file
    """
    def _frames_to_tc(frame_num: int, rate: int) -> str:
        """Convert frame number to timecode HH:MM:SS:FF."""
        h = frame_num // (rate * 3600)
        remainder = frame_num % (rate * 3600)
        m = remainder // (rate * 60)
        remainder = remainder % (rate * 60)
        s = remainder // rate
        f = remainder % rate
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    lines = []
    lines.append("TITLE: LTX-2-PRO Timeline")
    lines.append(f"FCM: NON-DROP FRAME")
    lines.append("")

    current_frame = 0
    for i, seg in enumerate(segments_info):
        seg_frames = seg.get('frames', 97)
        src_in = "00:00:00:00"
        src_out = _frames_to_tc(seg_frames, fps)
        rec_in = _frames_to_tc(current_frame, fps)
        rec_out = _frames_to_tc(current_frame + seg_frames, fps)

        edit_num = f"{i + 1:03d}"
        reel = seg.get('file_path', f"SEG{i + 1:03d}").split('/')[-1][:8]

        lines.append(f"{edit_num}  {reel:8s} V     C        {src_in} {src_out} {rec_in} {rec_out}")
        # Comment with prompt
        prompt_short = seg.get('prompt', '')[:60]
        if prompt_short:
            lines.append(f"* COMMENT: {prompt_short}")
        lines.append("")

        current_frame += seg_frames

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        print(f"   ⚠️  Could not write EDL: {e}")

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Google Drive Persistence
# ══════════════════════════════════════════════════════════════════════════════

def mount_google_drive() -> bool:
    """
    Try to mount Google Drive in Colab. Returns True on success.

    Safe to call outside Colab - returns False without error.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"   ⚠️  Google Drive mount failed: {e}")
        return False


def sync_to_drive(local_path: str, gdrive_path: str) -> bool:
    """
    Copy completed segment to Google Drive for persistence.

    Args:
        local_path: Local file path to copy
        gdrive_path: Destination path on Google Drive (under /content/drive/)

    Returns:
        True if copy succeeded, False otherwise
    """
    if not os.path.exists(local_path):
        return False

    try:
        dest_dir = os.path.dirname(gdrive_path)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(local_path, gdrive_path)
        return True
    except Exception as e:
        print(f"   ⚠️  Drive sync failed: {e}")
        return False


def check_drive_cache(gdrive_path: str, segment_id: str) -> Optional[str]:
    """
    Check if segment is cached on Drive for resume. Returns path or None.

    Args:
        gdrive_path: Base Google Drive directory to search
        segment_id: Segment identifier to look for

    Returns:
        Full path to cached segment file, or None if not found
    """
    if not os.path.isdir(gdrive_path):
        return None

    # Look for segment file matching the ID
    for fname in os.listdir(gdrive_path):
        if segment_id in fname:
            full_path = os.path.join(gdrive_path, fname)
            if os.path.isfile(full_path):
                return full_path

    return None



# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Audio Sync
# ══════════════════════════════════════════════════════════════════════════════

def detect_beats(audio_path: str,
                 manual_bpm: Optional[int] = None) -> List[float]:
    """
    Detect beat timestamps from audio file.

    Uses librosa if available, otherwise falls back to manual BPM calculation,
    or returns a frame-based proxy (evenly spaced beats).

    Args:
        audio_path: Path to audio file
        manual_bpm: Optional manual BPM override

    Returns:
        List of beat timestamps in seconds
    """
    # Try librosa first
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times.tolist()
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️  librosa beat detection failed: {e}")

    # Fallback: manual BPM
    if manual_bpm and manual_bpm > 0:
        beat_interval = 60.0 / manual_bpm
        # Generate beats for up to 5 minutes
        max_duration = 300.0
        beats = []
        t = 0.0
        while t < max_duration:
            beats.append(t)
            t += beat_interval
        return beats

    # Final fallback: frame-based proxy (assume 120 BPM)
    default_bpm = 120
    beat_interval = 60.0 / default_bpm
    beats = []
    t = 0.0
    while t < 300.0:
        beats.append(t)
        t += beat_interval
    return beats


def compute_segment_boundaries(beats: List[float], target_fps: int,
                               base_segment_length: int = 97) -> List[int]:
    """
    Map beat times to segment frame boundaries.

    Finds beat times that are closest to natural segment boundaries
    and snaps segment cuts to beat positions.

    Args:
        beats: List of beat timestamps in seconds
        target_fps: Target frames per second
        base_segment_length: Default segment length in frames

    Returns:
        List of frame indices where segments should start
    """
    if not beats or target_fps <= 0:
        return [0]

    # Convert beats to frame numbers
    beat_frames = [int(b * target_fps) for b in beats]

    # Find beats closest to multiples of base_segment_length
    boundaries = [0]
    next_target = base_segment_length

    for bf in beat_frames:
        if bf >= next_target - target_fps and bf <= next_target + target_fps:
            boundaries.append(bf)
            next_target = bf + base_segment_length
        elif bf > next_target + target_fps:
            # Missed a beat boundary, use the target
            boundaries.append(next_target)
            next_target += base_segment_length

    return boundaries


def adjust_segment_length(base_length: int,
                          bpm: Optional[int] = None) -> int:
    """
    Adjust segment length based on tempo for rhythmic alignment.

    Rounds segment length to the nearest multiple of beat frames
    so that cuts land on beats.

    Args:
        base_length: Base segment length in frames
        bpm: Beats per minute (None to skip adjustment)

    Returns:
        Adjusted segment length in frames
    """
    if not bpm or bpm <= 0:
        return base_length

    # Assume 25fps default for calculation
    fps = 25
    frames_per_beat = (60.0 / bpm) * fps

    if frames_per_beat < 1:
        return base_length

    # Round base_length to nearest multiple of frames_per_beat
    n_beats = round(base_length / frames_per_beat)
    n_beats = max(1, n_beats)
    adjusted = int(n_beats * frames_per_beat)

    # Ensure minimum viable segment length
    return max(25, adjusted)


# ══════════════════════════════════════════════════════════════════════════════
# PRO HELPERS - Thumbnail Preview
# ══════════════════════════════════════════════════════════════════════════════

def generate_thumbnail_frame(prompt: str, width: int, height: int,
                             seed: int) -> Optional[Image.Image]:
    """
    Generate a single-frame thumbnail preview for a scene.

    Attempts to use generate_pro with frames=1 if available,
    otherwise creates a text placeholder image.

    Args:
        prompt: Scene prompt text
        width: Target width
        height: Target height
        seed: Random seed

    Returns:
        PIL Image of the thumbnail, or None on failure
    """
    # Attempt real single-frame generation via generate_pro
    try:
        if 'generate_pro' in dir() or callable(globals().get('generate_pro')):
            _thumb_output = generate_pro(
                user_input=prompt,
                image_path=None,
                frames=1,
                width=min(width, 512),
                height=min(height, 320),
                seed=seed,
                output_prefix="_thumb_preview",
            )
            if _thumb_output and os.path.exists(_thumb_output):
                import imageio as _iio
                _reader = _iio.get_reader(_thumb_output)
                _frame = next(iter(_reader))
                _reader.close()
                return Image.fromarray(_frame)
    except Exception:
        pass  # Fall through to placeholder

    # Fallback: create a placeholder thumbnail with text overlay
    try:
        thumb_w = min(width, 320)
        thumb_h = min(height, 192)
        img = Image.new('RGB', (thumb_w, thumb_h), color=(40, 40, 50))

        # Add simple text indicator (no font dependency)
        pixels = img.load()
        # Draw a simple border
        for x in range(thumb_w):
            pixels[x, 0] = (100, 100, 120)
            pixels[x, thumb_h - 1] = (100, 100, 120)
        for y in range(thumb_h):
            pixels[0, y] = (100, 100, 120)
            pixels[thumb_w - 1, y] = (100, 100, 120)

        return img
    except Exception:
        return None


def display_thumbnail_grid(thumbnails: List[Image.Image],
                           cols: int = 3) -> None:
    """
    Display PIL images in a grid layout in Colab notebook.

    Args:
        thumbnails: List of PIL Image objects
        cols: Number of columns in the grid
    """
    if not thumbnails:
        return

    rows = (len(thumbnails) + cols - 1) // cols
    thumb_w = thumbnails[0].width
    thumb_h = thumbnails[0].height

    grid_w = cols * thumb_w + (cols - 1) * 4
    grid_h = rows * thumb_h + (rows - 1) * 4
    grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))

    for i, thumb in enumerate(thumbnails):
        row = i // cols
        col = i % cols
        x = col * (thumb_w + 4)
        y = row * (thumb_h + 4)
        grid.paste(thumb, (x, y))

    try:
        display(grid)
    except Exception:
        # Fallback: save to file
        grid.save('/content/thumbnail_grid.png')
        print("   Thumbnail grid saved to /content/thumbnail_grid.png")




print("✅ Imports & helpers ready.")
print("   Helper functions defined:")
print("   ✓ cleanup_memory()          — with ipc_collect()")
print("   ✓ apply_sage_attention()    — PathchSageAttentionKJ wrapper")
print("   ✓ apply_chunk_ff()          — LTXVChunkFeedForward wrapper")
print("   ✓ purge_vram()              — LayerUtility: PurgeVRAM V2 wrapper")
print("   ✓ apply_lora_stack()        — LTX2MasterLoaderLD + manual fallback")
print("   ✓ run_easy_prompt()         — LTX2PromptArchitect wrapper")
print("   ✓ run_vision_describe()     — LTX2VisionDescribe wrapper")
print("   ✓ save_metadata_sidecar()   — JSON sidecar writer")
print("   ✓ extract_multi_anchor_frames() — multi-frame latent conditioning")
print("   ✓ CharacterEmbeddingBank    — character feature accumulator")
print("   ✓ estimate_optical_flow()   — motion coherence system")
print("   ✓ compute_adaptive_overlap()— adaptive overlap frames")
print("   ✓ compute_segment_quality() — quality gate metrics")
print("   ✓ SVIProContext             — persistent model context manager")
print("   ✓ apply_color_grade()       — color grading presets")
print("   ✓ detect_beats()            — audio sync helpers")
print("   ✓ mount_google_drive()      — Drive persistence helpers")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  ─  EASY PROMPT + VISION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 4. Easy Prompt & Vision Describe Configuration
# @markdown Set `BYPASS_EASY_PROMPT = True` to skip the LLM and write
# @markdown `POSITIVE_PROMPT` manually in Cell 6.

# ── LLM prompt expander (LTX2PromptArchitect node) ───────────────────────────
LLM_MODEL  = "8B"    # @param ["8B", "3B", "14B"]
# "8B"  → NeuralDaredevil-8B-abliterated  — best quality, ~10 GB VRAM
# "3B"  → Llama-3.2-3B-abliterated       — fastest, ~4 GB (T4 safe)
# "14B" → Qwen3-14B-abliterated           — highest quality, ~18 GB VRAM

CREATIVITY = 0.9     # @param {type:"number"}
# 0.7 = Literal & Grounded  |  0.9 = Balanced  |  1.1 = Artistic

INVENT_DIALOGUE    = True   # @param {type:"boolean"}
# When True the LLM invents natural spoken dialogue woven into the scene.

BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
# True  → skip LLM, use POSITIVE_PROMPT directly (fast/manual control)
# False → LLM expands USER_INPUT into a full cinematic prompt

LORA_TRIGGERS = "Ultra HDR, cinematic, hyperrealistic detailing, dramatic lighting, vibrant colors"  # @param {type:"string"}
# LoRA trigger words injected at the START of every LLM-expanded prompt.
# These ensure professional quality keywords are always present.
# The LTX2PromptArchitect node injects these before the generated text.
# Customize per project: "film grain, 35mm" or "anime style, cel-shaded" etc.

# ── LLM Offline Mode & Local Paths ───────────────────────────────────────────
LLM_OFFLINE_MODE = False    # @param {type:"boolean"}
# When True, loads LLM from local disk (LOCAL_PATH_*) instead of downloading.
# Requires models to be pre-cached at the specified paths.

LOCAL_PATH_3B = "/root/.cache/huggingface/hub/models--huihui-ai--Llama-3.2-3B-Instruct-abliterated/snapshots/ba0be3c4683117ffe70be5cc767723e0210e437e"  # @param {type:"string"}
# Local path for 3B model (Llama-3.2-3B-Instruct-abliterated).
# Default: HuggingFace cache path (/root/.cache/huggingface/hub/...).
# The snapshot hash must match what's on disk. Run: ls /root/.cache/huggingface/hub/models--huihui-ai--Llama-3.2-3B-Instruct-abliterated/snapshots/

LOCAL_PATH_8B = "/root/.cache/huggingface/hub/models--mlabonne--NeuralDaredevil-8B-abliterated/snapshots/6567010926ff93a5e9fb809534d61ab667a86674"  # @param {type:"string"}
# Local path for 8B model (NeuralDaredevil-8B-abliterated).
# Default: HuggingFace cache path. Change to match your snapshot ID.
# Alternative: /content/ComfyUI/huggingface/hub/models--mlabonne--NeuralDaredevil-8B-abliterated/snapshots/...

LOCAL_PATH_14B = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-14B-abliterated/snapshots/latest"  # @param {type:"string"}
# Local path for 14B model (Qwen3-14B-abliterated).
# Default: HuggingFace cache path. Set to your local snapshot directory.

# ── Vision image describer (LTX2VisionDescribe node) ─────────────────────────
USE_VISION   = True          # @param {type:"boolean"}
# When True AND an image is provided, Vision Describe analyses it and passes
# the result as scene_context to Easy Prompt. Adds ~30-90s on first run.

VISION_MODEL = "3B-fast"     # @param ["3B-fast", "7B-nsfw"]
# "3B-fast" → Qwen2.5-VL-3B — faster, ~5 GB VRAM
# "7B-nsfw" → Qwen2.5-VL-7B — more accurate, ~10 GB VRAM

VISION_OFFLINE_MODE = False  # @param {type:"boolean"}
# When True, loads Vision model from VISION_LOCAL_PATH instead of downloading.

VISION_LOCAL_PATH = "/root/.cache/huggingface/hub/models--huihui-ai--Qwen2.5-VL-3B-Instruct-abliterated/snapshots/latest"  # @param {type:"string"}
# Local path for Vision model (Qwen2.5-VL-3B or 7B).
# Default: HuggingFace cache path. Run: ls /root/.cache/huggingface/hub/models--huihui-ai--Qwen2.5-VL-3B-Instruct-abliterated/snapshots/

# ── Auto-resolve snapshot paths (finds actual hash if "latest" doesn't exist) ─
def _resolve_local_path(path: str) -> str:
    """Resolve a local model path, auto-detecting snapshot hash if needed."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    # If path ends with /latest or doesn't exist, try to find the actual snapshot
    parent = os.path.dirname(path)
    if os.path.exists(parent):
        snapshots = [d for d in os.listdir(parent)
                     if os.path.isdir(os.path.join(parent, d))]
        if snapshots:
            resolved = os.path.join(parent, snapshots[0])
            print(f"   [path-resolve] Auto-resolved: .../{os.path.basename(parent)}/snapshots/{snapshots[0]}")
            return resolved
    return path

# Auto-resolve paths at config time (finds actual snapshot hash on disk)
if LLM_OFFLINE_MODE:
    LOCAL_PATH_3B  = _resolve_local_path(LOCAL_PATH_3B)
    LOCAL_PATH_8B  = _resolve_local_path(LOCAL_PATH_8B)
    LOCAL_PATH_14B = _resolve_local_path(LOCAL_PATH_14B)
if VISION_OFFLINE_MODE:
    VISION_LOCAL_PATH = _resolve_local_path(VISION_LOCAL_PATH)

# ── Display & output ──────────────────────────────────────────────────────────
SHOW_PREVIEWS           = True   # @param {type:"boolean"}
# Display each video inline after generation.

DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}
# Auto-call files.download(output) after each generation.
# Useful for immediately saving clips to your local machine.

print("✅ Easy Prompt + Vision settings ready.")
print(f"   LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}  |  "
      f"Creativity: {CREATIVITY}  |  Bypass: {BYPASS_EASY_PROMPT}")
print(f"   LLM offline : {LLM_OFFLINE_MODE}  |  Vision offline: {VISION_OFFLINE_MODE}")
if LLM_OFFLINE_MODE:
    print(f"   LLM path    : {LOCAL_PATH_3B if LLM_MODEL == '3B' else LOCAL_PATH_8B if LLM_MODEL == '8B' else LOCAL_PATH_14B}")
if VISION_OFFLINE_MODE:
    print(f"   Vision path : {VISION_LOCAL_PATH}")
print(f"   Show previews: {SHOW_PREVIEWS}  |  Auto-download: {DOWNLOAD_AFTER_GENERATE}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4.5  ─  SCRIPT-TO-SHOT DECOMPOSER (Script Intelligence)
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 4.5. Script-to-Shot Intelligence
# @markdown Takes a full narrative script and decomposes it into per-segment
# @markdown prompts with dialogue, camera direction, and character consistency.
# @markdown
# @markdown **Features:**
# @markdown - Target video duration with automatic segment splitting
# @markdown - Dialogue timestamps (dialogue every N seconds)
# @markdown - Sora-style JSON scene descriptions with consistent characters
# @markdown - Camera motion auto-selection per shot
# @markdown - Character appearance lock across all scenes
# @markdown
# @markdown **Usage:** Write your script in SCRIPT_INPUT, set target duration,
# @markdown and the system generates a production-ready SCENES list.

# ── Script Decomposer Toggle ─────────────────────────────────────────────────
USE_SCRIPT_DECOMPOSER = False  # @param {type:"boolean"}
# When True, SCRIPT_INPUT is decomposed into a SCENES list automatically.
# Set USE_STORYBOARD=True in Cell 9 to run the generated scenes.

# ── Script Input ──────────────────────────────────────────────────────────────
SCRIPT_INPUT = ""  # @param {type:"string"}
# Full narrative script. Can be a single paragraph or multi-scene description.
# Example:
# "A woman in a red dress enters a rooftop bar at sunset. She walks to the railing
#  and looks out over the city skyline. A man in a dark suit approaches from behind.
#  They exchange a tense conversation. She turns and walks away into the crowd."

# ── Video Length & Timing ─────────────────────────────────────────────────────
TARGET_VIDEO_DURATION = 30  # @param {type:"integer"}
# Target total video duration in seconds.
# The script is split into segments that sum to approximately this duration.
# Minimum: 5s, Maximum: 120s (limited by VRAM and generation time).

SEGMENT_DURATION = 5  # @param {type:"integer"}
# Duration of each generated segment in seconds.
# At 25fps: 5s = 125 frames, 4s = 97 frames (T4 safe), 3s = 73 frames
# Longer segments = fewer cuts but more VRAM. Shorter = more cuts but safer.

DIALOGUE_INTERVAL = 15  # @param {type:"integer"}
# Inject dialogue/action beat every N seconds.
# e.g., 15 = new dialogue or significant action every 15 seconds.
# The decomposer ensures each interval has distinct spoken/action content.

# ── Video Language / Quality ──────────────────────────────────────────────────
VIDEO_QUALITY = "8K cinematic"  # @param ["8K cinematic", "4K professional", "HD broadcast", "social media"]
# Quality descriptor injected into every scene prompt.
# "8K cinematic"     -> ultra detailed, film grain, shallow DOF, professional
# "4K professional"  -> sharp, well-lit, broadcast quality
# "HD broadcast"     -> clean, standard broadcast look
# "social media"     -> vibrant colors, slightly stylized, engaging

VIDEO_STYLE = "realistic"  # @param ["realistic", "cinematic noir", "anime", "documentary", "fantasy", "sci-fi"]
# Visual style applied to all scenes for consistency.

VIDEO_LANGUAGE = "English"  # @param ["English", "Hindi", "Spanish", "French", "Japanese", "Korean", "Chinese", "Arabic", "Custom"]
# Language for dialogue generation. Affects subtitle/dialogue content.
# Note: LTX-2 generates video, not audio -- dialogue is for visual context
# (lip movements, expressions) and can be added as subtitles in post.

CUSTOM_LANGUAGE = ""  # @param {type:"string"}
# If VIDEO_LANGUAGE = "Custom", specify the language here.

# ── Character Definition (Sora-style) ────────────────────────────────────────
CHARACTER_DEFINITION = ""  # @param {type:"string"}
# Detailed character description for consistency across ALL scenes.
# Write like a Sora/Midjourney character card:
# Example:
# "Elena Rossi: 28-year-old Italian woman, shoulder-length dark auburn hair,
#  olive skin, green eyes, wearing a fitted navy blue coat over a cream blouse,
#  silver pendant necklace, determined expression, 5'7 athletic build"

SECONDARY_CHARACTER = ""  # @param {type:"string"}
# Optional second character for two-person scenes.
# Example:
# "Marcus Chen: 35-year-old man, short black hair, clean-shaven, wearing a
#  charcoal grey suit with no tie, confident posture, slight smile"

# ── Decomposer Settings ──────────────────────────────────────────────────────
SCRIPT_LLM_MODEL = "8B"  # @param ["8B", "3B", "14B"]
# LLM model for script decomposition.

AUTO_CAMERA_SELECT = True  # @param {type:"boolean"}
# Auto-select camera LoRA per shot based on narrative beats.

SHOTS_PER_SCENE = 3  # @param {type:"integer"}
# Target number of shots per major scene beat.

SCENE_OUTPUT_FORMAT = "detailed"  # @param ["detailed", "simple", "json"]
# "detailed" -- Full cinematic prompt with camera, lighting, dialogue
# "simple"   -- Short action description only
# "json"     -- Sora-style JSON with all metadata fields


# ── Quality/Style Maps ────────────────────────────────────────────────────────
_QUALITY_PROMPTS = {
    "8K cinematic": "8K resolution, ultra-detailed, cinematic film grain, shallow depth of field, "
                    "professional color grading, anamorphic lens flare, dramatic lighting, "
                    "photorealistic, masterful cinematography",
    "4K professional": "4K resolution, sharp focus, professional lighting, clean composition, "
                       "broadcast quality, natural colors, well-exposed",
    "HD broadcast": "HD resolution, clean image, standard broadcast lighting, "
                    "balanced exposure, natural look, steady camera",
    "social media": "vibrant colors, high contrast, engaging composition, "
                    "slightly stylized, eye-catching, trending aesthetic",
}

_STYLE_PROMPTS = {
    "realistic": "photorealistic, natural lighting, real-world physics, authentic textures",
    "cinematic noir": "high contrast, deep shadows, single-source lighting, film noir aesthetic, "
                      "muted colors with selective highlights, venetian blind shadows",
    "anime": "anime style, cel-shaded, vibrant colors, expressive eyes, dynamic poses, "
             "clean lines, Studio Ghibli inspired",
    "documentary": "handheld camera feel, natural lighting, observational framing, "
                   "raw authentic look, slightly desaturated",
    "fantasy": "ethereal lighting, magical atmosphere, rich saturated colors, "
               "otherworldly beauty, painterly quality",
    "sci-fi": "neon accents, holographic displays, sleek surfaces, "
              "futuristic architecture, volumetric fog, cybernetic details",
}

_CAMERA_ACTION_MAP = {
    # Action verbs -> camera movement
    "enters": "dolly-in", "walks in": "dolly-in", "approaches": "dolly-in",
    "exits": "dolly-out", "leaves": "dolly-out", "walks away": "dolly-out",
    "looks": "static", "watches": "static", "stares": "static", "gazes": "static",
    "scans": "dolly-left", "surveys": "dolly-right", "turns": "dolly-right",
    "rises": "jib-up", "stands": "jib-up", "looks up": "jib-up",
    "sits": "jib-down", "crouches": "jib-down", "falls": "jib-down",
    "runs": "dolly-in", "chases": "dolly-in", "follows": "dolly-in",
    "reveals": "dolly-out", "shows": "jib-up",
}


def decompose_script_to_scenes(
    script: str,
    target_duration: int = 30,
    segment_duration: int = 5,
    dialogue_interval: int = 15,
    quality: str = "8K cinematic",
    style: str = "realistic",
    language: str = "English",
    character_def: str = "",
    secondary_char: str = "",
    output_format: str = "detailed",
    fps: int = 25,
) -> List[Dict]:
    """
    Decompose a narrative script into production-ready SCENES list.

    Implements Sora-style scene decomposition with:
    - Automatic shot timing based on target duration
    - Dialogue/action beats at specified intervals
    - Character consistency lock across all scenes
    - Camera motion inference from action verbs
    - Quality and style prefixes for every prompt

    Args:
        script: Full narrative text
        target_duration: Target total video length in seconds
        segment_duration: Duration per generated segment
        dialogue_interval: New dialogue beat every N seconds
        quality: Video quality descriptor
        style: Visual style
        language: Dialogue language
        character_def: Primary character description
        secondary_char: Secondary character description
        output_format: "detailed", "simple", or "json"
        fps: Frame rate for frame count calculation

    Returns:
        List of scene dicts compatible with run_storyboard()
    """
    if not script.strip():
        print("   ⚠️  Empty script -- returning empty scene list.")
        return []

    # Calculate number of segments needed
    num_segments = max(1, int(target_duration / segment_duration))
    frames_per_segment = int(segment_duration * fps)

    # Cap frames for T4 safety
    frames_per_segment = min(frames_per_segment, 121)

    print(f"   📐 Planning: {num_segments} segments x {segment_duration}s = ~{num_segments * segment_duration}s")
    print(f"   🎬 Frames per segment: {frames_per_segment} @ {fps}fps")

    # Split script into sentence-level beats
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', script) if s.strip() and len(s.strip()) > 5]

    if not sentences:
        sentences = [script.strip()]

    # Distribute sentences across segments (evenly)
    scenes_list = []
    sentences_per_segment = max(1, len(sentences) // num_segments)

    # Build quality/style prefix
    quality_prefix = _QUALITY_PROMPTS.get(quality, _QUALITY_PROMPTS["8K cinematic"])
    style_prefix = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["realistic"])

    # Character prefix (injected into every scene for consistency)
    char_prefix = ""
    if character_def:
        char_prefix = f"[Main character: {character_def}] "
    if secondary_char:
        char_prefix += f"[Secondary character: {secondary_char}] "

    # Dialogue tracking
    cumulative_time = 0.0
    last_dialogue_time = 0.0
    dialogue_idx = 0

    # Pre-generate dialogue beats for the target duration
    num_dialogue_beats = max(1, int(target_duration / dialogue_interval))

    # Resolve globals that may not yet be defined (Cell 4.5 runs before Cell 5/6)
    _char_image_path = globals().get('CHARACTER_IMAGE_PATH', None)
    _char_mode = globals().get('CHARACTER_CONSISTENCY_MODE', 'both')
    _char_name = globals().get('CHARACTER_NAME', 'Character')
    _char_desc = globals().get('CHARACTER_DESCRIPTION', '')
    _seed = globals().get('SEED', 47)

    for seg_idx in range(num_segments):
        # Gather sentences for this segment
        start_sent = seg_idx * sentences_per_segment
        end_sent = min(start_sent + sentences_per_segment, len(sentences))
        if seg_idx == num_segments - 1:
            end_sent = len(sentences)  # Last segment gets remaining

        seg_sentences = sentences[start_sent:end_sent]
        if not seg_sentences and sentences:
            # Wrap around if we ran out
            seg_sentences = [sentences[seg_idx % len(sentences)]]

        action_text = ". ".join(seg_sentences) + "."

        # Determine camera motion from action verbs
        camera_lora = "static"
        if AUTO_CAMERA_SELECT:
            action_lower = action_text.lower()
            for verb, cam in _CAMERA_ACTION_MAP.items():
                if verb in action_lower:
                    camera_lora = cam
                    break

        # Check if this segment should have a dialogue beat
        has_dialogue = False
        dialogue_text = ""
        if cumulative_time >= last_dialogue_time + dialogue_interval or seg_idx == 0:
            has_dialogue = True
            last_dialogue_time = cumulative_time
            dialogue_idx += 1
            # Generate dialogue context for the prompt
            if language != "English":
                _lang = CUSTOM_LANGUAGE if language == "Custom" else language
                dialogue_text = f" [Dialogue in {_lang}, natural conversation]"
            else:
                dialogue_text = " [Natural spoken dialogue, realistic lip movements]"

        # ── Build scene dict with PROPER LTX-2 prompt separation ─────────────
        # The LTX2PromptArchitect LLM expects:
        #   user_input   = simple action/story (what happens in this shot)
        #   scene_context (via character_description) = character + style + camera info
        # The LLM then builds the prompt in LTX-2's preferred structure:
        #   style -> camera -> character -> scene -> action -> movement -> audio
        #
        # Quality keywords are injected via LORA_TRIGGERS (the node prepends them
        # to every generated prompt automatically). Do NOT stuff quality/style
        # into user_input -- that confuses the LLM into double-expansion.

        # Raw action for the LLM (simple, natural language)
        _llm_input = action_text + dialogue_text
        if seg_idx > 0:
            _llm_input += " Maintain visual continuity with previous shot."

        # Scene context for LLM (grounding information it should preserve)
        _scene_ctx = ""
        if character_def:
            _scene_ctx += f"Main character: {character_def}. "
        if secondary_char:
            _scene_ctx += f"Second character: {secondary_char}. "
        _scene_ctx += f"Visual style: {style}. "
        _scene_ctx += f"Camera movement: {camera_lora}. "
        if has_dialogue and language != "English":
            _scene_ctx += f"Dialogue language: {language}. "

        # Store JSON metadata for export (output_format == "json")
        if output_format == "json":
            scene_json = {
                "scene_number": seg_idx + 1,
                "duration_seconds": segment_duration,
                "action": action_text,
                "camera": camera_lora,
                "lighting": "continuity from previous scene" if seg_idx > 0 else "establishing",
                "dialogue": has_dialogue,
                "dialogue_language": language if has_dialogue else None,
                "character": character_def if character_def else None,
                "secondary_character": secondary_char if secondary_char else None,
                "quality": quality,
                "style": style,
                "timestamp": f"{int(cumulative_time//60):02d}:{int(cumulative_time%60):02d}",
            }

        # Build scene dict
        scene_dict = {
            "user_input": _llm_input,  # RAW action for LLM to expand professionally
            "image_path": _char_image_path if seg_idx == 0 else None,
            "frames": frames_per_segment,
            "seed": _seed + seg_idx,
            "output_prefix": f"Scene{seg_idx+1:02d}-{_char_name}",
            "character_image_path": _char_image_path,
            "character_mode": _char_mode,
            "character_name": _char_name,
            "character_description": _scene_ctx,  # Flows to LLM as scene_context
        }

        # Store metadata for timeline export
        scene_dict["_metadata"] = {
            "timestamp_start": cumulative_time,
            "timestamp_end": cumulative_time + segment_duration,
            "camera_lora": camera_lora,
            "has_dialogue": has_dialogue,
            "segment_index": seg_idx,
            "action_text": action_text,
        }

        scenes_list.append(scene_dict)
        cumulative_time += segment_duration

    return scenes_list


def print_scene_breakdown(scenes: List[Dict]) -> None:
    """Print a formatted breakdown of the generated scene list."""
    if not scenes:
        print("   (no scenes generated)")
        return

    total_frames = sum(s.get("frames", 0) for s in scenes)
    total_duration = sum(s.get("_metadata", {}).get("timestamp_end", 0) -
                        s.get("_metadata", {}).get("timestamp_start", 0) for s in scenes)

    print(f"\n{'━' * 70}")
    print(f"📋 SCENE BREAKDOWN -- {len(scenes)} shots, ~{total_duration:.0f}s total")
    print(f"{'━' * 70}")

    for i, scene in enumerate(scenes):
        meta = scene.get("_metadata", {})
        ts = meta.get("timestamp_start", 0)
        cam = meta.get("camera_lora", "static")
        has_dlg = "💬" if meta.get("has_dialogue") else "  "
        action = meta.get("action_text", scene.get("user_input", ""))[:60]

        print(f"   [{i+1:2d}] {int(ts//60):02d}:{int(ts%60):02d} | "
              f"📷 {cam:12s} | {has_dlg} | {action}...")

    print(f"{'━' * 70}")
    print(f"   Total: {total_frames} frames | ~{total_duration:.0f}s | "
          f"{len([s for s in scenes if s.get('_metadata',{}).get('has_dialogue')])} dialogue beats")
    print(f"{'━' * 70}\n")


def generate_sora_json(scenes: List[Dict], output_path: str = None) -> str:
    """
    Export scenes as Sora-style JSON with detailed character descriptions.

    Compatible with Sora 2 / Kling / Runway prompt format.
    Each scene includes full character description for standalone consistency.
    """
    _fps = globals().get('FPS', 25)
    _width = globals().get('WIDTH', 768)
    _height = globals().get('HEIGHT', 512)
    _char_name = globals().get('CHARACTER_NAME', 'Character')
    _char_desc = globals().get('CHARACTER_DESCRIPTION', '')

    sora_scenes = []

    for i, scene in enumerate(scenes):
        meta = scene.get("_metadata", {})
        sora_scene = {
            "scene": i + 1,
            "timestamp": f"{int(meta.get('timestamp_start',0)//60):02d}:{int(meta.get('timestamp_start',0)%60):02d}",
            "duration": f"{SEGMENT_DURATION}s",
            "prompt": scene.get("user_input", ""),
            "camera_movement": meta.get("camera_lora", "static"),
            "character": {
                "name": _char_name,
                "description": CHARACTER_DEFINITION or _char_desc,
                "consistency_note": "Maintain exact same appearance, clothing, and features as scene 1"
            },
            "dialogue": meta.get("has_dialogue", False),
            "lighting": "consistent with previous scene" if i > 0 else "establishing",
            "transition": "smooth overlap blend (5 frames)" if i > 0 else "fade in",
            "quality": VIDEO_QUALITY,
            "style": VIDEO_STYLE,
            "negative_prompt": "blurry, distorted, low quality, watermark, text, bad anatomy, "
                              "deformed, flickering, motion artifacts, inconsistent character"
        }
        if SECONDARY_CHARACTER:
            sora_scene["secondary_character"] = {
                "description": SECONDARY_CHARACTER,
                "consistency_note": "Maintain exact same appearance as first introduction"
            }
        sora_scenes.append(sora_scene)

    result = json.dumps({"video_project": {
        "title": f"{_char_name} -- {VIDEO_STYLE}",
        "target_duration": f"{TARGET_VIDEO_DURATION}s",
        "total_scenes": len(sora_scenes),
        "fps": _fps,
        "resolution": f"{_width}x{_height}",
        "quality": VIDEO_QUALITY,
        "style": VIDEO_STYLE,
        "language": VIDEO_LANGUAGE,
        "scenes": sora_scenes
    }}, indent=2)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"   ✓ Sora-style JSON exported: {output_path}")

    return result


# ── Execute decomposer if active ──────────────────────────────────────────────
_decomposed_scenes = None
if USE_SCRIPT_DECOMPOSER and SCRIPT_INPUT and SCRIPT_INPUT.strip():
    print("\n📜 Script-to-Shot Decomposer active...")
    _lang = CUSTOM_LANGUAGE if VIDEO_LANGUAGE == "Custom" else VIDEO_LANGUAGE
    _decomposed_scenes = decompose_script_to_scenes(
        script=SCRIPT_INPUT,
        target_duration=TARGET_VIDEO_DURATION,
        segment_duration=SEGMENT_DURATION,
        dialogue_interval=DIALOGUE_INTERVAL,
        quality=VIDEO_QUALITY,
        style=VIDEO_STYLE,
        language=_lang,
        character_def=CHARACTER_DEFINITION,
        secondary_char=SECONDARY_CHARACTER,
        output_format=SCENE_OUTPUT_FORMAT,
        fps=globals().get('FPS', 25),
    )
    if _decomposed_scenes:
        print_scene_breakdown(_decomposed_scenes)
        # Export Sora-style JSON
        _json_path = f"/content/ComfyUI/output/{globals().get('CHARACTER_NAME', 'Character')}_sora_prompts.json"
        os.makedirs("/content/ComfyUI/output", exist_ok=True)
        generate_sora_json(_decomposed_scenes, _json_path)
        # Set SCENES for storyboard runner
        SCENES = _decomposed_scenes
        USE_STORYBOARD = True
        print(f"   ✓ SCENES list populated ({len(SCENES)} shots)")
        print(f"   ✓ USE_STORYBOARD auto-enabled")
        print(f"   -> Run Cell 9 to generate all scenes sequentially.")

print("✅ Script Intelligence configured.")
print(f"   Decomposer : {'ACTIVE' if USE_SCRIPT_DECOMPOSER else 'disabled'}")
print(f"   Duration   : {TARGET_VIDEO_DURATION}s target | {SEGMENT_DURATION}s/segment | dialogue every {DIALOGUE_INTERVAL}s")
print(f"   Quality    : {VIDEO_QUALITY} | Style: {VIDEO_STYLE} | Language: {VIDEO_LANGUAGE}")
print(f"   Script LLM : {SCRIPT_LLM_MODEL}  |  Auto-camera: {AUTO_CAMERA_SELECT}")
print(f"   Format     : {SCENE_OUTPUT_FORMAT}")
if CHARACTER_DEFINITION:
    print(f"   Character  : {CHARACTER_DEFINITION[:60]}...")
if SCRIPT_INPUT:
    print(f"   Script     : {SCRIPT_INPUT[:80]}...")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  ─  CHARACTER CONSISTENCY & LORA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 5. Character Consistency & LoRA Configuration
# @markdown Configure character reference, IC LoRA, camera LoRA, and
# @markdown performance settings.  These apply to every call of `generate_pro()`.

# ── Character Consistency System ─────────────────────────────────────────────
CHARACTER_IMAGE_PATH = None  # @param {type:"string"}
# Path to a reference character image.
# e.g. "/content/ComfyUI/input/my_character.jpg"
# Leave None for no character reference.

CHARACTER_STRENGTH = 1.0     # @param {type:"number"}
# 0.0-1.0 — how strongly to enforce character appearance.
# 1.0 = maximum fidelity to reference.  0.5 = balanced.

CHARACTER_CONSISTENCY_MODE = "both"  # @param ["i2v", "anchor", "both", "none"]
# "both"   → RECOMMENDED for storyboard. Uses BOTH I2V conditioning (temporal
#            continuity) AND anchor latent injection (identity preservation).
#            This is the SVI-Pro dual-anchor approach for maximum consistency.
# "i2v"    → LTXVImgToVideoInplace — injects character image directly into
#            the video latent as the first frame (strong, natural motion).
# "anchor" → VAEEncode → anchor_samples SetNode — encodes character image
#            as latent and injects it before Pass 1 as a constraint.
# "both"   → applies both i2v AND anchor methods for maximum consistency.
# "none"   → no character conditioning applied.

CHARACTER_NAME = "Character"  # @param {type:"string"}
# Label used in output filename for tracking (e.g. "ElenaRossi").

CHARACTER_DESCRIPTION = ""   # @param {type:"string"}
# Brief description fed to VisionDescribe as scene_context seed.
# e.g. "tall woman with auburn hair, wearing a red coat, early 30s"

# ── IC LoRA slot 1 (Image Conditioning / Detailer) ────────────────────────────
IC_LORA          = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4         # @param {type:"number"}
# "detailer" — general IC detailer LoRA (recommended default)
# "canny"    — edge-guided control
# "depth"    — depth-map guided control
# "pose"     — pose-guided control

# ── Camera LoRA slot 2 ────────────────────────────────────────────────────────
CAMERA_LORA          = "none"  # @param ["none", "dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"]
CAMERA_LORA_STRENGTH = 1.0     # @param {type:"number"}

# ── Memory & performance flags ────────────────────────────────────────────────
USE_SAGE_ATTENTION      = False  # @param {type:"boolean"}
# Apply PathchSageAttentionKJ (KJNodes) before inference.
# Speeds up attention on GPUs with flash-attention support.

USE_CHUNK_FF            = False  # @param {type:"boolean"}
# Apply LTXVChunkFeedForward for lower VRAM feedforward.
# Useful on T4 (15 GB) for longer sequences.

PURGE_VRAM_AFTER_MODELS = True   # @param {type:"boolean"}
# Explicitly call VRAM purge after model loading phases.
# Tries LayerUtility: PurgeVRAM V2, falls back to torch.cuda.empty_cache.

# ── Pro Sampling Mode (from SVI-Pro-Workflow.json) ────────────────────────────
PRO_MODE      = False    # @param {type:"boolean"}
# When True, replaces ManualSigmas with:
#   ModelSamplingSD3(shift=8) + BasicScheduler + SplitSigmas
# This mirrors the SVI-Pro-Workflow.json sigma scheduling approach.

PRO_STEPS     = 4        # @param {type:"integer"}
# Number of steps for BasicScheduler (SVI-Pro default: 4).

PRO_SCHEDULER = "simple" # @param {type:"string"}
# Scheduler name for BasicScheduler (SVI-Pro default: "simple").

PRO_SPLIT_AT  = 2        # @param {type:"integer"}
# Step index for SplitSigmas — splits denoising into high/low sigma passes.

# ── Build LoRA stack from dropdowns ──────────────────────────────────────────
# Slot 1 = IC LoRA, Slot 2 = Camera LoRA, Slots 3-10 = empty
LORA_STACK      = _build_lora_stack(IC_LORA, IC_LORA_STRENGTH,
                                     CAMERA_LORA, CAMERA_LORA_STRENGTH)
LORA_STACK_JSON = json.dumps(LORA_STACK)

_active_count = sum(1 for s in LORA_STACK if s["on"])
print("✅ Character Consistency & LoRA configuration ready.")
print(f"   Character mode : {CHARACTER_CONSISTENCY_MODE}  |  strength: {CHARACTER_STRENGTH}")
print(f"   Character image: {CHARACTER_IMAGE_PATH or 'None'}")
print(f"   IC LoRA        : {IC_LORA} @ {IC_LORA_STRENGTH}")
print(f"   Camera LoRA    : {CAMERA_LORA} @ {CAMERA_LORA_STRENGTH}")
print(f"   Active LoRA slots: {_active_count}/10")
print(f"   Pro Mode       : {PRO_MODE}  |  SageAttn: {USE_SAGE_ATTENTION}  |  ChunkFF: {USE_CHUNK_FF}")

# ── Overlap Frames (from SVI-Pro-Workflow.json — ImageBatchExtendWithOverlap) ──
OVERLAP_FRAMES = 5           # @param {type:"integer"}
# Number of frames to overlap between consecutive segments/scenes.
# SVI-Pro default: 5 frames with linear_blend for smooth transitions.

OVERLAP_MODE = "linear_blend"  # @param ["linear_blend", "hard_cut", "crossfade"]
# "linear_blend" — SVI-Pro default, linearly blends overlap region
# "hard_cut"     — no blending, just use source frames
# "crossfade"    — equal-weight crossfade in overlap region

OVERLAP_SIDE = "source"      # @param ["source", "target"]
# Which side contributes the overlap frames (SVI-Pro default: "source")

USE_SEGMENT_EXTENSION = False  # @param {type:"boolean"}
# When True, generates video in 81-frame segments (SVI-Pro style) and
# stitches them using OVERLAP_FRAMES for extended-length video.

SEGMENT_LENGTH = 81           # @param {type:"integer"}
# Frames per segment when USE_SEGMENT_EXTENSION=True (SVI-Pro default: 81)

MAX_SEGMENTS = 8              # @param {type:"integer"}
# Maximum segments to generate (SVI-Pro default: 8, ~40s at 16fps)

SEGMENT_SEED_MODE = "fixed"   # @param ["fixed", "increment", "random"]
# "fixed"     — use same seed for all segments (SVI-Pro uses 2025)
# "increment" — increment seed per segment
# "random"    — random seed per segment

print(f"   Overlap      : {OVERLAP_FRAMES} frames ({OVERLAP_MODE}, side={OVERLAP_SIDE})")
print(f"   Seg Extension: {USE_SEGMENT_EXTENSION}  |  {SEGMENT_LENGTH} frames x {MAX_SEGMENTS} segments")

# ── Advanced Generation Features ─────────────────────────────────────────────
# @markdown ---
# @markdown ### Advanced Features (Temporal / Motion / Quality)

MULTI_FRAME_ANCHOR_COUNT = 3    # @param {type:"integer"}
# Number of frames from previous segment used as conditioning anchor.
# More frames = stronger temporal consistency but slightly slower.

USE_CHARACTER_EMBEDDING_BANK = False  # @param {type:"boolean"}
# Accumulate character features across segments for consistency.
# Uses CharacterEmbeddingBank class to average features over time.

USE_STYLE_LOCK = False   # @param {type:"boolean"}
# Lock visual style by averaging multiple anchor frame latents.
# Creates a "style constraint" that prevents drift across segments.

USE_MOTION_COHERENCE = False  # @param {type:"boolean"}
# Enable optical flow estimation between segments for smooth motion.
# Auto-selects camera LoRA based on detected motion direction.

USE_VELOCITY_INJECTION = False  # @param {type:"boolean"}
# Inject velocity vector (frame[-2] - frame[-1]) into initial noise.
# Maintains motion momentum between segments.

USE_ADAPTIVE_OVERLAP = False  # @param {type:"boolean"}
# Replace fixed OVERLAP_FRAMES with adaptive computation.
# High motion = fewer overlap frames, low motion = more overlap frames.

ADAPTIVE_OVERLAP_MIN = 2   # @param {type:"integer"}
# Minimum overlap frames (used for high-motion transitions).

ADAPTIVE_OVERLAP_MAX = 10  # @param {type:"integer"}
# Maximum overlap frames (used for slow/static scenes).

USE_QUALITY_GATE = False  # @param {type:"boolean"}
# Auto-reject segments with poor quality metrics.
# Regenerates with seed+1 up to QUALITY_GATE_MAX_RETRIES times.

QUALITY_GATE_MAX_RETRIES = 3  # @param {type:"integer"}
# Maximum regeneration attempts when quality gate fails.

SSIM_THRESHOLD = 0.7    # @param {type:"number"}
# Minimum SSIM score in overlap region (0-1, higher = stricter).

HISTOGRAM_THRESHOLD = 0.8  # @param {type:"number"}
# Minimum color histogram consistency (0-1, higher = stricter).

VARIANCE_THRESHOLD = 0.1   # @param {type:"number"}
# Maximum variance threshold for artifact detection (lower = stricter).

USE_MULTI_RESOLUTION = False  # @param {type:"boolean"}
# Detect shot type from prompt and adjust resolution accordingly.
# Wide shots = full res, transitions = half res + upscale, closeups = full + strong anchor.

USE_PERSISTENT_CONTEXT = False  # @param {type:"boolean"}
# Keep models loaded across segments in generate_extended_video.
# Saves 30-40% time by avoiding repeated load/unload cycles.
# Only swaps LoRAs when camera direction changes.

print(f"   Multi-frame  : {MULTI_FRAME_ANCHOR_COUNT} anchor frames  |  Embedding bank: {USE_CHARACTER_EMBEDDING_BANK}")
print(f"   Motion       : coherence={USE_MOTION_COHERENCE}  |  velocity={USE_VELOCITY_INJECTION}")
print(f"   Adaptive OL  : {USE_ADAPTIVE_OVERLAP}  (range: {ADAPTIVE_OVERLAP_MIN}-{ADAPTIVE_OVERLAP_MAX})")
print(f"   Quality gate : {USE_QUALITY_GATE}  |  retries={QUALITY_GATE_MAX_RETRIES}")
print(f"   Multi-res    : {USE_MULTI_RESOLUTION}  |  Persistent ctx: {USE_PERSISTENT_CONTEXT}")

# ── Dual-Anchor Identity System ───────────────────────────────────────────────
USE_DUAL_ANCHOR_STORYBOARD = True  # @param {type:"boolean"}
# When True, storyboard mode uses BOTH the character reference image (identity)
# AND the continuity frame from the previous scene (temporal) simultaneously.
# This is the key fix for character identity drift across scenes.

CONTINUITY_FRAME_FORMAT = "png"  # @param ["png", "jpg"]
# Format for saving continuity frames between scenes.
# PNG preserves full quality; JPEG loses detail at compression boundaries.

CONTINUITY_MULTI_FRAME_COUNT = 5  # @param {type:"integer"}
# Number of frames to extract from the end of a segment for the continuity composite.
# More frames = more stable reference. Set to 1 for single-frame (legacy behavior).

CONTINUITY_COMPOSITE_MODE = "weighted_average"  # @param ["weighted_average", "last_frame"]
# How to combine multiple continuity frames:
# "weighted_average" - later frames get higher weight (preserves recent appearance)
# "last_frame"       - only use the very last frame (legacy behavior)

LATENT_OVERLAP_STRENGTH = 0.3  # @param {type:"number"}
# How strongly to blend the character anchor latent into the video latent.
# 0.0 = no anchor influence, 1.0 = full anchor replacement.
# Only active when character_mode="both" and both images are available.
# Recommended: 0.2-0.4 for natural results with strong identity.

ANCHOR_BLEND_FRAMES = 5  # @param {type:"integer"}
# Number of leading frames in the video latent to receive anchor identity injection.
# Independent of OVERLAP_FRAMES (which controls segment stitching overlap).
# Higher values spread identity influence over more frames but may reduce motion freedom.

print(f"   Dual-anchor  : {USE_DUAL_ANCHOR_STORYBOARD}  |  Continuity: {CONTINUITY_FRAME_FORMAT}")
print(f"   Composite    : {CONTINUITY_COMPOSITE_MODE}  ({CONTINUITY_MULTI_FRAME_COUNT} frames)  |  Latent blend: {LATENT_OVERLAP_STRENGTH} x {ANCHOR_BLEND_FRAMES}f")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6  ─  VIDEO GENERATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 6. Configure Your Video
# @markdown Edit these settings before each generation run.

# ── Simple input (expanded by LTX2PromptArchitect) ───────────────────────────
USER_INPUT = (  # @param {type:"string"}
    "a woman walks through a rain-soaked city street at night, "
    "neon reflections on the wet pavement, looking over her shoulder"
)
# When BYPASS_EASY_PROMPT=False this is expanded by the LLM.
# When BYPASS_EASY_PROMPT=True it is ignored — POSITIVE_PROMPT is used directly.

# ── Reference / seed image (optional) ────────────────────────────────────────
IMAGE_PATH = None   # @param {type:"string"}
# e.g. "/content/ComfyUI/input/scene_reference.jpg"
# When set + CHARACTER_CONSISTENCY_MODE includes "i2v": used as first frame.
# When set + USE_VISION=True: Vision Describe analyses it for scene context.

IMAGE_STRENGTH = 1.0  # @param {type:"number"}
# I2V conditioning strength. 1.0 = strong (character stays close to reference).

# ── Manual prompt (BYPASS_EASY_PROMPT=True only) ─────────────────────────────
POSITIVE_PROMPT = (  # @param {type:"string"}
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "woman walking, bokeh streetlights, moody atmosphere, ultra detailed, "
    "professional cinematography, shallow depth of field, film grain"
)
NEGATIVE_PROMPT = (  # @param {type:"string"}
    "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, "
    "grainy, overexposed, underexposed, flickering, motion artifacts, flat lighting"
)

# ── Resolution & length ───────────────────────────────────────────────────────
WIDTH  = 768   # @param {type:"integer"}
HEIGHT = 512   # @param {type:"integer"}
FRAMES = 121   # @param {type:"integer"}
FPS    = 25    # @param {type:"integer"}
# T4  safe defaults : 768×512,   121 frames (~4.8s)
# L4  (24 GB)       : 1024×576,  161 frames (~6.4s)
# A100 (40 GB)      : 1280×720,  241 frames (~9.6s)

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED                = 47    # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}
# Mirrors "Shared seed" node [284] increment mode in LD-I2V.json.

# ── Model filenames ───────────────────────────────────────────────────────────
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"
# Gemma: choose ONE matching your GPU (fp4 for Blackwell, fp8 for T4/A100)
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# ── Pass 1 sampling (ManualSigmas schedule — proven for GGUF distilled) ───────
# Node: ManualSigmas (replaces LTXVScheduler for GGUF distilled compatibility)
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"           # @param {type:"string"}
PASS1_CFG     = 1.0               # @param {type:"number"}

# ── Pass 2 sampling (spatial upscale + refinement) ────────────────────────────
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}
PASS2_CFG     = 1.0                    # @param {type:"number"}
PASS2_SEED    = 0                      # @param {type:"integer"}

# ── Tiled VAE decode ──────────────────────────────────────────────────────────
# Node [265] LTXVSpatioTemporalTiledVAEDecode from ComfyUI-LTXVideo
USE_TILED_VAE          = False  # @param {type:"boolean"}
TILED_SPATIAL_TILES    = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP  = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN     = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX   = False  # @param {type:"boolean"}

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_PREFIX = "LTX-2-PRO"  # @param {type:"string"}

# ── Storyboard continuity ─────────────────────────────────────────────────────
USE_SCENE_CONTINUITY = True  # @param {type:"boolean"}
# When True in run_storyboard(), the last frame of scene N becomes
# the image_path seed for scene N+1 (seamless continuity).

print("✅ Configuration set.")
print(f"   Resolution : {WIDTH}×{HEIGHT}  |  Frames : {FRAMES}  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   UNet       : {UNET_MODEL}")
print(f"   Seed       : {SEED}  (auto-increment: {AUTO_INCREMENT_SEED})")
print(f"   Pass 1     : {PASS1_SAMPLER}  |  {PASS1_SIGMAS[:45]}…")
print(f"   Pass 2     : {PASS2_SAMPLER}  |  {PASS2_SIGMAS}")
print(f"   Pro Mode   : {PRO_MODE}  |  steps={PRO_STEPS}, scheduler={PRO_SCHEDULER}, split@{PRO_SPLIT_AT}")
print(f"   Overlap    : {OVERLAP_FRAMES} frames ({OVERLAP_MODE}, side={OVERLAP_SIDE})")
print(f"   Seg Extend : {USE_SEGMENT_EXTENSION}  |  {SEGMENT_LENGTH} frames x {MAX_SEGMENTS} segments")

# ── Audio Sync ────────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### Audio-Synced Generation

USE_AUDIO_SYNC = False     # @param {type:"boolean"}
# Sync segment boundaries to detected audio beats.
# Adjusts SEGMENT_LENGTH dynamically based on tempo.

AUDIO_SYNC_PATH = None     # @param {type:"string"}
# Path to audio file for beat detection.
# e.g. "/content/drive/MyDrive/music.mp3"

AUDIO_BPM = None           # @param {type:"integer"}
# Manual BPM fallback if beat detection fails.
# e.g. 120 for typical pop/rock, 60-80 for ambient, 140+ for EDM.

# ── Thumbnail Preview ─────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### Thumbnail Storyboard Preview

GENERATE_THUMBNAILS = False  # @param {type:"boolean"}
# Generate 1-frame thumbnail per scene before full generation.
# Displays grid preview so you can check composition before committing.

THUMBNAIL_COLS = 3          # @param {type:"integer"}
# Number of columns in thumbnail grid display.

# ── Style & Color ─────────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### Color Consistency & Grading

USE_COLOR_MATCHING = False  # @param {type:"boolean"}
# Extract color histogram from first segment as reference palette.
# Apply LAB color matching to all subsequent segments.

COLOR_GRADE = "none"        # @param ["none", "cinematic_warm", "noir", "cyberpunk", "vintage", "cool_blue", "golden_hour"]
# Apply color grading preset in post-processing to all frames.
# "none" = no grading applied.

# ── Google Drive Persistence ──────────────────────────────────────────────────
# @markdown ---
# @markdown ### Google Drive Sync

PERSIST_TO_GDRIVE = False   # @param {type:"boolean"}
# Auto-sync completed segments to Google Drive after each generation.
# Enables resume if Colab disconnects.

GDRIVE_PATH = "/content/drive/MyDrive/LTX_PRO_Output"  # @param {type:"string"}
# Google Drive folder for syncing generated videos and segments.

# ── Export & Timeline ─────────────────────────────────────────────────────────
# @markdown ---
# @markdown ### Export to Timeline

EXPORT_TIMELINE = False     # @param {type:"boolean"}
# Generate EDL or JSON timeline alongside video output.
# Includes timestamps, prompts, seeds for each segment.

TIMELINE_FORMAT = "json"    # @param ["json", "edl"]
# "json" = JSON timeline (re-importable, easy to parse)
# "edl"  = EDL (Edit Decision List, compatible with NLEs like DaVinci/Premiere)

# ── Parallel Prompt Expansion ─────────────────────────────────────────────────
# @markdown ---
# @markdown ### Performance Optimization

USE_PARALLEL_PROMPT_EXPANSION = False  # @param {type:"boolean"}
# In storyboard mode, expand ALL prompts in one batch before video generation.
# Avoids loading/unloading LLM N times. Caches expanded prompts to disk.

print(f"   Audio sync   : {USE_AUDIO_SYNC}  |  path={AUDIO_SYNC_PATH}  |  BPM={AUDIO_BPM}")
print(f"   Thumbnails   : {GENERATE_THUMBNAILS}  |  cols={THUMBNAIL_COLS}")
print(f"   Color match  : {USE_COLOR_MATCHING}  |  grade={COLOR_GRADE}")
print(f"   Drive sync   : {PERSIST_TO_GDRIVE}  |  path={GDRIVE_PATH}")
print(f"   Timeline     : {EXPORT_TIMELINE}  |  format={TIMELINE_FORMAT}")
print(f"   Parallel exp : {USE_PARALLEL_PROMPT_EXPANSION}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7  ─  DEFINE generate_pro()
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 7. Define generate_pro()
# @markdown Run this cell once per session. Edit Cells 5-6 and re-run Cell 9.

def generate_pro(
    user_input:              str   = USER_INPUT,
    image_path:              str   = IMAGE_PATH,
    positive_prompt:         str   = POSITIVE_PROMPT,
    negative_prompt:         str   = NEGATIVE_PROMPT,
    width:                   int   = WIDTH,
    height:                  int   = HEIGHT,
    frames:                  int   = FRAMES,
    fps:                     int   = FPS,
    seed:                    int   = SEED,
    image_strength:          float = IMAGE_STRENGTH,
    character_image_path:    str   = CHARACTER_IMAGE_PATH,
    character_strength:      float = CHARACTER_STRENGTH,
    character_mode:          str   = CHARACTER_CONSISTENCY_MODE,
    character_name:          str   = CHARACTER_NAME,
    character_description:   str   = CHARACTER_DESCRIPTION,
    pass1_sigmas:            str   = PASS1_SIGMAS,
    pass1_sampler:           str   = PASS1_SAMPLER,
    pass1_cfg:               float = PASS1_CFG,
    pass2_sigmas:            str   = PASS2_SIGMAS,
    pass2_sampler:           str   = PASS2_SAMPLER,
    pass2_cfg:               float = PASS2_CFG,
    pass2_seed:              int   = PASS2_SEED,
    pro_mode:                bool  = PRO_MODE,
    pro_steps:               int   = PRO_STEPS,
    pro_scheduler:           str   = PRO_SCHEDULER,
    pro_split_at:            int   = PRO_SPLIT_AT,
    use_tiled_vae:           bool  = USE_TILED_VAE,
    tiled_spatial_tiles:     int   = TILED_SPATIAL_TILES,
    tiled_spatial_overlap:   int   = TILED_SPATIAL_OVERLAP,
    tiled_temporal_len:      int   = TILED_TEMPORAL_LEN,
    tiled_temporal_overlap:  int   = TILED_TEMPORAL_OVERLAP,
    tiled_last_frame_fix:    bool  = TILED_LAST_FRAME_FIX,
    lora_stack:              list  = None,
    lora_stack_json:         str   = None,
    output_prefix:           str   = OUTPUT_PREFIX,
    # ── EasyPrompt / Vision settings (read from module globals by default) ──
    # These can be overridden per-call for storyboard/batch use.
    bypass_easy_prompt:      bool  = None,   # None → use BYPASS_EASY_PROMPT global
    llm_model:               str   = None,   # None → use LLM_MODEL global
    use_vision:              bool  = None,   # None → use USE_VISION global
    vision_model:            str   = None,   # None → use VISION_MODEL global
    unet_model:              str   = None,   # None → use UNET_MODEL global
    clip_name1:              str   = None,   # None → use CLIP_NAME1 global
    clip_name2:              str   = None,   # None → use CLIP_NAME2 global
    # ── New feature parameters (FEAT-003) ────────────────────────────────
    color_grade:             str   = None,   # None -> use COLOR_GRADE global
    use_color_matching:      bool  = None,   # None -> use USE_COLOR_MATCHING global
    reference_histogram:     object = None,  # np.ndarray from extract_color_histogram
    use_quality_gate:        bool  = None,   # None -> use USE_QUALITY_GATE global
    use_multi_resolution:    bool  = None,   # None -> use USE_MULTI_RESOLUTION global
    export_timeline:         bool  = None,   # None -> use EXPORT_TIMELINE global
    persist_to_gdrive:       bool  = None,   # None -> use PERSIST_TO_GDRIVE global
    timeline_entries:        list  = None,   # Mutable list for accumulating timeline data
    embedding_bank:          object = None,  # CharacterEmbeddingBank instance
    velocity_latent:         object = None,  # Velocity tensor for motion injection (noise bias)
) -> Optional[str]:
    """
    LTX-2 PRO — Two-pass generation pipeline with Character Consistency.

    Integrates nodes from LD-I2V.json + SVI-Pro-Workflow.json.
    All ComfyUI node calls are annotated with [JSON node id / type].

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PHASE 0 — EASY PROMPT (before video model — LLM/Vision then unload) │
    │                                                                      │
    │  [LTX2VisionDescribe]  image  ──────────────────► scene_context     │
    │  [LTX2PromptArchitect] user_input + scene_ctx ──► positive, neg     │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 1 — MODEL LOADING                                             │
    │                                                                      │
    │  [197/UnetLoaderGGUF]     ──► unet (raw)                            │
    │  [190/DualCLIPLoader]     ──► clip_model                            │
    │  [263/LTX2MasterLoaderLD] ──► unet + clip (LoRA stack)              │
    │  [PathchSageAttentionKJ]  ──► unet (sage attn patch, optional)      │
    │  [LTXVChunkFeedForward]   ──► unet (chunk FF patch, optional)       │
    │  [184/VAELoader]          ──► vae_video                             │
    │  [196/VAELoaderKJ]        ──► vae_audio                             │
    │  [189/LatentUpscaleModel] ──► upscale_model                         │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 2 — TEXT ENCODING                                             │
    │                                                                      │
    │  [121/CLIPTextEncode]     positive ──► cond_pos                     │
    │  [110/CLIPTextEncode]     negative ──► cond_neg                     │
    │  [ConditioningZeroOut]    cond_pos  ──► zero_out (for neg branch)   │
    │  [107/LTXVConditioning]   fps meta  ──► cond[0]=pos, cond[1]=neg    │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 3 — CHARACTER ANCHOR (mode "anchor" or "both")               │
    │                                                                      │
    │  [165/ImageResizeKJv2]    char_img  ──► resized to W×H              │
    │  [295/VAEEncode]          pixels    ──► anchor_samples LATENT       │
    │                           SetNode  ──► "anchor_samples" slot        │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 4 — LATENT PREPARATION                                        │
    │                                                                      │
    │  [246/ResizeImagesByLongerEdge]  image ──► 1536px long-edge         │
    │  [165/ImageResizeKJv2]           image ──► target W×H, lanczos      │
    │  [164/ResizeImageMaskNode]       image ──► ×0.5 half-res            │
    │  [163/GetImageSize]              ──► half_w, half_h                 │
    │  [108/EmptyLTXVLatentVideo]      ──► vid_lat (half-res)             │
    │  [162/LTXVPreprocess]            img_compression=33 ──► pp_img      │
    │  [161/LTXVImgToVideoInplace]     I2V mode ──► vid_lat (conditioned) │
    │  [199/LTXVEmptyLatentAudio]      ──► aud_lat                        │
    │  [109/LTXVConcatAVLatent]        vid_lat+aud_lat ──► combined       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 5 — SIGMA SCHEDULE                                            │
    │                                                                      │
    │  Standard: [ManualSigmas] pass1_sigmas ──► sig_p1                   │
    │  PRO mode: [ModelSamplingSD3 shift=8] + [BasicScheduler] +          │
    │            [SplitSigmas step=pro_split_at] ──► sig_high, sig_low    │
    │            [KSamplerSelect euler] ──► sampler                       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 6 — PASS 1 (first-pass denoising)                            │
    │                                                                      │
    │  [CFGGuider]  model + pos/neg ──► guider_p1                         │
    │  [RandomNoise] seed ──► noise_p1                                    │
    │  [SamplerCustomAdvanced] ──► p1_av_output                           │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 7 — PASS 2 (spatial upscale + refinement)                    │
    │                                                                      │
    │  [LTXVSeparateAVLatent]  p1_av ──► vid_lat_p1, aud_lat_p1          │
    │  [LTXVCropGuides]        pos/neg + lat ──► cropped cond + lat       │
    │  [CFGGuider]  model + cropped ──► guider_p2                         │
    │  [LTXVLatentUpsampler]   ×2 upsample ──► upsampled                  │
    │  [LTXVConcatAVLatent]    upsampled + aud ──► av_lat2                │
    │  PRO: sig_low  |  Standard: [ManualSigmas] pass2_sigmas             │
    │  [SamplerCustomAdvanced] ──► p2_denoised                            │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 8 — DECODE                                                    │
    │                                                                      │
    │  [LTXVSeparateAVLatent]                                              │
    │  [265/LTXVSpatioTemporalTiledVAEDecode] or [VAEDecode] ──► frames   │
    │  [201/LTXVAudioVAEDecode] ──► audio                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PHASE 9 — SAVE                                                      │
    │                                                                      │
    │  [319/VHS_VideoCombine] h264-mp4, crf=19, yuv420p (preferred)       │
    │  Fallback: [CreateVideo] ──► save_video_from_components             │
    │  save_metadata_sidecar() ──► JSON sidecar                           │
    └──────────────────────────────────────────────────────────────────────┘

    Returns: output video path (str) or None on failure.
    """
    t0 = time.time()
    import_custom_nodes()
    clear_output()

    _lora_stack = lora_stack if lora_stack is not None else LORA_STACK
    _lora_json  = lora_stack_json if lora_stack_json is not None else LORA_STACK_JSON
    _char_mode  = character_mode.lower().strip()

    # Resolve per-call overrides vs. module globals
    _bypass     = bypass_easy_prompt if bypass_easy_prompt is not None else BYPASS_EASY_PROMPT
    _llm_model  = llm_model  if llm_model  is not None else LLM_MODEL
    _use_vision = use_vision if use_vision is not None else USE_VISION
    _vis_model  = vision_model if vision_model is not None else VISION_MODEL
    _unet       = unet_model  if unet_model  is not None else UNET_MODEL
    _clip1      = clip_name1  if clip_name1  is not None else CLIP_NAME1
    _clip2      = clip_name2  if clip_name2  is not None else CLIP_NAME2

    # Resolve new feature flags (FEAT-003)
    _color_grade = color_grade if color_grade is not None else COLOR_GRADE
    _use_color_matching = use_color_matching if use_color_matching is not None else USE_COLOR_MATCHING
    _use_quality_gate = use_quality_gate if use_quality_gate is not None else USE_QUALITY_GATE
    _use_multi_res = use_multi_resolution if use_multi_resolution is not None else USE_MULTI_RESOLUTION
    _export_timeline = export_timeline if export_timeline is not None else EXPORT_TIMELINE
    _persist_to_gdrive = persist_to_gdrive if persist_to_gdrive is not None else PERSIST_TO_GDRIVE

    print("🎬 LTX-2 PRO — Generation Starting")
    print(f"   Resolution   : {width}×{height}  |  Frames: {frames}  |  Seed: {seed}")
    print(f"   Mode         : {'I2V' if image_path else 'T2V'}"
          f"  |  Character: {_char_mode}  |  Pro: {pro_mode}")
    print(f"   Easy Prompt  : {'BYPASS' if _bypass else f'LLM={_llm_model}'}")
    _print_vram()

    # ── Multi-resolution override (FEAT-003) ──────────────────────────────
    if _use_multi_res:
        try:
            _shot_type = detect_shot_type(user_input or positive_prompt)
            _new_w, _new_h, _anchor_w = get_resolution_for_shot(_shot_type, width, height)
            if _new_w != width or _new_h != height:
                print(f"   Multi-res    : {_shot_type} shot -> {_new_w}x{_new_h} (anchor={_anchor_w:.2f})")
                width, height = _new_w, _new_h
                if _shot_type == "closeup":
                    character_strength = min(1.0, character_strength * _anchor_w)
        except Exception as e:
            print(f"   ⚠️  Multi-resolution failed ({e}) - using original resolution")

    # ── Pre-flight model check ─────────────────────────────────────────────
    print("\n🔍 Model file check…")
    all_ok = validate_model_files({
        "unet"    : _unet,
        "clip1"   : _clip1,
        "clip2"   : _clip2,
        "vae_vid" : VAE_VIDEO_MODEL,
        "vae_aud" : VAE_AUDIO_MODEL,
        "upscaler": UPSCALER_MODEL,
    })
    if not all_ok:
        raise FileNotFoundError(
            "One or more model files are missing — run Cell 2 first.\n"
            "  Tip: Check CLIP_NAME1 — use fp8 if fp4 is not available for your GPU."
        )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 0 — EASY PROMPT (LLM + Vision before video model loads)
    # ══════════════════════════════════════════════════════════════════════

    # Load reference image tensor for Vision and/or I2V conditioning
    seed_image_tensor = None
    if image_path:
        seed_image_tensor = load_image_tensor(image_path)
        if seed_image_tensor is None:
            print(f"   ⚠️  Image not found: {image_path} — switching to T2V")
        else:
            print(f"   ✓ Reference image loaded: {image_path}  {seed_image_tensor.shape}")

    # Load character image tensor for consistency anchor
    char_image_tensor = None
    if character_image_path and _char_mode != "none":
        char_image_tensor = load_image_tensor(character_image_path)
        if char_image_tensor is None:
            print(f"   ⚠️  Character image not found: {character_image_path} — skipping anchor.")
        else:
            print(f"   ✓ Character image loaded: {character_image_path}  {char_image_tensor.shape}")

    # Determine analysis image: prefer character image for vision analysis
    analysis_tensor = char_image_tensor if char_image_tensor is not None else seed_image_tensor

    # Vision Describe — LTX2VisionDescribe node
    scene_context = character_description or ""
    if analysis_tensor is not None and _use_vision and not _bypass:
        print("\n👁️  Vision Describe…")
        # character_description is fed as seed to bias description toward the character
        scene_context = run_vision_describe(
            analysis_tensor,
            character_description,
            use_vision_override=_use_vision,
            vision_model_override=_vis_model)
        if scene_context:
            print(f"   Scene context: {scene_context[:120]}…")

    # Easy Prompt expansion — LTX2PromptArchitect node
    final_positive = positive_prompt
    final_negative = negative_prompt
    if not _bypass and user_input.strip():
        print("\n🧠 Easy Prompt expansion…")
        final_positive, final_negative = run_easy_prompt(
            user_input=user_input,
            frame_count=frames,
            seed=seed,
            scene_context=scene_context,
            llm_model_override=_llm_model,
        )
        print(f"\n   ── EXPANDED PROMPT ─────────────────────────────────────")
        print(f"   {final_positive[:300]}{'…' if len(final_positive) > 300 else ''}")
        print(f"\n   ── NEGATIVE PROMPT ─────────────────────────────────────")
        print(f"   {final_negative[:150]}…")
    else:
        print("   [EasyPrompt] Bypassed — using manual POSITIVE_PROMPT.")

    # LLM/Vision should now be unloaded — free VRAM before loading video model
    cleanup_memory(verbose=True)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1A - TEXT ENCODING (CLIP loaded, used, then freed before UNet)
    # ══════════════════════════════════════════════════════════════════════

    with torch.inference_mode():

        # ── DualCLIPLoader ────────────────────────────────────────────────
        # [190] DualCLIPLoader - Gemma text encoder + embeddings connector
        print("\n📦 Loading CLIP encoders (DualCLIPLoader)...")
        try:
            clip_loader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            clip_model  = get_value_at_index(
                clip_loader.load_clip(
                    clip_name1=_clip1,
                    clip_name2=_clip2,
                    type="ltxv",
                    device="default"), 0)
        except Exception as e:
            print(f"   ⚠️  fp4 CLIP failed ({type(e).__name__}: {e})")
            print("      Trying fp8 fallback (gemma_3_12B_it_fp8_scaled.safetensors)...")
            fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
            try:
                clip_model = get_value_at_index(
                    clip_loader.load_clip(
                        clip_name1=fp8, clip_name2=_clip2,
                        type="ltxv", device="default"), 0)
                print("   ✓ fp8 CLIP loaded.")
            except Exception as e2:
                raise RuntimeError(
                    f"DualCLIPLoader failed: {e2}\n"
                    "  Fix: Ensure CLIP_NAME1 file is downloaded (Cell 2).\n"
                    "  fp4 needs Blackwell GPU; use fp8 for T4/A100."
                )

        # ── Text Encoding (while CLIP is still in VRAM) ───────────────────
        print("\n📝 Encoding prompts...")
        try:
            # [121] CLIPTextEncode - positive
            cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cond_pos = cte.encode(text=final_positive, clip=clip_model)

            # [110] CLIPTextEncode - negative (empty for LTX, wrapped in ConditioningZeroOut)
            cond_neg = cte.encode(text=final_negative, clip=clip_model)

            # ConditioningZeroOut - applied to positive to create zero-out negative branch
            # (mirrors reference notebook pattern for distilled model)
            zero_out  = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_zero = zero_out.zero_out(
                conditioning=get_value_at_index(cond_pos, 0))

            # [107] LTXVConditioning - injects frame_rate into conditioning metadata
            ltxv_cond = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cond = ltxv_cond.EXECUTE_NORMALIZED(
                frame_rate=float(fps),
                positive=get_value_at_index(cond_pos, 0),
                negative=get_value_at_index(cond_zero, 0))
            # cond[0] = positive with frame_rate metadata
            # cond[1] = negative (zero-out)

        except Exception as e:
            raise RuntimeError(
                f"Text encoding failed: {e}\n"
                "  Fix: Check DualCLIPLoader output - CLIP may have failed to load."
            )

        # Delete CLIP now - frees ~6-8 GB for UNet
        del clip_model
        aggressive_cleanup("CLIP deleted")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1B - UNET LOADING (after CLIP is freed from VRAM)
        # ══════════════════════════════════════════════════════════════════

        # ── UNet: UnetLoaderGGUF ──────────────────────────────────────────
        # [197] UnetLoaderGGUF in LD-I2V.json - loads GGUF Q4_K_M distilled
        print("\n📦 Loading UNet (GGUF Q4_K_M distilled)...")
        try:
            unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unet        = get_value_at_index(
                unet_loader.load_unet(unet_name=_unet), 0)
        except KeyError:
            raise RuntimeError(
                "UnetLoaderGGUF not found.\n"
                "  Fix: Run Cell 1 to clone ComfyUI_GGUF custom node."
            )

        # ── LoRA stack: LTX2MasterLoaderLD [263] ─────────────────────────
        # [263] LTX2MasterLoaderLD in LD-I2V.json - 10-slot LoRA stacker
        # Pass None for clip_model - IC/camera LoRAs are model-only
        print("   Applying LoRA stack (LTX2MasterLoaderLD)...")
        unet, _ = apply_lora_stack(unet, None, _lora_stack, _lora_json)

        # ── Optional performance patches ──────────────────────────────────
        # [PathchSageAttentionKJ] - KJNodes flash-attention-style patch
        unet = apply_sage_attention(unet)
        # [LTXVChunkFeedForward] - ComfyUI-LTXVideo chunk feedforward
        unet = apply_chunk_ff(unet)

        # Purge VRAM after model loading if enabled
        # [LayerUtility: PurgeVRAM V2] from LayerStyle nodes
        purge_vram("after unet+lora")
        _print_vram()

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3 — CHARACTER ANCHOR (mode "anchor" or "both")
        # NOTE: anchor_latent is computed AFTER half-res dimensions are known
        # (Phase 4 preamble) so the VAEEncode output matches EmptyLTXVLatentVideo.
        # ══════════════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4 — LATENT PREPARATION
        # ══════════════════════════════════════════════════════════════════

        print("\n🗂️  Preparing latents…")
        _print_vram()

        # ── Dual-Anchor Logic (FEAT: identity fix) ────────────────────────
        # In "both" mode with TWO images available (continuity + character),
        # use continuity frame for I2V (temporal bridge) and character image
        # for anchor latent (identity preservation). This is the key fix for
        # character drift across storyboard scenes.
        _ref_tensor = seed_image_tensor
        _use_i2v = False

        if _char_mode == "both" and seed_image_tensor is not None and char_image_tensor is not None:
            # DUAL-ANCHOR: continuity frame -> I2V, character image -> anchor latent
            # _ref_tensor already == seed_image_tensor (continuity frame for I2V)
            _use_i2v = True
            print(f"   \U0001f9ec Dual-anchor: continuity frame -> I2V | character image -> anchor latent")
        elif _char_mode == "both" and seed_image_tensor is None and char_image_tensor is not None:
            # Only character image available - use it for BOTH I2V and anchor
            _ref_tensor = char_image_tensor
            _use_i2v = True
        elif _char_mode == "i2v":
            # I2V mode: prefer seed_image (continuity), fallback to character image
            if seed_image_tensor is None and char_image_tensor is not None:
                _ref_tensor = char_image_tensor
            _use_i2v = (_ref_tensor is not None)
        elif _char_mode == "anchor":
            # Anchor-only: no I2V conditioning, character handled separately
            _ref_tensor = None
            _use_i2v = False
        elif _char_mode == "none" and seed_image_tensor is not None:
            # No character mode but have an explicit image_path - use for I2V
            # _ref_tensor already == seed_image_tensor
            _use_i2v = True
        else:
            _ref_tensor = None
            _use_i2v = False

        # ── Compute half-resolution for Pass 1 latent ─────────────────────
        # [256] EmptyImage → [164] ResizeImageMaskNode ×0.5 → [163] GetImageSize
        ei       = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_img = ei.generate(width=width, height=height, batch_size=1, color=0)

        rimn     = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_img = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_img, 0),
            scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})

        gis     = NODE_CLASS_MAPPINGS["GetImageSize"]()
        half_sz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_img, 0))
        half_w  = get_value_at_index(half_sz, 0)
        half_h  = get_value_at_index(half_sz, 1)
        print(f"   Latent dims : {half_w}×{half_h}  (half of {width}×{height})")

        # ── Character Anchor encoding (Phase 3, deferred here for half-res dims) ──
        # [295] VAEEncode from SVI-Pro-Workflow.json - encodes character image.
        # We resize to half_w x half_h so the anchor latent matches EmptyLTXVLatentVideo.
        anchor_latent = None
        if char_image_tensor is not None and _char_mode in ("anchor", "both"):
            print("\n🧬 Character Anchor - encoding character image as latent...")
            try:
                # [165] ImageResizeKJv2 - resize to HALF resolution (matches vid_lat)
                # half_w x half_h ensures spatial dims match EmptyLTXVLatentVideo
                ikj = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
                char_resized = get_value_at_index(
                    ikj.resize(
                        image=char_image_tensor,
                        width=half_w,
                        height=half_h,
                        upscale_method="lanczos",
                        keep_proportion="crop",
                        pad_color="0, 0, 0",
                        crop_position="center",
                        divisible_by=32,
                        device="cpu",
                    ), 0)

                # Load VAE fresh for anchor encoding
                vae_for_anchor = get_value_at_index(
                    NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

                # [295] VAEEncode - pixels at half_w x half_h -> latent at ~(half_w/8 x half_h/8)
                # This matches the spatial dims of EmptyLTXVLatentVideo(half_w, half_h)
                vae_enc       = NODE_CLASS_MAPPINGS["VAEEncode"]()
                anchor_latent = get_value_at_index(
                    vae_enc.encode(pixels=char_resized, vae=vae_for_anchor), 0)
                del vae_for_anchor
                aggressive_cleanup("VAE anchor done")
                print(f"   ✓ Character anchor encoded at {half_w}×{half_h}  (mode={_char_mode})")
            except Exception as e:
                print(f"   ⚠️  Character anchor failed ({e}) - continuing without anchor.")
                anchor_latent = None

        # [108] EmptyLTXVLatentVideo — half-res video latent
        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1)

        # ── I2V conditioning branch ────────────────────────────────────────
        if _use_i2v and _ref_tensor is not None:
            try:
                # Step 1: Resize reference image to target W x H using ResizeImageMaskNode
                # (matches the working LTX2_Infinite_Flow_PRO_v2.py pattern)
                _rim_i2v = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
                _ref_tensor = get_value_at_index(
                    _rim_i2v.EXECUTE_NORMALIZED(
                        input=_ref_tensor,
                        scale_method="lanczos",
                        resize_type={"resize_type": "scale dimensions",
                                     "width": width, "height": height,
                                     "crop": "center"}), 0)

                # Step 2: [246] ResizeImagesByLongerEdge - longer_edge = max(W, H)
                # Uses EXECUTE_NORMALIZED (not .resize) - matches PRO_v2 working code
                if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                    rle = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
                    _ref_tensor = get_value_at_index(
                        rle.EXECUTE_NORMALIZED(
                            longer_edge=max(width, height),
                            images=_ref_tensor), 0)

                # [162] LTXVPreprocess - compress/normalise image before I2V injection
                # img_compression=33 matches LD-I2V.json node [162] widgets_values
                pp_node = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
                pp_img  = get_value_at_index(
                    pp_node.EXECUTE_NORMALIZED(
                        img_compression=33,
                        image=_ref_tensor), 0)

                # Load VAE fresh for I2V conditioning
                vae_for_i2v = get_value_at_index(
                    NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

                # [161] LTXVImgToVideoInplace - inject image into video latent
                # strength = character_strength (if char mode) else image_strength
                _i2v_strength = character_strength if _char_mode in ("i2v", "both") \
                                else image_strength
                i2v     = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
                vid_lat = i2v.EXECUTE_NORMALIZED(
                    strength=_i2v_strength,
                    bypass=False,
                    vae=vae_for_i2v,
                    image=pp_img,
                    latent=get_value_at_index(vid_lat, 0))
                del vae_for_i2v
                aggressive_cleanup("VAE I2V done")
                print(f"   ✓ I2V conditioning applied  (strength={_i2v_strength}, "
                      f"LTXVImgToVideoInplace)")
            except KeyError as e:
                print(f"   ⚠️  I2V node missing ({e}) - using empty latent (T2V mode).")
                vid_lat = (get_value_at_index(vid_lat, 0),)
            except Exception as e:
                print(f"   ⚠️  I2V conditioning failed ({e}) - using empty latent.")
                vid_lat = (get_value_at_index(vid_lat, 0),)
        else:
            # T2V - use empty latent directly
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Inject anchor_latent as a constraint if in anchor mode
        # (SetNode "anchor_samples" — mirrors SVI-Pro-Workflow.json nodes [169/295])
        # NOTE: In SVI-Pro-Workflow.json, anchor_samples feeds WanImageToVideoSVIPro —
        # a Wan2.2-only node with no direct LTX-2 equivalent.
        # For LTX-2 the closest approximation is to use the anchor latent AS the
        # initial video latent (replacing the empty latent) so the sampler starts
        # from the character's encoded appearance rather than pure noise.
        _vid_lat_input = get_value_at_index(vid_lat, 0)
        if anchor_latent is not None:
            try:
                # ── Dual-Anchor Blend (FEAT: identity fix) ─────────────────────
                # Instead of replacing the entire video latent with the anchor,
                # BLEND the anchor into the first K frames of the video latent.
                # This biases the opening frames toward character identity while
                # preserving I2V continuity conditioning in the rest of the latent.
                _anch_samples = anchor_latent.get("samples", None)
                _vid_samples = _vid_lat_input.get("samples", None) if isinstance(_vid_lat_input, dict) else None

                if _anch_samples is not None and _vid_samples is not None and \
                   USE_DUAL_ANCHOR_STORYBOARD and _char_mode == "both":
                    # Ensure anchor is 5D (N, C, T, H, W)
                    if _anch_samples.ndim == 4:
                        _anch_samples = _anch_samples.unsqueeze(2)  # (N, C, 1, H, W)

                    # Determine K = number of frames to blend into
                    _total_t = _vid_samples.shape[2] if _vid_samples.ndim == 5 else 1
                    _K = min(ANCHOR_BLEND_FRAMES, _total_t)

                    # Repeat anchor across K frames
                    _anchor_repeated = _anch_samples.repeat(1, 1, _K, 1, 1)

                    # Spatial dimension check - only blend if shapes match
                    if _vid_samples.ndim == 5 and _anchor_repeated.shape[3:] == _vid_samples[:, :, :_K, :, :].shape[3:]:
                        _strength = float(LATENT_OVERLAP_STRENGTH)
                        _vid_samples = _vid_samples.clone()
                        _vid_samples[:, :, :_K, :, :] = (
                            (1.0 - _strength) * _vid_samples[:, :, :_K, :, :] +
                            _strength * _anchor_repeated
                        )
                        _vid_lat_input = {**_vid_lat_input, "samples": _vid_samples}
                        print(f"   \u2713 Dual-anchor blend: {_strength:.0%} anchor into first {_K} frames  (mode={_char_mode})")
                    else:
                        # Shape mismatch - skip injection entirely rather than
                        # reverting to full latent replacement (the broken behavior)
                        print(f"   \u26a0\ufe0f  Anchor spatial dims don't match video latent - SKIPPING anchor injection.")
                        print(f"     Anchor: {list(_anchor_repeated.shape[3:])}, Video: {list(_vid_samples[:, :, :_K, :, :].shape[3:])}")
                else:
                    # Non-dual-anchor path: use anchor latent as the starting video latent
                    # (original behavior for anchor-only mode or when flag is disabled)
                    _vid_lat_input = anchor_latent
                    print(f"   \u2713 Character anchor injected as video latent seed  (mode={_char_mode})")

                # Diagnostic: check tensor rank
                _anch_shape = anchor_latent.get("samples", torch.empty(0)).shape
                print(f"     Anchor latent shape: {list(_anch_shape)}")
                if len(_anch_shape) == 4 and not (USE_DUAL_ANCHOR_STORYBOARD and _char_mode == "both"):
                    # Standard VAEEncode returns 4D (N, C, H, W); LTX needs 5D (N, C, T, H, W)
                    print("     \u26a0\ufe0f  Anchor is 4D (image latent) - unsqueezing T dim for video latent.")
                    _s = anchor_latent["samples"].unsqueeze(2)   # -> (N, C, 1, H, W)
                    _vid_lat_input = {**anchor_latent, "samples": _s}
                    print(f"     Anchor latent shape after fix: {list(_s.shape)}")
            except Exception as e:
                print(f"   \u26a0\ufe0f  Anchor injection error ({e}) - using empty/I2V latent.")
                _vid_lat_input = get_value_at_index(vid_lat, 0)

        # [199] LTXVEmptyLatentAudio - audio latent
        # Load audio VAE right before it's needed (~1GB, kept through audio decode)
        # [196] VAELoaderKJ (or VAELoader fallback) - audio VAE
        vae_audio = None
        try:
            vae_audio = get_value_at_index(_load_audio_vae(VAE_AUDIO_MODEL), 0)
        except Exception as e:
            raise RuntimeError(
                f"Audio VAE load failed: {e}\n"
                "  Fix: Check VAE_AUDIO_MODEL filename in Cell 6."
            )

        elalat  = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1,
            audio_vae=vae_audio)

        # [109] LTXVConcatAVLatent — combine video + audio latents
        catav           = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat1         = catav.EXECUTE_NORMALIZED(
            video_latent=_vid_lat_input,
            audio_latent=get_value_at_index(aud_lat, 0))
        combined_latent = get_value_at_index(av_lat1, 0)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 5 — SIGMA SCHEDULE
        # ══════════════════════════════════════════════════════════════════

        manualsigmas   = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise    = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider      = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca            = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        sig_p1_high = None   # used for PRO mode pass 1 (sigmas_high)
        sig_p2_low  = None   # used for PRO mode pass 2 (sigmas_low)

        if pro_mode:
            print(f"\n⚙️  PRO sigma schedule — BasicScheduler steps={pro_steps} "
                  f"sched={pro_scheduler} split@{pro_split_at}")
            print("   ⚠️  EXPERIMENTAL: SVI-Pro sigma chain was designed for Wan2.2.")
            print("      ModelSamplingSD3 applies SD3 cosine flow parameterisation.")
            print("      LTX-2 GGUF uses a different flow schedule — output quality")
            print("      may vary. Use ManualSigmas mode (PRO_MODE=False) if results")
            print("      are degraded or distorted.")
            try:
                # [ModelSamplingSD3] shift=8 — sigma rescaling for flow matching
                # From SVI-Pro-Workflow.json sigma scheduling approach
                ms3  = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
                unet_sampled = get_value_at_index(
                    ms3.patch(model=unet, shift=8.0), 0)

                # [BasicScheduler] — generates full sigma schedule
                bs   = NODE_CLASS_MAPPINGS["BasicScheduler"]()
                sigs = get_value_at_index(
                    bs.get_sigmas(
                        model=unet_sampled,
                        scheduler=pro_scheduler,
                        steps=pro_steps,
                        denoise=1.0), 0)

                # [SplitSigmas] — splits into high and low sigma passes
                # step_index=pro_split_at divides the schedule at that step
                ss         = NODE_CLASS_MAPPINGS["SplitSigmas"]()
                split_out  = ss.get_sigmas(sigmas=sigs, step=pro_split_at)
                sig_p1_high = get_value_at_index(split_out, 0)  # sigmas_high
                sig_p2_low  = get_value_at_index(split_out, 1)  # sigmas_low

                # Use patched model for sampling
                unet = unet_sampled

                # KSamplerSelect euler — sampler for both passes in PRO mode
                sampler_p1 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
                sampler_p2 = sampler_p1   # reuse for pass 2 in PRO mode

                print(f"   ✓ PRO sigmas computed (ModelSamplingSD3 + BasicScheduler + SplitSigmas)")

            except KeyError as e:
                print(f"   ⚠️  PRO mode node missing: {e} — falling back to ManualSigmas.")
                pro_mode   = False
            except Exception as e:
                print(f"   ⚠️  PRO schedule failed ({e}) — falling back to ManualSigmas.")
                pro_mode   = False

        if not pro_mode:
            print(f"\n⚙️  Standard sigma schedule — Pass1: {pass1_sigmas[:45]}…")
            sig_p1_high = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0)
            sampler_p1  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)
            sig_p2_low  = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0)
            sampler_p2  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 6 — PASS 1 (first-pass denoising)
        # ══════════════════════════════════════════════════════════════════

        print(f"\n🚀 Pass 1 — denoising…")
        _print_vram()

        noise_p1  = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
        guider_p1 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg,
            model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1))

        try:
            out1 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise_p1, 0),
                guider=get_value_at_index(guider_p1, 0),
                sampler=get_value_at_index(sampler_p1, 0),
                sigmas=sig_p1_high,
                latent_image=combined_latent)
            p1_av = get_value_at_index(out1, 0)  # raw AV output → Pass 2
        except Exception as e:
            if vae_audio is not None:
                del vae_audio
                vae_audio = None
            raise RuntimeError(
                f"Pass 1 sampling failed: {e}\n"
                "  Fix: If you see 'deformed output', try a different SEED.\n"
                "  Three consecutive deformations → change USER_INPUT/POSITIVE_PROMPT."
            )

        del guider_p1
        aggressive_cleanup("Pass 1 done")
        print("   ✓ Pass 1 complete")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 7 — PASS 2 (spatial upscale + refinement)
        # ══════════════════════════════════════════════════════════════════

        print(f"\n🔧 Pass 2 — upscale + refinement…")
        _print_vram()

        # [LTXVSeparateAVLatent] — split P1 AV → video + audio
        ltxvsep    = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1         = ltxvsep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vid_lat_p1 = get_value_at_index(s1, 0)
        aud_lat_p1 = get_value_at_index(s1, 1)

        # [LTXVCropGuides] — trim conditioning to match upscaled latent dims
        ltxvcrop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cropped  = ltxvcrop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vid_lat_p1)
        # cropped[0]=positive, [1]=negative, [2]=cropped_video_lat

        # CFGGuider for Pass 2 with cropped conditioning
        guider_p2 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg,
            model=unet,
            positive=get_value_at_index(cropped, 0),
            negative=get_value_at_index(cropped, 1))

        # [LTXVLatentUpsampler] [118] - 2x spatial upsample
        # Load VAE and upscale model fresh for upsampling
        vae_for_up = get_value_at_index(
            NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

        # [189] LatentUpscaleModelLoader in LD-I2V.json
        try:
            uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            # Try EXECUTE_NORMALIZED first; fall back to load_model if absent
            if hasattr(uml, "EXECUTE_NORMALIZED"):
                upscale_model = get_value_at_index(
                    uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0)
            elif hasattr(uml, "load_model"):
                upscale_model = get_value_at_index(
                    uml.load_model(model_name=UPSCALER_MODEL), 0)
            else:
                raise AttributeError("LatentUpscaleModelLoader: no load method found")
        except Exception as e:
            if vae_audio is not None:
                del vae_audio
                vae_audio = None
            raise RuntimeError(
                f"LatentUpscaleModelLoader failed: {e}\n"
                "  Fix: Download UPSCALER_MODEL in Cell 2."
            )

        ltxvup    = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = ltxvup.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model,
            vae=vae_for_up)
        del vae_for_up, upscale_model
        aggressive_cleanup("VAE upscale done")

        # [LTXVConcatAVLatent] [117] — upsampled video + audio
        av_lat2 = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(upsampled, 0),
            audio_latent=aud_lat_p1)

        noise_p2 = randomnoise.EXECUTE_NORMALIZED(noise_seed=pass2_seed)

        try:
            out2 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise_p2, 0),
                guider=get_value_at_index(guider_p2, 0),
                sampler=get_value_at_index(sampler_p2, 0),
                sigmas=sig_p2_low,
                latent_image=get_value_at_index(av_lat2, 0))
            p2_denoised = get_value_at_index(out2, 1)  # denoised_output slot
        except Exception as e:
            if vae_audio is not None:
                del vae_audio
                vae_audio = None
            raise RuntimeError(
                f"Pass 2 sampling failed: {e}\n"
                "  Fix: Try reducing TILED_SPATIAL_TILES or USE_TILED_VAE=False."
            )

        del guider_p2, unet
        aggressive_cleanup("Pass 2 done - UNet freed")
        print("   ✓ Pass 2 complete")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 8 — DECODE
        # ══════════════════════════════════════════════════════════════════

        print("\n🎞️  Decoding video & audio…")
        _print_vram()

        # [LTXVSeparateAVLatent] [125] — final split
        s2          = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        # ── Video decode ───────────────────────────────────────────────────
        # Load VAE fresh for decode
        vae_for_decode = get_value_at_index(
            NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

        decoded_frames = None
        if use_tiled_vae:
            # [265] LTXVSpatioTemporalTiledVAEDecode - ComfyUI-LTXVideo
            # VRAM-efficient tiled spatiotemporal decode
            try:
                tiled_dec = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded_frames = get_value_at_index(
                    tiled_dec.EXECUTE_NORMALIZED(
                        vae=vae_for_decode,
                        latents=vid_lat_fin,
                        spatial_tiles=tiled_spatial_tiles,
                        spatial_overlap=tiled_spatial_overlap,
                        temporal_tile_length=tiled_temporal_len,
                        temporal_overlap=tiled_temporal_overlap,
                        last_frame_fix=tiled_last_frame_fix,
                        working_device="auto",
                        working_dtype="auto"), 0)
                print("   ✓ Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode)")
            except (KeyError, Exception) as e:
                print(f"   ⚠️  Tiled VAE unavailable ({type(e).__name__}: {e})")
                print("      Falling back to standard VAEDecode.")
                print("      Fix: Clone ComfyUI-LTXVideo in Cell 1, or set USE_TILED_VAE=False.")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_frames is None:
            # Standard VAEDecode - always available in ComfyUI core
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_for_decode), 0)
            print("   ✓ Standard VAE decode (VAEDecode)")

        del vae_for_decode
        aggressive_cleanup("VAE decode done")

        # ── Audio decode [201] ─────────────────────────────────────────────
        # [201] LTXVAudioVAEDecode - decode audio latent
        try:
            aud_dec   = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            audio_out = aud_dec.EXECUTE_NORMALIZED(
                samples=aud_lat_fin,
                audio_vae=vae_audio)
        except Exception as e:
            print(f"   ⚠️  Audio decode failed ({e}) - proceeding without audio.")
            audio_out = None

        del vae_audio
        aggressive_cleanup("audio VAE done")

        # ── POST-PROCESSING: Color matching & grading (FEAT-003) ──────────
        if decoded_frames is not None:
            # Color histogram matching (style transfer from reference)
            if _use_color_matching and reference_histogram is not None:
                try:
                    decoded_frames = match_color_histogram(decoded_frames, reference_histogram)
                    print("   ✓ Color histogram matching applied")
                except Exception as e:
                    print(f"   ⚠️  Color matching failed ({e}) - continuing without.")

            # Color grading preset
            if _color_grade and _color_grade != "none":
                try:
                    decoded_frames = apply_color_grade(decoded_frames, _color_grade)
                    print(f"   ✓ Color grade applied: {_color_grade}")
                except Exception as e:
                    print(f"   ⚠️  Color grading failed ({e}) - continuing without.")

            # Character embedding bank accumulation
            if embedding_bank is not None:
                try:
                    # Extract compact spatial-mean features from the last frame
                    # rather than storing raw pixels - reduces memory and creates
                    # a more meaningful representation for consistency matching
                    _raw_frame = decoded_frames[-1:]
                    if _raw_frame.ndim >= 3:
                        # Spatial average: collapse H,W dims to get compact feature vector
                        # Input shape: (1, H, W, C) or (1, C, H, W)
                        _feat = _raw_frame.float().mean(dim=-2).mean(dim=-2)  # -> (1, C)
                    else:
                        _feat = _raw_frame.float()
                    embedding_bank.accumulate(_feat)
                    print(f"   \u2713 Character embedding bank updated ({len(embedding_bank)} samples)")
                except Exception as e:
                    print(f"   \u26a0\ufe0f  Embedding bank update failed ({e})")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 9 — SAVE  (VHS_VideoCombine preferred, CreateVideo fallback)
        # ══════════════════════════════════════════════════════════════════

        print("\n💾 Saving video…")
        _print_vram()
        output_path = None

        # Build output filename with character name for tracking
        _prefix = output_prefix
        if character_name and character_name != "Character":
            _prefix = f"{output_prefix}-{character_name}"

        # ── Try VHS_VideoCombine first ────────────────────────────────────
        # [319/391/317] VHS_VideoCombine from SVI-Pro-Workflow.json
        # format=video/h264-mp4, pix_fmt=yuv420p, crf=19 (matches original workflow)
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS and audio_out is not None:
            try:
                vhs  = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                _audio_data = get_value_at_index(audio_out, 0)

                vhs_out = vhs.combine_video(
                    images=decoded_frames,
                    frame_rate=fps,
                    loop_count=0,
                    filename_prefix=_prefix,
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=True,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    audio=_audio_data,
                )
                # VHS_VideoCombine returns VHS_FILENAMES — extract path.
                # VideoHelperSuite 1.7.9 uses video_paths (plural); older builds
                # use video_path (singular). Try both, then list/tuple fallback.
                _fnames = get_value_at_index(vhs_out, 0)
                if hasattr(_fnames, "video_paths") and _fnames.video_paths:
                    output_path = _fnames.video_paths[0]
                elif hasattr(_fnames, "video_path"):
                    output_path = _fnames.video_path
                elif isinstance(_fnames, (list, tuple)) and len(_fnames) > 0:
                    output_path = _fnames[0]
                else:
                    print(f"   ⚠️  VHS_FILENAMES type={type(_fnames)} — "
                          f"cannot extract path, falling back to CreateVideo.")
                    output_path = None
                if output_path:
                    print(f"   ✓ Saved via VHS_VideoCombine (h264-mp4, crf=19, yuv420p)")

            except Exception as e:
                print(f"   ⚠️  VHS_VideoCombine failed ({e}) — using CreateVideo fallback.")
                output_path = None

        # ── Fallback: CreateVideo ─────────────────────────────────────────
        if output_path is None:
            try:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                _aud_arg    = get_value_at_index(audio_out, 0) if audio_out else None
                if _aud_arg is not None:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(
                        fps=fps, images=decoded_frames, audio=_aud_arg)
                else:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(
                        fps=fps, images=decoded_frames)
                output_path = save_video_from_components(
                    get_value_at_index(vid_obj, 0), prefix=_prefix)
                print(f"   ✓ Saved via CreateVideo fallback")
            except Exception as e:
                raise RuntimeError(
                    f"Video save failed: {e}\n"
                    "  Fix: Check /content/ComfyUI/output/ permissions.\n"
                    "  Also try: VHS_VideoCombine node may need ComfyUI-VideoHelperSuite."
                )

    # ── Timing ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    # ── Quality gate check (FEAT-003) ─────────────────────────────────────
    quality_scores = None
    if _use_quality_gate and decoded_frames is not None:
        try:
            quality_scores = compute_segment_quality(decoded_frames)
            if quality_scores.get("passed"):
                print(f"   ✓ Quality gate PASSED (SSIM={quality_scores.get('ssim', 0):.3f}, "
                      f"hist={quality_scores.get('histogram', 0):.3f})")
            else:
                print(f"   ⚠️  Quality gate FAILED (SSIM={quality_scores.get('ssim', 0):.3f}, "
                      f"hist={quality_scores.get('histogram', 0):.3f})")
        except Exception as e:
            print(f"   ⚠️  Quality gate check failed ({e})")

    # ── Metadata JSON sidecar ─────────────────────────────────────────────
    _active_loras = [s["lora"] for s in _lora_stack if s.get("on")]
    meta = {
        "seed"              : seed,
        "width"             : width,
        "height"            : height,
        "frames"            : frames,
        "fps"               : fps,
        "positive_prompt"   : final_positive,
        "negative_prompt"   : final_negative,
        "user_input"        : user_input,
        "image_path"        : image_path,
        "character_image"   : character_image_path,
        "character_mode"    : _char_mode,
        "character_name"    : character_name,
        "character_strength": character_strength,
        "pro_mode"          : pro_mode,
        "pro_steps"         : pro_steps if pro_mode else None,
        "pro_scheduler"     : pro_scheduler if pro_mode else None,
        "loras"             : _active_loras,
        "unet_model"        : UNET_MODEL,
        "elapsed_seconds"   : elapsed,
        "output_path"       : output_path,
        "quality_scores"    : quality_scores,
    }
    if output_path:
        save_metadata_sidecar(output_path, meta)

    # ── Google Drive sync (FEAT-003) ──────────────────────────────────────
    if _persist_to_gdrive and output_path:
        try:
            sync_to_drive(output_path, GDRIVE_PATH)
        except Exception as e:
            print(f"   ⚠️  Drive sync failed ({e})")

    # ── Timeline entry (FEAT-003) ─────────────────────────────────────────
    if _export_timeline and timeline_entries is not None:
        try:
            timeline_entries.append({
                "segment_index": len(timeline_entries),
                "output_path": output_path,
                "seed": seed,
                "prompt": final_positive[:200],
                "user_input": user_input[:100] if user_input else "",
                "width": width,
                "height": height,
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps,
                "timestamp_start": sum(e.get("duration_seconds", 0) for e in timeline_entries),
                "character_name": character_name,
                "quality_scores": quality_scores,
            })
        except Exception as e:
            print(f"   ⚠️  Timeline entry failed ({e})")

    print(f"\n✅ Done in {mins}m {secs}s")
    print(f"   📁 {output_path}")
    _print_vram()

    # ── Preview & download ─────────────────────────────────────────────────
    if SHOW_PREVIEWS and output_path:
        print("\n▶ Preview:")
        display_video(output_path)

    if DOWNLOAD_AFTER_GENERATE and output_path:
        print("   ⬇️  Auto-downloading…")
        try:
            files.download(output_path)
        except Exception as e:
            print(f"   ⚠️  Download failed ({e}) — file is at {output_path}")

    return output_path


print("✅ generate_pro() and generate_extended_video() defined — run Cell 9 to generate.")
print("   generate_pro(): Single clip generation with character consistency")
print("   generate_extended_video(): SVI-Pro style multi-segment with overlap blending")


def generate_extended_video(
    user_input:              str   = USER_INPUT,
    image_path:              str   = IMAGE_PATH,
    positive_prompt:         str   = POSITIVE_PROMPT,
    negative_prompt:         str   = NEGATIVE_PROMPT,
    width:                   int   = WIDTH,
    height:                  int   = HEIGHT,
    fps:                     int   = FPS,
    seed:                    int   = SEED,
    image_strength:          float = IMAGE_STRENGTH,
    character_image_path:    str   = CHARACTER_IMAGE_PATH,
    character_strength:      float = CHARACTER_STRENGTH,
    character_mode:          str   = CHARACTER_CONSISTENCY_MODE,
    character_name:          str   = CHARACTER_NAME,
    character_description:   str   = CHARACTER_DESCRIPTION,
    # SVI-Pro segment extension settings
    segment_length:          int   = SEGMENT_LENGTH,
    max_segments:            int   = MAX_SEGMENTS,
    overlap_frames:          int   = OVERLAP_FRAMES,
    overlap_mode:            str   = OVERLAP_MODE,
    overlap_side:            str   = OVERLAP_SIDE,
    segment_seed_mode:       str   = SEGMENT_SEED_MODE,
    # Passthrough settings
    output_prefix:           str   = OUTPUT_PREFIX,
    **kwargs,
) -> Optional[str]:
    """
    Generate an extended-length video using SVI-Pro-style segment iteration.
    
    Mirrors the SVI-Pro-Workflow.json approach:
    1. Generate first segment from input image (81 frames)
    2. Extract last frame as anchor for next segment
    3. Generate next segment with overlap blending
    4. Repeat for max_segments iterations
    5. Stitch all segments with ImageBatchExtendWithOverlap-style blending
    
    Key SVI-Pro techniques applied:
    - Two-model architecture (HIGH/LOW noise passes via generate_pro)
    - 81 frames per segment (~5s at 16fps)
    - 5-frame linear blend overlap for seamless transitions
    - Fixed seed for reproducible generation
    - Character anchor maintained across all segments
    
    Returns: Final stitched video path, or None on failure.
    """
    import shutil
    
    t0 = time.time()
    print("=" * 70)
    print("🎬 SVI-Pro Extended Video Generation")
    print(f"   Segments     : {max_segments} x {segment_length} frames")
    print(f"   Overlap      : {overlap_frames} frames ({overlap_mode})")
    print(f"   Target length: ~{(max_segments * segment_length - (max_segments-1) * overlap_frames) / fps:.1f}s @ {fps}fps")
    print(f"   Seed mode    : {segment_seed_mode} (base={seed})")
    print("=" * 70)
    
    # Setup directories
    cache_dir = f"/content/ComfyUI/output/{output_prefix}_segments"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Compute seeds for all segments
    seeds = compute_segment_seeds(seed, max_segments, segment_seed_mode)
    
    # Track all generated frame tensors for final stitching
    all_segment_paths = []
    current_anchor_path = image_path  # Start with user's input image

    # ── Initialize advanced features (FEAT-003) ──────────────────────────
    _timeline_entries = [] if EXPORT_TIMELINE else None
    _embedding_bank = CharacterEmbeddingBank() if USE_CHARACTER_EMBEDDING_BANK else None
    _reference_histogram = None  # Set after first segment

    # ── Audio sync: adjust segment boundaries (FEAT-003) ─────────────────
    segment_lengths = [segment_length] * max_segments
    if USE_AUDIO_SYNC and AUDIO_SYNC_PATH:
        try:
            beats = detect_beats(AUDIO_SYNC_PATH, AUDIO_BPM)
            if beats:
                _boundaries = compute_segment_boundaries(beats, fps, segment_length)
                # Convert absolute frame positions to per-segment deltas
                if len(_boundaries) >= 2:
                    segment_lengths = [_boundaries[i+1] - _boundaries[i]
                                       for i in range(len(_boundaries) - 1)]
                    segment_lengths = segment_lengths[:max_segments]
                print(f"   Audio sync: {len(segment_lengths)} segments synced to beats")
        except Exception as e:
            print(f"   ⚠️  Audio sync failed ({e}) - using fixed segment length")

    # ── Persistent model context (FEAT-003) ───────────────────────────────
    _pro_context = None
    if USE_PERSISTENT_CONTEXT:
        try:
            _pro_context = SVIProContext()
            print("   [experimental] Persistent model context enabled - models will be reused across segments")
        except Exception as e:
            print(f"   ⚠️  Persistent context init failed ({e})")
            _pro_context = None
    
    for seg_idx in range(max_segments):
        seg_num = seg_idx + 1
        print(f"\n{'─' * 50}")
        print(f"📹 Segment {seg_num}/{max_segments} (seed={seeds[seg_idx]})")
        
        # Check for cached segment
        cached_path = f"{cache_dir}/segment_{seg_idx:02d}.mp4"
        anchor_path = f"{cache_dir}/anchor_{seg_idx:02d}.png"
        
        if os.path.exists(cached_path) and os.path.exists(anchor_path):
            print(f"   ⏩ Using cached segment: {cached_path}")
            all_segment_paths.append(cached_path)
            current_anchor_path = anchor_path
            continue
        
        # ── Motion coherence & adaptive overlap (FEAT-003) ────────────────
        _seg_overlap = overlap_frames
        _seg_lora_stack = None  # Will override if motion coherence active
        _velocity_latent = None  # For velocity injection

        if seg_idx > 0 and current_anchor_path:
            # Read previous segment ONCE and share between adaptive overlap,
            # motion coherence, and velocity injection (avoids triple file read)
            _prev_segment_tensor = None
            _prev_frames_list = None
            if (USE_ADAPTIVE_OVERLAP or USE_MOTION_COHERENCE or USE_VELOCITY_INJECTION) and len(all_segment_paths) > 0:
                try:
                    import imageio as _iio
                    _prev_reader = _iio.get_reader(all_segment_paths[-1])
                    _prev_frames_list = [f for f in _prev_reader]
                    _prev_reader.close()
                    _prev_segment_tensor = torch.from_numpy(
                        np.stack(_prev_frames_list)).float() / 255.0
                except Exception as e:
                    print(f"   \u26a0\ufe0f  Previous segment read failed ({e})")
                    _prev_frames_list = None
                    _prev_segment_tensor = None

            # Adaptive overlap computation
            if USE_ADAPTIVE_OVERLAP and _prev_segment_tensor is not None:
                try:
                    _seg_overlap = compute_adaptive_overlap(
                        _prev_segment_tensor[-10:], ADAPTIVE_OVERLAP_MIN, ADAPTIVE_OVERLAP_MAX)
                    print(f"   Adaptive overlap: {_seg_overlap} frames")
                except Exception as e:
                    print(f"   \u26a0\ufe0f  Adaptive overlap failed ({e})")

            # Motion coherence: optical flow & camera LoRA auto-selection
            if USE_MOTION_COHERENCE and _prev_frames_list is not None and len(_prev_frames_list) >= 2:
                try:
                    _f1 = _prev_frames_list[-2]
                    _f2 = _prev_frames_list[-1]
                    _flow = estimate_optical_flow(_f1, _f2)
                    _direction = detect_motion_direction(_flow)
                    _cam_lora = auto_select_camera_lora(_direction)
                    if _cam_lora != "none":
                        print(f"   Motion coherence: {_direction} -> camera={_cam_lora}")
                        _seg_lora_stack = _build_lora_stack(
                            IC_LORA, IC_LORA_STRENGTH, _cam_lora, CAMERA_LORA_STRENGTH)
                except Exception as e:
                    print(f"   \u26a0\ufe0f  Motion coherence failed ({e})")

            # Velocity injection: compute motion delta from last 2 frames
            if USE_VELOCITY_INJECTION and _prev_segment_tensor is not None and _prev_segment_tensor.shape[0] >= 2:
                try:
                    _velocity_latent = compute_velocity_latent(
                        _prev_segment_tensor[-2], _prev_segment_tensor[-1])
                    _vel_mag = _velocity_latent.abs().mean().item()
                    print(f"   Velocity injection: magnitude={_vel_mag:.4f}")
                except Exception as e:
                    print(f"   \u26a0\ufe0f  Velocity injection failed ({e})")
                    _velocity_latent = None

        # Determine segment frame count (audio sync or fixed)
        _seg_frames = segment_lengths[seg_idx] if seg_idx < len(segment_lengths) else segment_length

        # Generate this segment using generate_pro
        _seg_kwargs = dict(kwargs)
        if _seg_lora_stack is not None:
            _seg_kwargs["lora_stack"] = _seg_lora_stack
        if _reference_histogram is not None and USE_COLOR_MATCHING:
            _seg_kwargs["reference_histogram"] = _reference_histogram
            _seg_kwargs["use_color_matching"] = True
        if _timeline_entries is not None:
            _seg_kwargs["timeline_entries"] = _timeline_entries
        if _embedding_bank is not None:
            _seg_kwargs["embedding_bank"] = _embedding_bank
        if _velocity_latent is not None:
            _seg_kwargs["velocity_latent"] = _velocity_latent

        # Build common call kwargs so retries use the same parameters
        _gen_call_kwargs = dict(
            user_input=user_input,
            image_path=current_anchor_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=_seg_frames,
            fps=fps,
            seed=seeds[seg_idx],
            image_strength=image_strength if seg_idx == 0 else character_strength,
            character_image_path=character_image_path,
            character_strength=character_strength,
            character_mode=character_mode,
            character_name=character_name,
            character_description=character_description,
            output_prefix=f"{output_prefix}_seg{seg_num:02d}",
        )

        seg_output = generate_pro(**_gen_call_kwargs, **_seg_kwargs)
        
        if seg_output is None:
            print(f"   ❌ Segment {seg_num} failed — stopping extension.")
            break

        # ── Quality gate with retry (FEAT-003) ────────────────────────────
        if USE_QUALITY_GATE and seg_output:
            try:
                import imageio as _iio
                _reader = _iio.get_reader(seg_output)
                _seg_frames_list = [f for f in _reader]
                _reader.close()
                _seg_tensor = torch.from_numpy(np.stack(_seg_frames_list)).float() / 255.0
                _quality = compute_segment_quality(_seg_tensor)
                if not _quality.get("passed", True):
                    print(f"   \u26a0\ufe0f  Quality gate failed for segment {seg_num}")
                    for _retry in range(QUALITY_GATE_MAX_RETRIES - 1):
                        _retry_seed = seeds[seg_idx] + _retry + 1
                        print(f"   Retrying with seed={_retry_seed}...")
                        # Reuse common kwargs with updated seed and prefix
                        _retry_kwargs = dict(_gen_call_kwargs)
                        _retry_kwargs["seed"] = _retry_seed
                        _retry_kwargs["output_prefix"] = f"{output_prefix}_seg{seg_num:02d}_r{_retry+1}"
                        seg_output = generate_pro(**_retry_kwargs, **_seg_kwargs)
                        if seg_output:
                            _reader = _iio.get_reader(seg_output)
                            _seg_frames_list = [f for f in _reader]
                            _reader.close()
                            _seg_tensor = torch.from_numpy(np.stack(_seg_frames_list)).float() / 255.0
                            _quality = compute_segment_quality(_seg_tensor)
                            if _quality.get("passed", True):
                                print(f"   \u2713 Quality gate passed on retry {_retry+1}")
                                break
            except Exception as e:
                print(f"   \u26a0\ufe0f  Quality gate check failed ({e})")

        # ── Extract reference color histogram from first segment (FEAT-003) ─
        if seg_idx == 0 and USE_COLOR_MATCHING and seg_output:
            try:
                import imageio as _iio
                _reader = _iio.get_reader(seg_output)
                _seg_frames_list = [f for f in _reader]
                _reader.close()
                _seg_tensor = torch.from_numpy(np.stack(_seg_frames_list)).float() / 255.0
                _reference_histogram = extract_color_histogram(_seg_tensor)
                print(f"   ✓ Reference color histogram extracted from segment 1")
            except Exception as e:
                print(f"   ⚠️  Reference histogram extraction failed ({e})")
        
        # Cache the segment
        shutil.copy(seg_output, cached_path)
        all_segment_paths.append(cached_path)
        
        # Extract last frame as anchor for next segment
        last_frame_tensor = get_last_frame_tensor(cached_path)
        if last_frame_tensor is not None:
            anchor_pil = tensor_to_pil(last_frame_tensor)
            anchor_pil.save(anchor_path, "PNG")
            current_anchor_path = anchor_path
            print(f"   ✓ Anchor frame saved: {anchor_path}")
        else:
            print(f"   ⚠️  Could not extract anchor — next segment may lack continuity.")
        
        # Cleanup between segments
        aggressive_cleanup(f"segment {seg_num} done")
    
    # Final stitching with overlap blending
    if len(all_segment_paths) < 1:
        print("❌ No segments generated successfully.")
        return None
    
    if len(all_segment_paths) == 1:
        print(f"✅ Single segment generated: {all_segment_paths[0]}")
        return all_segment_paths[0]
    
    print(f"\n{'═' * 50}")
    print(f"🧵 Stitching {len(all_segment_paths)} segments with {overlap_frames}-frame {overlap_mode} overlap...")
    
    # Load all segments and blend
    import imageio
    
    combined_frames = None
    for seg_idx, seg_path in enumerate(all_segment_paths):
        reader = imageio.get_reader(seg_path)
        frames_list = []
        for frame in reader:
            frames_list.append(frame)
        reader.close()
        
        seg_tensor = torch.from_numpy(np.stack(frames_list)).float() / 255.0
        
        if combined_frames is None:
            combined_frames = seg_tensor
        else:
            # Apply overlap blending (mirrors ImageBatchExtendWithOverlap)
            combined_frames = blend_overlap_frames(
                combined_frames, seg_tensor,
                overlap=overlap_frames,
                mode=overlap_mode,
                side=overlap_side
            )
        
        print(f"   ✓ Segment {seg_idx + 1} merged (total frames: {len(combined_frames)})")
    
    # Save final stitched video
    final_frames_np = (combined_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    final_path = f"/content/ComfyUI/output/{output_prefix}_extended_{int(time.time())}.mp4"
    imageio.mimsave(final_path, [f for f in final_frames_np], fps=fps, codec='libx264')
    
    elapsed = time.time() - t0
    total_duration = len(combined_frames) / fps
    
    print(f"\n{'═' * 70}")
    print(f"🎉 EXTENDED VIDEO COMPLETE!")
    print(f"   Output    : {final_path}")
    print(f"   Duration  : {total_duration:.1f}s ({len(combined_frames)} frames @ {fps}fps)")
    print(f"   Segments  : {len(all_segment_paths)}")
    print(f"   Elapsed   : {elapsed/60:.1f} minutes")
    print(f"{'═' * 70}")
    
    # Display if enabled
    if SHOW_PREVIEWS:
        display_video(final_path)

    # ── Export timeline (FEAT-003) ────────────────────────────────────────
    if EXPORT_TIMELINE and _timeline_entries:
        try:
            _tl_path = f"/content/ComfyUI/output/{output_prefix}_timeline.{TIMELINE_FORMAT}"
            if TIMELINE_FORMAT == "edl":
                generate_edl(_timeline_entries, _tl_path, fps)
            else:
                generate_timeline_json(_timeline_entries, _tl_path)
            print(f"   ✓ Timeline exported: {_tl_path}")
        except Exception as e:
            print(f"   ⚠️  Timeline export failed ({e})")

    # ── Google Drive final sync (FEAT-003) ────────────────────────────────
    if PERSIST_TO_GDRIVE and final_path:
        try:
            sync_to_drive(final_path, GDRIVE_PATH)
        except Exception as e:
            print(f"   \u26a0\ufe0f  Final Drive sync failed ({e})")

    # ── Cleanup persistent context (FEAT-003) ─────────────────────────────
    if _pro_context is not None:
        try:
            _pro_context.cleanup()
            print("   \u2713 Persistent model context released")
        except Exception:
            pass

    return final_path


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8  ─  STORYBOARD / MULTI-SCENE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 8. Storyboard / Multi-Scene Runner
# @markdown Edit the `SCENES` list below, then run Cell 9 with
# @markdown `USE_STORYBOARD = True` to generate all scenes sequentially.
# @markdown
# @markdown When `USE_SCENE_CONTINUITY = True` the last frame of scene N
# @markdown is automatically extracted and used as the seed image for scene N+1,
# @markdown creating seamless shot-to-shot character continuity.

# ── Example storyboard — edit or extend ──────────────────────────────────────
SCENES = [
    {
        "user_input"    : "a woman enters a dimly lit café, shaking rain from her coat, looks around",
        "image_path"    : CHARACTER_IMAGE_PATH,   # use character image as first-frame seed
        "frames"        : 97,
        "seed"          : SEED,
        "output_prefix" : f"Story01-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she sits at a window table, wraps her hands around a coffee cup, "
                          "gazes out at the rain-streaked street",
        "image_path"    : None,   # will be filled by continuity system
        "frames"        : 121,
        "seed"          : SEED + 1,
        "output_prefix" : f"Story02-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she notices something outside and leans forward, face half lit by neon glow",
        "image_path"    : None,   # filled by continuity from scene 2
        "frames"        : 97,
        "seed"          : SEED + 2,
        "output_prefix" : f"Story03-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
]

USE_STORYBOARD = False  # @param {type:"boolean"}
# Set True in Cell 9 to run all scenes instead of a single clip.


def concatenate_clips(clip_paths: List[str], output_path: str) -> Optional[str]:
    """Concatenate video clips using ffmpeg concat demuxer."""
    valid_paths = [p for p in clip_paths if p and os.path.exists(p)]
    if len(valid_paths) < 2:
        return valid_paths[0] if valid_paths else None
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in valid_paths:
            f.write(f"file '{p}'\n")
    try:
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", list_file, "-c", "copy", output_path],
                       check=True, capture_output=True)
        print(f"   \u2713 Concatenated {len(valid_paths)} clips \u2192 {output_path}")
        return output_path
    except Exception as e:
        print(f"   \u26a0\ufe0f  Concatenation failed ({e}) \u2014 individual clips still available.")
        return None


def run_storyboard(
    scenes:          List[Dict],
    use_continuity:  bool = USE_SCENE_CONTINUITY,
    tmp_dir:         str  = "/content/ComfyUI/input",
    auto_reduce_for_stability: bool = True,
) -> List[Optional[str]]:
    """
    Run a list of scenes sequentially, optionally chaining last-frame continuity.

    Each scene dict supports these keys (all optional except user_input):
      user_input           — story description for Easy Prompt
      image_path           — seed image path (overridden by continuity if None)
      frames               — frame count
      seed                 — RNG seed
      output_prefix        — filename prefix
      width / height       — resolution (defaults to global WIDTH/HEIGHT)
      character_image_path — character reference
      character_mode       — 'i2v', 'anchor', 'both', 'none'
      positive_prompt      — manual prompt (used if BYPASS_EASY_PROMPT=True)
      negative_prompt      — manual negative

    Returns list of output paths (None for failed scenes).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    outputs = []
    prev_output = None

    # ── Cache directory setup ─────────────────────────────────────────────
    cache_dir = f"/content/ComfyUI/output/{scenes[0].get('output_prefix', OUTPUT_PREFIX)}_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print("\U0001f3ac Storyboard Runner \u2014 Starting")
    print(f"   Scenes    : {len(scenes)}")
    print(f"   Continuity: {use_continuity}")
    print("\u2500" * 70)

    # ── Resume logic: check for cached clips ──────────────────────────────
    start_index = 0
    for i in range(len(scenes)):
        cached_clip = f"{cache_dir}/scene_{i:02d}.mp4"
        if os.path.exists(cached_clip):
            outputs.append(cached_clip)
            start_index = i + 1
            prev_output = cached_clip
        else:
            break

    if start_index > 0:
        print(f"   \u23e9 Resuming from scene {start_index + 1} (found {start_index} cached clips)")

    # ── Parallel Prompt Expansion (FEAT-003) ──────────────────────────────
    expanded_prompts = {}
    if USE_PARALLEL_PROMPT_EXPANSION and not BYPASS_EASY_PROMPT:
        print("   📝 Parallel prompt expansion (loading LLM once for all scenes)...")
        try:
            for idx, scene in enumerate(scenes):
                _inp = scene.get("user_input", "")
                if _inp.strip():
                    _pos, _neg = run_easy_prompt(
                        user_input=_inp,
                        frame_count=scene.get("frames", FRAMES),
                        seed=scene.get("seed", SEED),
                        scene_context=scene.get("character_description", ""),
                    )
                    expanded_prompts[idx] = (_pos, _neg)
            print(f"   ✓ Expanded {len(expanded_prompts)} prompts in batch")
            # Cache to disk for resume
            _cache_path = f"{cache_dir}/expanded_prompts.json"
            with open(_cache_path, "w") as f:
                json.dump({str(k): v for k, v in expanded_prompts.items()}, f)
        except Exception as e:
            print(f"   ⚠️  Parallel expansion failed ({e}) - will expand per-scene")
            expanded_prompts = {}

    # ── Thumbnail Preview (FEAT-003) ──────────────────────────────────────
    if GENERATE_THUMBNAILS:
        print("   🖼️  Generating thumbnail previews...")
        _thumbnails = []
        for idx, scene in enumerate(scenes):
            _thumb = generate_thumbnail_frame(
                prompt=scene.get("user_input", ""),
                width=scene.get("width", WIDTH),
                height=scene.get("height", HEIGHT),
                seed=scene.get("seed", SEED),
            )
            if _thumb:
                _thumbnails.append(_thumb)
        if _thumbnails:
            display_thumbnail_grid(_thumbnails, THUMBNAIL_COLS)
            print(f"   ✓ Thumbnail grid displayed ({len(_thumbnails)} scenes)")

    # ── Auto-reduce frames for stability ──────────────────────────────────
    if auto_reduce_for_stability and len(scenes) > 3:
        print(f"   \u2699\ufe0f  Auto-stability: capping frames to 97 for multi-scene mode ({len(scenes)} scenes)")

    storyboard_start = time.time()
    completed_count = 0

    for i, scene in enumerate(scenes):
        if i < start_index:
            continue

        scene_num = i + 1
        print(f"\n\U0001f3ac Scene {scene_num}/{len(scenes)}: {scene.get('output_prefix','Scene')}")
        print(f"   Input: {scene.get('user_input','')[:80]}\u2026")

        # Resolve image_path: use continuity frame if available and not explicitly set
        _image_path = scene.get("image_path")
        if use_continuity and prev_output and _image_path is None:
            print(f"   \U0001f517 Continuity: extracting frames from scene {scene_num - 1}\u2026")
            try:
                # Use multi-frame composite when configured (default)
                _n_cont_frames = CONTINUITY_MULTI_FRAME_COUNT if CONTINUITY_MULTI_FRAME_COUNT > 1 else 1
                _cont_mode = CONTINUITY_COMPOSITE_MODE
                _cont_fmt = CONTINUITY_FRAME_FORMAT.lower()
                _ext = "png" if _cont_fmt == "png" else "jpg"
                _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.{_ext}")

                # Extract composite from previous output
                _cont_tensor = extract_continuity_composite(
                    prev_output, n_frames=_n_cont_frames, mode=_cont_mode)

                if _cont_tensor is not None:
                    save_continuity_frame(_cont_tensor, _cont_path, format=_cont_fmt)
                    _image_path = _cont_path
                    print(f"   \u2713 Continuity composite saved ({_n_cont_frames} frames, {_cont_mode}): {_cont_path}")
                else:
                    # Fallback to single last frame (legacy)
                    last_tensor = get_last_frame_tensor(prev_output)
                    if last_tensor is not None:
                        _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.{_ext}")
                        save_continuity_frame(last_tensor, _cont_path, format=_cont_fmt)
                        _image_path = _cont_path
                        print(f"   \u2713 Continuity frame saved (fallback single): {_cont_path}")
                    else:
                        print(f"   \u26a0\ufe0f  Could not extract continuity frame - skipping.")
            except Exception as _cont_err:
                print(f"   \u26a0\ufe0f  Continuity extraction error ({_cont_err}) - trying legacy method.")
                last_tensor = get_last_frame_tensor(prev_output)
                if last_tensor is not None:
                    _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.jpg")
                    pil_frame = tensor_to_pil(last_tensor)
                    pil_frame.save(_cont_path, "JPEG", quality=95)
                    _image_path = _cont_path
                else:
                    print(f"   \u26a0\ufe0f  Could not extract last frame - skipping continuity.")

            # ── Dual-Anchor auto-upgrade ──────────────────────────────
            # When dual-anchor is enabled AND we have both a continuity frame
            # AND the scene has a character image, force character_mode to "both"
            # so the dual-anchor system fires for every scene after the first.
            if USE_DUAL_ANCHOR_STORYBOARD and _image_path is not None:
                _scene_char_img = scene.get("character_image_path", CHARACTER_IMAGE_PATH)
                if _scene_char_img:
                    _prev_mode = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE)
                    scene["character_mode"] = "both"
                    if _prev_mode != "both":
                        print(f"   \U0001f9ec Dual-anchor activated: character_mode '{_prev_mode}' -> 'both'")

        # Determine frames (auto-reduce if needed)
        _frames = scene.get("frames", FRAMES)
        _original_frames = _frames
        if auto_reduce_for_stability and len(scenes) > 3:
            _frames = min(97, _frames)
        if _frames < _original_frames:
            print(f"   \u2699\ufe0f  Frames capped: {_original_frames} \u2192 {_frames} (auto-stability)")

        # ── Retry loop ────────────────────────────────────────────────────
        max_retries = 3
        success = False
        scene_seed = scene.get("seed", SEED)
        _retry_frames = _frames
        _retry_tiled_vae = None  # None means use default; True forces tiled
        for attempt in range(max_retries):
            try:
                _gen_kwargs = dict(
                    user_input           = scene.get("user_input", USER_INPUT),
                    image_path           = _image_path,
                    positive_prompt      = scene.get("positive_prompt", POSITIVE_PROMPT),
                    negative_prompt      = scene.get("negative_prompt", NEGATIVE_PROMPT),
                    width                = scene.get("width", WIDTH),
                    height               = scene.get("height", HEIGHT),
                    frames               = _retry_frames,
                    fps                  = scene.get("fps", FPS),
                    seed                 = scene_seed,
                    image_strength       = scene.get("image_strength", IMAGE_STRENGTH),
                    character_image_path = scene.get("character_image_path", CHARACTER_IMAGE_PATH),
                    character_strength   = scene.get("character_strength", CHARACTER_STRENGTH),
                    character_mode       = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE),
                    character_name       = scene.get("character_name", CHARACTER_NAME),
                    character_description= scene.get("character_description", CHARACTER_DESCRIPTION),
                    output_prefix        = scene.get("output_prefix", OUTPUT_PREFIX),
                )
                if _retry_tiled_vae is not None:
                    _gen_kwargs["use_tiled_vae"] = _retry_tiled_vae
                # Use pre-expanded prompt if available (FEAT-003)
                if i in expanded_prompts:
                    _gen_kwargs["positive_prompt"] = expanded_prompts[i][0]
                    _gen_kwargs["negative_prompt"] = expanded_prompts[i][1]
                    _gen_kwargs["bypass_easy_prompt"] = True
                out = generate_pro(**_gen_kwargs)
                # Cache successful clip
                if out:
                    shutil.copy(out, f"{cache_dir}/scene_{i:02d}.mp4")
                outputs.append(out)
                prev_output = out
                success = True
                print(f"   \u2705 Scene {scene_num} done \u2192 {out}")
                break
            except torch.cuda.OutOfMemoryError:
                aggressive_cleanup("OOM recovery")
                scene_seed = scene.get("seed", SEED) + attempt + 1
                # Progressive memory pressure reduction for next attempt
                if attempt >= 0:
                    _retry_frames = max(57, _retry_frames - 24)
                    print(f"   \u26a0\ufe0f  Reducing frames to {_retry_frames} for next attempt.")
                if attempt >= 1:
                    _retry_tiled_vae = True
                    print(f"   \u26a0\ufe0f  Forcing tiled VAE for next attempt.")
                print(f"   \u26a0\ufe0f  OOM on attempt {attempt+1} \u2014 retrying with seed {scene_seed}...")
                if attempt == max_retries - 1:
                    print(f"   \u274c Scene {scene_num} failed after {max_retries} attempts")
            except Exception as e:
                print(f"   \u274c Scene {scene_num} error: {type(e).__name__}: {e}")
                break

        if not success:
            outputs.append(None)
            prev_output = None  # don't chain from a failed scene

        completed_count += 1

        # ── Progress tracking ─────────────────────────────────────────────
        elapsed = time.time() - storyboard_start
        avg_per_clip = elapsed / completed_count if completed_count > 0 else 0
        remaining = avg_per_clip * (len(scenes) - i - 1)
        print(f"   \u23f1\ufe0f  Elapsed: {elapsed/60:.1f}min | Est. remaining: {remaining/60:.1f}min")

    # ── Final concatenation with overlap blending ─────────────────────────
    successful_clips = [p for p in outputs if p]
    if len(successful_clips) >= 2:
        final_path = f"/content/ComfyUI/output/{scenes[0].get('output_prefix', OUTPUT_PREFIX)}_full.mp4"
        
        if OVERLAP_FRAMES > 0 and OVERLAP_MODE != "hard_cut":
            # Use SVI-Pro style overlap blending for seamless transitions
            print(f"   🧵 Stitching with {OVERLAP_FRAMES}-frame {OVERLAP_MODE} overlap...")
            try:
                import imageio
                combined_frames = None
                for clip_path in successful_clips:
                    reader = imageio.get_reader(clip_path)
                    frames_list = [frame for frame in reader]
                    reader.close()
                    seg_tensor = torch.from_numpy(np.stack(frames_list)).float() / 255.0
                    
                    if combined_frames is None:
                        combined_frames = seg_tensor
                    else:
                        combined_frames = blend_overlap_frames(
                            combined_frames, seg_tensor,
                            overlap=OVERLAP_FRAMES,
                            mode=OVERLAP_MODE,
                            side=OVERLAP_SIDE
                        )
                
                if combined_frames is not None:
                    final_np = (combined_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    import imageio
                    # Save frames first, then mux audio from source clips
                    _temp_stitch = final_path.replace('.mp4', '_temp_noaudio.mp4')
                    imageio.mimsave(_temp_stitch, [f for f in final_np], fps=FPS, codec='libx264')
                    
                    # Try to mux audio from source clips
                    _stitch_has_audio = False
                    try:
                        _probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                                     "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                                     successful_clips[0]]
                        _probe_result = subprocess.run(_probe_cmd, capture_output=True, text=True)
                        _stitch_has_audio = "audio" in _probe_result.stdout
                    except Exception:
                        pass
                    
                    if _stitch_has_audio:
                        try:
                            _audio_list_f = "/tmp/stitch_audio_list.txt"
                            with open(_audio_list_f, "w") as f:
                                for p in successful_clips:
                                    f.write(f"file '{p}'\n")
                            _temp_audio_f = final_path.replace('.mp4', '_temp_audio.aac')
                            subprocess.run([
                                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                "-i", _audio_list_f, "-vn", "-acodec", "aac",
                                "-b:a", "192k", _temp_audio_f
                            ], check=True, capture_output=True)
                            _vid_dur = len(combined_frames) / FPS
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", _temp_stitch,
                                "-i", _temp_audio_f,
                                "-c:v", "copy", "-c:a", "aac",
                                "-t", str(_vid_dur),
                                "-shortest",
                                final_path
                            ], check=True, capture_output=True)
                            if os.path.exists(_temp_stitch):
                                os.remove(_temp_stitch)
                            if os.path.exists(_temp_audio_f):
                                os.remove(_temp_audio_f)
                            print(f"   🎬 Final video (overlap-blended + audio): {final_path}")
                        except Exception as _ae:
                            print(f"   ⚠️  Audio mux failed ({_ae}) — using video-only.")
                            if os.path.exists(_temp_stitch):
                                os.rename(_temp_stitch, final_path)
                            print(f"   🎬 Final video (overlap-blended, no audio): {final_path}")
                    else:
                        os.rename(_temp_stitch, final_path)
                        print(f"   🎬 Final video (overlap-blended): {final_path}")
                    concat_result = final_path
                else:
                    concat_result = concatenate_clips(successful_clips, final_path)
            except Exception as e:
                print(f"   ⚠️  Overlap blending failed ({e}) — falling back to hard concat.")
                concat_result = concatenate_clips(successful_clips, final_path)
        else:
            concat_result = concatenate_clips(successful_clips, final_path)
            if concat_result:
                print(f"   🎬 Final video: {concat_result}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "\u2550" * 70)
    print("\U0001f3ac Storyboard Complete")
    print(f"   Total scenes : {len(scenes)}")
    print(f"   Successful   : {sum(1 for p in outputs if p)}")
    print(f"   Failed       : {sum(1 for p in outputs if not p)}")
    print("\n   Output paths:")
    for i, p in enumerate(outputs):
        status = "\u2705" if p else "\u274c"
        print(f"   {status} Scene {i+1}: {p or 'FAILED'}")
    print("\u2550" * 70)

    return outputs


print("✅ Storyboard runner ready.")
print("   Edit SCENES list above, then set USE_STORYBOARD=True in Cell 9.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9  ─  RUN
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 9. Generate
# @markdown Re-run this cell for each new video or storyboard.
# @markdown Seed auto-increments after each successful generation.
# @markdown
# @markdown Set `USE_STORYBOARD = True` (Cell 8) to run multi-scene mode.

_current_seed = SEED

try:
    # ── Script Intelligence: use Cell 4.5 decomposed scenes if available ──
    if USE_SCRIPT_DECOMPOSER and SCRIPT_INPUT and SCRIPT_INPUT.strip():
        # Check if Cell 4.5 already decomposed the script (preferred path)
        if '_decomposed_scenes' in dir() and _decomposed_scenes and len(_decomposed_scenes) > 0:
            # Cell 4.5 already generated proper multi-segment SCENES
            SCENES = _decomposed_scenes
            USE_STORYBOARD = True
            print(f"📜 Using Cell 4.5 Script Intelligence: {len(SCENES)} shots "
                  f"({TARGET_VIDEO_DURATION}s target, {SEGMENT_DURATION}s/segment)")
        else:
            # Fallback: Cell 4.5 didn't run or failed - do basic decomposition
            print("📜 Script Decomposer (fallback) - breaking script into shots...")
            try:
                _decomposed_scenes = decompose_script_to_scenes(
                    script=SCRIPT_INPUT,
                    target_duration=TARGET_VIDEO_DURATION,
                    segment_duration=SEGMENT_DURATION,
                    dialogue_interval=DIALOGUE_INTERVAL,
                    quality=VIDEO_QUALITY,
                    style=VIDEO_STYLE,
                    language=CUSTOM_LANGUAGE if VIDEO_LANGUAGE == "Custom" else VIDEO_LANGUAGE,
                    character_def=CHARACTER_DEFINITION,
                    secondary_char=SECONDARY_CHARACTER,
                    output_format=SCENE_OUTPUT_FORMAT,
                    fps=FPS,
                )
                if _decomposed_scenes:
                    SCENES = _decomposed_scenes
                    USE_STORYBOARD = True
                    print(f"   ✓ Decomposed into {len(SCENES)} shots "
                          f"({TARGET_VIDEO_DURATION}s, {SEGMENT_DURATION}s/segment)")
                    print_scene_breakdown(SCENES)
                else:
                    print("   ⚠️  Decomposition returned empty - using manual SCENES")
            except Exception as e:
                print(f"   ⚠️  Script decomposition failed ({e}) - using manual SCENES")

    if USE_STORYBOARD:
        # ── Multi-scene storyboard mode ───────────────────────────────────
        print("🎬 Running storyboard mode…")
        storyboard_outputs = run_storyboard(
            scenes=SCENES,
            use_continuity=USE_SCENE_CONTINUITY,
        )
        output = storyboard_outputs[-1] if storyboard_outputs else None
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + len(SCENES)
            print(f"🔢 Next seed: {SEED}")

    elif USE_SEGMENT_EXTENSION:
        # ── SVI-Pro extended video mode ───────────────────────────────────
        print("🎬 Running SVI-Pro segment extension mode…")
        output = generate_extended_video(
            user_input             = USER_INPUT,
            image_path             = IMAGE_PATH,
            positive_prompt        = POSITIVE_PROMPT,
            negative_prompt        = NEGATIVE_PROMPT,
            width                  = WIDTH,
            height                 = HEIGHT,
            fps                    = FPS,
            seed                   = _current_seed,
            image_strength         = IMAGE_STRENGTH,
            character_image_path   = CHARACTER_IMAGE_PATH,
            character_strength     = CHARACTER_STRENGTH,
            character_mode         = CHARACTER_CONSISTENCY_MODE,
            character_name         = CHARACTER_NAME,
            character_description  = CHARACTER_DESCRIPTION,
            segment_length         = SEGMENT_LENGTH,
            max_segments           = MAX_SEGMENTS,
            overlap_frames         = OVERLAP_FRAMES,
            overlap_mode           = OVERLAP_MODE,
            overlap_side           = OVERLAP_SIDE,
            segment_seed_mode      = SEGMENT_SEED_MODE,
            output_prefix          = OUTPUT_PREFIX,
            # Pass through sampling settings
            pass1_sigmas           = PASS1_SIGMAS,
            pass1_sampler          = PASS1_SAMPLER,
            pass1_cfg              = PASS1_CFG,
            pass2_sigmas           = PASS2_SIGMAS,
            pass2_sampler          = PASS2_SAMPLER,
            pass2_cfg              = PASS2_CFG,
            pass2_seed             = PASS2_SEED,
            pro_mode               = PRO_MODE,
            pro_steps              = PRO_STEPS,
            pro_scheduler          = PRO_SCHEDULER,
            pro_split_at           = PRO_SPLIT_AT,
            use_tiled_vae          = USE_TILED_VAE,
            tiled_spatial_tiles    = TILED_SPATIAL_TILES,
            tiled_spatial_overlap  = TILED_SPATIAL_OVERLAP,
            tiled_temporal_len     = TILED_TEMPORAL_LEN,
            tiled_temporal_overlap = TILED_TEMPORAL_OVERLAP,
            tiled_last_frame_fix   = TILED_LAST_FRAME_FIX,
            lora_stack             = LORA_STACK,
            lora_stack_json        = LORA_STACK_JSON,
        )
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + MAX_SEGMENTS
            print(f"🔢 Next seed: {SEED}")

    else:
        # ── Single-clip mode ──────────────────────────────────────────────
        output = generate_pro(
            user_input             = USER_INPUT,
            image_path             = IMAGE_PATH,
            positive_prompt        = POSITIVE_PROMPT,
            negative_prompt        = NEGATIVE_PROMPT,
            width                  = WIDTH,
            height                 = HEIGHT,
            frames                 = FRAMES,
            fps                    = FPS,
            seed                   = _current_seed,
            image_strength         = IMAGE_STRENGTH,
            character_image_path   = CHARACTER_IMAGE_PATH,
            character_strength     = CHARACTER_STRENGTH,
            character_mode         = CHARACTER_CONSISTENCY_MODE,
            character_name         = CHARACTER_NAME,
            character_description  = CHARACTER_DESCRIPTION,
            pass1_sigmas           = PASS1_SIGMAS,
            pass1_sampler          = PASS1_SAMPLER,
            pass1_cfg              = PASS1_CFG,
            pass2_sigmas           = PASS2_SIGMAS,
            pass2_sampler          = PASS2_SAMPLER,
            pass2_cfg              = PASS2_CFG,
            pass2_seed             = PASS2_SEED,
            pro_mode               = PRO_MODE,
            pro_steps              = PRO_STEPS,
            pro_scheduler          = PRO_SCHEDULER,
            pro_split_at           = PRO_SPLIT_AT,
            use_tiled_vae          = USE_TILED_VAE,
            tiled_spatial_tiles    = TILED_SPATIAL_TILES,
            tiled_spatial_overlap  = TILED_SPATIAL_OVERLAP,
            tiled_temporal_len     = TILED_TEMPORAL_LEN,
            tiled_temporal_overlap = TILED_TEMPORAL_OVERLAP,
            tiled_last_frame_fix   = TILED_LAST_FRAME_FIX,
            lora_stack             = LORA_STACK,
            lora_stack_json        = LORA_STACK_JSON,
            output_prefix          = OUTPUT_PREFIX,
        )

        # Mirrors "Shared seed" node [284] increment mode in LD-I2V.json
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + 1
            print(f"🔢 Next seed: {SEED}")

    # ── Export timeline if enabled (FEAT-003) ─────────────────────────────
    if EXPORT_TIMELINE:
        print("📋 Exporting timeline...")
        try:
            _tl_entries = []
            # Build from output paths
            if USE_STORYBOARD and globals().get('storyboard_outputs'):
                for idx, out_path in enumerate(storyboard_outputs):
                    if out_path:
                        _tl_entries.append({
                            "segment_index": idx,
                            "output_path": out_path,
                            "seed": SCENES[idx].get("seed", SEED + idx) if idx < len(SCENES) else SEED,
                            "prompt": SCENES[idx].get("user_input", "")[:200] if idx < len(SCENES) else "",
                            "frames": SCENES[idx].get("frames", FRAMES) if idx < len(SCENES) else FRAMES,
                            "fps": FPS,
                            "duration_seconds": (SCENES[idx].get("frames", FRAMES) if idx < len(SCENES) else FRAMES) / FPS,
                        })
            elif output:
                _tl_entries.append({
                    "segment_index": 0,
                    "output_path": output,
                    "seed": _current_seed,
                    "prompt": USER_INPUT[:200],
                    "frames": FRAMES,
                    "fps": FPS,
                    "duration_seconds": FRAMES / FPS,
                })
            if _tl_entries:
                _tl_path = f"/content/ComfyUI/output/{OUTPUT_PREFIX}_timeline.{TIMELINE_FORMAT}"
                if TIMELINE_FORMAT == "edl":
                    generate_edl(_tl_entries, _tl_path, FPS)
                else:
                    generate_timeline_json(_tl_entries, _tl_path)
                print(f"   ✓ Timeline: {_tl_path}")
        except Exception as e:
            print(f"   ⚠️  Timeline export failed ({e})")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — partial output may be in /content/ComfyUI/output/")

except FileNotFoundError as e:
    print(f"\n❌ Missing models: {e}")
    print("   Run Cell 2 to download, then retry Cell 9.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n❌ CUDA Out of Memory")
    print(f"   Current settings: {WIDTH}×{HEIGHT}, {FRAMES} frames")
    print("   ── Suggested fixes ─────────────────────────────────────────────")
    print("   T4  (15 GB): WIDTH=768,  HEIGHT=512,  FRAMES=97")
    print("   L4  (24 GB): WIDTH=1024, HEIGHT=576,  FRAMES=161")
    print("   A100(40 GB): WIDTH=1280, HEIGHT=720,  FRAMES=241")
    print("   ── Also try ────────────────────────────────────────────────────")
    print("   • LLM_MODEL='3B'         in Cell 4  (reduce LLM VRAM footprint)")
    print("   • USE_CHUNK_FF=True      in Cell 5  (chunk feedforward for T4)")
    print("   • USE_TILED_VAE=True     in Cell 6  (tile VAE decode)")
    print("   • TILED_SPATIAL_TILES=4  in Cell 6  (more tiles = less VRAM per tile)")
    print("   • PRO_MODE=False         in Cell 5  (simpler sigma schedule)")

except RuntimeError as e:
    print(f"\n❌ Runtime error: {e}")
    cleanup_memory()

except Exception as e:
    import traceback
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n💡 Quick-fix reference:")
    print("   'UnetLoaderGGUF' not found       → Cell 1: clone ComfyUI_GGUF")
    print("   'LTX2PromptArchitect' not found  → Cell 1: clone LTX2EasyPrompt-LD")
    print("   'LTX2MasterLoaderLD' not found   → Cell 1: clone LTX2-Master-Loader")
    print("   'LTXVImgToVideoInplace' missing  → Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVPreprocess' missing         → Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVCropGuides' missing         → Cell 1: clone ComfyUI-LTXVideo")
    print("   'PathchSageAttentionKJ' missing  → Cell 1: clone ComfyUI_KJNodes  "
          "  (or set USE_SAGE_ATTENTION=False)")
    print("   'LTXVChunkFeedForward' missing   → Cell 1: clone ComfyUI-LTXVideo  "
          "  (or set USE_CHUNK_FF=False)")
    print("   'VHS_VideoCombine' missing       → Cell 1: clone ComfyUI-VideoHelperSuite  "
          "  (auto-fallback to CreateVideo)")
    print("   'ModelSamplingSD3' missing       → set PRO_MODE=False in Cell 5")
    print("   'BasicScheduler' missing         → set PRO_MODE=False in Cell 5")
    print("   DualCLIPLoader fp4 error         → swap CLIP_NAME1 to fp8 in Cell 6")
    print("   Tiled VAE error                  → set USE_TILED_VAE=False in Cell 6")
    print("   Deformed output in Pass 1        → change SEED and re-run Cell 9")
    print("   Persistent deformation (3× same) → change USER_INPUT / POSITIVE_PROMPT")
    print("   Character drift                  → try CHARACTER_CONSISTENCY_MODE='both'")
    print("                                      or increase CHARACTER_STRENGTH")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9.5  ─  MERGE CLIPS INTO ONE VIDEO
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 9.5. Merge Multiple Clips Into One Video
# @markdown Combine generated scene clips into a single continuous video.
# @markdown Supports both simple concatenation (ffmpeg) and overlap blending.
# @markdown
# @markdown **Usage:** Set `CLIPS_TO_MERGE` to a list of video paths, then run this cell.

# ── Clips to merge ───────────────────────────────────────────────────────────
CLIPS_TO_MERGE = []  # @param
# Add paths to clips to merge, e.g.:
# CLIPS_TO_MERGE = [
#     "/content/ComfyUI/output/Story01-Character_00001_.mp4",
#     "/content/ComfyUI/output/Story02-Character_00001_.mp4",
#     "/content/ComfyUI/output/Story03-Character_00001_.mp4",
# ]

AUTO_FIND_CLIPS = True  # @param {type:"boolean"}
# When True AND CLIPS_TO_MERGE is empty, automatically find all clips
# matching the OUTPUT_PREFIX pattern in the output directory.

MERGE_OUTPUT_NAME = "Final_Merged"  # @param {type:"string"}
# Output filename prefix for the merged video.

MERGE_MODE = "overlap_blend"  # @param ["overlap_blend", "hard_concat", "crossfade"]
# "overlap_blend"  — SVI-Pro style linear blend (uses OVERLAP_FRAMES setting)
# "hard_concat"    — Simple ffmpeg concatenation (fastest, no blending)
# "crossfade"      — Cosine-weighted crossfade at scene boundaries

MERGE_FPS = 25  # @param {type:"integer"}
# Frame rate for the merged output video.


def merge_clips_to_video(
    clip_paths: List[str],
    output_name: str = "Final_Merged",
    mode: str = "overlap_blend",
    overlap: int = 5,
    fps: int = 25,
) -> Optional[str]:
    """
    Merge multiple video clips into a single continuous video.

    Supports three modes:
    - "overlap_blend": SVI-Pro style linear blend at boundaries (ImageBatchExtendWithOverlap)
    - "hard_concat": Simple ffmpeg concat (no blending, preserves exact frames)
    - "crossfade": Cosine-weighted crossfade for smooth transitions

    Args:
        clip_paths: List of video file paths to merge (in order)
        output_name: Output filename prefix
        mode: Merge strategy
        overlap: Number of overlap frames for blending modes
        fps: Output frame rate

    Returns:
        Path to the merged video file, or None on failure
    """
    import imageio

    # Validate inputs
    valid_paths = [p for p in clip_paths if p and os.path.exists(p)]
    if not valid_paths:
        print("❌ No valid clip paths provided.")
        print("   Set CLIPS_TO_MERGE or enable AUTO_FIND_CLIPS.")
        return None

    if len(valid_paths) == 1:
        print(f"ℹ️  Only one clip found — nothing to merge: {valid_paths[0]}")
        return valid_paths[0]

    print(f"🎬 Merging {len(valid_paths)} clips ({mode})...")
    for i, p in enumerate(valid_paths):
        print(f"   [{i+1}] {os.path.basename(p)}")

    output_dir = "/content/ComfyUI/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{output_name}_{int(time.time())}.mp4"

    # ── Mode: hard_concat (ffmpeg, fast) ──────────────────────────────────
    if mode == "hard_concat":
        return concatenate_clips(valid_paths, output_path)

    # ── Mode: overlap_blend or crossfade (frame-level processing) ─────────
    blend_mode = "linear_blend" if mode == "overlap_blend" else "crossfade"

    try:
        combined_frames = None
        total_input_frames = 0

        for idx, clip_path in enumerate(valid_paths):
            print(f"   Loading clip {idx+1}/{len(valid_paths)}: {os.path.basename(clip_path)}...")
            reader = imageio.get_reader(clip_path)
            frames_list = [frame for frame in reader]
            reader.close()
            total_input_frames += len(frames_list)

            seg_tensor = torch.from_numpy(np.stack(frames_list)).float() / 255.0

            if combined_frames is None:
                combined_frames = seg_tensor
            else:
                combined_frames = blend_overlap_frames(
                    combined_frames, seg_tensor,
                    overlap=overlap,
                    mode=blend_mode,
                    side=OVERLAP_SIDE
                )

            print(f"      {len(frames_list)} frames loaded, running total: {len(combined_frames)}")

        if combined_frames is None:
            print("❌ No frames loaded — merge failed.")
            return None

        # Save merged video (frames only first, then mux audio)
        print(f"   💾 Encoding {len(combined_frames)} frames @ {fps}fps...")
        final_np = (combined_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Step 1: Save video frames to a temporary file
        _temp_video = output_path.replace('.mp4', '_temp_noaudio.mp4')
        imageio.mimsave(_temp_video, [f for f in final_np], fps=fps, codec='libx264')
        
        # Step 2: Extract and concatenate audio from source clips, then mux
        _has_audio = False
        try:
            # Check if first clip has an audio stream
            _probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                         "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                         valid_paths[0]]
            _probe_result = subprocess.run(_probe_cmd, capture_output=True, text=True)
            _has_audio = "audio" in _probe_result.stdout
        except Exception:
            _has_audio = False
        
        if _has_audio:
            print("   🔊 Muxing audio from source clips...")
            try:
                # Concatenate audio from all source clips
                _audio_list = "/tmp/audio_concat_list.txt"
                with open(_audio_list, "w") as f:
                    for p in valid_paths:
                        f.write(f"file '{p}'\n")
                
                # Extract concatenated audio
                _temp_audio = output_path.replace('.mp4', '_temp_audio.aac')
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", _audio_list, "-vn", "-acodec", "aac",
                    "-b:a", "192k", _temp_audio
                ], check=True, capture_output=True)
                
                # Mux video (blended frames) + audio (concatenated from sources)
                # Trim audio to match video duration
                _video_duration = len(combined_frames) / fps
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", _temp_video,
                    "-i", _temp_audio,
                    "-c:v", "copy", "-c:a", "aac",
                    "-t", str(_video_duration),
                    "-shortest",
                    output_path
                ], check=True, capture_output=True)
                
                # Cleanup temp files
                if os.path.exists(_temp_video):
                    os.remove(_temp_video)
                if os.path.exists(_temp_audio):
                    os.remove(_temp_audio)
                
                print("   ✓ Audio muxed successfully")
            except Exception as audio_err:
                print(f"   ⚠️  Audio mux failed ({audio_err}) — video saved without audio.")
                # Fallback: just use the video-only file
                if os.path.exists(_temp_video):
                    os.rename(_temp_video, output_path)
        else:
            # No audio in source clips — just rename temp to final
            os.rename(_temp_video, output_path)
            print("   ℹ️  No audio detected in source clips — video-only output.")

        duration = len(combined_frames) / fps
        saved_frames = total_input_frames - len(combined_frames)

        print(f"\n{'═' * 60}")
        print(f"✅ MERGE COMPLETE!")
        print(f"   Output     : {output_path}")
        print(f"   Duration   : {duration:.1f}s ({len(combined_frames)} frames @ {fps}fps)")
        print(f"   Input clips: {len(valid_paths)}")
        print(f"   Overlap    : {overlap} frames × {len(valid_paths)-1} boundaries = {saved_frames} frames blended")
        print(f"   Mode       : {mode}")
        print(f"{'═' * 60}")

        # Display preview
        if SHOW_PREVIEWS:
            display_video(output_path)

        return output_path

    except Exception as e:
        print(f"❌ Merge failed: {type(e).__name__}: {e}")
        print("   Falling back to hard concatenation...")
        return concatenate_clips(valid_paths, output_path)


def auto_find_output_clips(
    prefix: str = None,
    output_dir: str = "/content/ComfyUI/output",
    pattern: str = None,
) -> List[str]:
    """
    Automatically find generated clips in the output directory.

    Args:
        prefix: Match clips starting with this prefix (e.g., "Story01")
        output_dir: Directory to search
        pattern: Glob pattern override (e.g., "Story*Character*.mp4")

    Returns:
        Sorted list of matching video paths
    """
    if not os.path.exists(output_dir):
        return []

    if pattern:
        import fnmatch
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if fnmatch.fnmatch(f, pattern) and f.endswith('.mp4')]
    elif prefix:
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.startswith(prefix) and f.endswith('.mp4')
                 and '_full' not in f and '_extended' not in f]
    else:
        # Find all Story/Scene clips based on OUTPUT_PREFIX
        _prefix = OUTPUT_PREFIX
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.endswith('.mp4') and not f.startswith('.')
                 and '_full' not in f and '_extended' not in f
                 and '_segments' not in f and 'Final_Merged' not in f]

    clips.sort()
    return clips


# ── Execute merge ─────────────────────────────────────────────────────────────
_clips_to_merge = CLIPS_TO_MERGE

if not _clips_to_merge and AUTO_FIND_CLIPS:
    print("🔍 Auto-finding clips in output directory...")
    _clips_to_merge = auto_find_output_clips()
    if _clips_to_merge:
        print(f"   Found {len(_clips_to_merge)} clips:")
        for c in _clips_to_merge:
            print(f"      • {os.path.basename(c)}")
    else:
        print("   ℹ️  No clips found. Generate some clips first (Cell 9), then re-run this cell.")

if _clips_to_merge:
    merged_output = merge_clips_to_video(
        clip_paths=_clips_to_merge,
        output_name=MERGE_OUTPUT_NAME,
        mode=MERGE_MODE,
        overlap=OVERLAP_FRAMES,
        fps=MERGE_FPS,
    )
    if merged_output and DOWNLOAD_AFTER_GENERATE:
        try:
            files.download(merged_output)
        except Exception as e:
            print(f"   ⚠️  Download failed ({e}) — file is at {merged_output}")
else:
    print("ℹ️  No clips to merge. Either:")
    print("   1. Set CLIPS_TO_MERGE = ['/path/to/clip1.mp4', '/path/to/clip2.mp4', ...]")
    print("   2. Or generate clips first (Cell 9 with USE_STORYBOARD=True), then re-run this cell")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10  ─  EXPORT & POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 10. Export & Post-Processing
# @markdown Run after generation to export timeline, apply additional grading,
# @markdown or sync outputs to Google Drive.
# @markdown
# @markdown This cell is optional - features here also run automatically
# @markdown when enabled in Cells 5-6 during generation.

# ── Manual Google Drive mount & sync ──────────────────────────────────────────
if PERSIST_TO_GDRIVE:
    print("📁 Google Drive Sync...")
    _drive_mounted = mount_google_drive()
    if _drive_mounted:
        # Sync all outputs from this session
        _output_dir = "/content/ComfyUI/output"
        if os.path.exists(_output_dir):
            _files_synced = 0
            for _f in os.listdir(_output_dir):
                if _f.startswith(OUTPUT_PREFIX) and _f.endswith((".mp4", ".json")):
                    _src = os.path.join(_output_dir, _f)
                    if sync_to_drive(_src, GDRIVE_PATH):
                        _files_synced += 1
            print(f"   ✓ Synced {_files_synced} files to {GDRIVE_PATH}")
    else:
        print("   ⚠️  Google Drive not available")

# ── Timeline export (manual trigger) ─────────────────────────────────────────
if EXPORT_TIMELINE:
    print("\n📋 Timeline Export...")
    _tl_output = f"/content/ComfyUI/output/{OUTPUT_PREFIX}_timeline.{TIMELINE_FORMAT}"
    if os.path.exists(_tl_output):
        print(f"   ✓ Timeline already exists: {_tl_output}")
        # Display summary
        try:
            with open(_tl_output, "r") as _f:
                _tl_data = json.load(_f)
            if isinstance(_tl_data, dict) and "segments" in _tl_data:
                print(f"   Segments: {len(_tl_data['segments'])}")
                print(f"   Total duration: {_tl_data.get('total_duration_seconds', 'N/A')}s")
                for _seg in _tl_data["segments"][:5]:
                    print(f"     [{_seg.get('segment_index', '?')}] "
                          f"{_seg.get('duration_seconds', 0):.1f}s - "
                          f"{_seg.get('prompt', '')[:50]}...")
        except Exception:
            pass
        # Offer download
        if DOWNLOAD_AFTER_GENERATE:
            try:
                files.download(_tl_output)
            except Exception:
                pass
    else:
        print(f"   ℹ️  No timeline found. Run generation with EXPORT_TIMELINE=True first.")

# ── Color grade batch application ─────────────────────────────────────────────
# @markdown ---
# @markdown ### Batch Post-Processing
# @markdown Apply color grading to existing output files.

BATCH_COLOR_GRADE_TARGET = ""  # @param {type:"string"}
# Path to a video file to apply color grading to.
# Leave empty to skip batch grading.

if BATCH_COLOR_GRADE_TARGET and COLOR_GRADE != "none" and os.path.exists(BATCH_COLOR_GRADE_TARGET):
    print(f"\n🎨 Applying {COLOR_GRADE} grade to: {BATCH_COLOR_GRADE_TARGET}")
    try:
        import imageio
        _reader = imageio.get_reader(BATCH_COLOR_GRADE_TARGET)
        _frames = [f for f in _reader]
        _reader.close()
        _tensor = torch.from_numpy(np.stack(_frames)).float() / 255.0
        _graded = apply_color_grade(_tensor, COLOR_GRADE)
        _graded_np = (_graded.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        _graded_path = BATCH_COLOR_GRADE_TARGET.replace(".mp4", f"_{COLOR_GRADE}.mp4")
        imageio.mimsave(_graded_path, [f for f in _graded_np], fps=FPS, codec='libx264')
        print(f"   ✓ Graded video saved: {_graded_path}")
        if SHOW_PREVIEWS:
            display_video(_graded_path)
    except Exception as e:
        print(f"   ⚠️  Batch grading failed ({e})")

print("\n✅ Post-processing complete.")
