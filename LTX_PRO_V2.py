# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 PRO V2 — Complete Pipeline v2.0                                     ║
# ║  Integrates: LD-I2V + SVI-Pro + EasyPrompt + Character Consistency          ║
# ║  Base engine: LTX-2 19B Distilled GGUF Q4_K_M (Colab T4/L4/A100 safe)     ║
# ║                                                                              ║
# ║  V2 IMPROVEMENTS (23):                                                       ║
# ║  1.  Session-level model cache (ModelCache dataclass + USE_MODEL_CACHE)     ║
# ║  2.  Real negative prompt encoding (USE_REAL_NEGATIVE)                      ║
# ║  3.  CFG > 1.0 guidance mode (USE_CFG_GUIDANCE)                             ║
# ║  4.  TAESD preview after Pass 1 (SHOW_TAESD_PREVIEW)                        ║
# ║  5.  Auto-resolution selection (AUTO_RESOLUTION)                            ║
# ║  6.  Node availability check table (check_nodes)                            ║
# ║  7.  OOM retry logic in Cell 9 (MAX_OOM_RETRIES)                            ║
# ║  8.  Sigma schedule validation (validate_sigmas)                            ║
# ║  9.  Dynamic sigma schedule generator (make_sigmas / SIGMA_SCHEDULE)        ║
# ║  10. Denoise strength / img2img partial denoising (DENOISE)                 ║
# ║  11. CPU offload toggle (USE_CPU_OFFLOAD)                                   ║
# ║  12. Batch generation function (generate_batch)                             ║
# ║  13. Prompt history CSV log (LOG_PATH / append_generation_log)              ║
# ║  14. Google Drive auto-save (SAVE_TO_DRIVE)                                 ║
# ║  15. VHS path coercion (str() for pathlib.Path objects)                     ║
# ║  16. Multi-character support (CHARACTERS / USE_MULTI_CHARACTER)             ║
# ║  17. Full type annotations on all helper functions                          ║
# ║  18. _node() wrapper with helpful KeyError messages                         ║
# ║  19. Frame interpolation via ffmpeg minterpolate (USE_FRAME_INTERPOLATION)  ║
# ║  20. Persistent prompt history display (show_generation_history)            ║
# ║  21. Improved VRAM reporting (reserved, peak, GPU name)                     ║
# ║  22. Storyboard continuity conflict resolution note                         ║
# ║  23. SeedGenerator node support (USE_SEED_GENERATOR)                        ║
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
dit_model = model_download(
    f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
    "/content/ComfyUI/models/unet")

# ── Text encoders ─────────────────────────────────────────────────────────────
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")
# Gemma fp8 — T4 / A100 / RTX 3000-4000 (uncomment if fp4 OOMs):
# text_encoder_model = model_download(
#     f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
#     "/content/ComfyUI/models/text_encoders")

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

# TaeEncoder preview VAE (fast latent preview / TAESD)
taeltx2_model = model_download(
    f"{KIJAI23}/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae")

# ── Spatial upscaler ──────────────────────────────────────────────────────────
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

# ── IC LoRAs + Camera Control LoRAs ──────────────────────────────────────────
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

import os, sys, gc, time, json, csv, shutil, random, math, warnings, subprocess, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
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

# ── IMPROVEMENT 1: Session-level model cache ─────────────────────────────────
@dataclass
class ModelCache:
    """Holds loaded model objects across generate_pro() calls to avoid reload."""
    unet:               Any   = None
    unet_name:          str   = ""
    clip:               Any   = None
    clip_names:         tuple = ()
    vae_video:          Any   = None
    vae_video_name:     str   = ""
    vae_audio:          Any   = None
    vae_audio_name:     str   = ""
    upscale_model:      Any   = None
    upscale_model_name: str   = ""

_model_cache = ModelCache()

def clear_model_cache() -> None:
    """Reset the session-level model cache, freeing all cached model objects."""
    global _model_cache
    _model_cache = ModelCache()
    cleanup_memory()
    print("   ✓ Model cache cleared.")

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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    if verbose:
        _print_vram()

# ── IMPROVEMENT 21: Enhanced VRAM reporting ──────────────────────────────────
def _print_vram() -> None:
    """Print VRAM usage including reserved, peak, and GPU name."""
    if not torch.cuda.is_available():
        return
    used     = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak     = torch.cuda.max_memory_allocated() / 1024**3
    total    = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    pct      = used / total * 100 if total > 0 else 0
    filled   = int(20 * used / total) if total > 0 else 0
    bar      = "█" * filled + "░" * (20 - filled)
    print(f"   GPU: {gpu_name} | VRAM [{bar}] alloc={used:.1f} reserved={reserved:.1f} "
          f"peak={peak:.1f} / {total:.1f} GB ({pct:.1f}%)")

# ── ComfyUI node output accessor ──────────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# ── IMPROVEMENT 18: Node call wrapper ────────────────────────────────────────
def _node(name: str) -> Any:
    """
    Instantiate a ComfyUI node by name with a helpful error if missing.
    For optional nodes, keep using 'if name in NODE_CLASS_MAPPINGS' guards.
    """
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(
            f"Node '{name}' not found. Run Cell 1 to install required custom nodes."
        )
    return NODE_CLASS_MAPPINGS[name]()

# ── Tensor / image conversion ─────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI NHWC float tensor."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """ComfyUI NHWC tensor -> PIL."""
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

def save_video_from_components(video_obj: Any, prefix: str = "LTX-2-PRO") -> str:
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
def apply_sage_attention(unet: Any) -> Any:
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
        node  = _node("PathchSageAttentionKJ")
        fn    = getattr(node, node.FUNCTION)
        unet  = get_value_at_index(fn(model=unet), 0)
        print("   ✓ SageAttention patch applied (PathchSageAttentionKJ)")
    except Exception as e:
        print(f"   ⚠️  SageAttention failed ({e}) — continuing without it.")
    return unet

def apply_chunk_ff(unet: Any) -> Any:
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
        node  = _node("LTXVChunkFeedForward")
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
            fn()
            print(f"   ✓ VRAM purged via PurgeVRAM V2{tag}")
            return
        except Exception as e:
            print(f"   ⚠️  PurgeVRAM V2 failed ({e}) — using torch fallback.")
    cleanup_memory()
    print(f"   ✓ VRAM cleared via torch.cuda.empty_cache{tag}")

# ── Upload helper ─────────────────────────────────────────────────────────────
def upload_image(save_dir: str = "/content/ComfyUI/input") -> Optional[str]:
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
def _load_audio_vae(vae_name: str) -> Any:
    """Load audio VAE. Prefers VAELoaderKJ (main_device, fp16), falls back to VAELoader."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=vae_name, device="main_device", weight_dtype="fp16")
    return _node("VAELoader").load_vae(vae_name=vae_name)

# ── IC LoRA filename lookup (slot 1) ─────────────────────────────────────────
_IC_LORA_FILES: Dict[str, str] = {
    "none":     "None",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    "canny":    "ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth":    "ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose":     "ltx-2-19b-ic-lora-pose-control.safetensors",
}

# ── Camera LoRA filename lookup (slot 2) ─────────────────────────────────────
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

def apply_lora_stack(unet: Any, clip_model: Any,
                     lora_stack: Optional[List[Dict]] = None,
                     lora_stack_json: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Apply LoRA stack via LTX2MasterLoaderLD node when available.
    Falls back to manual LoraLoaderModelOnly loop if the node is missing.
    Returns: (unet, clip_model)
    """
    stack  = lora_stack or []
    active = [s for s in stack
              if s.get("on") and s.get("lora") not in (None, "None", "")]

    if not active:
        print("   ℹ️  No active LoRAs in stack — skipping.")
        return unet, clip_model

    # ── Try LTX2MasterLoaderLD node [263] (LoRa Daddy) ───────────────────────
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        print(f"   [MasterLoader] {len(active)} LoRA(s) via LTX2MasterLoaderLD…")
        try:
            node   = _node("LTX2MasterLoaderLD")
            fn     = getattr(node, node.FUNCTION)
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

# ── IMPROVEMENT 8: Sigma schedule validation ─────────────────────────────────
def validate_sigmas(s: str) -> bool:
    """
    Validate a comma-separated sigma schedule string.
    Checks: all values in [0,1], strictly decreasing, ends with 0.0.
    Prints warnings for each violation.
    Returns True if valid, False otherwise.
    """
    try:
        vals = [float(v.strip()) for v in s.split(",") if v.strip()]
    except ValueError as e:
        print(f"   ⚠️  validate_sigmas: parse error: {e}")
        return False

    ok = True
    if not all(0.0 <= v <= 1.0 for v in vals):
        bad = [v for v in vals if not (0.0 <= v <= 1.0)]
        print(f"   ⚠️  validate_sigmas: values out of [0,1]: {bad}")
        ok = False
    for i in range(len(vals) - 1):
        if vals[i] <= vals[i + 1]:
            print(f"   ⚠️  validate_sigmas: not strictly decreasing at index {i}: "
                  f"{vals[i]} -> {vals[i+1]}")
            ok = False
    if vals and vals[-1] != 0.0:
        print(f"   ⚠️  validate_sigmas: schedule does not end with 0.0 (got {vals[-1]})")
        ok = False
    return ok

# ── IMPROVEMENT 9: Dynamic sigma schedule generator ──────────────────────────
def make_sigmas(steps: int, start: float = 1.0, end: float = 0.0,
                schedule: str = "cosine") -> str:
    """
    Generate a comma-separated sigma schedule string.

    schedule: "cosine" | "linear" | "karras"
    Always appends 0.0 at the end.
    """
    if steps < 1:
        steps = 1
    if schedule == "linear":
        vals = [start + (end - start) * i / steps for i in range(steps)]
    elif schedule == "karras":
        # Karras noise schedule: sigma(t) = (sigma_max^(1/rho) + t/(n-1)*(sigma_min^(1/rho)-sigma_max^(1/rho)))^rho
        rho = 7.0
        sigma_min = max(end, 1e-3)
        sigma_max = start
        n = steps
        vals = []
        for i in range(n):
            t = i / max(n - 1, 1)
            s = (sigma_max ** (1.0 / rho) + t * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
            vals.append(float(s))
    else:
        # cosine (default)
        vals = []
        for i in range(steps):
            t   = i / max(steps - 1, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            vals.append(start * cos + end * (1.0 - cos))

    vals.append(0.0)
    return ", ".join(f"{v:.6f}" for v in vals)

# ── IMPROVEMENT 5: Auto-resolution selection ─────────────────────────────────
def auto_select_resolution() -> Tuple[int, int, int]:
    """
    Select WIDTH, HEIGHT, FRAMES based on available GPU VRAM.
    Returns (width, height, frames).
    """
    if not torch.cuda.is_available():
        return 768, 512, 121
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram >= 38:
        return 1280, 720, 241
    if vram >= 22:
        return 1024, 576, 161
    return 768, 512, 121

# ── IMPROVEMENT 13: Prompt history CSV logger ────────────────────────────────
def append_generation_log(seed: int, width: int, height: int, frames: int,
                           elapsed_s: float, loras: List[str],
                           positive_prompt: str,
                           output_path: Optional[str]) -> None:
    """
    Append a generation record to the CSV log at LOG_PATH.
    Creates the file with a header row if it does not yet exist.
    """
    try:
        if not ENABLE_GENERATION_LOG:
            return
        file_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "seed", "width", "height", "frames",
                    "elapsed_s", "loras", "positive_prompt", "output_path"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                seed, width, height, frames,
                f"{elapsed_s:.1f}",
                "|".join(loras),
                positive_prompt[:100],
                output_path or "",
            ])
    except Exception as e:
        print(f"   ⚠️  Generation log write failed: {e}")

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
    Calls LTX2PromptArchitect to expand a simple story description into a
    dense cinematic prompt. Falls back to returning raw input if unavailable.
    Returns: (positive_prompt, negative_prompt)
    """
    if "LTX2PromptArchitect" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTX2PromptArchitect not found — using raw user_input.")
        return user_input, ""

    _model = llm_model_override if llm_model_override is not None else LLM_MODEL
    print(f"   [EasyPrompt] LLM={_model} | creativity={CREATIVITY} | frames={frame_count}")
    node = _node("LTX2PromptArchitect")
    result = node.generate(
        bypass=False,
        user_input=user_input,
        creativity=_creativity_label(CREATIVITY),
        seed=seed,
        invent_dialogue=INVENT_DIALOGUE,
        keep_model_loaded=False,
        offline_mode=False,
        frame_count=frame_count,
        model=_LLM_LABEL_MAP.get(_model, "8B - NeuralDaredevil (High Quality)"),
        local_path_8b="",
        local_path_3b="",
        local_path_14b="",
        scene_context=scene_context,
        lora_triggers=LORA_TRIGGERS,
    )
    prompt     = result[0]
    neg_prompt = result[2]
    print(f"   [EasyPrompt] ✓  {len(prompt.split())} words generated.")
    cleanup_memory()
    return prompt, neg_prompt


def run_vision_describe(image_tensor: torch.Tensor,
                        character_desc: str = "",
                        use_vision_override: bool = None,
                        vision_model_override: str = None) -> str:
    """
    Calls LTX2VisionDescribe to analyse the image and return a scene description.
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
    node = _node("LTX2VisionDescribe")
    result = node.describe(
        image=image_tensor,
        model_name=_VISION_LABEL_MAP.get(_vis_mod, "Qwen2.5-VL-3B — Fast (huihui abliterated)"),
        offline_mode=False,
        local_path="",
    )
    ctx = result[0]
    if character_desc:
        ctx = character_desc + " " + ctx
    print(f"   [VisionDescribe] ✓  {len(ctx.split())} words.")
    cleanup_memory()
    return ctx

# ── IMPROVEMENT 6: Node availability check ───────────────────────────────────
def check_nodes() -> None:
    """
    Print a formatted table of required and optional node availability.
    Missing required nodes show a hint to run Cell 1.
    """
    _required = [
        "UnetLoaderGGUF", "DualCLIPLoader", "CLIPTextEncode",
        "LTXVConditioning", "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent", "ManualSigmas", "KSamplerSelect", "CFGGuider",
        "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVCropGuides",
        "LTXVLatentUpsampler", "VAELoader", "VAEDecode", "LTXVAudioVAEDecode",
        "CreateVideo",
    ]
    _optional = [
        "LTX2PromptArchitect", "LTX2VisionDescribe", "LTX2MasterLoaderLD",
        "LTXVImgToVideoInplace", "LTXVPreprocess",
        "LTXVSpatioTemporalTiledVAEDecode", "VHS_VideoCombine",
        "PathchSageAttentionKJ", "LTXVChunkFeedForward", "ModelSamplingSD3",
        "BasicScheduler", "SplitSigmas", "VAELoaderKJ",
        "LatentUpscaleModelLoader", "ResizeImageMaskNode", "GetImageSize",
        "ImageResizeKJv2", "ResizeImagesByLongerEdge",
    ]
    missing_required = []
    print("\n── Node availability ────────────────────────────────────────────────────")
    print("   REQUIRED:")
    for name in _required:
        ok = name in NODE_CLASS_MAPPINGS
        print(f"   {'✅' if ok else '❌'}  {name}")
        if not ok:
            missing_required.append(name)
    print("   OPTIONAL:")
    for name in _optional:
        ok = name in NODE_CLASS_MAPPINGS
        print(f"   {'✅' if ok else '○ '}  {name}")
    if missing_required:
        print(f"\n   ⚠️  Missing required nodes -> clone in Cell 1:")
        for name in missing_required:
            print(f"      • {name}")
    else:
        print("\n   ✓ All required nodes present.")
    print("─" * 70)


print("✅ Imports & helpers ready.")
print("   Helper functions defined:")
print("   ✓ cleanup_memory()           — with ipc_collect()")
print("   ✓ _print_vram()              — alloc + reserved + peak + GPU name")
print("   ✓ _node()                    — safe NODE_CLASS_MAPPINGS wrapper")
print("   ✓ apply_sage_attention()     — PathchSageAttentionKJ wrapper")
print("   ✓ apply_chunk_ff()           — LTXVChunkFeedForward wrapper")
print("   ✓ purge_vram()               — LayerUtility: PurgeVRAM V2 wrapper")
print("   ✓ apply_lora_stack()         — LTX2MasterLoaderLD + manual fallback")
print("   ✓ run_easy_prompt()          — LTX2PromptArchitect wrapper")
print("   ✓ run_vision_describe()      — LTX2VisionDescribe wrapper")
print("   ✓ save_metadata_sidecar()    — JSON sidecar writer")
print("   ✓ validate_sigmas()          — sigma schedule validator")
print("   ✓ make_sigmas()              — dynamic sigma schedule generator")
print("   ✓ auto_select_resolution()   — VRAM-based resolution picker")
print("   ✓ check_nodes()              — node availability table")
print("   ✓ append_generation_log()    — CSV generation logger")
print("   ✓ clear_model_cache()        — session model cache reset")
print("   ✓ ModelCache (dataclass)     — session-level model cache")

# ── Print GPU name once at Cell 3 init ───────────────────────────────────────
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── IMPROVEMENT 6: Call check_nodes at end of Cell 3 ─────────────────────────
check_nodes()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  ─  EASY PROMPT + VISION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 4. Easy Prompt & Vision Describe Configuration
# @markdown Set `BYPASS_EASY_PROMPT = True` to skip the LLM and write
# @markdown `POSITIVE_PROMPT` manually in Cell 6.

# ── LLM prompt expander (LTX2PromptArchitect node) ───────────────────────────
LLM_MODEL  = "8B"    # @param ["8B", "3B", "14B"]
# "8B"  -> NeuralDaredevil-8B-abliterated  — best quality, ~10 GB VRAM
# "3B"  -> Llama-3.2-3B-abliterated       — fastest, ~4 GB (T4 safe)
# "14B" -> Qwen3-14B-abliterated           — highest quality, ~18 GB VRAM

CREATIVITY = 0.9     # @param {type:"number"}
# 0.7 = Literal & Grounded  |  0.9 = Balanced  |  1.1 = Artistic

INVENT_DIALOGUE    = True   # @param {type:"boolean"}
# When True the LLM invents natural spoken dialogue woven into the scene.

BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
# True  -> skip LLM, use POSITIVE_PROMPT directly (fast/manual control)
# False -> LLM expands USER_INPUT into a full cinematic prompt

LORA_TRIGGERS = ""          # @param {type:"string"}
# LoRA trigger words injected at the start of every expanded prompt.
# e.g. "ohwx woman" or "film grain, 35mm"

# ── Vision image describer (LTX2VisionDescribe node) ─────────────────────────
USE_VISION   = True          # @param {type:"boolean"}
# When True AND an image is provided, Vision Describe analyses it and passes
# the result as scene_context to Easy Prompt.

VISION_MODEL = "3B-fast"     # @param ["3B-fast", "7B-nsfw"]
# "3B-fast" -> Qwen2.5-VL-3B — faster, ~5 GB VRAM
# "7B-nsfw" -> Qwen2.5-VL-7B — more accurate, ~10 GB VRAM

# ── Display & output ──────────────────────────────────────────────────────────
SHOW_PREVIEWS           = True   # @param {type:"boolean"}
DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}

print("✅ Easy Prompt + Vision settings ready.")
print(f"   LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}  |  "
      f"Creativity: {CREATIVITY}  |  Bypass: {BYPASS_EASY_PROMPT}")
print(f"   Show previews: {SHOW_PREVIEWS}  |  Auto-download: {DOWNLOAD_AFTER_GENERATE}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  ─  CHARACTER CONSISTENCY & LORA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 5. Character Consistency & LoRA Configuration
# @markdown Configure character reference, IC LoRA, camera LoRA, performance,
# @markdown and V2 model-cache / guidance settings.

# ── Character Consistency System ─────────────────────────────────────────────
CHARACTER_IMAGE_PATH = None  # @param {type:"string"}
CHARACTER_STRENGTH   = 1.0   # @param {type:"number"}
CHARACTER_CONSISTENCY_MODE = "i2v"  # @param ["i2v", "anchor", "both", "none"]
CHARACTER_NAME        = "Character"  # @param {type:"string"}
CHARACTER_DESCRIPTION = ""           # @param {type:"string"}

# ── IC LoRA slot 1 ────────────────────────────────────────────────────────────
IC_LORA          = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4         # @param {type:"number"}

# ── Camera LoRA slot 2 ────────────────────────────────────────────────────────
CAMERA_LORA          = "none"  # @param ["none", "dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"]
CAMERA_LORA_STRENGTH = 1.0     # @param {type:"number"}

# ── Memory & performance flags ────────────────────────────────────────────────
USE_SAGE_ATTENTION      = False  # @param {type:"boolean"}
USE_CHUNK_FF            = False  # @param {type:"boolean"}
PURGE_VRAM_AFTER_MODELS = True   # @param {type:"boolean"}

# ── Pro Sampling Mode (from SVI-Pro-Workflow.json) ────────────────────────────
PRO_MODE      = False    # @param {type:"boolean"}
PRO_STEPS     = 4        # @param {type:"integer"}
PRO_SCHEDULER = "simple" # @param {type:"string"}
PRO_SPLIT_AT  = 2        # @param {type:"integer"}

# ── IMPROVEMENT 1: Session-level model cache ─────────────────────────────────
USE_MODEL_CACHE = True   # @param {type:"boolean"}
# When True, skip reloading models that are already in the session cache.
# Call clear_model_cache() to force a full reload.

# ── IMPROVEMENT 3: CFG > 1.0 guidance mode ───────────────────────────────────
USE_CFG_GUIDANCE = False  # @param {type:"boolean"}
# When True: forces USE_REAL_NEGATIVE=True and sets PASS1_CFG/PASS2_CFG=3.5
# CFG > 1 requires a real negative prompt to be meaningful.

# ── IMPROVEMENT 11: CPU offload toggle ───────────────────────────────────────
USE_CPU_OFFLOAD = False  # @param {type:"boolean"}
# USE_CPU_OFFLOAD: for T4 users generating >161 frames.
# Moves UNet to CPU after Pass 1 to free VRAM for Pass 2 setup, then back.

# ── IMPROVEMENT 16: Multi-character support ──────────────────────────────────
CHARACTERS: List[Dict] = [
    # {"name": "Elena", "image_path": "/content/elena.jpg", "strength": 1.0,
    #  "description": "tall woman, auburn hair", "mode": "i2v"},
]
USE_MULTI_CHARACTER = False  # @param {type:"boolean"}
# When True and CHARACTERS is non-empty, use CHARACTERS[0] as primary character
# and concatenate VisionDescribe results for additional characters into scene_context.

# ── Build LoRA stack from dropdowns ──────────────────────────────────────────
LORA_STACK      = _build_lora_stack(IC_LORA, IC_LORA_STRENGTH,
                                     CAMERA_LORA, CAMERA_LORA_STRENGTH)
LORA_STACK_JSON = json.dumps(LORA_STACK)

_active_count = sum(1 for s in LORA_STACK if s["on"])
print("✅ Character Consistency & LoRA configuration ready.")
print(f"   Character mode   : {CHARACTER_CONSISTENCY_MODE}  |  strength: {CHARACTER_STRENGTH}")
print(f"   Character image  : {CHARACTER_IMAGE_PATH or 'None'}")
print(f"   IC LoRA          : {IC_LORA} @ {IC_LORA_STRENGTH}")
print(f"   Camera LoRA      : {CAMERA_LORA} @ {CAMERA_LORA_STRENGTH}")
print(f"   Active LoRA slots: {_active_count}/10")
print(f"   Pro Mode         : {PRO_MODE}  |  SageAttn: {USE_SAGE_ATTENTION}  |  ChunkFF: {USE_CHUNK_FF}")
print(f"   Model Cache      : {USE_MODEL_CACHE}  |  CFG Guidance: {USE_CFG_GUIDANCE}  |  CPU Offload: {USE_CPU_OFFLOAD}")
print(f"   Multi-Character  : {USE_MULTI_CHARACTER}  ({len(CHARACTERS)} characters defined)")


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

# ── Reference / seed image (optional) ────────────────────────────────────────
IMAGE_PATH      = None   # @param {type:"string"}
IMAGE_STRENGTH  = 1.0    # @param {type:"number"}

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

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED                = 47    # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}

# ── Model filenames ───────────────────────────────────────────────────────────
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# ── Pass 1 sampling (ManualSigmas schedule) ───────────────────────────────────
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"           # @param {type:"string"}
PASS1_CFG     = 1.0               # @param {type:"number"}

# ── Pass 2 sampling (spatial upscale + refinement) ────────────────────────────
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}
PASS2_CFG     = 1.0                    # @param {type:"number"}
PASS2_SEED    = 0                      # @param {type:"integer"}

# ── Tiled VAE decode ──────────────────────────────────────────────────────────
USE_TILED_VAE          = True   # @param {type:"boolean"}
TILED_SPATIAL_TILES    = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP  = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN     = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX   = False  # @param {type:"boolean"}

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_PREFIX = "LTX-2-PRO"  # @param {type:"string"}

# ── Storyboard continuity ─────────────────────────────────────────────────────
USE_SCENE_CONTINUITY = True  # @param {type:"boolean"}

# ── IMPROVEMENT 2: Real negative prompt encoding ────────────────────────────
USE_REAL_NEGATIVE = False  # @param {type:"boolean"}
# When True (or when PASS1_CFG > 1.0), encodes actual NEGATIVE_PROMPT text
# for LTXVConditioning instead of ConditioningZeroOut.
# Automatically forced True when USE_CFG_GUIDANCE=True.

# ── IMPROVEMENT 4: TAESD preview after Pass 1 ────────────────────────────────
SHOW_TAESD_PREVIEW = True  # @param {type:"boolean"}
# After Pass 1 completes, decode the first frame using taeltx2_3.safetensors
# and display it inline. Falls back silently if TAESD file is unavailable.

# ── IMPROVEMENT 5: Auto-resolution ───────────────────────────────────────────
AUTO_RESOLUTION = False  # @param {type:"boolean"}
# When True, override WIDTH/HEIGHT/FRAMES based on available GPU VRAM.

# ── IMPROVEMENT 9: Dynamic sigma schedule ────────────────────────────────────
SIGMA_SCHEDULE = "manual"  # @param {type:"string"}
# "manual" -> use PASS1_SIGMAS string directly
# "cosine" / "linear" / "karras" -> generate using make_sigmas(PASS1_STEPS)
PASS1_STEPS = 8  # @param {type:"integer"}
# Number of steps for make_sigmas() when SIGMA_SCHEDULE != "manual"

# ── IMPROVEMENT 10: Denoise strength ─────────────────────────────────────────
DENOISE = 1.0  # @param {type:"number"}
# Range 0.0-1.0. When < 1.0, trims Pass 1 sigma schedule for img2img-style
# partial denoising. 1.0 = full denoising (default).

# ── IMPROVEMENT 13: Prompt history log ───────────────────────────────────────
LOG_PATH             = "/content/generation_log.csv"  # @param {type:"string"}
ENABLE_GENERATION_LOG = True  # @param {type:"boolean"}

# ── IMPROVEMENT 14: Google Drive auto-save ───────────────────────────────────
SAVE_TO_DRIVE = False  # @param {type:"boolean"}
DRIVE_FOLDER  = "LTX-2"  # @param {type:"string"}

# ── IMPROVEMENT 19: Frame interpolation ──────────────────────────────────────
USE_FRAME_INTERPOLATION = False  # @param {type:"boolean"}
INTERPOLATION_FACTOR    = 2      # @param {type:"integer"}
# When True, runs ffmpeg minterpolate after saving to double (or multiply) FPS.

# ── IMPROVEMENT 23: SeedGenerator node support ───────────────────────────────
USE_SEED_GENERATOR = False  # @param {type:"boolean"}
# When True, tries the SeedGenerator ComfyUI node to produce the seed,
# falling back to random.randint if the node is unavailable.

# ── Apply AUTO_RESOLUTION at Cell 6 execution time ───────────────────────────
if AUTO_RESOLUTION:
    WIDTH, HEIGHT, FRAMES = auto_select_resolution()
    print(f"   AUTO_RESOLUTION: selected {WIDTH}x{HEIGHT}, {FRAMES} frames")

# ── Apply CFG guidance defaults if enabled ───────────────────────────────────
if USE_CFG_GUIDANCE:
    USE_REAL_NEGATIVE = True   # CFG > 1 requires a real negative prompt to be meaningful.
    PASS1_CFG = 3.5
    PASS2_CFG = 3.5
    print("   ℹ️  USE_CFG_GUIDANCE=True: USE_REAL_NEGATIVE forced True, CFG set to 3.5")

print("✅ Configuration set.")
print(f"   Resolution : {WIDTH}x{HEIGHT}  |  Frames : {FRAMES}  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   UNet       : {UNET_MODEL}")
print(f"   Seed       : {SEED}  (auto-increment: {AUTO_INCREMENT_SEED})")
print(f"   Pass 1     : {PASS1_SAMPLER}  |  {PASS1_SIGMAS[:45]}...")
print(f"   Pass 2     : {PASS2_SAMPLER}  |  {PASS2_SIGMAS}")
print(f"   Pro Mode   : {PRO_MODE}  |  steps={PRO_STEPS}, scheduler={PRO_SCHEDULER}, split@{PRO_SPLIT_AT}")
print(f"   Sigma Sched: {SIGMA_SCHEDULE}  |  PASS1_STEPS={PASS1_STEPS}  |  DENOISE={DENOISE}")
print(f"   Real Neg   : {USE_REAL_NEGATIVE}  |  TAESD Preview: {SHOW_TAESD_PREVIEW}  |  Drive Save: {SAVE_TO_DRIVE}")


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
    bypass_easy_prompt:      bool  = None,
    llm_model:               str   = None,
    use_vision:              bool  = None,
    vision_model:            str   = None,
    unet_model:              str   = None,
    clip_name1:              str   = None,
    clip_name2:              str   = None,
    # ── V2 new params ──────────────────────────────────────────────────────
    use_real_negative:       bool  = None,   # None -> USE_REAL_NEGATIVE global
    denoise:                 float = None,   # None -> DENOISE global
    use_seed_generator:      bool  = None,   # None -> USE_SEED_GENERATOR global
    use_multi_character:     bool  = None,   # None -> USE_MULTI_CHARACTER global
    multi_characters:        Optional[List[Dict]] = None,  # None -> CHARACTERS global
    use_cpu_offload:         bool  = None,   # None -> USE_CPU_OFFLOAD global
    show_taesd_preview:      bool  = None,   # None -> SHOW_TAESD_PREVIEW global
    use_frame_interpolation: bool  = None,   # None -> USE_FRAME_INTERPOLATION global
    interpolation_factor:    int   = None,   # None -> INTERPOLATION_FACTOR global
) -> Optional[str]:
    """
    LTX-2 PRO V2 — Two-pass generation pipeline with Character Consistency.
    Includes all 23 V2 improvements.
    Returns: output video path (str) or None on failure.
    """
    t0 = time.time()
    import_custom_nodes()
    clear_output()

    _lora_stack = lora_stack if lora_stack is not None else LORA_STACK
    _lora_json  = lora_stack_json if lora_stack_json is not None else LORA_STACK_JSON
    _char_mode  = character_mode.lower().strip()

    # Resolve per-call overrides vs. module globals
    _bypass          = bypass_easy_prompt if bypass_easy_prompt is not None else BYPASS_EASY_PROMPT
    _llm_model       = llm_model       if llm_model       is not None else LLM_MODEL
    _use_vision      = use_vision      if use_vision      is not None else USE_VISION
    _vis_model       = vision_model    if vision_model    is not None else VISION_MODEL
    _unet            = unet_model      if unet_model      is not None else UNET_MODEL
    _clip1           = clip_name1      if clip_name1      is not None else CLIP_NAME1
    _clip2           = clip_name2      if clip_name2      is not None else CLIP_NAME2
    _use_real_neg    = use_real_negative  if use_real_negative  is not None else USE_REAL_NEGATIVE
    _denoise         = denoise         if denoise         is not None else DENOISE
    _use_seed_gen    = use_seed_generator if use_seed_generator is not None else USE_SEED_GENERATOR
    _use_multi_char  = use_multi_character if use_multi_character is not None else USE_MULTI_CHARACTER
    _characters      = multi_characters  if multi_characters  is not None else CHARACTERS
    _use_cpu_offload = use_cpu_offload  if use_cpu_offload  is not None else USE_CPU_OFFLOAD
    _show_taesd_preview      = show_taesd_preview      if show_taesd_preview      is not None else SHOW_TAESD_PREVIEW
    _use_frame_interpolation = use_frame_interpolation if use_frame_interpolation is not None else USE_FRAME_INTERPOLATION
    _interpolation_factor    = interpolation_factor    if interpolation_factor    is not None else INTERPOLATION_FACTOR

    # ── IMPROVEMENT 23: SeedGenerator node support ────────────────────────
    if _use_seed_gen:
        try:
            if "SeedGenerator" in NODE_CLASS_MAPPINGS:
                sg  = NODE_CLASS_MAPPINGS["SeedGenerator"]()
                fn  = getattr(sg, sg.FUNCTION)
                seed = get_value_at_index(fn(), 0)
                print(f"   Seed from SeedGenerator node: {seed}")
            else:
                seed = random.randint(0, 2**32 - 1)
                print(f"   SeedGenerator not found -- random seed: {seed}")
        except Exception as e:
            seed = random.randint(0, 2**32 - 1)
            print(f"   SeedGenerator failed ({e}) -- random seed: {seed}")

    # Dynamic sigma schedule (IMPROVEMENT 9)
    if SIGMA_SCHEDULE != "manual" and not pro_mode:
        pass1_sigmas = make_sigmas(PASS1_STEPS, schedule=SIGMA_SCHEDULE)
        print(f"   Sigma schedule ({SIGMA_SCHEDULE}): {pass1_sigmas[:60]}...")
        validate_sigmas(pass1_sigmas)

    # Sigma schedule validation (IMPROVEMENT 8)
    if not pro_mode:
        if SIGMA_SCHEDULE == "manual":
            validate_sigmas(pass1_sigmas)
        validate_sigmas(pass2_sigmas)

    print("🎬 LTX-2 PRO V2 — Generation Starting")
    print(f"   Resolution   : {width}x{height}  |  Frames: {frames}  |  Seed: {seed}")
    print(f"   Mode         : {'I2V' if image_path else 'T2V'}"
          f"  |  Character: {_char_mode}  |  Pro: {pro_mode}")
    print(f"   Easy Prompt  : {'BYPASS' if _bypass else f'LLM={_llm_model}'}")
    print(f"   Real Neg     : {_use_real_neg}  |  Denoise: {_denoise}  |  CFG: {pass1_cfg}")
    _print_vram()

    # ── Pre-flight model check ─────────────────────────────────────────────
    print("\n🔍 Model file check...")
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

    seed_image_tensor = None
    if image_path:
        seed_image_tensor = load_image_tensor(image_path)
        if seed_image_tensor is None:
            print(f"   ⚠️  Image not found: {image_path} — switching to T2V")
        else:
            print(f"   ✓ Reference image loaded: {image_path}  {seed_image_tensor.shape}")

    char_image_tensor = None
    if character_image_path and _char_mode != "none":
        char_image_tensor = load_image_tensor(character_image_path)
        if char_image_tensor is None:
            print(f"   ⚠️  Character image not found: {character_image_path} — skipping anchor.")
        else:
            print(f"   ✓ Character image loaded: {character_image_path}  {char_image_tensor.shape}")

    # ── IMPROVEMENT 16: Multi-character support ────────────────────────────
    if _use_multi_char and _characters:
        primary = _characters[0]
        # Override character params from primary character entry
        character_image_path = primary.get("image_path", character_image_path)
        character_strength   = primary.get("strength",   character_strength)
        character_mode       = primary.get("mode",        character_mode)
        character_description = primary.get("description", character_description)
        _char_mode = character_mode.lower().strip()
        char_image_tensor = load_image_tensor(character_image_path) if character_image_path else None
        print(f"   [MultiChar] Primary: {primary.get('name','?')} @ {character_image_path}")
        # Additional characters: run vision describe and concatenate
        for extra in _characters[1:]:
            _extra_img = load_image_tensor(extra.get("image_path", ""))
            if _extra_img is not None:
                _extra_ctx = run_vision_describe(
                    _extra_img,
                    extra.get("description", ""),
                    use_vision_override=True,
                    vision_model_override=_vis_model)
                character_description = (character_description + " " + _extra_ctx).strip()
                print(f"   [MultiChar] Extra: {extra.get('name','?')} — ctx appended")

    analysis_tensor = char_image_tensor if char_image_tensor is not None else seed_image_tensor

    scene_context = character_description or ""
    if analysis_tensor is not None and _use_vision and not _bypass:
        print("\n👁️  Vision Describe...")
        scene_context = run_vision_describe(
            analysis_tensor,
            character_description,
            use_vision_override=_use_vision,
            vision_model_override=_vis_model)
        if scene_context:
            print(f"   Scene context: {scene_context[:120]}...")

    final_positive = positive_prompt
    final_negative = negative_prompt
    if not _bypass and user_input.strip():
        print("\n🧠 Easy Prompt expansion...")
        final_positive, final_negative = run_easy_prompt(
            user_input=user_input,
            frame_count=frames,
            seed=seed,
            scene_context=scene_context,
            llm_model_override=_llm_model,
        )
        print(f"\n   ── EXPANDED PROMPT ─────────────────────────────────────")
        print(f"   {final_positive[:300]}{'...' if len(final_positive) > 300 else ''}")
        print(f"\n   ── NEGATIVE PROMPT ─────────────────────────────────────")
        print(f"   {final_negative[:150]}...")
    else:
        print("   [EasyPrompt] Bypassed — using manual POSITIVE_PROMPT.")

    cleanup_memory(verbose=True)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — MODEL LOADING (with session cache — IMPROVEMENT 1)
    # ══════════════════════════════════════════════════════════════════════

    with torch.inference_mode():

        # ── UNet: UnetLoaderGGUF ──────────────────────────────────────────
        print("\n📦 Loading UNet (GGUF Q4_K_M distilled)...")
        _unet_from_cache = False
        if USE_MODEL_CACHE and _model_cache.unet_name == _unet and _model_cache.unet is not None:
            unet = _model_cache.unet
            _unet_from_cache = True
            print("   down cached (UNet)")
        else:
            try:
                unet = get_value_at_index(_node("UnetLoaderGGUF").load_unet(unet_name=_unet), 0)
                print("   down loading (UNet)")
            except KeyError:
                raise RuntimeError(
                    "UnetLoaderGGUF not found.\n"
                    "  Fix: Run Cell 1 to clone ComfyUI_GGUF custom node."
                )
            if USE_MODEL_CACHE and not _unet_from_cache:
                _model_cache.unet      = unet
                _model_cache.unet_name = _unet

        # ── DualCLIPLoader ────────────────────────────────────────────────
        print("   Loading CLIP encoders (DualCLIPLoader)...")
        _clip_key = (_clip1, _clip2)
        if USE_MODEL_CACHE and _model_cache.clip_names == _clip_key and _model_cache.clip is not None:
            clip_model = _model_cache.clip
            print("   down cached (CLIP)")
        else:
            try:
                clip_loader = _node("DualCLIPLoader")
                clip_model  = get_value_at_index(
                    clip_loader.load_clip(
                        clip_name1=_clip1, clip_name2=_clip2,
                        type="ltxv", device="default"), 0)
                print("   down loading (CLIP)")
            except Exception as e:
                print(f"   ⚠️  fp4 CLIP failed ({type(e).__name__}: {e})")
                print("      Trying fp8 fallback (gemma_3_12B_it_fp8_scaled.safetensors)...")
                fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
                try:
                    clip_model = get_value_at_index(
                        _node("DualCLIPLoader").load_clip(
                            clip_name1=fp8, clip_name2=_clip2,
                            type="ltxv", device="default"), 0)
                    print("   ✓ fp8 CLIP loaded.")
                except Exception as e2:
                    raise RuntimeError(
                        f"DualCLIPLoader failed: {e2}\n"
                        "  Fix: Ensure CLIP_NAME1 file is downloaded (Cell 2).\n"
                        "  fp4 needs Blackwell GPU; use fp8 for T4/A100."
                    )
            if USE_MODEL_CACHE:
                _model_cache.clip       = clip_model
                _model_cache.clip_names = _clip_key

        # ── LoRA stack: LTX2MasterLoaderLD ────────────────────────────────
        print("   Applying LoRA stack (LTX2MasterLoaderLD)...")
        unet, clip_model = apply_lora_stack(unet, clip_model, _lora_stack, _lora_json)

        # ── Optional performance patches ──────────────────────────────────
        unet = apply_sage_attention(unet)
        unet = apply_chunk_ff(unet)

        purge_vram("after unet+lora")
        _print_vram()

        # ── VAELoader [184] — video VAE ───────────────────────────────────
        print("   Loading VAEs...")
        if USE_MODEL_CACHE and _model_cache.vae_video_name == VAE_VIDEO_MODEL and _model_cache.vae_video is not None:
            vae_video = _model_cache.vae_video
            print("   down cached (VAE video)")
        else:
            vae_video = get_value_at_index(_node("VAELoader").load_vae(vae_name=VAE_VIDEO_MODEL), 0)
            print("   down loading (VAE video)")
            if USE_MODEL_CACHE:
                _model_cache.vae_video      = vae_video
                _model_cache.vae_video_name = VAE_VIDEO_MODEL

        if USE_MODEL_CACHE and _model_cache.vae_audio_name == VAE_AUDIO_MODEL and _model_cache.vae_audio is not None:
            vae_audio = _model_cache.vae_audio
            print("   down cached (VAE audio)")
        else:
            try:
                vae_audio = get_value_at_index(_load_audio_vae(VAE_AUDIO_MODEL), 0)
                print("   down loading (VAE audio)")
            except Exception as e:
                raise RuntimeError(f"Audio VAE load failed: {e}\n  Fix: Check VAE_AUDIO_MODEL filename in Cell 6.")
            if USE_MODEL_CACHE:
                _model_cache.vae_audio      = vae_audio
                _model_cache.vae_audio_name = VAE_AUDIO_MODEL

        # ── Spatial upscaler ──────────────────────────────────────────────
        if USE_MODEL_CACHE and _model_cache.upscale_model_name == UPSCALER_MODEL and _model_cache.upscale_model is not None:
            upscale_model = _model_cache.upscale_model
            print("   down cached (upscaler)")
        else:
            try:
                uml = _node("LatentUpscaleModelLoader")
                if hasattr(uml, "EXECUTE_NORMALIZED"):
                    upscale_model = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0)
                elif hasattr(uml, "load_model"):
                    upscale_model = get_value_at_index(uml.load_model(model_name=UPSCALER_MODEL), 0)
                else:
                    raise AttributeError("LatentUpscaleModelLoader: no load method found")
                print("   down loading (upscaler)")
            except Exception as e:
                raise RuntimeError(f"LatentUpscaleModelLoader failed: {e}\n  Fix: Download UPSCALER_MODEL in Cell 2.")
            if USE_MODEL_CACHE:
                _model_cache.upscale_model      = upscale_model
                _model_cache.upscale_model_name = UPSCALER_MODEL

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2 — TEXT ENCODING
        # ══════════════════════════════════════════════════════════════════

        print("\n📝 Encoding prompts...")
        # ── IMPROVEMENT 2: Real negative prompt encoding ──────────────────
        # When USE_REAL_NEGATIVE is True OR pass1_cfg > 1.0: encode the actual
        # negative text and use it in LTXVConditioning instead of ConditioningZeroOut.
        _use_real_neg_effective = _use_real_neg or (pass1_cfg > 1.0)
        try:
            cte      = _node("CLIPTextEncode")
            cond_pos = cte.encode(text=final_positive, clip=clip_model)
            cond_neg_raw = cte.encode(text=final_negative, clip=clip_model)

            if _use_real_neg_effective:
                # Use the real encoded negative (no zero-out)
                ltxv_cond = _node("LTXVConditioning")
                cond = ltxv_cond.EXECUTE_NORMALIZED(
                    frame_rate=float(fps),
                    positive=get_value_at_index(cond_pos, 0),
                    negative=get_value_at_index(cond_neg_raw, 0))
                print(f"   [TextEncode] Real negative encoding (USE_REAL_NEGATIVE={_use_real_neg_effective})")
            else:
                # Default: ConditioningZeroOut applied to positive for zero-out negative branch
                zero_out  = _node("ConditioningZeroOut")
                cond_zero = zero_out.zero_out(
                    conditioning=get_value_at_index(cond_pos, 0))
                ltxv_cond = _node("LTXVConditioning")
                cond = ltxv_cond.EXECUTE_NORMALIZED(
                    frame_rate=float(fps),
                    positive=get_value_at_index(cond_pos, 0),
                    negative=get_value_at_index(cond_zero, 0))

        except Exception as e:
            raise RuntimeError(
                f"Text encoding failed: {e}\n"
                "  Fix: Check DualCLIPLoader output — CLIP may have failed to load."
            )

        del clip_model
        cleanup_memory()

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3+4 — CHARACTER ANCHOR + LATENT PREPARATION
        # ══════════════════════════════════════════════════════════════════

        print("\n🗂️  Preparing latents...")
        _print_vram()

        _ref_tensor = seed_image_tensor
        if _ref_tensor is None and _char_mode in ("i2v", "both"):
            _ref_tensor = char_image_tensor
        _use_i2v = (_ref_tensor is not None and _char_mode in ("i2v", "both")) or \
                   (_ref_tensor is not None and image_path is not None and _char_mode == "none")

        # ── Compute half-resolution ────────────────────────────────────────
        ei       = _node("EmptyImage")
        full_img = ei.generate(width=width, height=height, batch_size=1, color=0)

        rimn     = _node("ResizeImageMaskNode")
        half_img = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_img, 0),
            scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})

        gis     = _node("GetImageSize")
        half_sz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_img, 0))
        half_w  = get_value_at_index(half_sz, 0)
        half_h  = get_value_at_index(half_sz, 1)
        print(f"   Latent dims : {half_w}x{half_h}  (half of {width}x{height})")

        # ── Character Anchor encoding ──────────────────────────────────────
        anchor_latent = None
        if char_image_tensor is not None and _char_mode in ("anchor", "both"):
            print("\n🧬 Character Anchor — encoding character image as latent...")
            try:
                if "ImageResizeKJv2" in NODE_CLASS_MAPPINGS:
                    ikj = _node("ImageResizeKJv2")
                    char_resized = get_value_at_index(
                        ikj.resize(
                            image=char_image_tensor,
                            width=half_w, height=half_h,
                            upscale_method="lanczos",
                            keep_proportion="crop",
                            pad_color="0, 0, 0",
                            crop_position="center",
                            divisible_by=32,
                            device="cpu",
                        ), 0)
                else:
                    char_resized = char_image_tensor

                vae_enc       = _node("VAEEncode")
                anchor_latent = get_value_at_index(
                    vae_enc.encode(pixels=char_resized, vae=vae_video), 0)
                print(f"   ✓ Character anchor encoded at {half_w}x{half_h}  (mode={_char_mode})")
            except Exception as e:
                print(f"   ⚠️  Character anchor failed ({e}) — continuing without anchor.")
                anchor_latent = None

        # ── EmptyLTXVLatentVideo ───────────────────────────────────────────
        eltxv   = _node("EmptyLTXVLatentVideo")
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1)

        # ── I2V conditioning branch ────────────────────────────────────────
        if _use_i2v and _ref_tensor is not None:
            try:
                if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                    rle     = _node("ResizeImagesByLongerEdge")
                    _ref_tensor = get_value_at_index(
                        rle.resize(images=_ref_tensor, longer_edge=1536), 0)

                if "ImageResizeKJv2" in NODE_CLASS_MAPPINGS:
                    ikj2 = _node("ImageResizeKJv2")
                    _ref_tensor = get_value_at_index(
                        ikj2.resize(
                            image=_ref_tensor,
                            width=half_w * 2, height=half_h * 2,
                            upscale_method="lanczos",
                            keep_proportion="crop",
                            pad_color="0, 0, 0",
                            crop_position="center",
                            divisible_by=2, device="cpu",
                        ), 0)
                else:
                    rim2 = _node("ResizeImageMaskNode")
                    _orig_h, _orig_w = _ref_tensor.shape[1], _ref_tensor.shape[2]
                    _scale = max((half_w * 2) / max(_orig_w, 1), (half_h * 2) / max(_orig_h, 1))
                    _ref_tensor = get_value_at_index(
                        rim2.EXECUTE_NORMALIZED(
                            input=_ref_tensor,
                            scale_method="lanczos",
                            resize_type={"resize_type": "scale by multiplier", "multiplier": _scale}), 0)

                pp_node = _node("LTXVPreprocess")
                pp_img  = get_value_at_index(
                    pp_node.EXECUTE_NORMALIZED(img_compression=33, image=_ref_tensor), 0)

                _i2v_strength = character_strength if _char_mode in ("i2v", "both") else image_strength
                i2v     = _node("LTXVImgToVideoInplace")
                vid_lat = i2v.EXECUTE_NORMALIZED(
                    strength=_i2v_strength, bypass=False,
                    vae=vae_video, image=pp_img,
                    latent=get_value_at_index(vid_lat, 0))
                print(f"   ✓ I2V conditioning applied  (strength={_i2v_strength}, LTXVImgToVideoInplace)")
            except KeyError as e:
                print(f"   ⚠️  I2V node missing ({e}) — using empty latent (T2V mode).")
                vid_lat = (get_value_at_index(vid_lat, 0),)
            except Exception as e:
                print(f"   ⚠️  I2V conditioning failed ({e}) — using empty latent.")
                vid_lat = (get_value_at_index(vid_lat, 0),)
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # ── Character anchor injection ─────────────────────────────────────
        _vid_lat_input = get_value_at_index(vid_lat, 0)
        if anchor_latent is not None:
            try:
                _vid_lat_input = anchor_latent
                print(f"   ✓ Character anchor injected as video latent seed  (mode={_char_mode})")
                _anch_shape = anchor_latent.get("samples", torch.empty(0)).shape
                print(f"     Anchor latent shape: {list(_anch_shape)}")
                if len(_anch_shape) == 4:
                    print("     ⚠️  Anchor is 4D — unsqueezing T dim for video latent.")
                    _s = anchor_latent["samples"].unsqueeze(2)
                    _vid_lat_input = {**anchor_latent, "samples": _s}
                    print(f"     Anchor latent shape after fix: {list(_s.shape)}")
            except Exception as e:
                print(f"   ⚠️  Anchor injection error ({e}) — using empty/I2V latent.")
                _vid_lat_input = get_value_at_index(vid_lat, 0)

        # ── Audio latent ──────────────────────────────────────────────────
        elalat  = _node("LTXVEmptyLatentAudio")
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_audio)

        catav           = _node("LTXVConcatAVLatent")
        av_lat1         = catav.EXECUTE_NORMALIZED(
            video_latent=_vid_lat_input,
            audio_latent=get_value_at_index(aud_lat, 0))
        combined_latent = get_value_at_index(av_lat1, 0)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 5 — SIGMA SCHEDULE
        # ══════════════════════════════════════════════════════════════════

        manualsigmas   = _node("ManualSigmas")
        ksamplerselect = _node("KSamplerSelect")
        randomnoise    = _node("RandomNoise")
        cfgguider      = _node("CFGGuider")
        sca            = _node("SamplerCustomAdvanced")

        sig_p1_high = None
        sig_p2_low  = None

        if pro_mode:
            print(f"\n⚙️  PRO sigma schedule — BasicScheduler steps={pro_steps} "
                  f"sched={pro_scheduler} split@{pro_split_at}")
            print("   ⚠️  EXPERIMENTAL: SVI-Pro sigma chain was designed for Wan2.2.")
            try:
                ms3  = _node("ModelSamplingSD3")
                unet_sampled = get_value_at_index(ms3.patch(model=unet, shift=8.0), 0)
                bs   = _node("BasicScheduler")
                sigs = get_value_at_index(
                    bs.get_sigmas(model=unet_sampled, scheduler=pro_scheduler,
                                  steps=pro_steps, denoise=1.0), 0)
                ss         = _node("SplitSigmas")
                split_out  = ss.get_sigmas(sigmas=sigs, step=pro_split_at)
                sig_p1_high = get_value_at_index(split_out, 0)
                sig_p2_low  = get_value_at_index(split_out, 1)
                unet = unet_sampled
                sampler_p1 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
                sampler_p2 = sampler_p1
                print(f"   ✓ PRO sigmas computed")
            except KeyError as e:
                print(f"   ⚠️  PRO mode node missing: {e} — falling back to ManualSigmas.")
                pro_mode = False
            except Exception as e:
                print(f"   ⚠️  PRO schedule failed ({e}) — falling back to ManualSigmas.")
                pro_mode = False

        if not pro_mode:
            # ── IMPROVEMENT 10: Denoise strength (img2img partial denoising) ─
            _p1_sigmas_str = pass1_sigmas
            if _denoise < 1.0:
                # img2img-style partial denoising: denoise<1.0 trims sigma schedule
                try:
                    _sig_vals = [float(v.strip()) for v in _p1_sigmas_str.split(",") if v.strip()]
                    _keep = max(2, round(len(_sig_vals) * _denoise))
                    _sig_vals = _sig_vals[:_keep]
                    if _sig_vals[-1] != 0.0:
                        _sig_vals.append(0.0)
                    _p1_sigmas_str = ", ".join(str(v) for v in _sig_vals)
                    print(f"   [Denoise={_denoise:.2f}] Trimmed pass1_sigmas to {len(_sig_vals)} steps: "
                          f"{_p1_sigmas_str[:60]}...")
                except Exception as e:
                    print(f"   ⚠️  Denoise trim failed ({e}) — using full sigma schedule.")

            print(f"\n⚙️  Standard sigma schedule — Pass1: {_p1_sigmas_str[:45]}...")
            sig_p1_high = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=_p1_sigmas_str), 0)
            sampler_p1  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)
            sig_p2_low  = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0)
            sampler_p2  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 6 — PASS 1 (first-pass denoising)
        # ══════════════════════════════════════════════════════════════════

        print(f"\n🚀 Pass 1 — denoising...")
        _print_vram()

        noise_p1  = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
        guider_p1 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1))

        try:
            out1  = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise_p1, 0),
                guider=get_value_at_index(guider_p1, 0),
                sampler=get_value_at_index(sampler_p1, 0),
                sigmas=sig_p1_high,
                latent_image=combined_latent)
            p1_av = get_value_at_index(out1, 0)
        except Exception as e:
            raise RuntimeError(
                f"Pass 1 sampling failed: {e}\n"
                "  Fix: If you see 'deformed output', try a different SEED."
            )

        del guider_p1
        cleanup_memory()
        print("   ✓ Pass 1 complete")

        # ── IMPROVEMENT 4: TAESD preview after Pass 1 ────────────────────
        if _show_taesd_preview:
            try:
                ltxvsep_prev = _node("LTXVSeparateAVLatent")
                s_prev       = ltxvsep_prev.EXECUTE_NORMALIZED(av_latent=p1_av)
                vid_lat_prev = get_value_at_index(s_prev, 0)

                # Try to find taeltx2_3.safetensors in the vae folder
                _taesd_name = "taeltx2_3.safetensors"
                _taesd_found = any(
                    os.path.exists(os.path.join(b, _taesd_name))
                    for b in folder_paths.get_folder_paths("vae")
                )
                if not _taesd_found:
                    print(f"   ⚠️  TAESD: {_taesd_name} not found in vae folder — skipping preview.")
                else:
                    taesd_vae = get_value_at_index(
                        _node("VAELoader").load_vae(vae_name=_taesd_name), 0)
                    # Extract first frame latent (5D: N,C,T,H,W)
                    _lat_samples = vid_lat_prev
                    if hasattr(_lat_samples, "get"):
                        _lat_s = _lat_samples.get("samples", _lat_samples)
                    else:
                        _lat_s = _lat_samples
                    if _lat_s.ndim == 5:
                        _first_frame_lat = {"samples": _lat_s[:, :, :1, :, :].squeeze(2)}
                    else:
                        _first_frame_lat = {"samples": _lat_s}
                    vaedec_prev = _node("VAEDecode")
                    _preview_frames = get_value_at_index(
                        vaedec_prev.decode(samples=_first_frame_lat, vae=taesd_vae), 0)
                    _pil_preview = tensor_to_pil(
                        _preview_frames[0] if _preview_frames.ndim == 4 else _preview_frames)
                    print("   [TAESD Preview] Pass 1 first-frame preview:")
                    import io as _io
                    _buf = _io.BytesIO()
                    _pil_preview.save(_buf, format="JPEG", quality=85)
                    display(IPImage(data=_buf.getvalue(), width=min(_pil_preview.width, 512)))
                    del taesd_vae
                    cleanup_memory()
            except Exception as e:
                print(f"   ⚠️  TAESD preview failed ({e}) — skipping.")

        # ── IMPROVEMENT 11: CPU offload after Pass 1 ─────────────────────
        if _use_cpu_offload:
            try:
                if hasattr(unet, 'model'):
                    unet.model.to('cpu')
                cleanup_memory()
                print("   CPU offload: UNet moved to CPU after Pass 1")
            except Exception as e:
                print(f"   CPU offload warning: {e}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 7 — PASS 2 (spatial upscale + refinement)
        # ══════════════════════════════════════════════════════════════════

        print(f"\n🔧 Pass 2 — upscale + refinement...")
        _print_vram()

        # ── IMPROVEMENT 11: Move UNet back to GPU before Pass 2 ──────────
        if _use_cpu_offload:
            try:
                if hasattr(unet, 'model'):
                    unet.model.to('cuda')
                print("   CPU offload: UNet moved back to GPU for Pass 2")
            except Exception:
                pass

        ltxvsep    = _node("LTXVSeparateAVLatent")
        s1         = ltxvsep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vid_lat_p1 = get_value_at_index(s1, 0)
        aud_lat_p1 = get_value_at_index(s1, 1)

        ltxvcrop = _node("LTXVCropGuides")
        cropped  = ltxvcrop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vid_lat_p1)

        guider_p2 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg, model=unet,
            positive=get_value_at_index(cropped, 0),
            negative=get_value_at_index(cropped, 1))

        ltxvup    = _node("LTXVLatentUpsampler")
        upsampled = ltxvup.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model,
            vae=vae_video)
        del upscale_model
        cleanup_memory()

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
            p2_denoised = get_value_at_index(out2, 1)
        except Exception as e:
            raise RuntimeError(
                f"Pass 2 sampling failed: {e}\n"
                "  Fix: Try reducing TILED_SPATIAL_TILES or USE_TILED_VAE=False."
            )

        del guider_p2, unet
        cleanup_memory()
        print("   ✓ Pass 2 complete")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 8 — DECODE
        # ══════════════════════════════════════════════════════════════════

        print("\n🎞️  Decoding video & audio...")
        _print_vram()

        s2          = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        decoded_frames = None
        if use_tiled_vae:
            try:
                tiled_dec = _node("LTXVSpatioTemporalTiledVAEDecode")
                decoded_frames = get_value_at_index(
                    tiled_dec.EXECUTE_NORMALIZED(
                        vae=vae_video, latents=vid_lat_fin,
                        spatial_tiles=tiled_spatial_tiles,
                        spatial_overlap=tiled_spatial_overlap,
                        temporal_tile_length=tiled_temporal_len,
                        temporal_overlap=tiled_temporal_overlap,
                        last_frame_fix=tiled_last_frame_fix,
                        working_device="auto", working_dtype="auto"), 0)
                print("   ✓ Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode)")
            except (KeyError, Exception) as e:
                print(f"   ⚠️  Tiled VAE unavailable ({type(e).__name__}: {e})")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_frames is None:
            vaedecode = _node("VAEDecode")
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_video), 0)
            print("   ✓ Standard VAE decode (VAEDecode)")

        del vae_video
        cleanup_memory()

        try:
            aud_dec   = _node("LTXVAudioVAEDecode")
            audio_out = aud_dec.EXECUTE_NORMALIZED(samples=aud_lat_fin, audio_vae=vae_audio)
        except Exception as e:
            print(f"   ⚠️  Audio decode failed ({e}) — proceeding without audio.")
            audio_out = None

        del vae_audio
        cleanup_memory()

        # ══════════════════════════════════════════════════════════════════
        # PHASE 9 — SAVE
        # ══════════════════════════════════════════════════════════════════

        print("\n💾 Saving video...")
        _print_vram()
        output_path = None

        _prefix = output_prefix
        if character_name and character_name != "Character":
            _prefix = f"{output_prefix}-{character_name}"

        # ── Try VHS_VideoCombine first ────────────────────────────────────
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS and audio_out is not None:
            try:
                vhs  = _node("VHS_VideoCombine")
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
                _fnames = get_value_at_index(vhs_out, 0)
                if hasattr(_fnames, "video_paths") and _fnames.video_paths:
                    output_path = _fnames.video_paths[0]
                elif hasattr(_fnames, "video_path"):
                    output_path = _fnames.video_path
                elif isinstance(_fnames, (list, tuple)) and len(_fnames) > 0:
                    output_path = _fnames[0]
                else:
                    print(f"   ⚠️  VHS_FILENAMES type={type(_fnames)} — cannot extract path.")
                    output_path = None
                # IMPROVEMENT 15: VHS path coercion for pathlib.Path objects
                if output_path and not isinstance(output_path, str):
                    output_path = str(output_path)
                if output_path:
                    print(f"   ✓ Saved via VHS_VideoCombine (h264-mp4, crf=19, yuv420p)")
            except Exception as e:
                print(f"   ⚠️  VHS_VideoCombine failed ({e}) — using CreateVideo fallback.")
                output_path = None

        # ── Fallback: CreateVideo ─────────────────────────────────────────
        if output_path is None:
            try:
                createvideo = _node("CreateVideo")
                _aud_arg    = get_value_at_index(audio_out, 0) if audio_out else None
                if _aud_arg is not None:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(fps=fps, images=decoded_frames, audio=_aud_arg)
                else:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(fps=fps, images=decoded_frames)
                output_path = save_video_from_components(
                    get_value_at_index(vid_obj, 0), prefix=_prefix)
                print(f"   ✓ Saved via CreateVideo fallback")
            except Exception as e:
                raise RuntimeError(
                    f"Video save failed: {e}\n"
                    "  Fix: Check /content/ComfyUI/output/ permissions."
                )

    # ── IMPROVEMENT 19: Frame interpolation ──────────────────────────────
    if _use_frame_interpolation and output_path and os.path.exists(output_path):
        try:
            effective_fps = fps * _interpolation_factor
            _base, _ext = os.path.splitext(output_path)
            interp_path = f"{_base}_interp{effective_fps}fps{_ext if _ext else '.mp4'}"
            cmd = ["ffmpeg", "-y", "-i", output_path, "-vf",
                   f"minterpolate=fps={effective_fps}:mi_mode=mci",
                   interp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output_path = interp_path
                print(f"   Frame interpolation: {fps} -> {effective_fps} fps")
            else:
                print(f"   Frame interpolation failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"   Frame interpolation error: {e}")

    # ── Timing ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

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
        "denoise"           : _denoise,
        "use_real_negative" : _use_real_neg_effective,
    }
    if output_path:
        save_metadata_sidecar(output_path, meta)

    # ── IMPROVEMENT 13: Append generation log ────────────────────────────
    append_generation_log(
        seed=seed, width=width, height=height, frames=frames,
        elapsed_s=elapsed, loras=_active_loras,
        positive_prompt=final_positive, output_path=output_path)

    # ── IMPROVEMENT 14: Google Drive auto-save ────────────────────────────
    if SAVE_TO_DRIVE and output_path:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            drive_path = f"/content/drive/MyDrive/{DRIVE_FOLDER}/"
            os.makedirs(drive_path, exist_ok=True)
            shutil.copy2(output_path, drive_path)
            print(f"   Drive: copied to {drive_path}")
        except Exception as e:
            print(f"   Drive save failed: {e}")

    print(f"\n✅ Done in {mins}m {secs}s")
    print(f"   📁 {output_path}")
    _print_vram()

    if SHOW_PREVIEWS and output_path:
        print("\n▶ Preview:")
        display_video(output_path)

    if DOWNLOAD_AFTER_GENERATE and output_path:
        print("   ⬇️  Auto-downloading...")
        try:
            files.download(output_path)
        except Exception as e:
            print(f"   ⚠️  Download failed ({e}) — file is at {output_path}")

    return output_path


print("✅ generate_pro() defined — run Cell 9 to generate.")
print("   Signature: generate_pro(user_input, image_path, width, height, frames, ...)")
print("   V2 new params: use_real_negative, denoise, use_seed_generator,")
print("                  use_multi_character, multi_characters")


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
        "image_path"    : CHARACTER_IMAGE_PATH,
        "frames"        : 97,
        "seed"          : SEED,
        "output_prefix" : f"Story01-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she sits at a window table, wraps her hands around a coffee cup, "
                          "gazes out at the rain-streaked street",
        "image_path"    : None,
        "frames"        : 121,
        "seed"          : SEED + 1,
        "output_prefix" : f"Story02-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she notices something outside and leans forward, face half lit by neon glow",
        "image_path"    : None,
        "frames"        : 97,
        "seed"          : SEED + 2,
        "output_prefix" : f"Story03-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
]

USE_STORYBOARD = False  # @param {type:"boolean"}


def run_storyboard(
    scenes:          List[Dict],
    use_continuity:  bool = USE_SCENE_CONTINUITY,
    tmp_dir:         str  = "/content/ComfyUI/input",
) -> List[Optional[str]]:
    """
    Run a list of scenes sequentially, optionally chaining last-frame continuity.
    Returns list of output paths (None for failed scenes).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    outputs    = []
    prev_output = None

    print("🎬 Storyboard Runner — Starting")
    print(f"   Scenes    : {len(scenes)}")
    print(f"   Continuity: {use_continuity}")
    print("-" * 70)

    for i, scene in enumerate(scenes):
        scene_num = i + 1
        print(f"\n🎬 Scene {scene_num}/{len(scenes)}: {scene.get('output_prefix','Scene')}")
        print(f"   Input: {scene.get('user_input','')[:80]}...")

        _image_path = scene.get("image_path")
        _scene_char_mode = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE)
        _scene_char_img  = scene.get("character_image_path", CHARACTER_IMAGE_PATH)

        if use_continuity and prev_output and _image_path is None:
            # ── IMPROVEMENT 22: Storyboard continuity conflict resolution note ─
            if _scene_char_img and "i2v" in str(_scene_char_mode).lower():
                print(f"   Note: continuity frame -> image_path (scene reference), "
                      f"character image -> character anchor. They do NOT conflict.")
            print(f"   Continuity: extracting last frame from scene {scene_num - 1}...")
            last_tensor = get_last_frame_tensor(prev_output)
            if last_tensor is not None:
                _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.jpg")
                pil_frame = tensor_to_pil(last_tensor)
                pil_frame.save(_cont_path, "JPEG", quality=95)
                _image_path = _cont_path
                print(f"   ✓ Continuity frame saved: {_cont_path}")
            else:
                print(f"   ⚠️  Could not extract last frame — skipping continuity.")

        try:
            out = generate_pro(
                user_input           = scene.get("user_input", USER_INPUT),
                image_path           = _image_path,
                positive_prompt      = scene.get("positive_prompt", POSITIVE_PROMPT),
                negative_prompt      = scene.get("negative_prompt", NEGATIVE_PROMPT),
                width                = scene.get("width", WIDTH),
                height               = scene.get("height", HEIGHT),
                frames               = scene.get("frames", FRAMES),
                fps                  = scene.get("fps", FPS),
                seed                 = scene.get("seed", SEED),
                image_strength       = scene.get("image_strength", IMAGE_STRENGTH),
                character_image_path = _scene_char_img,
                character_strength   = scene.get("character_strength", CHARACTER_STRENGTH),
                character_mode       = _scene_char_mode,
                character_name       = scene.get("character_name", CHARACTER_NAME),
                character_description= scene.get("character_description", CHARACTER_DESCRIPTION),
                output_prefix        = scene.get("output_prefix", OUTPUT_PREFIX),
                use_real_negative    = scene.get("use_real_negative",    USE_REAL_NEGATIVE),
                denoise              = scene.get("denoise",              DENOISE),
                use_seed_generator   = scene.get("use_seed_generator",   USE_SEED_GENERATOR),
                use_multi_character  = scene.get("use_multi_character",  USE_MULTI_CHARACTER),
                multi_characters     = scene.get("multi_characters",     CHARACTERS),
            )
            outputs.append(out)
            prev_output = out
            print(f"   ✅ Scene {scene_num} done -> {out}")
        except Exception as e:
            import traceback
            print(f"   ❌ Scene {scene_num} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            outputs.append(None)
            prev_output = None

    print("\n" + "=" * 70)
    print("🎬 Storyboard Complete")
    print(f"   Total scenes : {len(scenes)}")
    print(f"   Successful   : {sum(1 for p in outputs if p)}")
    print(f"   Failed       : {sum(1 for p in outputs if not p)}")
    print("\n   Output paths:")
    for i, p in enumerate(outputs):
        status = "✅" if p else "❌"
        print(f"   {status} Scene {i+1}: {p or 'FAILED'}")
    print("=" * 70)
    return outputs


# ── IMPROVEMENT 12: Batch generation function ─────────────────────────────────
def generate_batch(prompts: List[str], seeds: Optional[List[int]] = None,
                   **kwargs) -> List[Optional[str]]:
    """
    Generate multiple videos from a list of prompts.
    Seeds are auto-generated from SEED global if not provided.
    Returns a list of output paths (None for failed items).
    """
    if seeds is None:
        _base = globals().get('SEED', random.randint(0, 2**32 - 1))
        seeds = [_base + i for i in range(len(prompts))]

    outputs: List[Optional[str]] = []
    print(f"🎬 Batch Generation — {len(prompts)} prompts")
    print("-" * 70)
    for i, (prompt, s) in enumerate(zip(prompts, seeds)):
        print(f"\n[{i+1}/{len(prompts)}] seed={s}  prompt={prompt[:60]}...")
        try:
            out = generate_pro(user_input=prompt, seed=s, **kwargs)
            outputs.append(out)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            outputs.append(None)

    print("\n" + "=" * 70)
    print("🎬 Batch Summary")
    print(f"   {'#':<4} {'Seed':<12} {'Prompt':<62} {'Result'}")
    print("-" * 70)
    for i, (prompt, s, out) in enumerate(zip(prompts, seeds, outputs)):
        result = os.path.basename(out) if out else "FAILED"
        print(f"   {i+1:<4} {s:<12} {prompt[:60]:<62} {result}")
    print("=" * 70)
    return outputs


# ── IMPROVEMENT 20: Persistent prompt history display ────────────────────────
def show_generation_history(n: int = 10) -> None:
    """
    Display the last n rows of the generation log CSV as an HTML table.
    Falls back to plain text if pandas is unavailable.
    """
    if not os.path.exists(LOG_PATH):
        print(f"   No generation log found at {LOG_PATH}. Run some generations first.")
        return
    try:
        import pandas as pd
        df = pd.read_csv(LOG_PATH)
        display(HTML(df.tail(n).to_html(index=False, border=1)))
    except ImportError:
        # Fallback to basic CSV -> HTML rendering
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows   = list(reader)
            if len(rows) < 2:
                print("   Log is empty.")
                return
            header  = rows[0]
            display_rows = rows[max(1, len(rows) - n):]
            html  = "<table border='1'><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
            for row in display_rows:
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            html += "</table>"
            display(HTML(html))
        except Exception as e:
            print(f"   Could not display history: {e}")


print("✅ Storyboard runner ready.")
print("   Edit SCENES list above, then set USE_STORYBOARD=True in Cell 9.")
print("   V2 additions:")
print("   ✓ generate_batch()           — multi-prompt batch generation")
print("   ✓ show_generation_history()  — display CSV log as HTML table")


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

# ── IMPROVEMENT 7: OOM retry logic ───────────────────────────────────────────
MAX_OOM_RETRIES = 2
_retry_frames   = FRAMES

try:
    if USE_STORYBOARD:
        # ── Multi-scene storyboard mode ───────────────────────────────────
        print("🎬 Running storyboard mode...")
        storyboard_outputs = run_storyboard(
            scenes=SCENES,
            use_continuity=USE_SCENE_CONTINUITY,
        )
        output = storyboard_outputs[-1] if storyboard_outputs else None
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + len(SCENES)
            print(f"🔢 Next seed: {SEED}")

    else:
        # ── Single-clip mode with OOM retry ──────────────────────────────
        for _oom_attempt in range(MAX_OOM_RETRIES + 1):
            try:
                output = generate_pro(
                    user_input             = USER_INPUT,
                    image_path             = IMAGE_PATH,
                    positive_prompt        = POSITIVE_PROMPT,
                    negative_prompt        = NEGATIVE_PROMPT,
                    width                  = WIDTH,
                    height                 = HEIGHT,
                    frames                 = _retry_frames,
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
                    use_real_negative      = USE_REAL_NEGATIVE,
                    denoise                = DENOISE,
                    use_seed_generator     = USE_SEED_GENERATOR,
                    use_multi_character    = USE_MULTI_CHARACTER,
                    multi_characters       = CHARACTERS,
                )
                break  # success
            except torch.cuda.OutOfMemoryError:
                cleanup_memory()
                _retry_frames = max(49, _retry_frames - 24)
                if _oom_attempt < MAX_OOM_RETRIES:
                    print(f"OOM -- retrying with {_retry_frames} frames "
                          f"(attempt {_oom_attempt + 1})")
                else:
                    raise

        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + 1
            print(f"🔢 Next seed: {SEED}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — partial output may be in /content/ComfyUI/output/")

except FileNotFoundError as e:
    print(f"\n❌ Missing models: {e}")
    print("   Run Cell 2 to download, then retry Cell 9.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n❌ CUDA Out of Memory (all OOM retries exhausted)")
    print(f"   Current settings: {WIDTH}x{HEIGHT}, {FRAMES} frames")
    print("   ── Suggested fixes ─────────────────────────────────────────────")
    print("   T4  (15 GB): WIDTH=768,  HEIGHT=512,  FRAMES=97")
    print("   L4  (24 GB): WIDTH=1024, HEIGHT=576,  FRAMES=161")
    print("   A100(40 GB): WIDTH=1280, HEIGHT=720,  FRAMES=241")
    print("   ── Also try ────────────────────────────────────────────────────")
    print("   • USE_CPU_OFFLOAD=True   in Cell 5  (T4 >161 frames)")
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
    print("   'UnetLoaderGGUF' not found       -> Cell 1: clone ComfyUI_GGUF")
    print("   'LTX2PromptArchitect' not found  -> Cell 1: clone LTX2EasyPrompt-LD")
    print("   'LTX2MasterLoaderLD' not found   -> Cell 1: clone LTX2-Master-Loader")
    print("   'LTXVImgToVideoInplace' missing  -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVPreprocess' missing         -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVCropGuides' missing         -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'PathchSageAttentionKJ' missing  -> Cell 1: clone ComfyUI_KJNodes")
    print("   'LTXVChunkFeedForward' missing   -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'VHS_VideoCombine' missing       -> Cell 1: clone ComfyUI-VideoHelperSuite")
    print("   'ModelSamplingSD3' missing       -> set PRO_MODE=False in Cell 5")
    print("   'BasicScheduler' missing         -> set PRO_MODE=False in Cell 5")
    print("   DualCLIPLoader fp4 error         -> swap CLIP_NAME1 to fp8 in Cell 6")
    print("   Tiled VAE error                  -> set USE_TILED_VAE=False in Cell 6")
    print("   Deformed output in Pass 1        -> change SEED and re-run Cell 9")
    print("   Persistent deformation (3x same) -> change USER_INPUT / POSITIVE_PROMPT")
    print("   Character drift                  -> try CHARACTER_CONSISTENCY_MODE='both'")
    print("                                       or increase CHARACTER_STRENGTH")
