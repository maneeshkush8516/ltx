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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()   # free IPC handles — reduces fragmentation
    if verbose:
        _print_vram()

def _print_vram() -> None:
    if not torch.cuda.is_available():
        return
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct   = used / total * 100 if total > 0 else 0
    filled = int(20 * used / total) if total > 0 else 0
    bar   = "█" * filled + "░" * (20 - filled)
    print(f"   💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%)")

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
        unet  = get_value_at_index(fn(model=unet), 0)
        print("   ✓ SageAttention patch applied (PathchSageAttentionKJ)")
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
            fn()
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
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
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


print("✅ Imports & helpers ready.")
print("   Helper functions defined:")
print("   ✓ cleanup_memory()       — with ipc_collect()")
print("   ✓ apply_sage_attention() — PathchSageAttentionKJ wrapper")
print("   ✓ apply_chunk_ff()       — LTXVChunkFeedForward wrapper")
print("   ✓ purge_vram()           — LayerUtility: PurgeVRAM V2 wrapper")
print("   ✓ apply_lora_stack()     — LTX2MasterLoaderLD + manual fallback")
print("   ✓ run_easy_prompt()      — LTX2PromptArchitect wrapper")
print("   ✓ run_vision_describe()  — LTX2VisionDescribe wrapper")
print("   ✓ save_metadata_sidecar() — JSON sidecar writer")


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

LORA_TRIGGERS = ""          # @param {type:"string"}
# LoRA trigger words injected at the start of every expanded prompt.
# e.g. "ohwx woman" or "film grain, 35mm"

# ── Vision image describer (LTX2VisionDescribe node) ─────────────────────────
USE_VISION   = True          # @param {type:"boolean"}
# When True AND an image is provided, Vision Describe analyses it and passes
# the result as scene_context to Easy Prompt. Adds ~30-90s on first run.

VISION_MODEL = "3B-fast"     # @param ["3B-fast", "7B-nsfw"]
# "3B-fast" → Qwen2.5-VL-3B — faster, ~5 GB VRAM
# "7B-nsfw" → Qwen2.5-VL-7B — more accurate, ~10 GB VRAM

# ── Display & output ──────────────────────────────────────────────────────────
SHOW_PREVIEWS           = True   # @param {type:"boolean"}
# Display each video inline after generation.

DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}
# Auto-call files.download(output) after each generation.
# Useful for immediately saving clips to your local machine.

print("✅ Easy Prompt + Vision settings ready.")
print(f"   LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}  |  "
      f"Creativity: {CREATIVITY}  |  Bypass: {BYPASS_EASY_PROMPT}")
print(f"   Show previews: {SHOW_PREVIEWS}  |  Auto-download: {DOWNLOAD_AFTER_GENERATE}")


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

CHARACTER_CONSISTENCY_MODE = "i2v"  # @param ["i2v", "anchor", "both", "none"]
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
# When True in run_storyboard(), the last frame of scene N becomes
# the image_path seed for scene N+1 (seamless continuity).

print("✅ Configuration set.")
print(f"   Resolution : {WIDTH}×{HEIGHT}  |  Frames : {FRAMES}  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   UNet       : {UNET_MODEL}")
print(f"   Seed       : {SEED}  (auto-increment: {AUTO_INCREMENT_SEED})")
print(f"   Pass 1     : {PASS1_SAMPLER}  |  {PASS1_SIGMAS[:45]}…")
print(f"   Pass 2     : {PASS2_SAMPLER}  |  {PASS2_SIGMAS}")
print(f"   Pro Mode   : {PRO_MODE}  |  steps={PRO_STEPS}, scheduler={PRO_SCHEDULER}, split@{PRO_SPLIT_AT}")


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

    print("🎬 LTX-2 PRO — Generation Starting")
    print(f"   Resolution   : {width}×{height}  |  Frames: {frames}  |  Seed: {seed}")
    print(f"   Mode         : {'I2V' if image_path else 'T2V'}"
          f"  |  Character: {_char_mode}  |  Pro: {pro_mode}")
    print(f"   Easy Prompt  : {'BYPASS' if _bypass else f'LLM={_llm_model}'}")
    _print_vram()

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
    # PHASE 1 — MODEL LOADING
    # ══════════════════════════════════════════════════════════════════════

    with torch.inference_mode():

        # ── UNet: UnetLoaderGGUF ──────────────────────────────────────────
        # [197] UnetLoaderGGUF in LD-I2V.json — loads GGUF Q4_K_M distilled
        print("\n📦 Loading UNet (GGUF Q4_K_M distilled)…")
        try:
            unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unet        = get_value_at_index(
                unet_loader.load_unet(unet_name=_unet), 0)
        except KeyError:
            raise RuntimeError(
                "UnetLoaderGGUF not found.\n"
                "  Fix: Run Cell 1 to clone ComfyUI_GGUF custom node."
            )

        # ── DualCLIPLoader ────────────────────────────────────────────────
        # [190] DualCLIPLoader — Gemma text encoder + embeddings connector
        print("   Loading CLIP encoders (DualCLIPLoader)…")
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
            print("      Trying fp8 fallback (gemma_3_12B_it_fp8_scaled.safetensors)…")
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

        # ── LoRA stack: LTX2MasterLoaderLD [263] ─────────────────────────
        # [263] LTX2MasterLoaderLD in LD-I2V.json — 10-slot LoRA stacker
        print("   Applying LoRA stack (LTX2MasterLoaderLD)…")
        unet, clip_model = apply_lora_stack(unet, clip_model, _lora_stack, _lora_json)

        # ── Optional performance patches ──────────────────────────────────
        # [PathchSageAttentionKJ] — KJNodes flash-attention-style patch
        unet = apply_sage_attention(unet)
        # [LTXVChunkFeedForward] — ComfyUI-LTXVideo chunk feedforward
        unet = apply_chunk_ff(unet)

        # Purge VRAM after model loading if enabled
        # [LayerUtility: PurgeVRAM V2] from LayerStyle nodes
        purge_vram("after unet+lora")
        _print_vram()

        # ── VAELoader [184] — video VAE ───────────────────────────────────
        # [184] VAELoader in LD-I2V.json
        print("   Loading VAEs…")
        vaeloader  = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_video  = get_value_at_index(
            vaeloader.load_vae(vae_name=VAE_VIDEO_MODEL), 0)

        # [196] VAELoaderKJ (or VAELoader fallback) — audio VAE
        try:
            vae_audio = get_value_at_index(_load_audio_vae(VAE_AUDIO_MODEL), 0)
        except Exception as e:
            raise RuntimeError(
                f"Audio VAE load failed: {e}\n"
                "  Fix: Check VAE_AUDIO_MODEL filename in Cell 6."
            )

        # ── Spatial upscaler [189] ────────────────────────────────────────
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
            raise RuntimeError(
                f"LatentUpscaleModelLoader failed: {e}\n"
                "  Fix: Download UPSCALER_MODEL in Cell 2."
            )

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2 — TEXT ENCODING
        # ══════════════════════════════════════════════════════════════════

        print("\n📝 Encoding prompts…")
        try:
            # [121] CLIPTextEncode — positive
            cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cond_pos = cte.encode(text=final_positive, clip=clip_model)

            # [110] CLIPTextEncode — negative (empty for LTX, wrapped in ConditioningZeroOut)
            cond_neg = cte.encode(text=final_negative, clip=clip_model)

            # ConditioningZeroOut — applied to positive to create zero-out negative branch
            # (mirrors reference notebook pattern for distilled model)
            zero_out  = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_zero = zero_out.zero_out(
                conditioning=get_value_at_index(cond_pos, 0))

            # [107] LTXVConditioning — injects frame_rate into conditioning metadata
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
                "  Fix: Check DualCLIPLoader output — CLIP may have failed to load."
            )

        del clip_model
        cleanup_memory()

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

        # Determine input image for latent prep (I2V or T2V)
        # For I2V: use image_path tensor. For anchor-only: use character image.
        _ref_tensor = seed_image_tensor
        if _ref_tensor is None and _char_mode in ("i2v", "both"):
            _ref_tensor = char_image_tensor
        _use_i2v = (_ref_tensor is not None and _char_mode in ("i2v", "both")) or \
                   (_ref_tensor is not None and image_path is not None and _char_mode == "none")

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
        # [295] VAEEncode from SVI-Pro-Workflow.json — encodes character image.
        # We resize to half_w × half_h so the anchor latent matches EmptyLTXVLatentVideo.
        anchor_latent = None
        if char_image_tensor is not None and _char_mode in ("anchor", "both"):
            print("\n🧬 Character Anchor — encoding character image as latent…")
            try:
                # [165] ImageResizeKJv2 — resize to HALF resolution (matches vid_lat)
                # half_w × half_h ensures spatial dims match EmptyLTXVLatentVideo
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

                # [295] VAEEncode — pixels at half_w × half_h → latent at ~(half_w/8 × half_h/8)
                # This matches the spatial dims of EmptyLTXVLatentVideo(half_w, half_h)
                vae_enc       = NODE_CLASS_MAPPINGS["VAEEncode"]()
                anchor_latent = get_value_at_index(
                    vae_enc.encode(pixels=char_resized, vae=vae_video), 0)
                print(f"   ✓ Character anchor encoded at {half_w}×{half_h}  (mode={_char_mode})")
            except Exception as e:
                print(f"   ⚠️  Character anchor failed ({e}) — continuing without anchor.")
                anchor_latent = None

        # [108] EmptyLTXVLatentVideo — half-res video latent
        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1)

        # ── I2V conditioning branch ────────────────────────────────────────
        if _use_i2v and _ref_tensor is not None:
            try:
                # Resize reference image to full target resolution first
                # [246] ResizeImagesByLongerEdge — longer_edge=1536
                if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                    rle     = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
                    _ref_tensor = get_value_at_index(
                        rle.resize(images=_ref_tensor, longer_edge=1536), 0)

                # [165] ImageResizeKJv2 — precise resize to half_w*2 × half_h*2
                if "ImageResizeKJv2" in NODE_CLASS_MAPPINGS:
                    ikj2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
                    _ref_tensor = get_value_at_index(
                        ikj2.resize(
                            image=_ref_tensor,
                            width=half_w * 2,
                            height=half_h * 2,
                            upscale_method="lanczos",
                            keep_proportion="crop",
                            pad_color="0, 0, 0",
                            crop_position="center",
                            divisible_by=2,
                            device="cpu",
                        ), 0)
                else:
                    # Fallback: ResizeImageMaskNode — use scale by multiplier to
                    # approximate the target size. Note: "scale dimensions" mode is
                    # not a valid resize_type for this node; only "scale by multiplier"
                    # and "scale to fit" are documented in LD-I2V.json widgets_values.
                    # We compute a representative scale factor for the longer edge.
                    rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
                    _orig_h, _orig_w = _ref_tensor.shape[1], _ref_tensor.shape[2]
                    _scale = max((half_w * 2) / max(_orig_w, 1),
                                 (half_h * 2) / max(_orig_h, 1))
                    _ref_tensor = get_value_at_index(
                        rim2.EXECUTE_NORMALIZED(
                            input=_ref_tensor,
                            scale_method="lanczos",
                            resize_type={"resize_type": "scale by multiplier",
                                         "multiplier": _scale}), 0)

                # [162] LTXVPreprocess — compress/normalise image before I2V injection
                # img_compression=33 matches LD-I2V.json node [162] widgets_values
                pp_node = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
                pp_img  = get_value_at_index(
                    pp_node.EXECUTE_NORMALIZED(
                        img_compression=33,
                        image=_ref_tensor), 0)

                # [161] LTXVImgToVideoInplace — inject image into video latent
                # strength = character_strength (if char mode) else image_strength
                _i2v_strength = character_strength if _char_mode in ("i2v", "both") \
                                else image_strength
                i2v     = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
                vid_lat = i2v.EXECUTE_NORMALIZED(
                    strength=_i2v_strength,
                    bypass=False,
                    vae=vae_video,
                    image=pp_img,
                    latent=get_value_at_index(vid_lat, 0))
                print(f"   ✓ I2V conditioning applied  (strength={_i2v_strength}, "
                      f"LTXVImgToVideoInplace)")
            except KeyError as e:
                print(f"   ⚠️  I2V node missing ({e}) — using empty latent (T2V mode).")
                vid_lat = (get_value_at_index(vid_lat, 0),)
            except Exception as e:
                print(f"   ⚠️  I2V conditioning failed ({e}) — using empty latent.")
                vid_lat = (get_value_at_index(vid_lat, 0),)
        else:
            # T2V — use empty latent directly
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
                # Use anchor_latent as the starting video latent for Pass 1.
                # This biases the denoising toward the character's appearance.
                # (The anchor replaces the empty latent — it is a valid LATENT dict.)
                _vid_lat_input = anchor_latent
                print(f"   ✓ Character anchor injected as video latent seed  (mode={_char_mode})")
                print(f"     Note: LTX-2 has no WanImageToVideoSVIPro equivalent;")
                print(f"     anchor is used as the initial latent for Pass 1 denoising.")

                # Diagnostic: check tensor rank (LTX video VAE should produce T dim)
                _anch_shape = anchor_latent.get("samples", torch.empty(0)).shape
                print(f"     Anchor latent shape: {list(_anch_shape)}")
                if len(_anch_shape) == 4:
                    # Standard VAEEncode returns 4D (N, C, H, W); LTX needs 5D (N, C, T, H, W)
                    # Unsqueeze temporal dimension (T=1) to make it a single-frame video latent
                    print("     ⚠️  Anchor is 4D (image latent) — unsqueezing T dim for video latent.")
                    _s = anchor_latent["samples"].unsqueeze(2)   # → (N, C, 1, H, W)
                    _vid_lat_input = {**anchor_latent, "samples": _s}
                    print(f"     Anchor latent shape after fix: {list(_s.shape)}")
            except Exception as e:
                print(f"   ⚠️  Anchor injection error ({e}) — using empty/I2V latent.")
                _vid_lat_input = get_value_at_index(vid_lat, 0)

        # [199] LTXVEmptyLatentAudio — audio latent
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
            raise RuntimeError(
                f"Pass 1 sampling failed: {e}\n"
                "  Fix: If you see 'deformed output', try a different SEED.\n"
                "  Three consecutive deformations → change USER_INPUT/POSITIVE_PROMPT."
            )

        del guider_p1
        cleanup_memory()
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

        # [LTXVLatentUpsampler] [118] — 2× spatial upsample
        ltxvup    = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = ltxvup.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model,
            vae=vae_video)
        del upscale_model
        cleanup_memory()

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

        print("\n🎞️  Decoding video & audio…")
        _print_vram()

        # [LTXVSeparateAVLatent] [125] — final split
        s2          = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        # ── Video decode ───────────────────────────────────────────────────
        decoded_frames = None
        if use_tiled_vae:
            # [265] LTXVSpatioTemporalTiledVAEDecode — ComfyUI-LTXVideo
            # VRAM-efficient tiled spatiotemporal decode
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
                        working_dtype="auto"), 0)
                print("   ✓ Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode)")
            except (KeyError, Exception) as e:
                print(f"   ⚠️  Tiled VAE unavailable ({type(e).__name__}: {e})")
                print("      Falling back to standard VAEDecode.")
                print("      Fix: Clone ComfyUI-LTXVideo in Cell 1, or set USE_TILED_VAE=False.")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_frames is None:
            # Standard VAEDecode — always available in ComfyUI core
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_video), 0)
            print("   ✓ Standard VAE decode (VAEDecode)")

        del vae_video
        cleanup_memory()

        # ── Audio decode [201] ─────────────────────────────────────────────
        # [201] LTXVAudioVAEDecode — decode audio latent
        try:
            aud_dec   = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            audio_out = aud_dec.EXECUTE_NORMALIZED(
                samples=aud_lat_fin,
                audio_vae=vae_audio)
        except Exception as e:
            print(f"   ⚠️  Audio decode failed ({e}) — proceeding without audio.")
            audio_out = None

        del vae_audio
        cleanup_memory()

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
    }
    if output_path:
        save_metadata_sidecar(output_path, meta)

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


print("✅ generate_pro() defined — run Cell 9 to generate.")
print("   Signature: generate_pro(user_input, image_path, width, height, frames, …)")


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


def run_storyboard(
    scenes:          List[Dict],
    use_continuity:  bool = USE_SCENE_CONTINUITY,
    tmp_dir:         str  = "/content/ComfyUI/input",
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

    print("🎬 Storyboard Runner — Starting")
    print(f"   Scenes    : {len(scenes)}")
    print(f"   Continuity: {use_continuity}")
    print("─" * 70)

    for i, scene in enumerate(scenes):
        scene_num = i + 1
        print(f"\n🎬 Scene {scene_num}/{len(scenes)}: {scene.get('output_prefix','Scene')}")
        print(f"   Input: {scene.get('user_input','')[:80]}…")

        # Resolve image_path: use continuity frame if available and not explicitly set
        _image_path = scene.get("image_path")
        if use_continuity and prev_output and _image_path is None:
            print(f"   🔗 Continuity: extracting last frame from scene {scene_num - 1}…")
            last_tensor = get_last_frame_tensor(prev_output)
            if last_tensor is not None:
                _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.jpg")
                # Save last frame as JPEG for use as seed image
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
                character_image_path = scene.get("character_image_path", CHARACTER_IMAGE_PATH),
                character_strength   = scene.get("character_strength", CHARACTER_STRENGTH),
                character_mode       = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE),
                character_name       = scene.get("character_name", CHARACTER_NAME),
                character_description= scene.get("character_description", CHARACTER_DESCRIPTION),
                output_prefix        = scene.get("output_prefix", OUTPUT_PREFIX),
            )
            outputs.append(out)
            prev_output = out
            print(f"   ✅ Scene {scene_num} done → {out}")
        except Exception as e:
            import traceback
            print(f"   ❌ Scene {scene_num} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            outputs.append(None)
            prev_output = None  # don't chain from a failed scene

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("🎬 Storyboard Complete")
    print(f"   Total scenes : {len(scenes)}")
    print(f"   Successful   : {sum(1 for p in outputs if p)}")
    print(f"   Failed       : {sum(1 for p in outputs if not p)}")
    print("\n   Output paths:")
    for i, p in enumerate(outputs):
        status = "✅" if p else "❌"
        print(f"   {status} Scene {i+1}: {p or 'FAILED'}")
    print("═" * 70)

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
