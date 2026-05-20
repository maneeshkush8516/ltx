# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 LD — Complete Pipeline with Easy Prompt & Vision Describe        ║
# ║  Integrates: LTX2EasyPrompt-LD  +  LTX2-Master-Loader (LoRa Daddy)    ║
# ║  Base engine: LD T2V / I2V pipeline (Kijai GGUF Q4_K_M distilled)      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# CELL ORDER:
#   Cell 1 — Install environment + custom nodes  (once per Colab session)
#   Cell 2 — Download all model weights          (once, skip if cached)
#   Cell 3 — Imports, helpers, LoRA stack        (run every session)
#   Cell 4 — Easy Prompt / Vision settings       (edit to taste)
#   Cell 5 — Video generation config             (edit per video)
#   Cell 6 — Define generate_ld()               (run once per session)
#   Cell 7 — Run generation                      (re-run for each clip)


# ══════════════════════════════════════════════════════════════════════════
# CELL 1  ─  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 1. Prepare Environment & Install Custom Nodes
# @markdown Clones all required ComfyUI repos including **LTX2EasyPrompt-LD**
# @markdown and **LTX2-Master-Loader** (LoRa Daddy nodes).

# ── Base Python packages ───────────────────────────────────────────────────
!pip install torch torchvision torchaudio

%cd /content
from IPython.display import clear_output
clear_output()

!pip install -q torchsde einops diffusers accelerate nest_asyncio
!pip install -q av spandrel albumentations onnx opencv-python onnxruntime
!pip install -q imageio imageio-ffmpeg

# Extra packages required by Easy Prompt & Vision Describe nodes
!pip install -q transformers>=4.43.0 accelerate qwen-vl-utils huggingface_hub

# ── ComfyUI (pinned branch — matches reference notebook) ──────────────────
!git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt -q
clear_output()

# ── Custom nodes ───────────────────────────────────────────────────────────
%cd /content/ComfyUI/custom_nodes

# Core nodes (pinned builds from reference notebook)
!git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git

# LTXVideo nodes — required for tiled VAE decode + AV helpers
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git

# ── LoRa Daddy custom nodes ────────────────────────────────────────────────
# LTX2EasyPrompt-LD : LTX2PromptArchitect + LTX2VisionDescribe nodes
!git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
# LTX2-Master-Loader : LTX2MasterLoaderLD node (multi-slot LoRA stacker)
!git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git

# Video helper suite (optional — for VHS_VideoCombine compatibility)
!git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# ── Install node requirements ──────────────────────────────────────────────
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

# ── System tools ───────────────────────────────────────────────────────────
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

# ── Final setup ────────────────────────────────────────────────────────────
%cd /content/ComfyUI
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

clear_output()
print("✅ Environment setup complete.")
print("   Custom nodes installed:")
print("   ✓ ComfyUI_KJNodes    (kj_1.2.6)")
print("   ✓ ComfyUI_GGUF       (22_01_2026)")
print("   ✓ ComfyUI-LTXVideo   (Lightricks)")
print("   ✓ LTX2EasyPrompt-LD  (LoRa Daddy)")
print("   ✓ LTX2-Master-Loader (LoRa Daddy)")
print("   ✓ ComfyUI-VideoHelperSuite")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2  ─  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 2. Download All Model Weights
# @markdown Uses aria2c (fast parallel download). Skips files already cached.

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

# ── Source base URLs ───────────────────────────────────────────────────────
KIJAI    = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"

print("── Core model downloads ──────────────────────────────────────────────")

# ── UNet: GGUF Q4_K_M distilled  (works on T4/L4/A100) ───────────────────
# LTX-2 19B distilled — baked-in distillation, no separate distill LoRA needed
dit_model = model_download(
    f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
    "/content/ComfyUI/models/unet")

# ── Text encoders ──────────────────────────────────────────────────────────
# Gemma fp4  — for RTX 5000 Blackwell. Comment out and use fp8 for T4/A100.
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")
# Gemma fp8  — for T4 / A100 / RTX 3000-4000 (uncomment if fp4 OOMs):
# text_encoder_model = model_download(
#     f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
#     "/content/ComfyUI/models/text_encoders")

# Embeddings connector — distilled version (must match GGUF, NOT the _dev_ one)
text_encoder2_model = model_download(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders")

# ── VAEs ───────────────────────────────────────────────────────────────────
vae_model = model_download(
    f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

vae_audio_model = model_download(
    f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

# TaeEncoder preview VAE (for fast latent preview)
taeltx2_model = model_download(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae")

# ── Spatial upscaler ───────────────────────────────────────────────────────
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

# ── IC LoRAs + Camera Control LoRAs ───────────────────────────────────────
LORA_URLS = {
    "Detailer":     f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":        f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":        f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":         f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":     f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Left":   f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Out":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Right":  f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Down":     f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Jib-Up":       f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Static":       f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}

LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(LORA_DIR, exist_ok=True)
print(f"\n── LoRA batch download ({len(LORA_URLS)} files) ──────────────────────────────")
for name, url in LORA_URLS.items():
    r = model_download(url, LORA_DIR)
    print(f"   {'✅' if r else '❌'}  {name}")

print("\n✅ All model files downloaded.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3  ─  IMPORTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 3. Imports, Helpers & LoRA Stack

import os, sys, gc, time, json, shutil, warnings, subprocess, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, Image as IPImage, clear_output
from google.colab import files

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

# ── ComfyUI core ───────────────────────────────────────────────────────────
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# ── Async node loader (Jupyter/Colab safe) ─────────────────────────────────
def import_custom_nodes() -> None:
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

# ── VRAM helpers ───────────────────────────────────────────────────────────
def cleanup_memory(verbose: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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

# ── ComfyUI node output accessor ───────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# ── Tensor / image conversion ──────────────────────────────────────────────
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
    """Load an image file as a ComfyUI NHWC tensor."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))

def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    """Extract last frame of a video as NHWC tensor."""
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

# ── Video display & saving ─────────────────────────────────────────────────
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

def save_video_from_components(video_obj, prefix="LTX-2") -> str:
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext  = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path

# ── Model file validator ───────────────────────────────────────────────────
def validate_models(model_dict: dict) -> bool:
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
        fk = folder_map.get(label, "loras")
        found = any(
            os.path.exists(os.path.join(base, filename))
            for base in folder_paths.get_folder_paths(fk)
        )
        print(f"   {'✅' if found else '❌'} [{label:9s}] {filename}")
        if not found:
            ok = False
    return ok

# ── Upload helpers ─────────────────────────────────────────────────────────
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

# ── Audio VAE loader (KJNodes or fallback) ─────────────────────────────────
def load_audio_vae(vae_name: str):
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=vae_name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)

# ──────────────────────────────────────────────────────────────────────────
# LTX2-Master-Loader LoRA Stack
# ──────────────────────────────────────────────────────────────────────────
# Mirrors the LTX2MasterLoaderLD node (node 263 in LD JSON workflow).
# 10-slot stack. Each slot: on, lora, guard, strength.
# "guard" = skip LoRA if model doesn't have matching keys (safe mode).
#
# Slot 1: IC Detailer @ 0.4  (active in original LD workflow)
# Slots 2-10: add camera LoRAs or style LoRAs here.
#
# CAMERA LORA FILENAMES (all downloaded in Cell 2):
#   "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
#   "ltx-2-19b-lora-camera-control-dolly-out.safetensors"
#   "ltx-2-19b-lora-camera-control-dolly-left.safetensors"
#   "ltx-2-19b-lora-camera-control-dolly-right.safetensors"
#   "ltx-2-19b-lora-camera-control-jib-up.safetensors"
#   "ltx-2-19b-lora-camera-control-jib-down.safetensors"
#   "ltx-2-19b-lora-camera-control-static.safetensors"
# ──────────────────────────────────────────────────────────────────────────
LD_LORA_STACK = [
    {"on": True,  "lora": "ltx-2-19b-ic-lora-detailer.safetensors", "guard": False, "strength": 0.4},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 2
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 3
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 4
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 5
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 6
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 7
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 8
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 9
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # slot 10
]

# Serialised for passing to LTX2MasterLoaderLD node widget
LORA_STACK_JSON = json.dumps(LD_LORA_STACK)

print("✅ Imports & helpers ready.")
print(f"   Active LoRA slots: "
      f"{sum(1 for s in LD_LORA_STACK if s['on'] and s['lora'] not in ('None',''))}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4  ─  EASY PROMPT + VISION SETTINGS  (LoRa Daddy nodes)
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 4. Easy Prompt & Vision Describe Configuration
# @markdown Set `BYPASS_EASY_PROMPT = True` to skip the LLM and write
# @markdown `POSITIVE_PROMPT` manually in Cell 5.

# ── LLM prompt expander (LTX2PromptArchitect) ─────────────────────────────
LLM_MODEL   = "8B"    # @param ["8B","3B","14B"]
# "8B"  → NeuralDaredevil-8B-abliterated  — best quality, needs ~10 GB VRAM
# "3B"  → Llama-3.2-3B-abliterated       — fastest, ~4 GB VRAM (T4 safe)
# "14B" → Qwen3-14B-abliterated           — highest quality, needs ~18 GB VRAM

CREATIVITY  = 0.9     # @param {type:"number"}  0.7=literal / 0.9=balanced / 1.1=creative

INVENT_DIALOGUE   = True   # @param {type:"boolean"}
# When True the LLM invents natural spoken dialogue woven into the scene prose.
# When False it only uses quotes you typed, or generates no dialogue.

BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
# True  → skip LLM, use POSITIVE_PROMPT from Cell 5 directly (fast, manual control)
# False → LLM expands USER_INPUT into a full cinematic prompt

LORA_TRIGGERS = ""    # @param {type:"string"}
# LoRA trigger words injected at the start of every expanded prompt.
# e.g. "ohwx woman" or "film grain, 35mm"

# ── Vision image describer (LTX2VisionDescribe) ────────────────────────────
USE_VISION    = True   # @param {type:"boolean"}
# When True AND an image is provided, Vision Describe analyses it and passes
# the result as scene_context to Easy Prompt. Adds ~30-90s on first run
# (downloads Qwen VL model). Set False to skip and save VRAM.

VISION_MODEL  = "3B-fast"  # @param ["3B-fast","7B-nsfw"]
# "3B-fast"  → Qwen2.5-VL-3B  — faster, uses ~5 GB VRAM, good general accuracy
# "7B-nsfw"  → Qwen2.5-VL-7B  — more accurate for adult/NSFW content, ~10 GB VRAM

SHOW_PREVIEWS = True   # @param {type:"boolean"}
# Display each video inline after generation.

# ── Internal label maps (do not edit) ─────────────────────────────────────
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


# ── Easy Prompt wrapper ────────────────────────────────────────────────────
def run_easy_prompt(
    user_input:    str,
    frame_count:   int,
    seed:          int,
    scene_context: str = "",
) -> tuple:
    """
    Calls LTX2PromptArchitect (LTX2EasyPrompt-LD node) to expand a simple
    story description into a dense cinematic prompt + negative prompt.

    Falls back to returning the raw input if the node is unavailable.
    LLM is loaded, run, then unloaded to free VRAM for the video model.

    Returns: (positive_prompt: str, negative_prompt: str)
    """
    if "LTX2PromptArchitect" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTX2PromptArchitect not found — using raw user_input.")
        return user_input, ""

    print(f"   [EasyPrompt] LLM={LLM_MODEL} | creativity={CREATIVITY} | frames={frame_count}")
    node = NODE_CLASS_MAPPINGS["LTX2PromptArchitect"]()

    result = node.generate(
        bypass=False,
        user_input=user_input,
        creativity=_creativity_label(CREATIVITY),
        seed=seed,
        invent_dialogue=INVENT_DIALOGUE,
        keep_model_loaded=False,   # always unload after — VRAM for video model
        offline_mode=False,
        frame_count=frame_count,
        model=_LLM_LABEL_MAP.get(LLM_MODEL, "8B - NeuralDaredevil (High Quality)"),
        local_path_8b="",
        local_path_3b="",
        local_path_14b="",
        scene_context=scene_context,
        lora_triggers=LORA_TRIGGERS,
    )

    prompt     = result[0]  # PROMPT
    neg_prompt = result[2]  # NEG_PROMPT
    print(f"   [EasyPrompt] ✓  {len(prompt.split())} words generated.")
    cleanup_memory()
    return prompt, neg_prompt


# ── Vision Describe wrapper ────────────────────────────────────────────────
def run_vision_describe(image_tensor: torch.Tensor) -> str:
    """
    Calls LTX2VisionDescribe (LTX2EasyPrompt-LD node) to analyse the image
    and return a 100-130 word scene description for use as scene_context.

    Model is unloaded immediately after inference to free VRAM.
    Returns: scene_context string (empty string on failure).
    """
    if not USE_VISION:
        return ""
    if "LTX2VisionDescribe" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTX2VisionDescribe not found — skipping vision analysis.")
        return ""

    print(f"   [VisionDescribe] model={VISION_MODEL} | image shape={image_tensor.shape}")
    node = NODE_CLASS_MAPPINGS["LTX2VisionDescribe"]()

    result = node.describe(
        image=image_tensor,
        model_name=_VISION_LABEL_MAP.get(VISION_MODEL, "Qwen2.5-VL-3B — Fast (huihui abliterated)"),
        offline_mode=False,
        local_path="",
    )

    ctx = result[0]  # scene_context string
    print(f"   [VisionDescribe] ✓  {len(ctx.split())} words.")
    cleanup_memory()
    return ctx


# ── LTX2MasterLoaderLD wrapper ─────────────────────────────────────────────
def apply_lora_stack(unet, clip_model):
    """
    Applies the LD_LORA_STACK via LTX2MasterLoaderLD node when available.
    Falls back to manual LoraLoaderModelOnly loop if the node is missing.

    Returns: (unet, clip_model)  — clip unchanged for IC/camera LoRAs.
    """
    active = [s for s in LD_LORA_STACK
              if s.get("on") and s.get("lora") not in (None, "None", "")]

    if not active:
        print("   ℹ️  No active LoRAs in stack — skipping.")
        return unet, clip_model

    # ── Try LTX2MasterLoaderLD node (LoRa Daddy) ──────────────────────────
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        print(f"   [MasterLoader] Applying {len(active)} LoRA(s) via LTX2MasterLoaderLD…")
        try:
            node    = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            fn_name = node.FUNCTION                          # e.g. "apply" or "load"
            result  = getattr(node, fn_name)(
                model=unet,
                clip=clip_model,
                stack_data=LORA_STACK_JSON,
            )
            unet       = get_value_at_index(result, 0)
            # clip_model = get_value_at_index(result, 1)  # IC LoRAs don't modify CLIP
            print("   [MasterLoader] ✓  Stack applied.")
            return unet, clip_model
        except Exception as e:
            print(f"   [MasterLoader] ⚠️  Node failed ({e}) — falling back to manual loop.")

    # ── Fallback: LoraLoaderModelOnly loop ────────────────────────────────
    print(f"   [MasterLoader] Applying {len(active)} LoRA(s) manually…")
    for slot in active:
        name, strength, guard = slot["lora"], slot["strength"], slot.get("guard", False)
        try:
            ll   = LoraLoaderModelOnly()
            unet = ll.load_lora_model_only(unet, name, strength)[0]
            print(f"      ✓ {name} @ {strength}")
        except Exception as e:
            if guard:
                print(f"      ⚠️  {name} skipped (guard active): {e}")
            else:
                print(f"      ❌ {name} failed: {e}")

    return unet, clip_model


print("✅ Easy Prompt + Vision settings ready.")
print(f"   LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}  |  "
      f"Creativity: {CREATIVITY}  |  Bypass: {BYPASS_EASY_PROMPT}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5  ─  VIDEO GENERATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 5. Configure Your Video

# ── Simple input (used by Easy Prompt) ────────────────────────────────────
USER_INPUT = (  # @param {type:"string"}
    "a woman walks through a rain-soaked city street at night, "
    "neon reflections on the wet pavement, looking over her shoulder"
)
# When BYPASS_EASY_PROMPT=False this is expanded by the LLM.
# When BYPASS_EASY_PROMPT=True it is ignored and POSITIVE_PROMPT is used directly.

# ── Reference / seed image (optional — enables Image-to-Video + Vision) ───
IMAGE_PATH = None   # @param {type:"string"}
# Set to a local path, e.g. "/content/ComfyUI/input/my_photo.jpg"
# When set: Vision Describe analyses it → Easy Prompt uses the description
# as authoritative scene context.  Leave None for text-to-video mode.

# ── Manual prompt (used only when BYPASS_EASY_PROMPT=True) ────────────────
POSITIVE_PROMPT = (  # @param {type:"string"}
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "woman walking, bokeh streetlights, moody atmosphere, ultra detailed, "
    "professional cinematography, shallow depth of field, film grain"
)
NEGATIVE_PROMPT = (  # @param {type:"string"}
    "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, "
    "grainy, overexposed, underexposed, flickering, motion artifacts, flat lighting"
)

# ── Resolution & length ────────────────────────────────────────────────────
WIDTH  = 768   # @param {type:"integer"}
HEIGHT = 512   # @param {type:"integer"}
FRAMES = 121   # @param {type:"integer"}
FPS    = 25    # @param {type:"integer"}
# T4  safe defaults : 768×512, 121 frames (~4.8s)
# L4  (24 GB)       : 1024×576, 161 frames
# A100 (40 GB)      : 1280×720, 241 frames

# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 47                  # @param {type:"integer"}
AUTO_INCREMENT_SEED = True # @param {type:"boolean"}

# ── Model filenames (must match what was downloaded in Cell 2) ─────────────
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"   # swap to fp8 if fp4 OOMs
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# ── Sampling schedules ─────────────────────────────────────────────────────
# Pass 1 — first-pass denoising (ManualSigmas replaces LTXVScheduler for GGUF)
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"
PASS1_CFG     = 1.0

# Pass 2 — spatial upscale + refinement
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"
PASS2_CFG     = 1.0
PASS2_SEED    = 0          # fixed for reproducible refinement

# ── Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode) ───────────────────
USE_TILED_VAE           = True   # @param {type:"boolean"}
TILED_SPATIAL_TILES     = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP   = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN      = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP  = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX    = False  # @param {type:"boolean"}

# ── I2V conditioning strength ──────────────────────────────────────────────
IMAGE_STRENGTH = 1.0   # @param {type:"number"}
# 1.0 = strong image guidance (character stays close to reference)
# 0.5 = balanced (some drift allowed for motion)

OUTPUT_PREFIX = "LTX-2"  # @param {type:"string"}

print("✅ Generation config set.")
print(f"   Resolution : {WIDTH}×{HEIGHT}  |  {FRAMES} frames  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   Mode       : {'I2V' if IMAGE_PATH else 'T2V'}  |  Seed: {SEED}")
print(f"   Easy Prompt: {'BYPASS' if BYPASS_EASY_PROMPT else LLM_MODEL}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 6  ─  GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 6. Define generate_ld()

def generate_ld(
    user_input:             str   = USER_INPUT,
    image_path:             str   = IMAGE_PATH,
    # ── Used only when BYPASS_EASY_PROMPT=True ──
    positive_prompt:        str   = POSITIVE_PROMPT,
    negative_prompt:        str   = NEGATIVE_PROMPT,
    # ── Video settings ──
    width:                  int   = WIDTH,
    height:                 int   = HEIGHT,
    frames:                 int   = FRAMES,
    fps:                    int   = FPS,
    seed:                   int   = SEED,
    image_strength:         float = IMAGE_STRENGTH,
    # ── Sampling ──
    pass1_sigmas:           str   = PASS1_SIGMAS,
    pass1_sampler:          str   = PASS1_SAMPLER,
    pass1_cfg:              float = PASS1_CFG,
    pass2_sigmas:           str   = PASS2_SIGMAS,
    pass2_sampler:          str   = PASS2_SAMPLER,
    pass2_cfg:              float = PASS2_CFG,
    pass2_seed:             int   = PASS2_SEED,
    # ── Tiled VAE ──
    use_tiled_vae:          bool  = USE_TILED_VAE,
    tiled_spatial_tiles:    int   = TILED_SPATIAL_TILES,
    tiled_spatial_overlap:  int   = TILED_SPATIAL_OVERLAP,
    tiled_temporal_len:     int   = TILED_TEMPORAL_LEN,
    tiled_temporal_overlap: int   = TILED_TEMPORAL_OVERLAP,
    tiled_last_frame_fix:   bool  = TILED_LAST_FRAME_FIX,
    output_prefix:          str   = OUTPUT_PREFIX,
) -> Optional[str]:
    """
    Two-pass LTX-2 LD generation pipeline with integrated:
      • LTX2VisionDescribe  — auto-analyses seed image
      • LTX2PromptArchitect — expands USER_INPUT into cinematic prompt
      • LTX2MasterLoaderLD  — applies LD LoRA stack (with manual fallback)

    Pipeline node graph
    ───────────────────
    [EasyPrompt phase] (before video model loads)
      LTX2VisionDescribe  →  scene_context
      LTX2PromptArchitect →  positive_prompt, negative_prompt

    [Model loading]
      UnetLoaderGGUF      →  unet (raw)
      LTX2MasterLoaderLD  →  unet + clip (LoRA stack applied)
      DualCLIPLoader      →  clip_model
      CLIPTextEncode ×2   →  cond_pos, cond_neg (zero-out)
      LTXVConditioning    →  cond (with frame_rate)
      VAELoader           →  vae_video
      VAELoaderKJ         →  vae_audio
      LatentUpscaleModel  →  upscale_model

    [Latent prep]
      EmptyImage + ResizeImageMaskNode × 0.5 + GetImageSize
      EmptyLTXVLatentVideo  →  vid_lat  (half-res)
      LTXVImgToVideoInplace →  vid_lat  (if I2V mode)
      LTXVEmptyLatentAudio  →  aud_lat
      LTXVConcatAVLatent    →  combined AV latent

    [Pass 1 — first-pass]
      ManualSigmas + KSamplerSelect(euler) + RandomNoise(seed)
      CFGGuider + SamplerCustomAdvanced  →  p1_av

    [Pass 2 — upscale + refinement]
      LTXVSeparateAVLatent + LTXVCropGuides
      LTXVLatentUpsampler (×2) + LTXVConcatAVLatent
      ManualSigmas + KSamplerSelect(gradient_estimation)
      CFGGuider + SamplerCustomAdvanced  →  p2_denoised

    [Decode]
      LTXVSeparateAVLatent
      LTXVSpatioTemporalTiledVAEDecode (or VAEDecode)
      LTXVAudioVAEDecode
      CreateVideo → save → display
    """
    t0 = time.time()
    import_custom_nodes()
    clear_output()

    print("🎬 LTX-2 LD — Generation Starting")
    print(f"   Mode       : {'I2V' if image_path else 'T2V'}")
    print(f"   Resolution : {width}×{height}  |  {frames} frames  |  seed {seed}")
    print(f"   Easy Prompt: {'BYPASS' if BYPASS_EASY_PROMPT else f'LLM={LLM_MODEL}'}")
    _print_vram()

    # ── Pre-flight ──────────────────────────────────────────────────────────
    print("\n🔍 Checking model files…")
    all_ok = validate_models({
        "unet"    : UNET_MODEL,
        "clip1"   : CLIP_NAME1,
        "clip2"   : CLIP_NAME2,
        "vae_vid" : VAE_VIDEO_MODEL,
        "vae_aud" : VAE_AUDIO_MODEL,
        "upscaler": UPSCALER_MODEL,
    })
    if not all_ok:
        raise FileNotFoundError("Missing model files — run Cell 2.")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — EASY PROMPT (LLM + Vision, before video model loads)
    # ══════════════════════════════════════════════════════════════════════

    # ── Load reference image (for Vision Describe and/or I2V conditioning) ─
    seed_image_tensor = None
    if image_path:
        seed_image_tensor = load_image_tensor(image_path)
        if seed_image_tensor is None:
            print(f"   ⚠️  Image not found: {image_path} — switching to T2V")
        else:
            print(f"   ✓  Image loaded: {image_path}  {seed_image_tensor.shape}")

    # ── Vision Describe ─────────────────────────────────────────────────────
    scene_context = ""
    if seed_image_tensor is not None and USE_VISION and not BYPASS_EASY_PROMPT:
        print("\n👁️  Vision Describe…")
        scene_context = run_vision_describe(seed_image_tensor)
        if scene_context:
            print(f"   Scene context: {scene_context[:120]}…")

    # ── Easy Prompt expansion ───────────────────────────────────────────────
    final_positive = positive_prompt
    final_negative = negative_prompt

    if not BYPASS_EASY_PROMPT and user_input.strip():
        print("\n🧠 Easy Prompt expansion…")
        final_positive, final_negative = run_easy_prompt(
            user_input=user_input,
            frame_count=frames,
            seed=seed,
            scene_context=scene_context,
        )
        print(f"\n   ── EXPANDED PROMPT ──────────────────────────────")
        print(f"   {final_positive[:300]}{'…' if len(final_positive)>300 else ''}")
        print(f"\n   ── NEGATIVE PROMPT ─────────────────────────────")
        print(f"   {final_negative[:150]}…")
    else:
        print("\n   [EasyPrompt] Bypassed — using manual POSITIVE_PROMPT.")

    # VRAM should now be clear of LLM
    cleanup_memory(verbose=True)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — VIDEO GENERATION (two-pass LTX-2 pipeline)
    # ══════════════════════════════════════════════════════════════════════

    with torch.inference_mode():

        # ── 1. MODEL LOADING ──────────────────────────────────────────────

        # UnetLoaderGGUF — loads GGUF Q4_K_M distilled checkpoint
        print("\n📦 Loading UNet (GGUF distilled Q4_K_M)…")
        unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet        = get_value_at_index(unet_loader.load_unet(unet_name=UNET_MODEL), 0)

        # DualCLIPLoader — Gemma + embeddings connector
        print("   Loading CLIP encoders…")
        clip_loader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        try:
            clip_model = get_value_at_index(
                clip_loader.load_clip(
                    clip_name1=CLIP_NAME1, clip_name2=CLIP_NAME2,
                    type="ltxv", device="default"), 0)
        except Exception as e:
            print(f"   ⚠️  fp4 CLIP load failed ({e}) — trying fp8 fallback…")
            fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
            clip_model = get_value_at_index(
                clip_loader.load_clip(
                    clip_name1=fp8, clip_name2=CLIP_NAME2,
                    type="ltxv", device="default"), 0)

        # LTX2MasterLoaderLD — apply LoRA stack (or manual fallback)
        print("   Applying LoRA stack via LTX2MasterLoaderLD…")
        unet, clip_model = apply_lora_stack(unet, clip_model)
        _print_vram()

        # VAEs
        print("   Loading VAEs…")
        vaeloader   = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_video   = get_value_at_index(vaeloader.load_vae(vae_name=VAE_VIDEO_MODEL), 0)
        vae_audio   = get_value_at_index(load_audio_vae(VAE_AUDIO_MODEL), 0)

        # Spatial upscaler
        uml           = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        upscale_model = get_value_at_index(
            uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0)

        # ── 2. TEXT ENCODING ──────────────────────────────────────────────
        print("\n📝 Encoding prompts…")
        cte       = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos  = cte.encode(text=final_positive, clip=clip_model)
        zero_out  = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        cond_zero = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))

        ltxv_cond_node = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond = ltxv_cond_node.EXECUTE_NORMALIZED(
            frame_rate=float(fps),
            positive=get_value_at_index(cond_pos, 0),
            negative=get_value_at_index(cond_zero, 0))

        del clip_model
        cleanup_memory()

        # ── 3. LATENT PREPARATION ─────────────────────────────────────────
        print("🗂️  Preparing latents…")

        # EmptyImage at target res → resize ×0.5 → GetImageSize → half dims
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

        # EmptyLTXVLatentVideo at half resolution
        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1)

        # ── I2V: LTXVImgToVideoInplace — apply image conditioning ────────
        # Uses LTXVPreprocess first to normalise the seed image compression.
        img_bypass   = seed_image_tensor is None
        img_strength = image_strength if not img_bypass else 0.0

        if not img_bypass:
            # Resize seed image to match latent dimensions × 2 (half-res input space)
            rim2  = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ri2   = rim2.EXECUTE_NORMALIZED(
                input=seed_image_tensor,
                scale_method="lanczos",
                resize_type={
                    "resize_type": "scale dimensions",
                    "width": half_w * 2, "height": half_h * 2, "crop": "center"})
            pp_node = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp_img  = get_value_at_index(
                pp_node.EXECUTE_NORMALIZED(
                    img_compression=33,
                    image=get_value_at_index(ri2, 0)), 0)

            i2v     = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            vid_lat = i2v.EXECUTE_NORMALIZED(
                strength=img_strength, bypass=False,
                vae=vae_video,
                image=pp_img,
                latent=get_value_at_index(vid_lat, 0))
            print(f"   ✓ I2V conditioning applied (strength={img_strength})")
        else:
            # T2V — just use the empty latent directly
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Audio latent
        elalat  = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1,
            audio_vae=vae_audio)

        # Concatenate AV latent
        catav  = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(vid_lat, 0)
                         if not img_bypass else get_value_at_index(vid_lat, 0),
            audio_latent=get_value_at_index(aud_lat, 0))
        combined_latent = get_value_at_index(av_lat, 0)

        # ── 4. PASS 1 — first-pass denoising ─────────────────────────────
        print(f"\n🚀 Pass 1 — {pass1_sampler}  sigmas: {pass1_sigmas[:55]}…")
        _print_vram()

        manualsigmas   = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise    = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider      = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca            = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        sig_p1     = manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas)
        sampler_p1 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)
        noise_p1   = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
        guider_p1  = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1))

        out1 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise_p1, 0),
            guider=get_value_at_index(guider_p1, 0),
            sampler=get_value_at_index(sampler_p1, 0),
            sigmas=get_value_at_index(sig_p1, 0),
            latent_image=combined_latent)
        p1_av = get_value_at_index(out1, 0)

        del guider_p1
        cleanup_memory()
        print("   ✓ Pass 1 done")

        # ── 5. PASS 2 — spatial upscale + refinement ─────────────────────
        print(f"\n🔧 Pass 2 — {pass2_sampler}  sigmas: {pass2_sigmas}")
        _print_vram()

        ltxvsep      = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1           = ltxvsep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vid_lat_p1   = get_value_at_index(s1, 0)
        aud_lat_p1   = get_value_at_index(s1, 1)

        ltxvcrop     = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cropped      = ltxvcrop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vid_lat_p1)

        guider_p2    = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg, model=unet,
            positive=get_value_at_index(cropped, 0),
            negative=get_value_at_index(cropped, 1))

        ltxvup       = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled    = ltxvup.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model,
            vae=vae_video)
        del upscale_model
        cleanup_memory()

        av_lat2 = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(upsampled, 0),
            audio_latent=aud_lat_p1)

        sig_p2     = manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas)
        sampler_p2 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)
        noise_p2   = randomnoise.EXECUTE_NORMALIZED(noise_seed=pass2_seed)

        out2        = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(noise_p2, 0),
            guider=get_value_at_index(guider_p2, 0),
            sampler=get_value_at_index(sampler_p2, 0),
            sigmas=get_value_at_index(sig_p2, 0),
            latent_image=get_value_at_index(av_lat2, 0))
        p2_denoised = get_value_at_index(out2, 1)  # denoised_output

        del guider_p2, unet
        cleanup_memory()
        print("   ✓ Pass 2 done")

        # ── 6. DECODE ─────────────────────────────────────────────────────
        print("\n🎞️  Decoding…")
        _print_vram()

        s2          = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        # ── Tiled VAE (preferred) or standard VAEDecode ───────────────────
        decoded_frames = None
        if use_tiled_vae:
            try:
                td = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded_frames = get_value_at_index(
                    td.EXECUTE_NORMALIZED(
                        vae=vae_video,
                        latents=vid_lat_fin,
                        spatial_tiles=tiled_spatial_tiles,
                        spatial_overlap=tiled_spatial_overlap,
                        temporal_tile_length=tiled_temporal_len,
                        temporal_overlap=tiled_temporal_overlap,
                        last_frame_fix=tiled_last_frame_fix,
                        working_device="auto",
                        working_dtype="auto"), 0)
                print("   ✓ Tiled VAE decode")
            except (KeyError, Exception) as e:
                print(f"   ⚠️  Tiled VAE unavailable ({type(e).__name__}) — fallback…")
                use_tiled_vae = False

        if not use_tiled_vae:
            vaedecode      = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_video), 0)
            print("   ✓ Standard VAE decode")

        del vae_video
        cleanup_memory()

        # ── Audio decode ──────────────────────────────────────────────────
        aud_dec   = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio_out = aud_dec.EXECUTE_NORMALIZED(
            samples=aud_lat_fin, audio_vae=vae_audio)
        del vae_audio
        cleanup_memory()

        # ── Save ──────────────────────────────────────────────────────────
        print("\n💾 Saving video…")
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        vid_obj     = createvideo.EXECUTE_NORMALIZED(
            fps=fps,
            images=decoded_frames,
            audio=get_value_at_index(audio_out, 0))
        output_path = save_video_from_components(
            get_value_at_index(vid_obj, 0), prefix=output_prefix)

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n✅ Done in {mins}m {secs}s")
    print(f"   📁 {output_path}")
    _print_vram()

    if SHOW_PREVIEWS:
        print("\n▶ Preview:")
        display_video(output_path)

    return output_path


print("✅ generate_ld() defined — run Cell 7 to generate.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 7  ─  RUN
# ══════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 7. Generate
# @markdown Re-run this cell for each new video.

_current_seed = SEED

try:
    output = generate_ld(
        user_input      = USER_INPUT,
        image_path      = IMAGE_PATH,
        positive_prompt = POSITIVE_PROMPT,
        negative_prompt = NEGATIVE_PROMPT,
        width           = WIDTH,
        height          = HEIGHT,
        frames          = FRAMES,
        fps             = FPS,
        seed            = _current_seed,
        image_strength  = IMAGE_STRENGTH,
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

    # Auto-increment seed (mirrors "Shared seed" node [284] increment mode)
    if AUTO_INCREMENT_SEED:
        SEED = _current_seed + 1
        print(f"🔢 Next seed: {SEED}")

    # Optional: download the clip immediately after generation
    # files.download(output)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted.")

except FileNotFoundError as e:
    print(f"\n❌ Missing models: {e}")
    print("   → Run Cell 2 to download, then retry Cell 7.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n❌ CUDA Out of Memory")
    print(f"   Current: {WIDTH}×{HEIGHT}, {FRAMES} frames")
    print("   T4  (15 GB) → try 768×512, 121 frames")
    print("   L4  (24 GB) → try 1024×576, 161 frames")
    print("   A100 (40GB) → try 1280×720, 241 frames")
    print("   Also try: LLM_MODEL='3B' to reduce LLM VRAM usage in Cell 4")

except Exception as e:
    import traceback
    print(f"\n❌ {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n💡 Quick fixes:")
    print("   UnetLoaderGGUF missing       → Cell 1: ComfyUI_GGUF not cloned")
    print("   LTX2PromptArchitect missing  → Cell 1: LTX2EasyPrompt-LD not cloned")
    print("   LTX2MasterLoaderLD missing   → Cell 1: LTX2-Master-Loader not cloned")
    print("   Tiled VAE error              → Set USE_TILED_VAE=False in Cell 5")
    print("   DualCLIPLoader error         → fp4 needs Blackwell GPU; switch to fp8")
    print("   Deformed output (Pass 1)     → change SEED and re-run Cell 7")
    print("   Persistent deformation (3×)  → change USER_INPUT and re-run Cell 7")
