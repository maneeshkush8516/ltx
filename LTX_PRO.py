"""
LTX-2 Infinite Flow Engine v2.0 - Thin Orchestrator.

All implementation lives in the ltx_pro/ package.
This file provides Colab @param widgets and orchestration logic.

GENERATION MODES:
    1. Single Clip    - generate_pro()
    2. Storyboard     - run_storyboard()
    3. Infinite Flow  - generate_infinite_flow()
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1  ─  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 1. Prepare Environment & Install Custom Nodes

# -- Colab shell commands (uncomment in Google Colab) --
# !pip install torch torchvision torchaudio
# %cd /content
from IPython.display import clear_output
clear_output()
# !pip install -q torchsde einops diffusers accelerate nest_asyncio
# !pip install -q av spandrel albumentations onnx opencv-python onnxruntime
# !pip install -q imageio imageio-ffmpeg
# !pip install -q transformers>=4.43.0 accelerate qwen-vl-utils huggingface_hub
# !git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
# !pip install -r /content/ComfyUI/requirements.txt -q
clear_output()
# %cd /content/ComfyUI/custom_nodes
# !git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
# !git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git
# !git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
# !git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
# !git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git
# !git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
# %cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
# !pip install -r requirements.txt -q
# %cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
# !pip install -r requirements.txt -q
# %cd /content/ComfyUI/custom_nodes/ComfyUI-LTXVideo
# !pip install -r requirements.txt -q 2>/dev/null || true
# %cd /content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD
# !pip install -r requirements.txt -q 2>/dev/null || true
# %cd /content/ComfyUI/custom_nodes/LTX2-Master-Loader
# !pip install -r requirements.txt -q 2>/dev/null || true

import subprocess
def install_apt_packages():
    packages = ["aria2", "ffmpeg"]
    try:
        subprocess.run(["apt-get", "-y", "install", "-qq"] + packages,
                       check=True, capture_output=True)
        print("  apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"  apt error: {e.stderr.decode().strip() or 'unknown'}")

install_apt_packages()

# %cd /content/ComfyUI
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
sys.path.insert(0, "/content/ComfyUI")
clear_output()
print("Environment setup complete.")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2  ─  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 2. Download All Model Weights

import os, subprocess
from pathlib import Path

def model_download(url: str, dest_dir: str, filename: str = None,
                   silent: bool = True) -> str:
    """aria2c download with skip-if-cached logic."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return filename
    cmd = ["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16",
           "-k", "1M", "-d", dest_dir, "-o", filename]
    if silent:
        cmd += ["--summary-interval=0", "--quiet"]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return filename

KIJAI    = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
KIJAI23  = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"

dit_model = model_download(f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
                           "/content/ComfyUI/models/unet")
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")
text_encoder2_model = model_download(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders")
vae_model = model_download(f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",
                           "/content/ComfyUI/models/vae")
vae_audio_model = model_download(f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",
                                 "/content/ComfyUI/models/vae")
taeltx2_model = model_download(f"{KIJAI23}/vae/taeltx2_3.safetensors",
                               "/content/ComfyUI/models/vae")
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

LORA_URLS = {
    "Detailer":    f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":        f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Left":  f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
}
LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(LORA_DIR, exist_ok=True)
for name, url in LORA_URLS.items():
    model_download(url, LORA_DIR)
print("All model files downloaded.")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3  ─  IMPORTS FROM ltx_pro PACKAGE
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 3. Imports from ltx_pro Package

import os, sys, gc, time, json, shutil, warnings, subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from IPython.display import display, HTML, clear_output
from google.colab import files
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/ComfyUI")

from ltx_pro.config import *
from ltx_pro.vram import (VRAMManager, vram_guard, cleanup_memory,
    aggressive_cleanup, force_unload_all_models, purge_vram,
    _deep_unload_model, _print_vram, _verify_vram_clear, _VRAM_MGR)
from ltx_pro.utils import (get_value_at_index, pil_to_tensor, tensor_to_pil,
    load_image_tensor, get_last_frame_tensor, display_video,
    save_video_from_components, save_metadata_sidecar, validate_model_files,
    upload_image, import_custom_nodes)
from ltx_pro.prompt_architect import (InlinePromptArchitect,
    _get_inline_prompt_architect, run_easy_prompt)
from ltx_pro.vision_describe import (InlineVisionDescribe,
    _get_inline_vision_describer, run_vision_describe)
from ltx_pro.character import (PersistentLatentSeed, CharacterPromptAnchor,
    CharacterFeatureExtractor, CharacterEmbeddingBank,
    extract_multi_anchor_frames, create_style_lock,
    extract_continuity_composite, save_continuity_frame,
    extract_character_profiles_from_script)
from ltx_pro.lora import (_IC_LORA_FILES, _CAMERA_LORA_FILES,
    _build_lora_stack, apply_lora_stack, apply_sage_attention,
    apply_chunk_ff, _load_audio_vae)
from ltx_pro.overlap import (blend_overlap_frames, extract_anchor_frame,
    compute_segment_seeds)
from ltx_pro.quality import (compute_frame_ssim, compute_histogram_consistency,
    compute_segment_quality, quality_gate_check, detect_shot_type,
    get_resolution_for_shot)
from ltx_pro.motion import (estimate_optical_flow, detect_motion_direction,
    auto_select_camera_lora, compute_velocity_latent, compute_adaptive_overlap)
from ltx_pro.color import (extract_color_histogram, match_color_histogram,
    apply_color_grade)
from ltx_pro.audio import (AudioSyncEngine, detect_beats,
    compute_segment_boundaries, adjust_segment_length)
from ltx_pro.export import (generate_timeline_json, generate_edl,
    mount_google_drive, sync_to_drive, generate_thumbnail_frame,
    display_thumbnail_grid, concatenate_clips, merge_clips_to_video,
    auto_find_output_clips)
from ltx_pro.pipeline import generate_pro, SVIProContext
from ltx_pro.extended import (generate_extended_video, generate_infinite_flow,
    FlowState, InfiniteFlowConfig)
from ltx_pro.storyboard import (run_storyboard, decompose_script_to_scenes,
    print_scene_breakdown, generate_sora_json)
from ltx_pro.story_to_video import StoryToVideo

try:
    from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
    import folder_paths
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    LoraLoaderModelOnly = None
    folder_paths = None

print("All ltx_pro modules imported.")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3.5  ─  QUICK SETUP DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 3.5. Quick Setup Dashboard

GENERATION_PRESET = "Standard"  # @param ["Quick Draft", "Standard", "Cinema Quality", "T4 Safe"]
RESOLUTION_PRESET = "Landscape 16:9"  # @param ["Portrait 9:16", "Landscape 16:9", "Square 1:1", "Widescreen 21:9", "Custom"]
GENERATION_MODE = "Single Clip"  # @param ["Single Clip", "Storyboard", "Infinite Flow"]
DIFFICULTY_LEVEL = "Intermediate"  # @param ["Beginner", "Intermediate", "Expert"]

print_current_config()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3.6  ─  FEATURE TOGGLES
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 3.6. Feature Toggles

ENABLE_VIDEO_GENERATION    = True   # @param {type:"boolean"}
ENABLE_UPSCALING           = True   # @param {type:"boolean"}
ENABLE_AUDIO_GENERATION    = True   # @param {type:"boolean"}
ENABLE_CHARACTER_SYSTEM    = True   # @param {type:"boolean"}
ENABLE_POST_PROCESSING     = True   # @param {type:"boolean"}
ENABLE_QUALITY_CHECKS      = True   # @param {type:"boolean"}
ENABLE_LLM_PROMPT_EXPANSION = True  # @param {type:"boolean"}
ENABLE_VISION_ANALYSIS     = True   # @param {type:"boolean"}
ENABLE_SCENE_CONTINUITY    = True   # @param {type:"boolean"}
ENABLE_SEGMENT_EXTENSION   = False  # @param {type:"boolean"}

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  ─  EASY PROMPT + VISION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 4. Easy Prompt & Vision Describe Configuration

LLM_MODEL  = "3B"    # @param ["8B", "3B", "14B"]
CREATIVITY = 0.9     # @param {type:"number"}
INVENT_DIALOGUE    = True   # @param {type:"boolean"}
BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
LORA_TRIGGERS = "Ultra HDR, cinematic, hyperrealistic detailing, dramatic lighting, vibrant colors"  # @param {type:"string"}
LLM_OFFLINE_MODE = False    # @param {type:"boolean"}
LOCAL_PATH_3B = "/root/.cache/huggingface/hub/models--huihui-ai--Llama-3.2-3B-Instruct-abliterated/snapshots/ba0be3c4683117ffe70be5cc767723e0210e437e"  # @param {type:"string"}
LOCAL_PATH_8B = "/root/.cache/huggingface/hub/models--mlabonne--NeuralDaredevil-8B-abliterated/snapshots/6567010926ff93a5e9fb809534d61ab667a86674"  # @param {type:"string"}
LOCAL_PATH_14B = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-14B-abliterated/snapshots/latest"  # @param {type:"string"}
USE_VISION   = True          # @param {type:"boolean"}
VISION_MODEL = "3B-fast"     # @param ["3B-fast", "7B-nsfw"]
VISION_OFFLINE_MODE = False  # @param {type:"boolean"}
VISION_LOCAL_PATH = "/root/.cache/huggingface/hub/models--huihui-ai--Qwen2.5-VL-3B-Instruct-abliterated/snapshots/latest"  # @param {type:"string"}
SHOW_PREVIEWS           = True   # @param {type:"boolean"}
DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4.5  ─  SCRIPT-TO-SHOT DECOMPOSER (Script Intelligence)
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 4.5. Script-to-Shot Intelligence

USE_SCRIPT_DECOMPOSER = False  # @param {type:"boolean"}
SCRIPT_INPUT = ""  # @param {type:"string"}
TARGET_VIDEO_DURATION = 30  # @param {type:"integer"}
SEGMENT_DURATION = 5  # @param {type:"integer"}
DIALOGUE_INTERVAL = 15  # @param {type:"integer"}
VIDEO_QUALITY = "8K cinematic"  # @param ["8K cinematic", "4K professional", "HD broadcast", "social media"]
VIDEO_STYLE = "realistic"  # @param ["realistic", "cinematic noir", "anime", "documentary", "fantasy", "sci-fi"]
VIDEO_LANGUAGE = "English"  # @param ["English", "Hindi", "Spanish", "French", "Japanese", "Korean", "Chinese", "Arabic", "Custom"]
CUSTOM_LANGUAGE = ""  # @param {type:"string"}
CHARACTER_DEFINITION = ""  # @param {type:"string"}
SECONDARY_CHARACTER = ""  # @param {type:"string"}
SCRIPT_LLM_MODEL = "3B"  # @param ["8B", "3B", "14B"]
AUTO_CAMERA_SELECT = True  # @param {type:"boolean"}
SHOTS_PER_SCENE = 3  # @param {type:"integer"}
SCENE_OUTPUT_FORMAT = "detailed"  # @param ["detailed", "simple", "json"]

_decomposed_scenes = None
if USE_SCRIPT_DECOMPOSER and SCRIPT_INPUT and SCRIPT_INPUT.strip():
    _lang = CUSTOM_LANGUAGE if VIDEO_LANGUAGE == "Custom" else VIDEO_LANGUAGE
    _decomposed_scenes = decompose_script_to_scenes(
        script=SCRIPT_INPUT, target_duration=TARGET_VIDEO_DURATION,
        segment_duration=SEGMENT_DURATION, dialogue_interval=DIALOGUE_INTERVAL,
        quality=VIDEO_QUALITY, style=VIDEO_STYLE, language=_lang,
        character_def=CHARACTER_DEFINITION, secondary_char=SECONDARY_CHARACTER,
        output_format=SCENE_OUTPUT_FORMAT, fps=globals().get('FPS', 25))
    if _decomposed_scenes:
        print_scene_breakdown(_decomposed_scenes)
        _json_path = f"/content/ComfyUI/output/{globals().get('CHARACTER_NAME', 'Character')}_sora_prompts.json"
        os.makedirs("/content/ComfyUI/output", exist_ok=True)
        generate_sora_json(_decomposed_scenes, _json_path)
        SCENES = _decomposed_scenes
        USE_STORYBOARD = True

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  ─  CHARACTER CONSISTENCY & LORA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 5. Character Consistency & LoRA Configuration

CHARACTER_IMAGE_PATH = None  # @param {type:"string"}
CHARACTER_STRENGTH = 1.0     # @param {type:"number"}
CHARACTER_CONSISTENCY_MODE = "both"  # @param ["i2v", "anchor", "both", "none"]
CHARACTER_NAME = "Character"  # @param {type:"string"}
CHARACTER_DESCRIPTION = ""   # @param {type:"string"}
CHARACTER_PROFILE_EXTRACTION = False  # @param {type:"boolean"}
CHARACTER_EXTRACTED_PROFILE = {}
USE_IDENTITY_REINFORCEMENT = True  # @param {type:"boolean"}
IC_LORA          = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4         # @param {type:"number"}
CAMERA_LORA          = "none"  # @param ["none", "dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"]
CAMERA_LORA_STRENGTH = 1.0     # @param {type:"number"}
USE_SAGE_ATTENTION      = False  # @param {type:"boolean"}
USE_CHUNK_FF            = False  # @param {type:"boolean"}
PURGE_VRAM_AFTER_MODELS = True   # @param {type:"boolean"}
PRO_MODE      = False    # @param {type:"boolean"}
PRO_STEPS     = 4        # @param {type:"integer"}
PRO_SCHEDULER = "simple" # @param {type:"string"}
PRO_SPLIT_AT  = 2        # @param {type:"integer"}

LORA_STACK = _build_lora_stack(IC_LORA, IC_LORA_STRENGTH, CAMERA_LORA, CAMERA_LORA_STRENGTH)
LORA_STACK_JSON = json.dumps(LORA_STACK)

OVERLAP_FRAMES = 5           # @param {type:"integer"}
OVERLAP_MODE = "linear_blend"  # @param ["linear_blend", "hard_cut", "crossfade"]
OVERLAP_SIDE = "source"      # @param ["source", "target"]
USE_SEGMENT_EXTENSION = False  # @param {type:"boolean"}
SEGMENT_LENGTH = 81           # @param {type:"integer"}
MAX_SEGMENTS = 8              # @param {type:"integer"}
SEGMENT_SEED_MODE = "fixed"   # @param ["fixed", "increment", "random"]
MULTI_FRAME_ANCHOR_COUNT = 3    # @param {type:"integer"}
USE_CHARACTER_EMBEDDING_BANK = True  # @param {type:"boolean"}
USE_STYLE_LOCK = True   # @param {type:"boolean"}
USE_MOTION_COHERENCE = True  # @param {type:"boolean"}
USE_VELOCITY_INJECTION = True  # @param {type:"boolean"}
USE_ADAPTIVE_OVERLAP = True  # @param {type:"boolean"}
ADAPTIVE_OVERLAP_MIN = 2   # @param {type:"integer"}
ADAPTIVE_OVERLAP_MAX = 10  # @param {type:"integer"}
USE_QUALITY_GATE = False  # @param {type:"boolean"}
QUALITY_GATE_MAX_RETRIES = 3  # @param {type:"integer"}
SSIM_THRESHOLD = 0.7    # @param {type:"number"}
HISTOGRAM_THRESHOLD = 0.8  # @param {type:"number"}
VARIANCE_THRESHOLD = 0.1   # @param {type:"number"}
USE_MULTI_RESOLUTION = False  # @param {type:"boolean"}
USE_PERSISTENT_CONTEXT = False  # @param {type:"boolean"}
USE_DUAL_ANCHOR_STORYBOARD = True  # @param {type:"boolean"}
CONTINUITY_FRAME_FORMAT = "png"  # @param ["png", "jpg"]
CONTINUITY_MULTI_FRAME_COUNT = 5  # @param {type:"integer"}
CONTINUITY_COMPOSITE_MODE = "weighted_average"  # @param ["weighted_average", "last_frame"]
LATENT_OVERLAP_STRENGTH = 0.3  # @param {type:"number"}
ANCHOR_BLEND_FRAMES = 5  # @param {type:"integer"}

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6  ─  VIDEO GENERATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 6. Configure Your Video

USER_INPUT = (  # @param {type:"string"}
    "a woman walks through a rain-soaked city street at night, "
    "neon reflections on the wet pavement, looking over her shoulder"
)
IMAGE_PATH = None   # @param {type:"string"}
IMAGE_STRENGTH = 1.0  # @param {type:"number"}
POSITIVE_PROMPT = (  # @param {type:"string"}
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "woman walking, bokeh streetlights, moody atmosphere, ultra detailed, "
    "professional cinematography, shallow depth of field, film grain"
)
NEGATIVE_PROMPT = (  # @param {type:"string"}
    "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, "
    "grainy, overexposed, underexposed, flickering, motion artifacts, flat lighting"
)
WIDTH  = 768   # @param {type:"integer"}
HEIGHT = 512   # @param {type:"integer"}
FRAMES = 121   # @param {type:"integer"}
FPS    = 25    # @param {type:"integer"}
SEED                = 47    # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"           # @param {type:"string"}
PASS1_CFG     = 1.0               # @param {type:"number"}
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}
PASS2_CFG     = 1.0                    # @param {type:"number"}
PASS2_SEED    = 0                      # @param {type:"integer"}
USE_TILED_VAE          = False  # @param {type:"boolean"}
TILED_SPATIAL_TILES    = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP  = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN     = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX   = False  # @param {type:"boolean"}
OUTPUT_PREFIX = "LTX-2-PRO"  # @param {type:"string"}
USE_SCENE_CONTINUITY = True  # @param {type:"boolean"}
if not ENABLE_SCENE_CONTINUITY:
    USE_SCENE_CONTINUITY = False
USE_AUDIO_SYNC = False     # @param {type:"boolean"}
AUDIO_SYNC_PATH = None     # @param {type:"string"}
AUDIO_BPM = None           # @param {type:"integer"}
GENERATE_THUMBNAILS = False  # @param {type:"boolean"}
THUMBNAIL_COLS = 3          # @param {type:"integer"}
USE_COLOR_MATCHING = False  # @param {type:"boolean"}
COLOR_GRADE = "none"        # @param ["none", "cinematic_warm", "noir", "cyberpunk", "vintage", "cool_blue", "golden_hour"]
PERSIST_TO_GDRIVE = False   # @param {type:"boolean"}
GDRIVE_PATH = "/content/drive/MyDrive/LTX_PRO_Output"  # @param {type:"string"}
EXPORT_TIMELINE = False     # @param {type:"boolean"}
TIMELINE_FORMAT = "json"    # @param ["json", "edl"]
USE_PARALLEL_PROMPT_EXPANSION = False  # @param {type:"boolean"}

# Apply preset AFTER all @param declarations so preset values are not overwritten
apply_preset()
_config_valid = validate_config()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7  ─  DEFINE generate_pro()
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 7. Generation Functions (imported from ltx_pro.pipeline)
# generate_pro, generate_extended_video, generate_infinite_flow, and
# run_storyboard are all imported from the ltx_pro package in Cell 3.

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8  ─  STORYBOARD / MULTI-SCENE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 8. Storyboard / Multi-Scene Runner
# @markdown Edit the `SCENES` list below, then run Cell 9 with
# @markdown `USE_STORYBOARD = True` to generate all scenes sequentially.

SCENES = [
    {
        "user_input"    : "a woman enters a dimly lit cafe, shaking rain from her coat, looks around",
        "image_path"    : CHARACTER_IMAGE_PATH,
        "frames"        : 121,
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
        "frames"        : 121,
        "seed"          : SEED + 2,
        "output_prefix" : f"Story03-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
]

USE_STORYBOARD = True  # @param {type:"boolean"}

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9  ─  RUN
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 9. Generate
# @markdown Re-run this cell for each new video or storyboard.

import torch
_current_seed = SEED

try:
    # Script Intelligence: use decomposed scenes if available
    if USE_SCRIPT_DECOMPOSER and SCRIPT_INPUT and SCRIPT_INPUT.strip():
        if '_decomposed_scenes' in dir() and _decomposed_scenes and len(_decomposed_scenes) > 0:
            SCENES = _decomposed_scenes
            USE_STORYBOARD = True
        else:
            try:
                _decomposed_scenes = decompose_script_to_scenes(
                    script=SCRIPT_INPUT, target_duration=TARGET_VIDEO_DURATION,
                    segment_duration=SEGMENT_DURATION, dialogue_interval=DIALOGUE_INTERVAL,
                    quality=VIDEO_QUALITY, style=VIDEO_STYLE,
                    language=CUSTOM_LANGUAGE if VIDEO_LANGUAGE == "Custom" else VIDEO_LANGUAGE,
                    character_def=CHARACTER_DEFINITION, secondary_char=SECONDARY_CHARACTER,
                    output_format=SCENE_OUTPUT_FORMAT, fps=FPS)
                if _decomposed_scenes:
                    SCENES = _decomposed_scenes
                    USE_STORYBOARD = True
                    print_scene_breakdown(SCENES)
            except Exception as e:
                print(f"   Script decomposition failed ({e}) - using manual SCENES")

    if USE_STORYBOARD:
        print("Running storyboard mode...")
        storyboard_outputs = run_storyboard(scenes=SCENES, use_continuity=USE_SCENE_CONTINUITY)
        output = storyboard_outputs[-1] if storyboard_outputs else None
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + len(SCENES)

    elif USE_SEGMENT_EXTENSION:
        print("Running Infinite Flow segment extension mode...")
        output = generate_infinite_flow(
            user_input=USER_INPUT, image_path=IMAGE_PATH,
            positive_prompt=POSITIVE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH, height=HEIGHT, fps=FPS, seed=_current_seed,
            image_strength=IMAGE_STRENGTH,
            character_image_path=CHARACTER_IMAGE_PATH,
            character_strength=CHARACTER_STRENGTH,
            character_mode=CHARACTER_CONSISTENCY_MODE,
            character_name=CHARACTER_NAME,
            character_description=CHARACTER_DESCRIPTION,
            output_prefix=OUTPUT_PREFIX,
            pass1_sigmas=PASS1_SIGMAS, pass1_sampler=PASS1_SAMPLER,
            pass1_cfg=PASS1_CFG, pass2_sigmas=PASS2_SIGMAS,
            pass2_sampler=PASS2_SAMPLER, pass2_cfg=PASS2_CFG,
            pass2_seed=PASS2_SEED, pro_mode=PRO_MODE, pro_steps=PRO_STEPS,
            pro_scheduler=PRO_SCHEDULER, pro_split_at=PRO_SPLIT_AT,
            use_tiled_vae=USE_TILED_VAE,
            tiled_spatial_tiles=TILED_SPATIAL_TILES,
            tiled_spatial_overlap=TILED_SPATIAL_OVERLAP,
            tiled_temporal_len=TILED_TEMPORAL_LEN,
            tiled_temporal_overlap=TILED_TEMPORAL_OVERLAP,
            tiled_last_frame_fix=TILED_LAST_FRAME_FIX,
            lora_stack=LORA_STACK, lora_stack_json=LORA_STACK_JSON)
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + MAX_SEGMENTS

    else:
        # Single-clip mode
        output = generate_pro(
            user_input=USER_INPUT, image_path=IMAGE_PATH,
            positive_prompt=POSITIVE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS,
            seed=_current_seed, image_strength=IMAGE_STRENGTH,
            character_image_path=CHARACTER_IMAGE_PATH,
            character_strength=CHARACTER_STRENGTH,
            character_mode=CHARACTER_CONSISTENCY_MODE,
            character_name=CHARACTER_NAME,
            character_description=CHARACTER_DESCRIPTION,
            pass1_sigmas=PASS1_SIGMAS, pass1_sampler=PASS1_SAMPLER,
            pass1_cfg=PASS1_CFG, pass2_sigmas=PASS2_SIGMAS,
            pass2_sampler=PASS2_SAMPLER, pass2_cfg=PASS2_CFG,
            pass2_seed=PASS2_SEED, pro_mode=PRO_MODE, pro_steps=PRO_STEPS,
            pro_scheduler=PRO_SCHEDULER, pro_split_at=PRO_SPLIT_AT,
            use_tiled_vae=USE_TILED_VAE,
            tiled_spatial_tiles=TILED_SPATIAL_TILES,
            tiled_spatial_overlap=TILED_SPATIAL_OVERLAP,
            tiled_temporal_len=TILED_TEMPORAL_LEN,
            tiled_temporal_overlap=TILED_TEMPORAL_OVERLAP,
            tiled_last_frame_fix=TILED_LAST_FRAME_FIX,
            lora_stack=LORA_STACK, lora_stack_json=LORA_STACK_JSON,
            output_prefix=OUTPUT_PREFIX)
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + 1

    # Export timeline if enabled
    if EXPORT_TIMELINE:
        try:
            _tl_entries = []
            if USE_STORYBOARD and 'storyboard_outputs' in dir() and storyboard_outputs:
                for idx, out_path in enumerate(storyboard_outputs):
                    if out_path:
                        _tl_entries.append({
                            "segment_index": idx, "output_path": out_path,
                            "seed": SCENES[idx].get("seed", SEED + idx) if idx < len(SCENES) else SEED,
                            "prompt": SCENES[idx].get("user_input", "")[:200] if idx < len(SCENES) else "",
                            "frames": SCENES[idx].get("frames", FRAMES) if idx < len(SCENES) else FRAMES,
                            "fps": FPS,
                            "duration_seconds": (SCENES[idx].get("frames", FRAMES) if idx < len(SCENES) else FRAMES) / FPS,
                        })
            elif output:
                _tl_entries.append({"segment_index": 0, "output_path": output,
                    "seed": _current_seed, "prompt": USER_INPUT[:200],
                    "frames": FRAMES, "fps": FPS, "duration_seconds": FRAMES / FPS})
            if _tl_entries:
                _tl_path = f"/content/ComfyUI/output/{OUTPUT_PREFIX}_timeline.{TIMELINE_FORMAT}"
                if TIMELINE_FORMAT == "edl":
                    generate_edl(_tl_entries, _tl_path, FPS)
                else:
                    generate_timeline_json(_tl_entries, _tl_path)
        except Exception as e:
            print(f"   Timeline export failed ({e})")

except KeyboardInterrupt:
    print("\nInterrupted - partial output may be in /content/ComfyUI/output/")

except FileNotFoundError as e:
    print(f"\nMissing models: {e}")
    print("   Run Cell 2 to download, then retry Cell 9.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print(f"\nCUDA Out of Memory. Current: {WIDTH}x{HEIGHT}, {FRAMES} frames")
    print("   T4: WIDTH=768, HEIGHT=512, FRAMES=97")
    print("   L4: WIDTH=1024, HEIGHT=576, FRAMES=161")
    print("   A100: WIDTH=1280, HEIGHT=720, FRAMES=241")
    print("   Also try: LLM_MODEL='3B', USE_CHUNK_FF=True, USE_TILED_VAE=True")

except RuntimeError as e:
    print(f"\nRuntime error: {e}")
    cleanup_memory()

except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\nQuick-fix reference:")
    print("   'UnetLoaderGGUF' not found       -> Cell 1: clone ComfyUI_GGUF")
    print("   'LTX2PromptArchitect' not found  -> Cell 1: clone LTX2EasyPrompt-LD")
    print("   'LTX2MasterLoaderLD' not found   -> Cell 1: clone LTX2-Master-Loader")
    print("   'LTXVImgToVideoInplace' missing  -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'PathchSageAttentionKJ' missing  -> Cell 1: clone ComfyUI_KJNodes")
    print("   DualCLIPLoader fp4 error         -> swap CLIP_NAME1 to fp8 in Cell 6")
    print("   Character drift                  -> CHARACTER_CONSISTENCY_MODE='both'")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9.5  ─  MERGE CLIPS INTO ONE VIDEO
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 9.5. Merge Multiple Clips Into One Video

CLIPS_TO_MERGE = []  # @param
AUTO_FIND_CLIPS = True  # @param {type:"boolean"}
MERGE_OUTPUT_NAME = "Final_Merged"  # @param {type:"string"}
MERGE_MODE = "overlap_blend"  # @param ["overlap_blend", "hard_concat", "crossfade"]
MERGE_FPS = 25  # @param {type:"integer"}

_clips_to_merge = CLIPS_TO_MERGE
if not _clips_to_merge and AUTO_FIND_CLIPS:
    _clips_to_merge = auto_find_output_clips()
    if _clips_to_merge:
        print(f"   Found {len(_clips_to_merge)} clips to merge.")

if _clips_to_merge:
    merged_output = merge_clips_to_video(
        clip_paths=_clips_to_merge, output_name=MERGE_OUTPUT_NAME,
        mode=MERGE_MODE, overlap=OVERLAP_FRAMES, fps=MERGE_FPS)
    if merged_output and DOWNLOAD_AFTER_GENERATE:
        try:
            files.download(merged_output)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10  ─  EXPORT & POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 10. Export & Post-Processing

if PERSIST_TO_GDRIVE:
    _drive_mounted = mount_google_drive()
    if _drive_mounted:
        _output_dir = "/content/ComfyUI/output"
        if os.path.exists(_output_dir):
            _files_synced = 0
            for _f in os.listdir(_output_dir):
                if _f.startswith(OUTPUT_PREFIX) and _f.endswith((".mp4", ".json")):
                    if sync_to_drive(os.path.join(_output_dir, _f), GDRIVE_PATH):
                        _files_synced += 1
            print(f"   Synced {_files_synced} files to {GDRIVE_PATH}")

if EXPORT_TIMELINE:
    _tl_output = f"/content/ComfyUI/output/{OUTPUT_PREFIX}_timeline.{TIMELINE_FORMAT}"
    if os.path.exists(_tl_output):
        print(f"   Timeline exists: {_tl_output}")
        if DOWNLOAD_AFTER_GENERATE:
            try:
                files.download(_tl_output)
            except Exception:
                pass

# @markdown ---
# @markdown ### Batch Post-Processing
BATCH_COLOR_GRADE_TARGET = ""  # @param {type:"string"}

if BATCH_COLOR_GRADE_TARGET and COLOR_GRADE != "none" and os.path.exists(BATCH_COLOR_GRADE_TARGET):
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
        print(f"   Graded video saved: {_graded_path}")
        if SHOW_PREVIEWS:
            display_video(_graded_path)
    except Exception as e:
        print(f"   Batch grading failed ({e})")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11  ─  USAGE INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# @title  { "single-column": true }
# @markdown ## 11. Usage Instructions
# @markdown
# @markdown ### Story-to-Prompt Workflow
# @markdown 1. Set `SCRIPT_INPUT` in Cell 4.5 with your narrative script
# @markdown 2. Set `CHARACTER_DEFINITION` with detailed character description
# @markdown 3. Choose `VIDEO_QUALITY` and `VIDEO_STYLE` in Cell 4.5
# @markdown 4. Set `USE_SCRIPT_DECOMPOSER = True` and run Cell 4.5
# @markdown 5. Run Cell 9 to generate all scenes
# @markdown
# @markdown ### Three Generation Modes
# @markdown 1. **Single Clip**: USE_STORYBOARD=False, USE_SEGMENT_EXTENSION=False
# @markdown 2. **Storyboard**: Edit SCENES in Cell 8, USE_STORYBOARD=True
# @markdown 3. **Infinite Flow**: USE_SEGMENT_EXTENSION=True
# @markdown
# @markdown ### Key Tips
# @markdown - T4 GPUs: frames <= 97, 3B LLM, tiled VAE
# @markdown - Character consistency: mode="both" + clear reference image
# @markdown - Long videos: Infinite Flow mode with overlap_frames=5
# @markdown - Best prompts: BYPASS_EASY_PROMPT=False (LLM expansion)

print("Three modes available: Single Clip, Storyboard, Infinite Flow")
