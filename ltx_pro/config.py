"""
Configuration constants, preset definitions, feature toggles, and helper functions.

Extracted from LTX_PRO.py Cells 3.5, 3.6, and 6. Contains all generation
parameters, resolution/preset definitions, ENABLE_* feature toggles, model
filename defaults, sampling defaults, and utility functions for applying
presets, validating config, and estimating resource usage.
"""

__all__ = [
    # Preset definitions
    "_PRESET_DEFINITIONS",
    "_RESOLUTION_DEFINITIONS",
    # Default constants
    "GENERATION_PRESET",
    "RESOLUTION_PRESET",
    "GENERATION_MODE",
    "DIFFICULTY_LEVEL",
    "WIDTH",
    "HEIGHT",
    "FRAMES",
    "FPS",
    "SEED",
    "AUTO_INCREMENT_SEED",
    "USER_INPUT",
    "IMAGE_PATH",
    "IMAGE_STRENGTH",
    "POSITIVE_PROMPT",
    "NEGATIVE_PROMPT",
    "OUTPUT_PREFIX",
    # Model filenames
    "UNET_MODEL",
    "CLIP_NAME1",
    "CLIP_NAME2",
    "VAE_VIDEO_MODEL",
    "VAE_AUDIO_MODEL",
    "UPSCALER_MODEL",
    # Sampling defaults
    "PASS1_SIGMAS",
    "PASS1_SAMPLER",
    "PASS1_CFG",
    "PASS2_SIGMAS",
    "PASS2_SAMPLER",
    "PASS2_CFG",
    "PASS2_SEED",
    # Tiled VAE
    "USE_TILED_VAE",
    "TILED_SPATIAL_TILES",
    "TILED_SPATIAL_OVERLAP",
    "TILED_TEMPORAL_LEN",
    "TILED_TEMPORAL_OVERLAP",
    "TILED_LAST_FRAME_FIX",
    # Performance flags
    "USE_SAGE_ATTENTION",
    "USE_CHUNK_FF",
    "PURGE_VRAM_AFTER_MODELS",
    # Pro sampling mode
    "PRO_MODE",
    "PRO_STEPS",
    "PRO_SCHEDULER",
    "PRO_SPLIT_AT",
    # LLM settings
    "LLM_MODEL",
    "CREATIVITY",
    "INVENT_DIALOGUE",
    "BYPASS_EASY_PROMPT",
    # Scene continuity
    "USE_SCENE_CONTINUITY",
    # Segment extension
    "USE_SEGMENT_EXTENSION",
    "SEGMENT_LENGTH",
    "MAX_SEGMENTS",
    "SEGMENT_SEED_MODE",
    # Overlap
    "OVERLAP_FRAMES",
    "OVERLAP_MODE",
    "OVERLAP_SIDE",
    # Feature toggles
    "ENABLE_VIDEO_GENERATION",
    "ENABLE_UPSCALING",
    "ENABLE_AUDIO_GENERATION",
    "ENABLE_CHARACTER_SYSTEM",
    "ENABLE_POST_PROCESSING",
    "ENABLE_QUALITY_CHECKS",
    "ENABLE_LLM_PROMPT_EXPANSION",
    "ENABLE_VISION_ANALYSIS",
    "ENABLE_SCENE_CONTINUITY",
    "ENABLE_SEGMENT_EXTENSION",
    # Quality keywords
    "STORY_QUALITY_KEYWORDS",
    # Functions
    "apply_preset",
    "print_current_config",
    "validate_config",
    "disable_all_extras",
    "enable_recommended",
    "show_help",
    "estimate_generation_time",
    "estimate_vram_usage",
    "suggest_settings",
]

# ══════════════════════════════════════════════════════════════════════════════
# PRESET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

_PRESET_DEFINITIONS = {
    "Quick Draft": {
        "width": 512,
        "height": 512,
        "frames": 41,
        "llm_model": "3B",
        "bypass_easy_prompt": True,
        "use_tiled_vae": True,
        "use_chunk_ff": False,
    },
    "Standard": {
        "width": 768,
        "height": 512,
        "frames": 121,
        "llm_model": "3B",
        "bypass_easy_prompt": False,
        "use_tiled_vae": False,
        "use_chunk_ff": False,
    },
    "Cinema Quality": {
        "width": 1024,
        "height": 576,
        "frames": 161,
        "llm_model": "8B",
        "bypass_easy_prompt": False,
        "use_tiled_vae": False,
        "use_chunk_ff": False,
    },
    "T4 Safe": {
        "width": 768,
        "height": 512,
        "frames": 97,
        "llm_model": "3B",
        "bypass_easy_prompt": False,
        "use_tiled_vae": True,
        "use_chunk_ff": True,
    },
}

_RESOLUTION_DEFINITIONS = {
    "Portrait 9:16": (576, 1024),
    "Landscape 16:9": (768, 512),
    "Square 1:1": (512, 512),
    "Widescreen 21:9": (896, 384),
    "Custom": None,  # Keep user-specified WIDTH/HEIGHT
}

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION PRESET & MODE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_PRESET = "Standard"
RESOLUTION_PRESET = "Landscape 16:9"
GENERATION_MODE = "Single Clip"
DIFFICULTY_LEVEL = "Intermediate"

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUTION & LENGTH DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

WIDTH = 768
HEIGHT = 512
FRAMES = 121
FPS = 25

# ══════════════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════════════

SEED = 47
AUTO_INCREMENT_SEED = True

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS & IMAGE INPUT
# ══════════════════════════════════════════════════════════════════════════════

USER_INPUT = (
    "a woman walks through a rain-soaked city street at night, "
    "neon reflections on the wet pavement, looking over her shoulder"
)
IMAGE_PATH = None
IMAGE_STRENGTH = 1.0
POSITIVE_PROMPT = (
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "woman walking, bokeh streetlights, moody atmosphere, ultra detailed, "
    "professional cinematography, shallow depth of field, film grain"
)
NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, "
    "grainy, overexposed, underexposed, flickering, motion artifacts, flat lighting"
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL FILENAMES
# ══════════════════════════════════════════════════════════════════════════════

UNET_MODEL = "ltx-2-19b-distilled_Q4_K_M.gguf"
CLIP_NAME1 = "gemma_3_12B_it_fp4_mixed.safetensors"
CLIP_NAME2 = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"
PASS1_CFG = 1.0

PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"
PASS2_CFG = 1.0
PASS2_SEED = 0

# ══════════════════════════════════════════════════════════════════════════════
# TILED VAE DECODE
# ══════════════════════════════════════════════════════════════════════════════

USE_TILED_VAE = False
TILED_SPATIAL_TILES = 2
TILED_SPATIAL_OVERLAP = 8
TILED_TEMPORAL_LEN = 48
TILED_TEMPORAL_OVERLAP = 4
TILED_LAST_FRAME_FIX = False

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE FLAGS
# ══════════════════════════════════════════════════════════════════════════════

USE_SAGE_ATTENTION = False
USE_CHUNK_FF = False
PURGE_VRAM_AFTER_MODELS = True

# ══════════════════════════════════════════════════════════════════════════════
# PRO SAMPLING MODE
# ══════════════════════════════════════════════════════════════════════════════

PRO_MODE = False
PRO_STEPS = 4
PRO_SCHEDULER = "simple"
PRO_SPLIT_AT = 2

# ══════════════════════════════════════════════════════════════════════════════
# LLM SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

LLM_MODEL = "3B"
CREATIVITY = 0.9
INVENT_DIALOGUE = True
BYPASS_EASY_PROMPT = False

# ══════════════════════════════════════════════════════════════════════════════
# SCENE CONTINUITY & SEGMENT EXTENSION
# ══════════════════════════════════════════════════════════════════════════════

USE_SCENE_CONTINUITY = True
USE_SEGMENT_EXTENSION = False
SEGMENT_LENGTH = 81
MAX_SEGMENTS = 8
SEGMENT_SEED_MODE = "fixed"

# ══════════════════════════════════════════════════════════════════════════════
# OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

OVERLAP_FRAMES = 5
OVERLAP_MODE = "linear_blend"
OVERLAP_SIDE = "source"

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_PREFIX = "LTX-2-PRO"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE TOGGLES (ENABLE_* flags)
# ══════════════════════════════════════════════════════════════════════════════

ENABLE_VIDEO_GENERATION = True
ENABLE_UPSCALING = True
ENABLE_AUDIO_GENERATION = True
ENABLE_CHARACTER_SYSTEM = True
ENABLE_POST_PROCESSING = True
ENABLE_QUALITY_CHECKS = True
ENABLE_LLM_PROMPT_EXPANSION = True
ENABLE_VISION_ANALYSIS = True
ENABLE_SCENE_CONTINUITY = True
ENABLE_SEGMENT_EXTENSION = False

# ══════════════════════════════════════════════════════════════════════════════
# QUALITY KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════

STORY_QUALITY_KEYWORDS = (
    "Ultra HDR, 3D intricate details, vibrant colors, realistic lighting, "
    "Dramatic Lighting, Enhanced Clarity, Brilliant Highlights, "
    "Hyperrealistic Detailing, cinematic"
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_preset():
    """Apply GENERATION_PRESET and RESOLUTION_PRESET to global variables.

    Reads the current values of GENERATION_PRESET, RESOLUTION_PRESET, and
    GENERATION_MODE module-level variables, then updates WIDTH, HEIGHT, FRAMES,
    LLM_MODEL, BYPASS_EASY_PROMPT, USE_TILED_VAE, USE_CHUNK_FF,
    USE_STORYBOARD (via GENERATION_MODE), and USE_SEGMENT_EXTENSION accordingly.
    """
    global WIDTH, HEIGHT, FRAMES, LLM_MODEL, BYPASS_EASY_PROMPT
    global USE_TILED_VAE, USE_CHUNK_FF, USE_SEGMENT_EXTENSION
    global RESOLUTION_PRESET

    # Apply generation preset
    preset = _PRESET_DEFINITIONS.get(GENERATION_PRESET)
    if preset:
        WIDTH = preset["width"]
        HEIGHT = preset["height"]
        FRAMES = preset["frames"]
        LLM_MODEL = preset["llm_model"]
        BYPASS_EASY_PROMPT = preset["bypass_easy_prompt"]
        USE_TILED_VAE = preset["use_tiled_vae"]
        USE_CHUNK_FF = preset["use_chunk_ff"]
        # Generation preset locks resolution - set to Custom to prevent override
        RESOLUTION_PRESET = "Custom"

    # Apply resolution preset (overrides preset width/height unless Custom)
    res = _RESOLUTION_DEFINITIONS.get(RESOLUTION_PRESET)
    if res is not None:
        WIDTH, HEIGHT = res

    # Apply generation mode
    if GENERATION_MODE == "Single Clip":
        USE_SEGMENT_EXTENSION = False
    elif GENERATION_MODE == "Storyboard":
        USE_SEGMENT_EXTENSION = False
    elif GENERATION_MODE == "Infinite Flow":
        USE_SEGMENT_EXTENSION = True


def print_current_config():
    """Display all key settings in a formatted table."""
    print("  +------------------------------------------------------------+")
    print("  |              CURRENT CONFIGURATION                         |")
    print("  +------------------------------------------------------------+")
    print(f"  |  Preset          : {GENERATION_PRESET:<38} |")
    print(f"  |  Resolution      : {WIDTH}x{HEIGHT} ({RESOLUTION_PRESET})"
          f"{' ' * max(0, 24 - len(f'{WIDTH}x{HEIGHT} ({RESOLUTION_PRESET})'))}|")
    print(f"  |  Frames          : {FRAMES:<38} |")
    print(f"  |  LLM Model       : {LLM_MODEL:<38} |")
    print(f"  |  Generation Mode : {GENERATION_MODE:<38} |")
    print(f"  |  Difficulty       : {DIFFICULTY_LEVEL:<37} |")
    print("  +------------------------------------------------------------+")
    print(f"  |  Tiled VAE       : {'ON' if USE_TILED_VAE else 'OFF':<38} |")
    print(f"  |  Chunk FF        : {'ON' if USE_CHUNK_FF else 'OFF':<38} |")
    print(f"  |  Bypass Prompt   : {'ON' if BYPASS_EASY_PROMPT else 'OFF':<38} |")
    print("  +------------------------------------------------------------+")


def validate_config():
    """Check for configuration conflicts and print warnings.

    Uses the vram module's _VRAM_MGR singleton to determine GPU capabilities.
    Returns True if config is valid, prints warnings and returns False otherwise.
    """
    from ltx_pro.vram import _VRAM_MGR

    warnings_found = []

    # Check T4 GPU with Cinema preset
    if _VRAM_MGR.is_t4 and GENERATION_PRESET == "Cinema Quality":
        warnings_found.append(
            "  WARNING: T4 GPU detected with 'Cinema Quality' preset - VRAM may be "
            "insufficient. Consider 'T4 Safe' or 'Standard' preset."
        )

    # Check high resolution + many frames
    pixel_count = WIDTH * HEIGHT * FRAMES
    if pixel_count > 768 * 512 * 161 and _VRAM_MGR.is_low_vram:
        warnings_found.append(
            f"  WARNING: High resolution ({WIDTH}x{HEIGHT}) + {FRAMES} frames may exceed "
            "available VRAM. Consider reducing resolution or frame count."
        )

    # Check segment extension with very high frame count
    if USE_SEGMENT_EXTENSION and FRAMES > 200:
        warnings_found.append(
            f"  WARNING: Segment extension with {FRAMES} frames may produce very long "
            "generation times. Consider FRAMES <= 161 for segment mode."
        )

    # Check incompatible CLIP precision for detected GPU
    if _VRAM_MGR.is_t4 and LLM_MODEL == "14B":
        warnings_found.append(
            "  WARNING: 14B LLM model requires ~18 GB VRAM - incompatible with T4 GPU. "
            "Use 3B or 8B model instead."
        )

    if warnings_found:
        print("  +--- CONFIGURATION WARNINGS ----------------------------+")
        for w in warnings_found:
            print(f"  | {w}")
        print("  +-------------------------------------------------------+")
        return False

    return True


def disable_all_extras():
    """Disable all feature toggles except the master video generation switch."""
    global ENABLE_VIDEO_GENERATION, ENABLE_UPSCALING, ENABLE_AUDIO_GENERATION
    global ENABLE_CHARACTER_SYSTEM, ENABLE_POST_PROCESSING, ENABLE_QUALITY_CHECKS
    global ENABLE_LLM_PROMPT_EXPANSION, ENABLE_VISION_ANALYSIS
    global ENABLE_SCENE_CONTINUITY, ENABLE_SEGMENT_EXTENSION

    ENABLE_VIDEO_GENERATION = True
    ENABLE_UPSCALING = False
    ENABLE_AUDIO_GENERATION = False
    ENABLE_CHARACTER_SYSTEM = False
    ENABLE_POST_PROCESSING = False
    ENABLE_QUALITY_CHECKS = False
    ENABLE_LLM_PROMPT_EXPANSION = False
    ENABLE_VISION_ANALYSIS = False
    ENABLE_SCENE_CONTINUITY = False
    ENABLE_SEGMENT_EXTENSION = False


def enable_recommended():
    """Enable features appropriate for the detected GPU using _VRAM_MGR."""
    from ltx_pro.vram import _VRAM_MGR

    global ENABLE_VIDEO_GENERATION, ENABLE_UPSCALING, ENABLE_AUDIO_GENERATION
    global ENABLE_CHARACTER_SYSTEM, ENABLE_POST_PROCESSING, ENABLE_QUALITY_CHECKS
    global ENABLE_LLM_PROMPT_EXPANSION, ENABLE_VISION_ANALYSIS
    global ENABLE_SCENE_CONTINUITY, ENABLE_SEGMENT_EXTENSION

    ENABLE_VIDEO_GENERATION = True
    ENABLE_SCENE_CONTINUITY = True
    ENABLE_LLM_PROMPT_EXPANSION = True

    if _VRAM_MGR.is_t4:
        # T4: minimal extras to stay within 16 GB
        ENABLE_UPSCALING = False
        ENABLE_AUDIO_GENERATION = False
        ENABLE_CHARACTER_SYSTEM = False
        ENABLE_POST_PROCESSING = False
        ENABLE_QUALITY_CHECKS = False
        ENABLE_VISION_ANALYSIS = False
        ENABLE_SEGMENT_EXTENSION = False
    elif _VRAM_MGR.is_low_vram:
        # L4 / 24 GB: moderate extras
        ENABLE_UPSCALING = True
        ENABLE_AUDIO_GENERATION = True
        ENABLE_CHARACTER_SYSTEM = True
        ENABLE_POST_PROCESSING = True
        ENABLE_QUALITY_CHECKS = False
        ENABLE_VISION_ANALYSIS = True
        ENABLE_SEGMENT_EXTENSION = False
    else:
        # A100+ / 40 GB+: enable everything
        ENABLE_UPSCALING = True
        ENABLE_AUDIO_GENERATION = True
        ENABLE_CHARACTER_SYSTEM = True
        ENABLE_POST_PROCESSING = True
        ENABLE_QUALITY_CHECKS = True
        ENABLE_VISION_ANALYSIS = True
        ENABLE_SEGMENT_EXTENSION = True


def show_help(topic):
    """Print guidance for a given topic.

    Supported topics: presets, resolution, character, quality, performance

    Args:
        topic: One of 'presets', 'resolution', 'character', 'quality', 'performance'
    """
    _help_topics = {
        "presets": (
            "  PRESETS\n"
            "  -------\n"
            "  Quick Draft   : Fastest output, low resolution, good for testing prompts.\n"
            "  Standard      : Balanced quality/speed, works on most GPUs.\n"
            "  Cinema Quality: Highest quality, requires L4/A100 (24+ GB VRAM).\n"
            "  T4 Safe       : Optimized for T4 GPUs (16 GB), enables memory savers."
        ),
        "resolution": (
            "  RESOLUTION\n"
            "  ----------\n"
            "  Portrait 9:16  : 576x1024 - vertical video (mobile, TikTok)\n"
            "  Landscape 16:9 : 768x512  - horizontal (YouTube, standard)\n"
            "  Square 1:1     : 512x512  - social media posts\n"
            "  Widescreen 21:9: 896x384  - cinematic ultrawide\n"
            "  Custom         : Use WIDTH/HEIGHT from Cell 6 directly"
        ),
        "character": (
            "  CHARACTER SYSTEM\n"
            "  ----------------\n"
            "  Maintains character consistency across scenes using multi-frame\n"
            "  anchor extraction and an embedding bank. Enable for storyboard\n"
            "  mode or multi-scene narratives. Adds ~2 GB VRAM overhead."
        ),
        "quality": (
            "  QUALITY CHECKS\n"
            "  ---------------\n"
            "  When enabled, each generated segment is scored for motion coherence,\n"
            "  artifact density, and color consistency. Segments below threshold\n"
            "  are regenerated with seed+1 (up to max retries)."
        ),
        "performance": (
            "  PERFORMANCE TIPS\n"
            "  -----------------\n"
            "  - Enable Tiled VAE and Chunk FF on T4 GPUs\n"
            "  - Use 3B LLM model for fastest prompt expansion\n"
            "  - Reduce FRAMES to 41-97 for quick previews\n"
            "  - Disable unused features with disable_all_extras()\n"
            "  - Use 'T4 Safe' preset for memory-constrained environments"
        ),
    }
    topic_lower = topic.lower().strip()
    if topic_lower in _help_topics:
        print(_help_topics[topic_lower])
    else:
        print(f"  Unknown topic: '{topic}'. Available: {', '.join(_help_topics.keys())}")


def estimate_generation_time():
    """Estimate generation time in seconds based on current settings and GPU.

    Returns estimated seconds as a float. Prints a summary to stdout.
    """
    from ltx_pro.vram import _VRAM_MGR

    # Base time per frame at 768x512 on A100 (approx 0.8s/frame)
    base_time_per_frame = 0.8

    # Scale by resolution relative to baseline (768x512 = 393216 pixels)
    pixel_ratio = (WIDTH * HEIGHT) / 393216.0

    # GPU speed multiplier
    if _VRAM_MGR.is_t4:
        gpu_multiplier = 3.0  # T4 is roughly 3x slower
    elif _VRAM_MGR.is_low_vram:
        gpu_multiplier = 1.5  # L4 is roughly 1.5x slower
    else:
        gpu_multiplier = 1.0  # A100 baseline

    # LLM overhead (one-time)
    llm_overhead = {"3B": 5.0, "8B": 12.0, "14B": 25.0}.get(LLM_MODEL, 10.0)

    estimated = (base_time_per_frame * FRAMES * pixel_ratio * gpu_multiplier) + llm_overhead

    minutes = estimated / 60.0
    print(f"  Estimated generation time: ~{estimated:.0f}s ({minutes:.1f} min)")
    print(f"    [{FRAMES} frames @ {WIDTH}x{HEIGHT} on {_VRAM_MGR.gpu_name}]")

    return estimated


def estimate_vram_usage():
    """Estimate peak VRAM usage in GB for current settings.

    Returns estimated peak VRAM in GB as a float. Prints a summary to stdout.
    """
    from ltx_pro.vram import _VRAM_MGR

    # Base model footprint
    base_model_gb = 4.5  # LTX-Video transformer

    # LLM VRAM
    llm_gb = {"3B": 4.0, "8B": 10.0, "14B": 18.0}.get(LLM_MODEL, 5.0)

    # Resolution/frames scaling for latent space
    # Latents are roughly (frames/8) * (height/32) * (width/32) * channels * dtype_size
    latent_elements = (FRAMES / 8.0) * (HEIGHT / 32.0) * (WIDTH / 32.0) * 128
    latent_gb = (latent_elements * 4) / (1024**3)  # fp32

    # VAE decode VRAM (peaks during decode)
    vae_gb = 2.0 if USE_TILED_VAE else (latent_gb * 2.5)

    # Extras
    extras_gb = 0.0
    if ENABLE_CHARACTER_SYSTEM:
        extras_gb += 1.5
    if ENABLE_VISION_ANALYSIS:
        extras_gb += 2.0
    if ENABLE_UPSCALING:
        extras_gb += 2.5

    # Peak is typically model + LLM (if not offloaded) or model + VAE decode
    peak_with_llm = base_model_gb + llm_gb + latent_gb + extras_gb
    peak_at_decode = base_model_gb + vae_gb + latent_gb + extras_gb
    peak = max(peak_with_llm, peak_at_decode)

    print(f"  Estimated peak VRAM: ~{peak:.1f} GB")
    print(f"    Model: {base_model_gb:.1f} GB | LLM: {llm_gb:.1f} GB | "
          f"Latents: {latent_gb:.1f} GB | VAE: {vae_gb:.1f} GB")
    if extras_gb > 0:
        print(f"    Extras: {extras_gb:.1f} GB (character, vision, upscale)")
    print(f"    Available: {_VRAM_MGR.total_vram_gb:.1f} GB [{_VRAM_MGR.gpu_name}]")

    if peak > _VRAM_MGR.total_vram_gb and _VRAM_MGR.total_vram_gb > 0:
        print(f"  WARNING: Estimated peak ({peak:.1f} GB) exceeds available VRAM!")

    return peak


def suggest_settings(goal):
    """Suggest optimal settings for a given goal.

    Supported goals: "fast preview", "max quality", "long video"

    Args:
        goal: One of 'fast preview', 'max quality', 'long video'
    """
    goal_lower = goal.lower().strip()

    if goal_lower == "fast preview":
        print("  SUGGESTED SETTINGS: Fast Preview")
        print("  ---------------------------------")
        print("  GENERATION_PRESET  = 'Quick Draft'")
        print("  RESOLUTION_PRESET  = 'Square 1:1'")
        print("  GENERATION_MODE    = 'Single Clip'")
        print("  ENABLE_UPSCALING   = False")
        print("  ENABLE_AUDIO       = False")
        print("  TIP: Use BYPASS_EASY_PROMPT=True for instant prompt passthrough")
    elif goal_lower == "max quality":
        print("  SUGGESTED SETTINGS: Max Quality")
        print("  --------------------------------")
        print("  GENERATION_PRESET  = 'Cinema Quality'")
        print("  RESOLUTION_PRESET  = 'Widescreen 21:9'")
        print("  GENERATION_MODE    = 'Single Clip'")
        print("  ENABLE_QUALITY_CHECKS = True")
        print("  ENABLE_CHARACTER_SYSTEM = True")
        print("  TIP: Requires A100/L4 (24+ GB VRAM)")
    elif goal_lower == "long video":
        print("  SUGGESTED SETTINGS: Long Video")
        print("  -------------------------------")
        print("  GENERATION_PRESET  = 'Standard'")
        print("  RESOLUTION_PRESET  = 'Landscape 16:9'")
        print("  GENERATION_MODE    = 'Infinite Flow'")
        print("  ENABLE_SEGMENT_EXTENSION = True")
        print("  ENABLE_SCENE_CONTINUITY  = True")
        print("  TIP: Set FRAMES=161 for each segment, total length via repetitions")
    else:
        print(f"  Unknown goal: '{goal}'. Available: 'fast preview', 'max quality', 'long video'")
