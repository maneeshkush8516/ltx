"""
Storyboard and multi-scene sequential generation runner.

Contains the script-to-shot decomposition pipeline, scene breakdown display,
Sora-style JSON export, and the run_storyboard() function for sequential
multi-scene generation with character continuity and retry logic.

All heavy imports (torch, imageio, subprocess, etc.) are guarded inside
function bodies so this module passes py_compile without those packages.
"""

import os
import re
import json
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    import torch

__all__ = [
    "decompose_script_to_scenes",
    "print_scene_breakdown",
    "generate_sora_json",
    "run_storyboard",
]


# ---- Quality/Style Maps ----
_QUALITY_PROMPTS = {
    "8K cinematic": (
        "8K resolution, ultra-detailed, cinematic film grain, shallow depth of field, "
        "professional color grading, anamorphic lens flare, dramatic lighting, "
        "photorealistic, masterful cinematography"
    ),
    "4K professional": (
        "4K resolution, sharp focus, professional lighting, clean composition, "
        "broadcast quality, natural colors, well-exposed"
    ),
    "HD broadcast": (
        "HD resolution, clean image, standard broadcast lighting, "
        "balanced exposure, natural look, steady camera"
    ),
    "social media": (
        "vibrant colors, high contrast, engaging composition, "
        "slightly stylized, eye-catching, trending aesthetic"
    ),
}

_STYLE_PROMPTS = {
    "realistic": "photorealistic, natural lighting, real-world physics, authentic textures",
    "cinematic noir": (
        "high contrast, deep shadows, single-source lighting, film noir aesthetic, "
        "muted colors with selective highlights, venetian blind shadows"
    ),
    "anime": (
        "anime style, cel-shaded, vibrant colors, expressive eyes, dynamic poses, "
        "clean lines, Studio Ghibli inspired"
    ),
    "documentary": (
        "handheld camera feel, natural lighting, observational framing, "
        "raw authentic look, slightly desaturated"
    ),
    "fantasy": (
        "ethereal lighting, magical atmosphere, rich saturated colors, "
        "otherworldly beauty, painterly quality"
    ),
    "sci-fi": (
        "neon accents, holographic displays, sleek surfaces, "
        "futuristic architecture, volumetric fog, cybernetic details"
    ),
}

_CAMERA_ACTION_MAP = {
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
    script,
    target_duration=30,
    segment_duration=5,
    dialogue_interval=15,
    quality="8K cinematic",
    style="realistic",
    language="English",
    character_def="",
    secondary_char="",
    output_format="detailed",
    fps=25,
):
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
    from ltx_pro.character import extract_character_profiles_from_script

    _g = globals()

    if not script.strip():
        print("   Warning: Empty script -- returning empty scene list.")
        return []

    num_segments = max(1, int(target_duration / segment_duration))
    frames_per_segment = int(segment_duration * fps)
    frames_per_segment = min(frames_per_segment, 121)

    print(f"   Planning: {num_segments} segments x {segment_duration}s = ~{num_segments * segment_duration}s")
    print(f"   Frames per segment: {frames_per_segment} @ {fps}fps")

    sentences = [s.strip() for s in re.split(r'[.!?]+', script) if s.strip() and len(s.strip()) > 5]
    if not sentences:
        sentences = [script.strip()]

    scenes_list = []
    sentences_per_segment = max(1, len(sentences) // num_segments)

    quality_prefix = _QUALITY_PROMPTS.get(quality, _QUALITY_PROMPTS["8K cinematic"])
    style_prefix = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["realistic"])

    _mandatory_quality = _g.get("STORY_QUALITY_KEYWORDS",
        "Ultra HDR, 3D intricate details, vibrant colors, realistic lighting, "
        "Dramatic Lighting, Enhanced Clarity, Brilliant Highlights, "
        "Hyperrealistic Detailing, cinematic")

    _extracted_profiles = extract_character_profiles_from_script(script, character_def)
    if _extracted_profiles:
        print(f"   Characters extracted: {list(_extracted_profiles.keys())}")

    char_prefix = ""
    if character_def:
        char_prefix = f"[Main character: {character_def}] "
    if secondary_char:
        char_prefix += f"[Secondary character: {secondary_char}] "

    cumulative_time = 0.0
    last_dialogue_time = 0.0
    dialogue_idx = 0
    num_dialogue_beats = max(1, int(target_duration / dialogue_interval))

    AUTO_CAMERA_SELECT = _g.get('AUTO_CAMERA_SELECT', True)
    CUSTOM_LANGUAGE = _g.get('CUSTOM_LANGUAGE', language)
    _char_image_path = _g.get('CHARACTER_IMAGE_PATH', None)
    _char_mode = _g.get('CHARACTER_CONSISTENCY_MODE', 'both')
    _char_name = _g.get('CHARACTER_NAME', 'Character')
    _char_desc = _g.get('CHARACTER_DESCRIPTION', '')
    _seed = _g.get('SEED', 47)
    SEGMENT_DURATION = _g.get('SEGMENT_DURATION', segment_duration)
    VIDEO_QUALITY = _g.get('VIDEO_QUALITY', quality)
    VIDEO_STYLE = _g.get('VIDEO_STYLE', style)
    VIDEO_LANGUAGE = _g.get('VIDEO_LANGUAGE', language)
    CHARACTER_DEFINITION = _g.get('CHARACTER_DEFINITION', character_def)
    SECONDARY_CHARACTER = _g.get('SECONDARY_CHARACTER', secondary_char)

    for seg_idx in range(num_segments):
        start_sent = seg_idx * sentences_per_segment
        end_sent = min(start_sent + sentences_per_segment, len(sentences))
        if seg_idx == num_segments - 1:
            end_sent = len(sentences)

        seg_sentences = sentences[start_sent:end_sent]
        if not seg_sentences and sentences:
            seg_sentences = [sentences[seg_idx % len(sentences)]]

        action_text = ". ".join(seg_sentences) + "."

        # Camera motion from action verbs
        camera_lora = "static"
        if AUTO_CAMERA_SELECT:
            action_lower = action_text.lower()
            for verb, cam in _CAMERA_ACTION_MAP.items():
                if verb in action_lower:
                    camera_lora = cam
                    break

        # Dialogue beat check
        has_dialogue = False
        dialogue_text = ""
        if cumulative_time >= last_dialogue_time + dialogue_interval or seg_idx == 0:
            has_dialogue = True
            last_dialogue_time = cumulative_time
            dialogue_idx += 1
            if language != "English":
                _lang = CUSTOM_LANGUAGE if language == "Custom" else language
                dialogue_text = f" [Dialogue in {_lang}, natural conversation]"
            else:
                dialogue_text = " [Natural spoken dialogue, realistic lip movements]"

        _llm_input = action_text + dialogue_text
        if seg_idx > 0:
            _llm_input += " Maintain visual continuity with previous shot."

        _scene_ctx = ""
        if character_def:
            _scene_ctx += f"Main character: {character_def}. "
        if secondary_char:
            _scene_ctx += f"Second character: {secondary_char}. "
        _scene_ctx += f"Visual style: {style}. "
        _scene_ctx += f"Camera movement: {camera_lora}. "
        _scene_ctx += f"Quality: {_mandatory_quality}. "
        if has_dialogue and language != "English":
            _scene_ctx += f"Dialogue language: {language}. "

        _profile_ctx = ""
        if _extracted_profiles:
            _profile_ctx = " ".join(
                f"[{name}: {desc}]" for name, desc in _extracted_profiles.items()
            ) + " "

        scene_dict = {
            "user_input": _llm_input + " " + _mandatory_quality,
            "image_path": _char_image_path if seg_idx == 0 else None,
            "frames": frames_per_segment,
            "seed": _seed + seg_idx,
            "output_prefix": f"Scene{seg_idx+1:02d}-{_char_name}",
            "character_image_path": _char_image_path,
            "character_mode": _char_mode,
            "character_name": _char_name,
            "character_description": _profile_ctx + _scene_ctx,
        }

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


def print_scene_breakdown(scenes):
    """Print a formatted breakdown of the generated scene list.

    Args:
        scenes: List of scene dicts from decompose_script_to_scenes().
    """
    if not scenes:
        print("   (no scenes generated)")
        return

    total_frames = sum(s.get("frames", 0) for s in scenes)
    total_duration = sum(s.get("_metadata", {}).get("timestamp_end", 0) -
                        s.get("_metadata", {}).get("timestamp_start", 0) for s in scenes)

    print(f"\n{'_' * 70}")
    print(f"   SCENE BREAKDOWN -- {len(scenes)} shots, ~{total_duration:.0f}s total")
    print(f"{'_' * 70}")

    for i, scene in enumerate(scenes):
        meta = scene.get("_metadata", {})
        ts = meta.get("timestamp_start", 0)
        cam = meta.get("camera_lora", "static")
        has_dlg = "[dlg]" if meta.get("has_dialogue") else "     "
        action = meta.get("action_text", scene.get("user_input", ""))[:60]

        print(f"   [{i+1:2d}] {int(ts//60):02d}:{int(ts%60):02d} | "
              f"cam: {cam:12s} | {has_dlg} | {action}...")

    print(f"{'_' * 70}")
    print(f"   Total: {total_frames} frames | ~{total_duration:.0f}s | "
          f"{len([s for s in scenes if s.get('_metadata',{}).get('has_dialogue')])} dialogue beats")
    print(f"{'_' * 70}\n")


def generate_sora_json(scenes, output_path=None):
    """
    Export scenes as Sora-style JSON with detailed character descriptions.

    Compatible with Sora 2 / Kling / Runway prompt format.
    Each scene includes full character description for standalone consistency.

    Args:
        scenes: List of scene dicts from decompose_script_to_scenes().
        output_path: Optional file path for JSON output.

    Returns:
        JSON string of the exported project.
    """
    _g = globals()
    _fps = _g.get('FPS', 25)
    _width = _g.get('WIDTH', 768)
    _height = _g.get('HEIGHT', 512)
    _char_name = _g.get('CHARACTER_NAME', 'Character')
    _char_desc = _g.get('CHARACTER_DESCRIPTION', '')
    SEGMENT_DURATION = _g.get('SEGMENT_DURATION', 5)
    CHARACTER_DEFINITION = _g.get('CHARACTER_DEFINITION', _char_desc)
    SECONDARY_CHARACTER = _g.get('SECONDARY_CHARACTER', '')
    VIDEO_QUALITY = _g.get('VIDEO_QUALITY', '8K cinematic')
    VIDEO_STYLE = _g.get('VIDEO_STYLE', 'realistic')
    VIDEO_LANGUAGE = _g.get('VIDEO_LANGUAGE', 'English')
    TARGET_VIDEO_DURATION = _g.get('TARGET_VIDEO_DURATION', 30)

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
            "negative_prompt": (
                "blurry, distorted, low quality, watermark, text, bad anatomy, "
                "deformed, flickering, motion artifacts, inconsistent character"
            ),
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
        print(f"   Sora-style JSON exported: {output_path}")

    return result


def run_storyboard(
    scenes,
    use_continuity=True,
    tmp_dir="/content/ComfyUI/input",
    auto_reduce_for_stability=True,
):
    """
    Run a list of scenes sequentially, optionally chaining last-frame continuity.

    Each scene dict supports these keys (all optional except user_input):
      user_input           -- story description for Easy Prompt
      image_path           -- seed image path (overridden by continuity if None)
      frames               -- frame count
      seed                 -- RNG seed
      output_prefix        -- filename prefix
      width / height       -- resolution (defaults to global WIDTH/HEIGHT)
      character_image_path -- character reference
      character_mode       -- 'i2v', 'anchor', 'both', 'none'
      positive_prompt      -- manual prompt (used if BYPASS_EASY_PROMPT=True)
      negative_prompt      -- manual negative

    Args:
        scenes: List of scene dicts defining the storyboard.
        use_continuity: If True, extract last frame of scene N as seed for scene N+1.
        tmp_dir: Directory for temporary continuity frames.
        auto_reduce_for_stability: If True, cap frames in multi-scene mode.

    Returns:
        List of output paths (None for failed scenes).
    """
    import shutil
    import subprocess
    import torch
    import numpy as np

    from ltx_pro.pipeline import generate_pro
    from ltx_pro.vram import aggressive_cleanup
    from ltx_pro.utils import get_last_frame_tensor, tensor_to_pil, display_video
    from ltx_pro.overlap import blend_overlap_frames
    from ltx_pro.prompt_architect import run_easy_prompt
    from ltx_pro.character import extract_continuity_composite, save_continuity_frame
    from ltx_pro.export import generate_thumbnail_frame, display_thumbnail_grid

    _g = globals()

    # Resolve global config
    USE_SCENE_CONTINUITY = _g.get("USE_SCENE_CONTINUITY", use_continuity)
    USE_DUAL_ANCHOR_STORYBOARD = _g.get("USE_DUAL_ANCHOR_STORYBOARD", False)
    USE_PARALLEL_PROMPT_EXPANSION = _g.get("USE_PARALLEL_PROMPT_EXPANSION", False)
    BYPASS_EASY_PROMPT = _g.get("BYPASS_EASY_PROMPT", False)
    GENERATE_THUMBNAILS = _g.get("GENERATE_THUMBNAILS", False)
    THUMBNAIL_COLS = _g.get("THUMBNAIL_COLS", 4)
    CONTINUITY_MULTI_FRAME_COUNT = _g.get("CONTINUITY_MULTI_FRAME_COUNT", 1)
    CONTINUITY_COMPOSITE_MODE = _g.get("CONTINUITY_COMPOSITE_MODE", "last")
    CONTINUITY_FRAME_FORMAT = _g.get("CONTINUITY_FRAME_FORMAT", "png")
    OVERLAP_FRAMES = _g.get("OVERLAP_FRAMES", 5)
    OVERLAP_MODE = _g.get("OVERLAP_MODE", "linear_blend")
    OVERLAP_SIDE = _g.get("OVERLAP_SIDE", "source")
    SHOW_PREVIEWS = _g.get("SHOW_PREVIEWS", False)
    OUTPUT_PREFIX = _g.get("OUTPUT_PREFIX", "LTX-2-PRO")
    FRAMES = _g.get("FRAMES", 81)
    FPS = _g.get("FPS", 25)
    SEED = _g.get("SEED", 47)
    WIDTH = _g.get("WIDTH", 768)
    HEIGHT = _g.get("HEIGHT", 512)
    USER_INPUT = _g.get("USER_INPUT", "")
    POSITIVE_PROMPT = _g.get("POSITIVE_PROMPT", "")
    NEGATIVE_PROMPT = _g.get("NEGATIVE_PROMPT", "")
    IMAGE_STRENGTH = _g.get("IMAGE_STRENGTH", 1.0)
    CHARACTER_IMAGE_PATH = _g.get("CHARACTER_IMAGE_PATH", None)
    CHARACTER_STRENGTH = _g.get("CHARACTER_STRENGTH", 1.0)
    CHARACTER_CONSISTENCY_MODE = _g.get("CHARACTER_CONSISTENCY_MODE", "both")
    CHARACTER_NAME = _g.get("CHARACTER_NAME", "Character")
    CHARACTER_DESCRIPTION = _g.get("CHARACTER_DESCRIPTION", "")

    os.makedirs(tmp_dir, exist_ok=True)
    outputs = []
    prev_output = None

    # Cache directory setup
    cache_dir = f"/content/ComfyUI/output/{scenes[0].get('output_prefix', OUTPUT_PREFIX)}_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print("Storyboard Runner -- Starting")
    print(f"   Scenes    : {len(scenes)}")
    print(f"   Continuity: {use_continuity}")
    print("-" * 70)

    # Resume logic
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
        print(f"   Resuming from scene {start_index + 1} (found {start_index} cached clips)")

    # Parallel Prompt Expansion
    expanded_prompts = {}
    if USE_PARALLEL_PROMPT_EXPANSION and not BYPASS_EASY_PROMPT:
        print("   Parallel prompt expansion (loading LLM once for all scenes)...")
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
            print(f"   Expanded {len(expanded_prompts)} prompts in batch")
            _cache_path = f"{cache_dir}/expanded_prompts.json"
            with open(_cache_path, "w") as f:
                json.dump({str(k): v for k, v in expanded_prompts.items()}, f)
        except Exception as e:
            print(f"   Warning: Parallel expansion failed ({e}) - will expand per-scene")
            expanded_prompts = {}

    # Thumbnail Preview
    if GENERATE_THUMBNAILS:
        print("   Generating thumbnail previews...")
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
            print(f"   Thumbnail grid displayed ({len(_thumbnails)} scenes)")

    # Auto-reduce frames for stability
    if auto_reduce_for_stability and len(scenes) > 3:
        print(f"   Auto-stability: capping frames to 97 for multi-scene mode ({len(scenes)} scenes)")

    storyboard_start = time.time()
    completed_count = 0

    for i, scene in enumerate(scenes):
        if i < start_index:
            continue

        scene_num = i + 1
        print(f"\n   Scene {scene_num}/{len(scenes)}: {scene.get('output_prefix','Scene')}")
        print(f"   Input: {scene.get('user_input','')[:80]}...")

        # Resolve image_path: use continuity frame if available and not explicitly set
        _image_path = scene.get("image_path")
        if use_continuity and prev_output and _image_path is None:
            print(f"   Continuity: extracting frames from scene {scene_num - 1}...")
            try:
                _n_cont_frames = CONTINUITY_MULTI_FRAME_COUNT if CONTINUITY_MULTI_FRAME_COUNT > 1 else 1
                _cont_mode = CONTINUITY_COMPOSITE_MODE
                _cont_fmt = CONTINUITY_FRAME_FORMAT.lower()
                _ext = "png" if _cont_fmt == "png" else "jpg"
                _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.{_ext}")

                _cont_tensor = extract_continuity_composite(
                    prev_output, n_frames=_n_cont_frames, mode=_cont_mode)

                if _cont_tensor is not None:
                    save_continuity_frame(_cont_tensor, _cont_path, format=_cont_fmt)
                    _image_path = _cont_path
                    print(f"   Continuity composite saved ({_n_cont_frames} frames, {_cont_mode}): {_cont_path}")
                else:
                    last_tensor = get_last_frame_tensor(prev_output)
                    if last_tensor is not None:
                        _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.{_ext}")
                        save_continuity_frame(last_tensor, _cont_path, format=_cont_fmt)
                        _image_path = _cont_path
                        print(f"   Continuity frame saved (fallback single): {_cont_path}")
                    else:
                        print(f"   Warning: Could not extract continuity frame - skipping.")
            except Exception as _cont_err:
                print(f"   Warning: Continuity extraction error ({_cont_err}) - trying legacy method.")
                last_tensor = get_last_frame_tensor(prev_output)
                if last_tensor is not None:
                    _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.jpg")
                    pil_frame = tensor_to_pil(last_tensor)
                    pil_frame.save(_cont_path, "JPEG", quality=95)
                    _image_path = _cont_path
                else:
                    print(f"   Warning: Could not extract last frame - skipping continuity.")

            # Dual-Anchor auto-upgrade
            if USE_DUAL_ANCHOR_STORYBOARD and _image_path is not None:
                _scene_char_img = scene.get("character_image_path", CHARACTER_IMAGE_PATH)
                if _scene_char_img:
                    _prev_mode = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE)
                    scene["character_mode"] = "both"
                    if _prev_mode != "both":
                        print(f"   Dual-anchor activated: character_mode '{_prev_mode}' -> 'both'")

        # Determine frames (auto-reduce if needed)
        _frames = scene.get("frames", FRAMES)
        _original_frames = _frames
        if auto_reduce_for_stability and len(scenes) > 3:
            _frames = min(97, _frames)
        if _frames < _original_frames:
            print(f"   Frames capped: {_original_frames} -> {_frames} (auto-stability)")

        # Retry loop
        max_retries = 3
        success = False
        scene_seed = scene.get("seed", SEED)
        _retry_frames = _frames
        _retry_tiled_vae = None
        for attempt in range(max_retries):
            try:
                _gen_kwargs = dict(
                    user_input=scene.get("user_input", USER_INPUT),
                    image_path=_image_path,
                    positive_prompt=scene.get("positive_prompt", POSITIVE_PROMPT),
                    negative_prompt=scene.get("negative_prompt", NEGATIVE_PROMPT),
                    width=scene.get("width", WIDTH),
                    height=scene.get("height", HEIGHT),
                    frames=_retry_frames,
                    fps=scene.get("fps", FPS),
                    seed=scene_seed,
                    image_strength=scene.get("image_strength", IMAGE_STRENGTH),
                    character_image_path=scene.get("character_image_path", CHARACTER_IMAGE_PATH),
                    character_strength=scene.get("character_strength", CHARACTER_STRENGTH),
                    character_mode=scene.get("character_mode", CHARACTER_CONSISTENCY_MODE),
                    character_name=scene.get("character_name", CHARACTER_NAME),
                    character_description=scene.get("character_description", CHARACTER_DESCRIPTION),
                    output_prefix=scene.get("output_prefix", OUTPUT_PREFIX),
                )
                if _retry_tiled_vae is not None:
                    _gen_kwargs["use_tiled_vae"] = _retry_tiled_vae
                # Use pre-expanded prompt if available
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
                print(f"   Scene {scene_num} done -> {out}")
                break
            except torch.cuda.OutOfMemoryError:
                aggressive_cleanup("OOM recovery")
                scene_seed = scene.get("seed", SEED) + attempt + 1
                if attempt >= 0:
                    _retry_frames = max(57, _retry_frames - 24)
                    print(f"   Warning: Reducing frames to {_retry_frames} for next attempt.")
                if attempt >= 1:
                    _retry_tiled_vae = True
                    print(f"   Warning: Forcing tiled VAE for next attempt.")
                print(f"   Warning: OOM on attempt {attempt+1} -- retrying with seed {scene_seed}...")
                if attempt == max_retries - 1:
                    print(f"   Scene {scene_num} failed after {max_retries} attempts")
            except Exception as e:
                print(f"   Scene {scene_num} error: {type(e).__name__}: {e}")
                break

        if not success:
            outputs.append(None)
            prev_output = None

        completed_count += 1

        # Progress tracking
        elapsed = time.time() - storyboard_start
        avg_per_clip = elapsed / completed_count if completed_count > 0 else 0
        remaining = avg_per_clip * (len(scenes) - i - 1)
        print(f"   Elapsed: {elapsed/60:.1f}min | Est. remaining: {remaining/60:.1f}min")

    # Final concatenation with overlap blending
    successful_clips = [p for p in outputs if p]
    if len(successful_clips) >= 2:
        final_path = f"/content/ComfyUI/output/{scenes[0].get('output_prefix', OUTPUT_PREFIX)}_full.mp4"

        if OVERLAP_FRAMES > 0 and OVERLAP_MODE != "hard_cut":
            print(f"   Stitching with {OVERLAP_FRAMES}-frame {OVERLAP_MODE} overlap...")
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
                            print(f"   Final video (overlap-blended + audio): {final_path}")
                        except Exception as _ae:
                            print(f"   Warning: Audio mux failed ({_ae}) -- using video-only.")
                            if os.path.exists(_temp_stitch):
                                os.rename(_temp_stitch, final_path)
                            print(f"   Final video (overlap-blended, no audio): {final_path}")
                    else:
                        os.rename(_temp_stitch, final_path)
                        print(f"   Final video (overlap-blended): {final_path}")
                    concat_result = final_path
                else:
                    concat_result = _concatenate_clips(successful_clips, final_path)
            except Exception as e:
                print(f"   Warning: Overlap blending failed ({e}) -- falling back to hard concat.")
                concat_result = _concatenate_clips(successful_clips, final_path)
        else:
            concat_result = _concatenate_clips(successful_clips, final_path)
            if concat_result:
                print(f"   Final video: {concat_result}")

    # Summary
    print("\n" + "=" * 70)
    print("Storyboard Complete")
    print(f"   Total scenes : {len(scenes)}")
    print(f"   Successful   : {sum(1 for p in outputs if p)}")
    print(f"   Failed       : {sum(1 for p in outputs if not p)}")
    print("\n   Output paths:")
    for i, p in enumerate(outputs):
        status = "[OK]" if p else "[FAIL]"
        print(f"   {status} Scene {i+1}: {p or 'FAILED'}")
    print("=" * 70)

    return outputs


def _concatenate_clips(clip_paths, output_path):
    """Concatenate video clips using ffmpeg concat demuxer.

    Args:
        clip_paths: List of video file paths to concatenate.
        output_path: Destination path for the merged video.

    Returns:
        The output path on success, or the single valid path if less than 2 clips.
    """
    import subprocess

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
        print(f"   Concatenated {len(valid_paths)} clips -> {output_path}")
        return output_path
    except Exception as e:
        print(f"   Warning: Concatenation failed ({e}) -- individual clips still available.")
        return None
