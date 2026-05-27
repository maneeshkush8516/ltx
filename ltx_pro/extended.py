"""
Extended video generation using SVI-Pro-style segment iteration.

Contains FlowState and InfiniteFlowConfig dataclasses for tracking generation
state, generate_extended_video() for multi-segment generation with overlap
blending, and generate_infinite_flow() as the v2.0 wrapper with FlowState.

All heavy imports (torch, numpy, imageio, etc.) are guarded inside function
bodies so this module passes py_compile without those packages installed.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ltx_pro.config import *  # noqa: F401,F403 - populate module globals for globals().get()

__all__ = [
    "FlowState",
    "InfiniteFlowConfig",
    "generate_extended_video",
    "generate_infinite_flow",
]


@dataclass
class FlowState:
    """Tracks all state between segments in infinite flow generation."""
    current_anchor: Optional[str] = None
    embedding_bank: Any = None
    color_histogram: Any = None
    segment_index: int = 0
    total_frames: int = 0
    quality_scores: List[Dict] = field(default_factory=list)
    segment_paths: List[str] = field(default_factory=list)
    prompts_used: List[str] = field(default_factory=list)
    seeds_used: List[int] = field(default_factory=list)


@dataclass
class InfiniteFlowConfig:
    """Groups all extension settings for infinite flow generation."""
    segment_length: int = 81
    max_segments: int = 8
    overlap_frames: int = 5
    overlap_mode: str = "linear_blend"
    overlap_side: str = "source"
    seed_mode: str = "fixed"
    base_seed: int = 2025
    use_prompt_variation: bool = False
    use_camera_variation: bool = False
    continuation_prompt_template: str = ""


def generate_extended_video(
    user_input="",
    image_path="",
    positive_prompt="",
    negative_prompt="",
    width=768,
    height=512,
    fps=25,
    seed=47,
    image_strength=1.0,
    character_image_path=None,
    character_strength=1.0,
    character_mode="both",
    character_name="Character",
    character_description="",
    segment_length=81,
    max_segments=8,
    overlap_frames=5,
    overlap_mode="linear_blend",
    overlap_side="source",
    segment_seed_mode="fixed",
    output_prefix="LTX-2-PRO",
    **kwargs,
):
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

    Returns:
        Final stitched video path, or None on failure.
    """
    import shutil
    import numpy as np
    import torch

    from ltx_pro.pipeline import generate_pro, SVIProContext
    from ltx_pro.overlap import blend_overlap_frames, compute_segment_seeds
    from ltx_pro.vram import aggressive_cleanup, _print_vram
    from ltx_pro.utils import (
        get_last_frame_tensor, tensor_to_pil, load_image_tensor, display_video,
    )
    from ltx_pro.quality import compute_segment_quality
    from ltx_pro.color import extract_color_histogram
    from ltx_pro.motion import (
        estimate_optical_flow, detect_motion_direction,
        auto_select_camera_lora, compute_velocity_latent, compute_adaptive_overlap,
    )
    from ltx_pro.audio import detect_beats, compute_segment_boundaries
    from ltx_pro.export import generate_timeline_json, generate_edl, sync_to_drive
    from ltx_pro.character import CharacterEmbeddingBank
    from ltx_pro.lora import _build_lora_stack

    _g = globals()

    t0 = time.time()
    print("=" * 70)
    print("SVI-Pro Extended Video Generation")
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

    all_segment_paths = []
    current_anchor_path = image_path

    # Initialize advanced features
    USE_EXPORT_TIMELINE = _g.get("EXPORT_TIMELINE", False)
    USE_CHARACTER_EMBEDDING_BANK = _g.get("USE_CHARACTER_EMBEDDING_BANK", False)
    USE_COLOR_MATCHING = _g.get("USE_COLOR_MATCHING", False)
    USE_QUALITY_GATE = _g.get("USE_QUALITY_GATE", False)
    USE_AUDIO_SYNC = _g.get("USE_AUDIO_SYNC", False)
    AUDIO_SYNC_PATH = _g.get("AUDIO_SYNC_PATH", "")
    AUDIO_BPM = _g.get("AUDIO_BPM", 120)
    USE_PERSISTENT_CONTEXT = _g.get("USE_PERSISTENT_CONTEXT", False)
    USE_IDENTITY_REINFORCEMENT = _g.get("USE_IDENTITY_REINFORCEMENT", False)
    USE_ADAPTIVE_OVERLAP = _g.get("USE_ADAPTIVE_OVERLAP", False)
    ADAPTIVE_OVERLAP_MIN = _g.get("ADAPTIVE_OVERLAP_MIN", 3)
    ADAPTIVE_OVERLAP_MAX = _g.get("ADAPTIVE_OVERLAP_MAX", 12)
    USE_MOTION_COHERENCE = _g.get("USE_MOTION_COHERENCE", False)
    USE_VELOCITY_INJECTION = _g.get("USE_VELOCITY_INJECTION", False)
    IC_LORA = _g.get("IC_LORA", "")
    IC_LORA_STRENGTH = _g.get("IC_LORA_STRENGTH", 0.5)
    CAMERA_LORA_STRENGTH = _g.get("CAMERA_LORA_STRENGTH", 0.5)
    QUALITY_GATE_MAX_RETRIES = _g.get("QUALITY_GATE_MAX_RETRIES", 3)
    SHOW_PREVIEWS = _g.get("SHOW_PREVIEWS", False)
    PERSIST_TO_GDRIVE = _g.get("PERSIST_TO_GDRIVE", False)
    GDRIVE_PATH = _g.get("GDRIVE_PATH", "")
    TIMELINE_FORMAT = _g.get("TIMELINE_FORMAT", "json")

    _timeline_entries = [] if USE_EXPORT_TIMELINE else None
    _embedding_bank = CharacterEmbeddingBank() if USE_CHARACTER_EMBEDDING_BANK else None
    _reference_histogram = None

    # Audio sync: adjust segment boundaries
    segment_lengths = [segment_length] * max_segments
    if USE_AUDIO_SYNC and AUDIO_SYNC_PATH:
        try:
            beats = detect_beats(AUDIO_SYNC_PATH, AUDIO_BPM)
            if beats:
                _boundaries = compute_segment_boundaries(beats, fps, segment_length)
                if len(_boundaries) >= 2:
                    segment_lengths = [_boundaries[i+1] - _boundaries[i]
                                       for i in range(len(_boundaries) - 1)]
                    segment_lengths = segment_lengths[:max_segments]
                print(f"   Audio sync: {len(segment_lengths)} segments synced to beats")
        except Exception as e:
            print(f"   Warning: Audio sync failed ({e}) - using fixed segment length")

    # Persistent model context
    _pro_context = None
    if USE_PERSISTENT_CONTEXT:
        try:
            _pro_context = SVIProContext()
            print("   [experimental] Persistent model context enabled")
        except Exception as e:
            print(f"   Warning: Persistent context init failed ({e})")
            _pro_context = None

    # Cache character image tensor for identity reinforcement
    _cached_char_tensor_for_reinforce = None
    if USE_IDENTITY_REINFORCEMENT and character_image_path and os.path.exists(str(character_image_path)):
        try:
            _cached_char_tensor_for_reinforce = load_image_tensor(character_image_path)
        except Exception as e:
            print(f"   Warning: Could not pre-load character image for reinforcement ({e})")

    for seg_idx in range(max_segments):
        seg_num = seg_idx + 1
        print(f"\n{'_' * 50}")
        print(f"   Segment {seg_num}/{max_segments} (seed={seeds[seg_idx]})")

        cached_path = f"{cache_dir}/segment_{seg_idx:02d}.mp4"
        anchor_path = f"{cache_dir}/anchor_{seg_idx:02d}.png"

        if os.path.exists(cached_path) and os.path.exists(anchor_path):
            print(f"   Using cached segment: {cached_path}")
            all_segment_paths.append(cached_path)
            current_anchor_path = anchor_path
            continue

        # Motion coherence & adaptive overlap
        _seg_overlap = overlap_frames
        _seg_lora_stack = None
        _velocity_latent = None

        if seg_idx > 0 and current_anchor_path:
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
                    print(f"   Warning: Previous segment read failed ({e})")
                    _prev_frames_list = None
                    _prev_segment_tensor = None

            if USE_ADAPTIVE_OVERLAP and _prev_segment_tensor is not None:
                try:
                    _seg_overlap = compute_adaptive_overlap(
                        _prev_segment_tensor[-10:], ADAPTIVE_OVERLAP_MIN, ADAPTIVE_OVERLAP_MAX)
                    print(f"   Adaptive overlap: {_seg_overlap} frames")
                except Exception as e:
                    print(f"   Warning: Adaptive overlap failed ({e})")

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
                    print(f"   Warning: Motion coherence failed ({e})")

            if USE_VELOCITY_INJECTION and _prev_segment_tensor is not None and _prev_segment_tensor.shape[0] >= 2:
                try:
                    _velocity_latent = compute_velocity_latent(
                        _prev_segment_tensor[-2], _prev_segment_tensor[-1])
                    _vel_mag = _velocity_latent.abs().mean().item()
                    print(f"   Velocity injection: magnitude={_vel_mag:.4f}")
                except Exception as e:
                    print(f"   Warning: Velocity injection failed ({e})")
                    _velocity_latent = None

        _seg_frames = segment_lengths[seg_idx] if seg_idx < len(segment_lengths) else segment_length

        # Generate this segment
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
            print(f"   Segment {seg_num} failed -- stopping extension.")
            break

        # Quality gate with retry
        if USE_QUALITY_GATE and seg_output:
            try:
                import imageio as _iio
                _reader = _iio.get_reader(seg_output)
                _seg_frames_list = [f for f in _reader]
                _reader.close()
                _seg_tensor = torch.from_numpy(np.stack(_seg_frames_list)).float() / 255.0
                _quality = compute_segment_quality(_seg_tensor)
                if not _quality.get("passed", True):
                    print(f"   Warning: Quality gate failed for segment {seg_num}")
                    for _retry in range(QUALITY_GATE_MAX_RETRIES - 1):
                        _retry_seed = seeds[seg_idx] + _retry + 1
                        print(f"   Retrying with seed={_retry_seed}...")
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
                                print(f"   Quality gate passed on retry {_retry+1}")
                                break
            except Exception as e:
                print(f"   Warning: Quality gate check failed ({e})")

        # Extract reference color histogram from first segment
        if seg_idx == 0 and USE_COLOR_MATCHING and seg_output:
            try:
                import imageio as _iio
                _reader = _iio.get_reader(seg_output)
                _seg_frames_list = [f for f in _reader]
                _reader.close()
                _seg_tensor = torch.from_numpy(np.stack(_seg_frames_list)).float() / 255.0
                _reference_histogram = extract_color_histogram(_seg_tensor)
                print(f"   Reference color histogram extracted from segment 1")
            except Exception as e:
                print(f"   Warning: Reference histogram extraction failed ({e})")

        # Cache the segment
        shutil.copy(seg_output, cached_path)
        all_segment_paths.append(cached_path)

        # Extract last frame as anchor for next segment
        last_frame_tensor = get_last_frame_tensor(cached_path)
        if last_frame_tensor is not None:
            anchor_pil = tensor_to_pil(last_frame_tensor)
            anchor_pil.save(anchor_path, "PNG")
            current_anchor_path = anchor_path
            print(f"   Anchor frame saved: {anchor_path}")

            # Identity reinforcement at segment boundary
            if USE_IDENTITY_REINFORCEMENT and character_image_path and os.path.exists(str(character_image_path)):
                try:
                    if _cached_char_tensor_for_reinforce is not None:
                        _reinforce_strength = character_strength * 0.1
                        _rgen = torch.Generator()
                        _rgen.manual_seed(seed + int(abs(_cached_char_tensor_for_reinforce.mean().item()) * 1000))
                        _pixel_noise = torch.randn_like(last_frame_tensor, generator=_rgen)
                        _reinforced = (1.0 - _reinforce_strength) * last_frame_tensor + _reinforce_strength * _pixel_noise
                        _reinforced = _reinforced.clamp(0.0, 1.0)
                        _reinforced_pil = tensor_to_pil(_reinforced)
                        _reinforced_pil.save(anchor_path, "PNG")
                        print(f"   Identity reinforcement applied to anchor (strength={_reinforce_strength:.3f})")
                except Exception as e:
                    print(f"   Warning: Identity reinforcement failed ({e}) - using original anchor.")
        else:
            print(f"   Warning: Could not extract anchor -- next segment may lack continuity.")

        aggressive_cleanup(f"segment {seg_num} done")

    # Final stitching with overlap blending
    if len(all_segment_paths) < 1:
        print("No segments generated successfully.")
        return None

    if len(all_segment_paths) == 1:
        print(f"Single segment generated: {all_segment_paths[0]}")
        return all_segment_paths[0]

    print(f"\n{'=' * 50}")
    print(f"   Stitching {len(all_segment_paths)} segments with {overlap_frames}-frame {overlap_mode} overlap...")
    _expected_total = sum(segment_lengths[:len(all_segment_paths)]) - max(0, len(all_segment_paths) - 1) * overlap_frames
    print(f"   Expected total frames after overlap subtraction: ~{_expected_total}")

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
            combined_frames = blend_overlap_frames(
                combined_frames, seg_tensor,
                overlap=overlap_frames,
                mode=overlap_mode,
                side=overlap_side
            )

        print(f"   Segment {seg_idx + 1} merged (total frames: {len(combined_frames)})")

    # Save final stitched video
    final_frames_np = (combined_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    final_path = f"/content/ComfyUI/output/{output_prefix}_extended_{int(time.time())}.mp4"
    imageio.mimsave(final_path, [f for f in final_frames_np], fps=fps, codec='libx264')

    elapsed = time.time() - t0
    total_duration = len(combined_frames) / fps

    print(f"\n{'=' * 70}")
    print(f"   EXTENDED VIDEO COMPLETE!")
    print(f"   Output    : {final_path}")
    print(f"   Duration  : {total_duration:.1f}s ({len(combined_frames)} frames @ {fps}fps)")
    print(f"   Segments  : {len(all_segment_paths)}")
    print(f"   Elapsed   : {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    if SHOW_PREVIEWS:
        display_video(final_path)

    # Export timeline
    if USE_EXPORT_TIMELINE and _timeline_entries:
        try:
            _tl_path = f"/content/ComfyUI/output/{output_prefix}_timeline.{TIMELINE_FORMAT}"
            if TIMELINE_FORMAT == "edl":
                generate_edl(_timeline_entries, _tl_path, fps)
            else:
                generate_timeline_json(_timeline_entries, _tl_path)
            print(f"   Timeline exported: {_tl_path}")
        except Exception as e:
            print(f"   Warning: Timeline export failed ({e})")

    # Google Drive final sync
    if PERSIST_TO_GDRIVE and final_path:
        try:
            sync_to_drive(final_path, GDRIVE_PATH)
        except Exception as e:
            print(f"   Warning: Final Drive sync failed ({e})")

    # Cleanup persistent context
    if _pro_context is not None:
        try:
            _pro_context.cleanup()
            print("   Persistent model context released")
        except Exception:
            pass

    return final_path


# Module-level storage for the most recent FlowState
_LAST_FLOW_STATE = None


def generate_infinite_flow(
    user_input=None,
    image_path=None,
    positive_prompt=None,
    negative_prompt=None,
    width=None,
    height=None,
    fps=None,
    seed=None,
    image_strength=None,
    character_image_path=None,
    character_strength=None,
    character_mode=None,
    character_name=None,
    character_description=None,
    config=None,
    flow_state=None,
    output_prefix=None,
    **kwargs,
):
    """
    Generate an infinite-length video using enhanced SVI-Pro-style segment iteration.

    This is the v2.0 replacement for generate_extended_video with FlowState tracking
    and InfiniteFlowConfig support. Adds segment-level prompt variation and
    continuation prompt generation for long videos.

    Args:
        user_input: Story description
        image_path: Starting image path
        config: InfiniteFlowConfig instance (overrides individual settings)
        flow_state: FlowState instance for tracking (created if None)
        **kwargs: Additional arguments passed to generate_pro()

    Returns:
        Final stitched video path, or None on failure.
    """
    global _LAST_FLOW_STATE

    _g = globals()

    # Resolve defaults from globals
    _user_input = user_input if user_input is not None else _g.get('USER_INPUT', '')
    _image_path = image_path if image_path is not None else _g.get('IMAGE_PATH')
    _pos_prompt = positive_prompt if positive_prompt is not None else _g.get('POSITIVE_PROMPT', '')
    _neg_prompt = negative_prompt if negative_prompt is not None else _g.get('NEGATIVE_PROMPT', '')
    _width = width if width is not None else _g.get('WIDTH', 768)
    _height = height if height is not None else _g.get('HEIGHT', 512)
    _fps = fps if fps is not None else _g.get('FPS', 25)
    _seed = seed if seed is not None else _g.get('SEED', 47)
    _img_str = image_strength if image_strength is not None else _g.get('IMAGE_STRENGTH', 1.0)
    _char_img = character_image_path if character_image_path is not None else _g.get('CHARACTER_IMAGE_PATH')
    _char_str = character_strength if character_strength is not None else _g.get('CHARACTER_STRENGTH', 1.0)
    _char_mode = character_mode if character_mode is not None else _g.get('CHARACTER_CONSISTENCY_MODE', 'both')
    _char_name = character_name if character_name is not None else _g.get('CHARACTER_NAME', 'Character')
    _char_desc = character_description if character_description is not None else _g.get('CHARACTER_DESCRIPTION', '')
    _prefix = output_prefix if output_prefix is not None else _g.get('OUTPUT_PREFIX', 'LTX-2-PRO')

    # Use InfiniteFlowConfig if provided
    if config is None:
        config = InfiniteFlowConfig(
            segment_length=_g.get('SEGMENT_LENGTH', 81),
            max_segments=_g.get('MAX_SEGMENTS', 8),
            overlap_frames=_g.get('OVERLAP_FRAMES', 5),
            overlap_mode=_g.get('OVERLAP_MODE', 'linear_blend'),
            overlap_side=_g.get('OVERLAP_SIDE', 'source'),
            seed_mode=_g.get('SEGMENT_SEED_MODE', 'fixed'),
            base_seed=_seed,
        )

    # Initialize FlowState if not provided
    if flow_state is None:
        flow_state = FlowState()

    # Delegate to generate_extended_video with FlowState tracking
    result = generate_extended_video(
        user_input=_user_input,
        image_path=_image_path,
        positive_prompt=_pos_prompt,
        negative_prompt=_neg_prompt,
        width=_width,
        height=_height,
        fps=_fps,
        seed=config.base_seed,
        image_strength=_img_str,
        character_image_path=_char_img,
        character_strength=_char_str,
        character_mode=_char_mode,
        character_name=_char_name,
        character_description=_char_desc,
        segment_length=config.segment_length,
        max_segments=config.max_segments,
        overlap_frames=config.overlap_frames,
        overlap_mode=config.overlap_mode,
        overlap_side=config.overlap_side,
        segment_seed_mode=config.seed_mode,
        output_prefix=_prefix,
        **kwargs,
    )

    # Update FlowState
    if result:
        flow_state.segment_index = config.max_segments
        _cache_dir = f"/content/ComfyUI/output/{_prefix}_segments"
        if os.path.isdir(_cache_dir):
            import glob
            _seg_files = sorted(glob.glob(os.path.join(_cache_dir, "segment_*.mp4")))
            flow_state.segment_paths = _seg_files
            _actual_segments = len(_seg_files) if _seg_files else config.max_segments
            flow_state.total_frames = (
                _actual_segments * config.segment_length
                - max(0, _actual_segments - 1) * config.overlap_frames
            )
        else:
            flow_state.total_frames = config.max_segments * config.segment_length
        flow_state.prompts_used.append(_user_input or _pos_prompt)
        flow_state.seeds_used.append(config.base_seed)

    # Store the FlowState at module level
    _LAST_FLOW_STATE = flow_state

    return result
