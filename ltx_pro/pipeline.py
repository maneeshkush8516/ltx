"""
Core two-pass generation pipeline for LTX-2 PRO.

Contains the SVIProContext class for persistent model management across segments
and the generate_pro() function which orchestrates the full two-pass generation
pipeline with character consistency, Easy Prompt expansion, and quality gating.

All heavy imports (torch, ComfyUI nodes, etc.) are guarded inside function bodies
so this module passes py_compile without those packages installed.
"""

import gc
import os
import time
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ltx_pro.config import *  # noqa: F401,F403 - populate module globals for globals().get()

__all__ = [
    "SVIProContext",
    "generate_pro",
]


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
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    @property
    def is_loaded(self) -> bool:
        """Check if models are currently loaded."""
        return self._loaded


def generate_pro(
    user_input="",
    image_path="",
    positive_prompt="",
    negative_prompt="",
    width=768,
    height=512,
    frames=81,
    fps=25,
    seed=47,
    image_strength=1.0,
    character_image_path=None,
    character_strength=1.0,
    character_mode="both",
    character_name="Character",
    character_description="",
    pass1_sigmas="",
    pass1_sampler="euler",
    pass1_cfg=1.0,
    pass2_sigmas="",
    pass2_sampler="euler",
    pass2_cfg=1.0,
    pass2_seed=47,
    pro_mode=True,
    pro_steps=10,
    pro_scheduler="simple",
    pro_split_at=4,
    use_tiled_vae=False,
    tiled_spatial_tiles=2,
    tiled_spatial_overlap=64,
    tiled_temporal_len=16,
    tiled_temporal_overlap=4,
    tiled_last_frame_fix=False,
    lora_stack=None,
    lora_stack_json=None,
    output_prefix="LTX-2-PRO",
    bypass_easy_prompt=None,
    llm_model=None,
    use_vision=None,
    vision_model=None,
    unet_model=None,
    clip_name1=None,
    clip_name2=None,
    color_grade=None,
    use_color_matching=None,
    reference_histogram=None,
    use_quality_gate=None,
    use_multi_resolution=None,
    export_timeline=None,
    persist_to_gdrive=None,
    timeline_entries=None,
    embedding_bank=None,
    velocity_latent=None,
):
    """
    LTX-2 PRO -- Two-pass generation pipeline with Character Consistency.

    Integrates nodes from LD-I2V.json + SVI-Pro-Workflow.json.
    All ComfyUI node calls are annotated with [JSON node id / type].

    Pipeline Phases:
        PHASE 0 -- EASY PROMPT (before video model -- LLM/Vision then unload)
            [LTX2VisionDescribe]  image -> scene_context
            [LTX2PromptArchitect] user_input + scene_ctx -> positive, neg

        PHASE 1 -- MODEL LOADING
            [197/UnetLoaderGGUF]     -> unet (raw)
            [190/DualCLIPLoader]     -> clip_model
            [263/LTX2MasterLoaderLD] -> unet + clip (LoRA stack)
            [PathchSageAttentionKJ]  -> unet (sage attn patch, optional)
            [LTXVChunkFeedForward]   -> unet (chunk FF patch, optional)
            [184/VAELoader]          -> vae_video
            [196/VAELoaderKJ]        -> vae_audio
            [189/LatentUpscaleModel] -> upscale_model

        PHASE 2 -- TEXT ENCODING
            [121/CLIPTextEncode]     positive -> cond_pos
            [110/CLIPTextEncode]     negative -> cond_neg
            [ConditioningZeroOut]    cond_pos -> zero_out (for neg branch)
            [107/LTXVConditioning]   fps meta -> cond[0]=pos, cond[1]=neg

        PHASE 3 -- CHARACTER ANCHOR (mode "anchor" or "both")
            [165/ImageResizeKJv2]    char_img -> resized to WxH
            [295/VAEEncode]          pixels   -> anchor_samples LATENT
            SetNode -> "anchor_samples" slot

        PHASE 4 -- LATENT PREPARATION
            [246/ResizeImagesByLongerEdge]  image -> 1536px long-edge
            [165/ImageResizeKJv2]           image -> target WxH, lanczos
            [164/ResizeImageMaskNode]       image -> x0.5 half-res
            [163/GetImageSize]              -> half_w, half_h
            [108/EmptyLTXVLatentVideo]      -> vid_lat (half-res)
            [162/LTXVPreprocess]            img_compression=33 -> pp_img
            [161/LTXVImgToVideoInplace]     I2V mode -> vid_lat (conditioned)
            [199/LTXVEmptyLatentAudio]      -> aud_lat
            [109/LTXVConcatAVLatent]        vid_lat+aud_lat -> combined

        PHASE 5 -- SIGMA SCHEDULE
            Standard: [ManualSigmas] pass1_sigmas -> sig_p1
            PRO mode: [ModelSamplingSD3 shift=8] + [BasicScheduler] +
                      [SplitSigmas step=pro_split_at] -> sig_high, sig_low
                      [KSamplerSelect euler] -> sampler

        PHASE 6 -- PASS 1 (first-pass denoising)
            [CFGGuider]  model + pos/neg -> guider_p1
            [RandomNoise] seed -> noise_p1
            [SamplerCustomAdvanced] -> p1_av_output

        PHASE 7 -- PASS 2 (spatial upscale + refinement)
            [LTXVSeparateAVLatent]  p1_av -> vid_lat_p1, aud_lat_p1
            [LTXVCropGuides]        pos/neg + lat -> cropped cond + lat
            [CFGGuider]  model + cropped -> guider_p2
            [LTXVLatentUpsampler]   x2 upsample -> upsampled
            [LTXVConcatAVLatent]    upsampled + aud -> av_lat2
            PRO: sig_low  |  Standard: [ManualSigmas] pass2_sigmas
            [SamplerCustomAdvanced] -> p2_denoised

        PHASE 8 -- DECODE
            [LTXVSeparateAVLatent]
            [265/LTXVSpatioTemporalTiledVAEDecode] or [VAEDecode] -> frames
            [201/LTXVAudioVAEDecode] -> audio

        PHASE 9 -- SAVE
            [319/VHS_VideoCombine] h264-mp4, crf=19, yuv420p (preferred)
            Fallback: [CreateVideo] -> save_video_from_components
            save_metadata_sidecar() -> JSON sidecar

    Returns:
        Output video path (str) or None on failure.
    """
    # -- Heavy imports inside function body --
    import torch
    from nodes import NODE_CLASS_MAPPINGS
    from ltx_pro.utils import (
        get_value_at_index, load_image_tensor, get_last_frame_tensor,
        display_video, save_video_from_components, save_metadata_sidecar,
        validate_model_files,
    )
    from ltx_pro.vram import (
        vram_guard, cleanup_memory, aggressive_cleanup,
        force_unload_all_models, purge_vram, _print_vram, _verify_vram_clear,
        _VRAM_MGR,
    )
    from ltx_pro.lora import apply_lora_stack, apply_sage_attention, apply_chunk_ff, _load_audio_vae
    from ltx_pro.prompt_architect import run_easy_prompt
    from ltx_pro.vision_describe import run_vision_describe
    from ltx_pro.character import (
        CharacterPromptAnchor, PersistentLatentSeed, CharacterEmbeddingBank,
    )
    from ltx_pro.quality import (
        compute_segment_quality, detect_shot_type, get_resolution_for_shot,
    )
    from ltx_pro.color import match_color_histogram, apply_color_grade
    from ltx_pro.export import sync_to_drive

    try:
        from IPython.display import clear_output
    except ImportError:
        def clear_output(wait=False):
            pass

    try:
        from google.colab import files
    except ImportError:
        files = None

    def import_custom_nodes():
        """Load all built-in and external custom nodes in a Jupyter/Colab-safe way."""
        import asyncio
        import nest_asyncio
        from nodes import init_builtin_extra_nodes, init_external_custom_nodes

        async def _load():
            failed = await init_builtin_extra_nodes()
            await init_external_custom_nodes()
            if failed:
                print(f"   Warning: Some nodes failed: {[str(n) for n in failed]}")
        try:
            asyncio.run(_load())
        except RuntimeError:
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(_load())

    t0 = time.time()
    import_custom_nodes()
    clear_output()

    # -- Resolve defaults from module globals --
    _g = globals()
    _lora_stack = lora_stack if lora_stack is not None else _g.get("LORA_STACK", [])
    _lora_json = lora_stack_json if lora_stack_json is not None else _g.get("LORA_STACK_JSON", None)
    _char_mode = character_mode.lower().strip()

    _bypass = bypass_easy_prompt if bypass_easy_prompt is not None else _g.get("BYPASS_EASY_PROMPT", False)
    _llm_model = llm_model if llm_model is not None else _g.get("LLM_MODEL", "")
    _use_vision = use_vision if use_vision is not None else _g.get("USE_VISION", False)
    _vis_model = vision_model if vision_model is not None else _g.get("VISION_MODEL", "")
    _unet = unet_model if unet_model is not None else _g.get("UNET_MODEL", "")
    _clip1 = clip_name1 if clip_name1 is not None else _g.get("CLIP_NAME1", "")
    _clip2 = clip_name2 if clip_name2 is not None else _g.get("CLIP_NAME2", "")

    _color_grade = color_grade if color_grade is not None else _g.get("COLOR_GRADE", "none")
    _use_color_matching = use_color_matching if use_color_matching is not None else _g.get("USE_COLOR_MATCHING", False)
    _use_quality_gate = use_quality_gate if use_quality_gate is not None else _g.get("USE_QUALITY_GATE", False)
    _use_multi_res = use_multi_resolution if use_multi_resolution is not None else _g.get("USE_MULTI_RESOLUTION", False)
    _export_timeline = export_timeline if export_timeline is not None else _g.get("EXPORT_TIMELINE", False)
    _persist_to_gdrive = persist_to_gdrive if persist_to_gdrive is not None else _g.get("PERSIST_TO_GDRIVE", False)

    # Feature toggles
    _enable_video = _g.get("ENABLE_VIDEO_GENERATION", True)
    _enable_upscaling = _g.get("ENABLE_UPSCALING", True)
    _enable_audio = _g.get("ENABLE_AUDIO_GENERATION", True)
    _enable_character = _g.get("ENABLE_CHARACTER_SYSTEM", True)
    _enable_post_processing = _g.get("ENABLE_POST_PROCESSING", True)
    _enable_quality_checks = _g.get("ENABLE_QUALITY_CHECKS", True)

    if not _g.get("ENABLE_LLM_PROMPT_EXPANSION", True):
        _bypass = True
    if not _g.get("ENABLE_VISION_ANALYSIS", True):
        _use_vision = False
    if not _enable_quality_checks:
        _use_quality_gate = False

    # VRAMManager optimal settings
    _vram_settings = _VRAM_MGR.get_optimal_settings()
    if _VRAM_MGR.is_t4 or _VRAM_MGR.is_low_vram:
        _vram_max_frames = _vram_settings.get("max_frames", frames)
        if frames > _vram_max_frames:
            frames = _vram_max_frames
        if use_tiled_vae is None or use_tiled_vae == _g.get("USE_TILED_VAE", False):
            if _vram_settings.get("use_tiled_vae", False):
                use_tiled_vae = True

    print("LTX-2 PRO -- Generation Starting")
    print(f"   Resolution   : {width}x{height}  |  Frames: {frames}  |  Seed: {seed}")
    print(f"   Mode         : {'I2V' if image_path else 'T2V'}"
          f"  |  Character: {_char_mode}  |  Pro: {pro_mode}")
    print(f"   Easy Prompt  : {'BYPASS' if _bypass else f'LLM={_llm_model}'}")
    _print_vram()

    # -- Multi-resolution override --
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
            print(f"   Warning: Multi-resolution failed ({e}) - using original resolution")

    # -- Pre-flight model check --
    print("\n Model file check...")
    all_ok = validate_model_files({
        "unet": _unet,
        "clip1": _clip1,
        "clip2": _clip2,
        "vae_vid": _g.get("VAE_VIDEO_MODEL", ""),
        "vae_aud": _g.get("VAE_AUDIO_MODEL", ""),
        "upscaler": _g.get("UPSCALER_MODEL", ""),
    })
    if not all_ok:
        raise FileNotFoundError(
            "One or more model files are missing -- run Cell 2 first.\n"
            "  Tip: Check CLIP_NAME1 -- use fp8 if fp4 is not available for your GPU."
        )

    # ================================================================
    # PHASE 0 -- EASY PROMPT (LLM + Vision before video model loads)
    # ================================================================

    seed_image_tensor = None
    if image_path:
        seed_image_tensor = load_image_tensor(image_path)
        if seed_image_tensor is None:
            print(f"   Warning: Image not found: {image_path} -- switching to T2V")
        else:
            print(f"   Reference image loaded: {image_path}  {seed_image_tensor.shape}")

    char_image_tensor = None
    if character_image_path and _char_mode != "none" and _enable_character:
        char_image_tensor = load_image_tensor(character_image_path)
        if char_image_tensor is None:
            print(f"   Warning: Character image not found: {character_image_path} -- skipping anchor.")
        else:
            print(f"   Character image loaded: {character_image_path}  {char_image_tensor.shape}")
    elif not _enable_character and character_image_path and _char_mode != "none":
        print("   [SKIPPED] Character system disabled via ENABLE_CHARACTER_SYSTEM=False")

    analysis_tensor = char_image_tensor if char_image_tensor is not None else seed_image_tensor

    # Vision Describe
    scene_context = character_description or ""
    if analysis_tensor is not None and _use_vision and not _bypass:
        print("\n   Vision Describe...")
        scene_context = run_vision_describe(
            analysis_tensor,
            character_description,
            use_vision_override=_use_vision,
            vision_model_override=_vis_model)
        if scene_context:
            print(f"   Scene context: {scene_context[:120]}...")

    # Easy Prompt expansion
    final_positive = positive_prompt
    final_negative = negative_prompt
    if not _bypass and user_input.strip():
        print("\n   Easy Prompt expansion...")
        final_positive, final_negative = run_easy_prompt(
            user_input=user_input,
            frame_count=frames,
            seed=seed,
            scene_context=scene_context,
            llm_model_override=_llm_model,
        )
        print(f"\n   EXPANDED PROMPT:")
        print(f"   {final_positive[:300]}{'...' if len(final_positive) > 300 else ''}")
        print(f"\n   NEGATIVE PROMPT:")
        print(f"   {final_negative[:150]}...")
    else:
        print("   [EasyPrompt] Bypassed -- using manual POSITIVE_PROMPT.")

    # Feature Toggle: Video Generation Master Switch
    if not _enable_video:
        print("\n" + "=" * 70)
        print("   [FEATURE TOGGLE] Video generation DISABLED")
        print("   Prompt expansion complete. Returning expanded prompt only.")
        print("=" * 70)
        print(f"\n   Final positive prompt:\n   {final_positive[:500]}")
        print(f"\n   Final negative prompt:\n   {final_negative[:200]}")
        return None

    # Aggressive VRAM cleanup before CLIP
    print("\n   Aggressive VRAM cleanup (pre-CLIP)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    aggressive_cleanup("pre-CLIP cleanup")
    _VRAM_MGR.release_stage("Vision")
    _VRAM_MGR.release_stage("LLM")

    try:
        _verify_vram_clear("post-Vision-unload", max_allowed_gb=2.0 if _VRAM_MGR.is_t4 else 3.0)
    except RuntimeError as e:
        print(f"   WARNING: {e}")
        force_unload_all_models()
        cleanup_memory(force=True)
        aggressive_cleanup("post-Vision retry")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _allocated = torch.cuda.memory_allocated() / 1024**3
        if _allocated > 2.0:
            print(f"   WARNING: {_allocated:.1f} GB still allocated! Running force_unload...")
            force_unload_all_models()
            cleanup_memory(force=True)
            aggressive_cleanup("pre-CLIP retry")
            _allocated = torch.cuda.memory_allocated() / 1024**3
            if _allocated > 2.0:
                print(f"   CRITICAL: {_allocated:.1f} GB still in use after force cleanup!")

    _print_vram()

    try:
        _verify_vram_clear("pre-CLIP-load", max_allowed_gb=2.0 if _VRAM_MGR.is_t4 else 4.0)
    except RuntimeError as e:
        print(f"   WARNING: {e}")

    if _VRAM_MGR.is_t4:
        _VRAM_MGR.enforce_sequential_loading("CLIP", 10.0)
    else:
        _VRAM_MGR.enforce_sequential_loading("CLIP", 8.0)

    _VRAM_MGR.register_stage("CLIP", 10.0 if _VRAM_MGR.is_t4 else 8.0)

    # ================================================================
    # PHASE 1A - TEXT ENCODING (CLIP loaded, used, then freed before UNet)
    # ================================================================

    with torch.inference_mode():

        # DualCLIPLoader
        print("\n   Loading CLIP encoders (DualCLIPLoader)...")
        try:
            clip_loader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            clip_model = get_value_at_index(
                clip_loader.load_clip(
                    clip_name1=_clip1,
                    clip_name2=_clip2,
                    type="ltxv",
                    device="default"), 0)
        except Exception as e:
            print(f"   Warning: fp4 CLIP failed ({type(e).__name__}: {e})")
            print("      Trying fp8 fallback (gemma_3_12B_it_fp4_mixed.safetensors)...")
            fp8 = "gemma_3_12B_it_fp4_mixed.safetensors"
            try:
                clip_model = get_value_at_index(
                    clip_loader.load_clip(
                        clip_name1=fp8, clip_name2=_clip2,
                        type="ltxv", device="default"), 0)
                print("   fp8 CLIP loaded.")
            except Exception as e2:
                raise RuntimeError(
                    f"DualCLIPLoader failed: {e2}\n"
                    "  Fix: Ensure CLIP_NAME1 file is downloaded (Cell 2).\n"
                    "  fp4 needs Blackwell GPU; use fp8 for T4/A100."
                )

        # Character Prompt Anchor (apply profile-based prefix)
        CHARACTER_EXTRACTED_PROFILE = _g.get("CHARACTER_EXTRACTED_PROFILE", None)
        if _char_mode in ("anchor", "both") and (character_description or CHARACTER_EXTRACTED_PROFILE):
            _anchor = CharacterPromptAnchor(
                name=character_name,
                description=character_description,
                enabled=True,
                profile=CHARACTER_EXTRACTED_PROFILE if CHARACTER_EXTRACTED_PROFILE else None
            )
            final_positive = _anchor.anchor_prompt(final_positive)
            print(f"   Character prompt anchor applied (profile={'yes' if CHARACTER_EXTRACTED_PROFILE else 'no'})")

        # Text Encoding
        print("\n   Encoding prompts...")
        try:
            cte = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cond_pos = cte.encode(text=final_positive, clip=clip_model)
            cond_neg = cte.encode(text=final_negative, clip=clip_model)
            zero_out = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_zero = zero_out.zero_out(
                conditioning=get_value_at_index(cond_pos, 0))
            ltxv_cond = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cond = ltxv_cond.EXECUTE_NORMALIZED(
                frame_rate=float(fps),
                positive=get_value_at_index(cond_pos, 0),
                negative=get_value_at_index(cond_zero, 0))
        except Exception as e:
            raise RuntimeError(
                f"Text encoding failed: {e}\n"
                "  Fix: Check DualCLIPLoader output - CLIP may have failed to load."
            )

        del clip_model
        aggressive_cleanup("CLIP deleted")

        if _VRAM_MGR.is_t4:
            _VRAM_MGR.enforce_sequential_loading("UNet", 6.0)

        # ================================================================
        # PHASE 1B - UNET LOADING (after CLIP is freed from VRAM)
        # ================================================================

        print("\n   Loading UNet (GGUF Q4_K_M distilled)...")
        try:
            unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unet = get_value_at_index(
                unet_loader.load_unet(unet_name=_unet), 0)
        except KeyError:
            raise RuntimeError(
                "UnetLoaderGGUF not found.\n"
                "  Fix: Run Cell 1 to clone ComfyUI_GGUF custom node."
            )

        _VRAM_MGR.register_stage("UNet", 6.0 if _VRAM_MGR.is_t4 else 8.0)

        # LoRA stack
        print("   Applying LoRA stack (LTX2MasterLoaderLD)...")
        unet, _ = apply_lora_stack(unet, None, _lora_stack, _lora_json)

        # Optional performance patches
        unet = apply_sage_attention(unet)
        unet = apply_chunk_ff(unet)
        purge_vram("after unet+lora")
        _print_vram()

        # ================================================================
        # PHASE 4 -- LATENT PREPARATION
        # ================================================================

        print("\n   Preparing latents...")
        _print_vram()

        # Dual-Anchor Logic
        _ref_tensor = seed_image_tensor
        _use_i2v = False

        if _char_mode == "both" and seed_image_tensor is not None and char_image_tensor is not None:
            _use_i2v = True
            print(f"   Dual-anchor: continuity frame -> I2V | character image -> anchor latent")
        elif _char_mode == "both" and seed_image_tensor is None and char_image_tensor is not None:
            _ref_tensor = char_image_tensor
            _use_i2v = True
        elif _char_mode == "i2v":
            if seed_image_tensor is None and char_image_tensor is not None:
                _ref_tensor = char_image_tensor
            _use_i2v = (_ref_tensor is not None)
        elif _char_mode == "anchor":
            _ref_tensor = None
            _use_i2v = False
        elif _char_mode == "none" and seed_image_tensor is not None:
            _use_i2v = True
        else:
            _ref_tensor = None
            _use_i2v = False

        # Compute half-resolution for Pass 1 latent
        ei = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_img = ei.generate(width=width, height=height, batch_size=1, color=0)

        rimn = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_img = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_img, 0),
            scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})

        gis = NODE_CLASS_MAPPINGS["GetImageSize"]()
        half_sz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_img, 0))
        half_w = get_value_at_index(half_sz, 0)
        half_h = get_value_at_index(half_sz, 1)
        print(f"   Latent dims : {half_w}x{half_h}  (half of {width}x{height})")

        # Character Anchor encoding (Phase 3, deferred for half-res dims)
        VAE_VIDEO_MODEL = _g.get("VAE_VIDEO_MODEL", "")
        anchor_latent = None
        if char_image_tensor is not None and _char_mode in ("anchor", "both") and _enable_character:
            print("\n   Character Anchor - encoding character image as latent...")
            try:
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

                vae_for_anchor = get_value_at_index(
                    NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

                vae_enc = NODE_CLASS_MAPPINGS["VAEEncode"]()
                anchor_latent = get_value_at_index(
                    vae_enc.encode(pixels=char_resized, vae=vae_for_anchor), 0)
                del vae_for_anchor
                aggressive_cleanup("VAE anchor done")
                print(f"   Character anchor encoded at {half_w}x{half_h}  (mode={_char_mode})")
            except Exception as e:
                print(f"   Warning: Character anchor failed ({e}) - continuing without anchor.")
                anchor_latent = None

        # EmptyLTXVLatentVideo -- half-res video latent
        eltxv = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=half_w, height=half_h, length=frames, batch_size=1)

        # I2V conditioning branch
        if _use_i2v and _ref_tensor is not None:
            try:
                _rim_i2v = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
                _ref_tensor = get_value_at_index(
                    _rim_i2v.EXECUTE_NORMALIZED(
                        input=_ref_tensor,
                        scale_method="lanczos",
                        resize_type={"resize_type": "scale dimensions",
                                     "width": width, "height": height,
                                     "crop": "center"}), 0)

                if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                    rle = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
                    _ref_tensor = get_value_at_index(
                        rle.EXECUTE_NORMALIZED(
                            longer_edge=max(width, height),
                            images=_ref_tensor), 0)

                pp_node = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
                pp_img = get_value_at_index(
                    pp_node.EXECUTE_NORMALIZED(
                        img_compression=33,
                        image=_ref_tensor), 0)

                vae_for_i2v = get_value_at_index(
                    NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

                _i2v_strength = character_strength if _char_mode in ("i2v", "both") \
                                else image_strength
                i2v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
                vid_lat = i2v.EXECUTE_NORMALIZED(
                    strength=_i2v_strength,
                    bypass=False,
                    vae=vae_for_i2v,
                    image=pp_img,
                    latent=get_value_at_index(vid_lat, 0))
                del vae_for_i2v
                aggressive_cleanup("VAE I2V done")
                print(f"   I2V conditioning applied  (strength={_i2v_strength}, "
                      f"LTXVImgToVideoInplace)")
            except KeyError as e:
                print(f"   Warning: I2V node missing ({e}) - using empty latent (T2V mode).")
                vid_lat = (get_value_at_index(vid_lat, 0),)
            except Exception as e:
                print(f"   Warning: I2V conditioning failed ({e}) - using empty latent.")
                vid_lat = (get_value_at_index(vid_lat, 0),)
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Anchor latent injection
        USE_DUAL_ANCHOR_STORYBOARD = _g.get("USE_DUAL_ANCHOR_STORYBOARD", False)
        ANCHOR_BLEND_FRAMES = _g.get("ANCHOR_BLEND_FRAMES", 8)
        LATENT_OVERLAP_STRENGTH = _g.get("LATENT_OVERLAP_STRENGTH", 0.5)

        _vid_lat_input = get_value_at_index(vid_lat, 0)
        if anchor_latent is not None:
            try:
                _anch_samples = anchor_latent.get("samples", None)
                _vid_samples = _vid_lat_input.get("samples", None) if isinstance(_vid_lat_input, dict) else None

                if _anch_samples is not None and _vid_samples is not None and \
                   USE_DUAL_ANCHOR_STORYBOARD and _char_mode == "both":
                    if _anch_samples.ndim == 4:
                        _anch_samples = _anch_samples.unsqueeze(2)

                    _total_t = _vid_samples.shape[2] if _vid_samples.ndim == 5 else 1
                    _K = min(ANCHOR_BLEND_FRAMES, _total_t)
                    _anchor_repeated = _anch_samples.repeat(1, 1, _K, 1, 1)

                    if _vid_samples.ndim == 5 and _anchor_repeated.shape[3:] == _vid_samples[:, :, :_K, :, :].shape[3:]:
                        _strength = float(LATENT_OVERLAP_STRENGTH)
                        _vid_samples = _vid_samples.clone()
                        _vid_samples[:, :, :_K, :, :] = (
                            (1.0 - _strength) * _vid_samples[:, :, :_K, :, :] +
                            _strength * _anchor_repeated
                        )
                        _vid_lat_input = {**_vid_lat_input, "samples": _vid_samples}
                        print(f"   Dual-anchor blend: {_strength:.0%} anchor into first {_K} frames  (mode={_char_mode})")
                    else:
                        print(f"   Warning: Anchor spatial dims don't match video latent - SKIPPING anchor injection.")
                        print(f"     Anchor: {list(_anchor_repeated.shape[3:])}, Video: {list(_vid_samples[:, :, :_K, :, :].shape[3:])}")
                else:
                    _vid_lat_input = anchor_latent
                    print(f"   Character anchor injected as video latent seed  (mode={_char_mode})")

                _anch_shape = anchor_latent.get("samples", torch.empty(0)).shape
                print(f"     Anchor latent shape: {list(_anch_shape)}")
                if len(_anch_shape) == 4 and not (USE_DUAL_ANCHOR_STORYBOARD and _char_mode == "both"):
                    print("     Warning: Anchor is 4D (image latent) - unsqueezing T dim for video latent.")
                    _s = anchor_latent["samples"].unsqueeze(2)
                    _vid_lat_input = {**anchor_latent, "samples": _s}
                    print(f"     Anchor latent shape after fix: {list(_s.shape)}")
            except Exception as e:
                print(f"   Warning: Anchor injection error ({e}) - using empty/I2V latent.")
                _vid_lat_input = get_value_at_index(vid_lat, 0)

        # PersistentLatentSeed: blend identity noise into first N frames
        OVERLAP_FRAMES_CHARACTER = _g.get("OVERLAP_FRAMES_CHARACTER", 8)
        if char_image_tensor is not None and _char_mode in ("anchor", "both") and _enable_character:
            try:
                _persistent_seed = PersistentLatentSeed(
                    reference_image=char_image_tensor,
                    seed=seed,
                    strength=character_strength * 0.15
                )
                _vid_samples = _vid_lat_input.get("samples", None) if isinstance(_vid_lat_input, dict) else None
                if _vid_samples is not None and _vid_samples.ndim == 5:
                    _blended = _persistent_seed.blend_into_latent(
                        _vid_samples, num_frames=OVERLAP_FRAMES_CHARACTER)
                    _vid_lat_input = {**_vid_lat_input, "samples": _blended}
                    print(f"   PersistentLatentSeed: identity noise blended into first "
                          f"{OVERLAP_FRAMES_CHARACTER} frames (strength={_persistent_seed.strength:.3f})")
            except Exception as e:
                print(f"   Warning: PersistentLatentSeed blend failed ({e}) - continuing without identity noise.")

        # Audio latent
        VAE_AUDIO_MODEL = _g.get("VAE_AUDIO_MODEL", "")
        vae_audio = None
        try:
            vae_audio = get_value_at_index(_load_audio_vae(VAE_AUDIO_MODEL), 0)
        except Exception as e:
            raise RuntimeError(
                f"Audio VAE load failed: {e}\n"
                "  Fix: Check VAE_AUDIO_MODEL filename in Cell 6."
            )

        elalat = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1,
            audio_vae=vae_audio)

        # LTXVConcatAVLatent -- combine video + audio latents
        catav = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat1 = catav.EXECUTE_NORMALIZED(
            video_latent=_vid_lat_input,
            audio_latent=get_value_at_index(aud_lat, 0))
        combined_latent = get_value_at_index(av_lat1, 0)

        # ================================================================
        # PHASE 5 -- SIGMA SCHEDULE
        # ================================================================

        manualsigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        sig_p1_high = None
        sig_p2_low = None

        if pro_mode:
            print(f"\n   PRO sigma schedule -- BasicScheduler steps={pro_steps} "
                  f"sched={pro_scheduler} split@{pro_split_at}")
            print("   EXPERIMENTAL: SVI-Pro sigma chain was designed for Wan2.2.")
            print("      ModelSamplingSD3 applies SD3 cosine flow parameterisation.")
            print("      LTX-2 GGUF uses a different flow schedule -- output quality")
            print("      may vary. Use ManualSigmas mode (PRO_MODE=False) if results")
            print("      are degraded or distorted.")
            try:
                ms3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
                unet_sampled = get_value_at_index(
                    ms3.patch(model=unet, shift=8.0), 0)

                bs = NODE_CLASS_MAPPINGS["BasicScheduler"]()
                sigs = get_value_at_index(
                    bs.get_sigmas(
                        model=unet_sampled,
                        scheduler=pro_scheduler,
                        steps=pro_steps,
                        denoise=1.0), 0)

                ss = NODE_CLASS_MAPPINGS["SplitSigmas"]()
                split_out = ss.get_sigmas(sigmas=sigs, step=pro_split_at)
                sig_p1_high = get_value_at_index(split_out, 0)
                sig_p2_low = get_value_at_index(split_out, 1)

                unet = unet_sampled
                sampler_p1 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
                sampler_p2 = sampler_p1

                print(f"   PRO sigmas computed (ModelSamplingSD3 + BasicScheduler + SplitSigmas)")

            except KeyError as e:
                print(f"   Warning: PRO mode node missing: {e} -- falling back to ManualSigmas.")
                pro_mode = False
            except Exception as e:
                print(f"   Warning: PRO schedule failed ({e}) -- falling back to ManualSigmas.")
                pro_mode = False

        if not pro_mode:
            print(f"\n   Standard sigma schedule -- Pass1: {pass1_sigmas[:45]}...")
            sig_p1_high = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0)
            sampler_p1 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)
            sig_p2_low = get_value_at_index(
                manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0)
            sampler_p2 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)

        # ================================================================
        # PHASE 6 -- PASS 1 (first-pass denoising)
        # ================================================================

        print(f"\n   Pass 1 -- denoising...")
        _print_vram()

        noise_p1 = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
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
            p1_av = get_value_at_index(out1, 0)
        except Exception as e:
            if vae_audio is not None:
                del vae_audio
                vae_audio = None
            raise RuntimeError(
                f"Pass 1 sampling failed: {e}\n"
                "  Fix: If you see 'deformed output', try a different SEED.\n"
                "  Three consecutive deformations -> change USER_INPUT/POSITIVE_PROMPT."
            )

        del guider_p1
        aggressive_cleanup("Pass 1 done")
        print("   Pass 1 complete")

        # ================================================================
        # PHASE 7 -- PASS 2 (spatial upscale + refinement)
        # ================================================================

        if _enable_upscaling:
            print(f"\n   Pass 2 -- upscale + refinement...")
            _print_vram()

            ltxvsep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            s1 = ltxvsep.EXECUTE_NORMALIZED(av_latent=p1_av)
            vid_lat_p1 = get_value_at_index(s1, 0)
            aud_lat_p1 = get_value_at_index(s1, 1)

            ltxvcrop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
            cropped = ltxvcrop.EXECUTE_NORMALIZED(
                positive=get_value_at_index(cond, 0),
                negative=get_value_at_index(cond, 1),
                latent=vid_lat_p1)

            guider_p2 = cfgguider.EXECUTE_NORMALIZED(
                cfg=pass2_cfg,
                model=unet,
                positive=get_value_at_index(cropped, 0),
                negative=get_value_at_index(cropped, 1))

            vae_for_up = get_value_at_index(
                NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

            UPSCALER_MODEL = _g.get("UPSCALER_MODEL", "")
            try:
                uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
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

            ltxvup = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
            upsampled = ltxvup.upsample_latent(
                samples=get_value_at_index(cropped, 2),
                upscale_model=upscale_model,
                vae=vae_for_up)
            del vae_for_up, upscale_model
            aggressive_cleanup("VAE upscale done")

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
                if vae_audio is not None:
                    del vae_audio
                    vae_audio = None
                raise RuntimeError(
                    f"Pass 2 sampling failed: {e}\n"
                    "  Fix: Try reducing TILED_SPATIAL_TILES or USE_TILED_VAE=False."
                )

            del guider_p2, unet
            aggressive_cleanup("Pass 2 done - UNet freed")
            print("   Pass 2 complete")
        else:
            print("\n   [SKIPPED] Phase 7 - Upscaling disabled via ENABLE_UPSCALING=False")
            print("   Decoding directly from Pass 1 output.")
            p2_denoised = p1_av
            ltxvsep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            del unet
            aggressive_cleanup("Pass 2 skipped - UNet freed")

        # ================================================================
        # PHASE 8 -- DECODE
        # ================================================================

        print("\n   Decoding video & audio...")
        _print_vram()

        s2 = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        # Video decode
        vae_for_decode = get_value_at_index(
            NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=VAE_VIDEO_MODEL), 0)

        _VRAM_MGR.register_stage("VAE", 2.0)

        decoded_frames = None
        if use_tiled_vae:
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
                print("   Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode)")
            except (KeyError, Exception) as e:
                print(f"   Warning: Tiled VAE unavailable ({type(e).__name__}: {e})")
                print("      Falling back to standard VAEDecode.")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_frames is None:
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_for_decode), 0)
            print("   Standard VAE decode (VAEDecode)")

        del vae_for_decode
        aggressive_cleanup("VAE decode done")

        # Audio decode
        audio_out = None
        if _enable_audio:
            try:
                aud_dec = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
                audio_out = aud_dec.EXECUTE_NORMALIZED(
                    samples=aud_lat_fin,
                    audio_vae=vae_audio)
            except Exception as e:
                print(f"   Warning: Audio decode failed ({e}) - proceeding without audio.")
                audio_out = None
        else:
            print("   [SKIPPED] Audio decode disabled via ENABLE_AUDIO_GENERATION=False")

        try:
            del vae_audio
        except NameError:
            pass
        aggressive_cleanup("audio VAE done")

        # POST-PROCESSING: Color matching & grading
        if decoded_frames is not None and _enable_post_processing:
            if _use_color_matching and reference_histogram is not None:
                try:
                    decoded_frames = match_color_histogram(decoded_frames, reference_histogram)
                    print("   Color histogram matching applied")
                except Exception as e:
                    print(f"   Warning: Color matching failed ({e}) - continuing without.")

            if _color_grade and _color_grade != "none":
                try:
                    decoded_frames = apply_color_grade(decoded_frames, _color_grade)
                    print(f"   Color grade applied: {_color_grade}")
                except Exception as e:
                    print(f"   Warning: Color grading failed ({e}) - continuing without.")
        elif decoded_frames is not None and not _enable_post_processing:
            print("   [SKIPPED] Post-processing disabled via ENABLE_POST_PROCESSING=False")

        # Character embedding bank accumulation
        if decoded_frames is not None and embedding_bank is not None and _enable_character:
            try:
                _raw_frame = decoded_frames[-1:]
                if _raw_frame.ndim >= 3:
                    _feat = _raw_frame.float().mean(dim=-2).mean(dim=-2)
                else:
                    _feat = _raw_frame.float()
                embedding_bank.accumulate(_feat)
                print(f"   Character embedding bank updated ({len(embedding_bank)} samples)")
            except Exception as e:
                print(f"   Warning: Embedding bank update failed ({e})")

        # ================================================================
        # PHASE 9 -- SAVE (VHS_VideoCombine preferred, CreateVideo fallback)
        # ================================================================

        print("\n   Saving video...")
        _print_vram()
        output_path = None

        _prefix = output_prefix
        if character_name and character_name != "Character":
            _prefix = f"{output_prefix}-{character_name}"

        # Try VHS_VideoCombine first
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS and audio_out is not None:
            try:
                vhs = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
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
                    print(f"   Warning: VHS_FILENAMES type={type(_fnames)} -- "
                          f"cannot extract path, falling back to CreateVideo.")
                    output_path = None
                if output_path:
                    print(f"   Saved via VHS_VideoCombine (h264-mp4, crf=19, yuv420p)")

            except Exception as e:
                print(f"   Warning: VHS_VideoCombine failed ({e}) -- using CreateVideo fallback.")
                output_path = None

        # Fallback: CreateVideo
        if output_path is None:
            try:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                _aud_arg = get_value_at_index(audio_out, 0) if audio_out else None
                if _aud_arg is not None:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(
                        fps=fps, images=decoded_frames, audio=_aud_arg)
                else:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(
                        fps=fps, images=decoded_frames)
                output_path = save_video_from_components(
                    get_value_at_index(vid_obj, 0), prefix=_prefix)
                print(f"   Saved via CreateVideo fallback")
            except Exception as e:
                raise RuntimeError(
                    f"Video save failed: {e}\n"
                    "  Fix: Check /content/ComfyUI/output/ permissions.\n"
                    "  Also try: VHS_VideoCombine node may need ComfyUI-VideoHelperSuite."
                )

    # Timing
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    # Quality gate check
    quality_scores = None
    if _use_quality_gate and decoded_frames is not None:
        try:
            quality_scores = compute_segment_quality(decoded_frames)
            if quality_scores.get("passed"):
                print(f"   Quality gate PASSED (SSIM={quality_scores.get('ssim', 0):.3f}, "
                      f"hist={quality_scores.get('histogram', 0):.3f})")
            else:
                print(f"   Warning: Quality gate FAILED (SSIM={quality_scores.get('ssim', 0):.3f}, "
                      f"hist={quality_scores.get('histogram', 0):.3f})")
        except Exception as e:
            print(f"   Warning: Quality gate check failed ({e})")

    # Metadata JSON sidecar
    _active_loras = [s["lora"] for s in _lora_stack if s.get("on")]
    meta = {
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "positive_prompt": final_positive,
        "negative_prompt": final_negative,
        "user_input": user_input,
        "image_path": image_path,
        "character_image": character_image_path,
        "character_mode": _char_mode,
        "character_name": character_name,
        "character_strength": character_strength,
        "pro_mode": pro_mode,
        "pro_steps": pro_steps if pro_mode else None,
        "pro_scheduler": pro_scheduler if pro_mode else None,
        "loras": _active_loras,
        "unet_model": _g.get("UNET_MODEL", ""),
        "elapsed_seconds": elapsed,
        "output_path": output_path,
        "quality_scores": quality_scores,
    }
    if output_path:
        save_metadata_sidecar(output_path, meta)

    # Google Drive sync
    GDRIVE_PATH = _g.get("GDRIVE_PATH", "")
    if _persist_to_gdrive and output_path:
        try:
            sync_to_drive(output_path, GDRIVE_PATH)
        except Exception as e:
            print(f"   Warning: Drive sync failed ({e})")

    # Timeline entry
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
            print(f"   Warning: Timeline entry failed ({e})")

    print(f"\n   Done in {mins}m {secs}s")
    print(f"   Output: {output_path}")
    _print_vram()

    # Preview & download
    SHOW_PREVIEWS = _g.get("SHOW_PREVIEWS", False)
    DOWNLOAD_AFTER_GENERATE = _g.get("DOWNLOAD_AFTER_GENERATE", False)
    if SHOW_PREVIEWS and output_path:
        print("\n   Preview:")
        display_video(output_path)

    if DOWNLOAD_AFTER_GENERATE and output_path and files is not None:
        print("   Auto-downloading...")
        try:
            files.download(output_path)
        except Exception as e:
            print(f"   Warning: Download failed ({e}) -- file is at {output_path}")

    return output_path
