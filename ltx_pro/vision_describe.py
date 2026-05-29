"""
Inline Vision Describer for image-to-text description.

Contains InlineVisionDescribe class that uses Qwen2.5-VL models for
image analysis and description. Also provides singleton accessor and
the run_vision_describe() wrapper for ComfyUI node calls.

All heavy imports (torch, transformers, numpy, PIL, qwen_vl_utils) are
guarded inside function bodies so this module passes py_compile without
those packages.
"""

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image

__all__ = [
    "InlineVisionDescribe",
    "_get_inline_vision_describer",
    "run_vision_describe",
]

class InlineVisionDescribe:
    """
    Embedded vision description logic from LTX2VisionEasyPromptLD.py.
    Used as fallback when the ComfyUI LTX2VisionDescribe node is not available.

    Loads Qwen2.5-VL models for image-to-text description with proper
    VRAM management (load -> describe -> unload pattern).
    """

    DESCRIBE_PROMPT = (
        "Describe this image in one paragraph of plain sentences, around 100-130 words. "
        "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
        "Then describe the person -- your FIRST sentence about the person MUST explicitly state "
        "their ethnicity and skin tone using plain terms. "
        "Then continue with their age, hair colour and style, body type, "
        "what they are wearing or doing, and any exposed body parts. "
        "Describe their pose, what they are on or interacting with, "
        "the camera framing and angle, the lighting and time of day, and the setting. "
        "Write it as one flowing paragraph. Do not use bullet points, lists, or labels. "
        "If there is no person in the image, describe the scene instead."
    )

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }

    def __init__(self):
        self.processor = None
        self.model = None
        self.source = None

    def _load_model(self, model_key="3B-fast", offline_mode=False, local_path=""):
        """Load vision model."""
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        hf_id = self.MODEL_OPTIONS.get(model_key, self.MODEL_OPTIONS["3B-fast"])
        source = local_path.strip() if local_path and local_path.strip() else hf_id

        if not offline_mode and not (local_path and local_path.strip()):
            try:
                from huggingface_hub import snapshot_download
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id

        if self.model is not None and self.source == source:
            return

        self._unload()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=offline_mode)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, device_map="auto", torch_dtype=dtype, local_files_only=offline_mode)
        self.model.eval()
        self.source = source

    def _unload(self):
        """Unload model and free VRAM."""
        if self.model is not None:
            from ltx_pro.vram import _deep_unload_model
            _deep_unload_model(self.model, label="InlineVisionDescribe")
        self.model = None
        self.processor = None
        self.source = None

    def describe(self, image_tensor, model_key="3B-fast", offline_mode=False, local_path=""):
        """
        Describe an image tensor and return a text description.

        Args:
            image_tensor: ComfyUI NHWC tensor or PIL Image
            model_key: "3B-fast" or "7B-nsfw"
            offline_mode: Use cached models only
            local_path: Path to local model snapshot

        Returns:
            String description of the image
        """
        import torch
        import numpy as np
        from PIL import Image

        self._load_model(model_key, offline_mode, local_path)

        # Convert tensor to PIL
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.ndim == 4:
                image_tensor = image_tensor[0]
            arr = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(arr, "RGB")
        else:
            pil_image = image_tensor

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            self._unload()
            return ""

        messages = [
            {"role": "system", "content": "You are an image description tool for an AI video pipeline."},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": self.DESCRIBE_PROMPT},
            ]},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        tok = self.processor.tokenizer
        stop_ids = []
        if tok.eos_token_id is not None:
            stop_ids.append(tok.eos_token_id)
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=180, temperature=0.3,
                do_sample=True, top_p=0.9, pad_token_id=pad_id,
                eos_token_id=stop_ids)

        new_tokens = out[0][input_len:]
        description = tok.decode(new_tokens, skip_special_tokens=True).strip()
        del out, inputs

        self._unload()
        return description



# Global inline vision describer instance (lazy-loaded)
_INLINE_VISION_DESCRIBER = None

def _get_inline_vision_describer():
    """Get or create the singleton InlineVisionDescribe instance."""
    global _INLINE_VISION_DESCRIBER
    if _INLINE_VISION_DESCRIBER is None:
        _INLINE_VISION_DESCRIBER = InlineVisionDescribe()
    return _INLINE_VISION_DESCRIBER


# ══════════════════════════════════════════════════════════════════════════════

_VISION_LABEL_MAP = {
    "3B-fast": "Qwen2.5-VL-3B — Fast (huihui abliterated)",
    "7B-nsfw": "Qwen2.5-VL-7B — Better NSFW (prithiv caption)",
}

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
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}
    from ltx_pro.vram import cleanup_memory

    USE_VISION = globals().get("USE_VISION", True)
    VISION_MODEL = globals().get("VISION_MODEL", "3B-fast")
    VISION_OFFLINE_MODE = globals().get("VISION_OFFLINE_MODE", False)
    VISION_LOCAL_PATH = globals().get("VISION_LOCAL_PATH", "")

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




