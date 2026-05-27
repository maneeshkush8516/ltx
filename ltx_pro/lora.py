"""
LoRA stack management and performance patches for the LTX-2 pipeline.

Contains IC LoRA and Camera LoRA filename lookups, stack building helpers,
LoRA application via LTX2MasterLoaderLD or manual fallback, SageAttention
and ChunkFeedForward performance patches, and audio VAE loader.

All heavy imports (torch, comfy, nodes) are guarded inside function bodies
so this module passes py_compile without those packages.
"""

import os
import json
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "_IC_LORA_FILES",
    "_CAMERA_LORA_FILES",
    "_build_lora_stack",
    "apply_lora_stack",
    "apply_sage_attention",
    "apply_chunk_ff",
    "_load_audio_vae",
]

def apply_sage_attention(unet):
    """
    Apply PathchSageAttentionKJ (KJNodes) for flash-attention-style speedup.
    Node: PathchSageAttentionKJ from ComfyUI_KJNodes.
    Falls back silently if node is not available.
    """
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}
    from ltx_pro.utils import get_value_at_index
    USE_SAGE_ATTENTION = globals().get("USE_SAGE_ATTENTION", False)
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
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}
    from ltx_pro.utils import get_value_at_index
    USE_CHUNK_FF = globals().get("USE_CHUNK_FF", False)
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


def _load_audio_vae(vae_name: str):
    """Load audio VAE. Prefers VAELoaderKJ (main_device, fp16), falls back to VAELoader."""
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=vae_name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)




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
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}
    from ltx_pro.utils import get_value_at_index

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
            from nodes import LoraLoaderModelOnly
            ll   = LoraLoaderModelOnly()
            unet = ll.load_lora_model_only(unet, name, strength)[0]
            print(f"      ✓ {name} @ {strength}")
        except Exception as e:
            if guard:
                print(f"      ⚠️  {name} skipped (guard): {e}")
            else:
                print(f"      ❌ {name} failed: {e}")

    return unet, clip_model



