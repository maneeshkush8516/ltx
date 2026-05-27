"""
LTX-2 Infinite Flow Engine v2.0 - Modular Package.

Provides AI video generation pipeline components extracted from the monolithic
LTX_PRO.py notebook, organized into importable submodules for maintainability
and reuse.

Submodules:
    config  - Presets, feature toggles, generation defaults, helper functions
    vram    - VRAMManager, cleanup utilities, OOM guard decorator
    utils   - Tensor/image conversion, file I/O, ComfyUI node helpers
"""

__version__ = "2.0.0"

__all__ = [
    "config",
    "vram",
    "utils",
]
