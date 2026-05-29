"""
VRAM management utilities for GPU memory optimization.

Provides the VRAMManager class for auto-detecting GPU type and selecting
optimal generation strategies, cleanup functions for releasing VRAM between
model loading phases, and the vram_guard decorator for OOM retry logic.

All heavy imports (torch, transformers, accelerate) are guarded inside
function bodies so this module passes py_compile without those packages.
"""

import gc
import time
import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "VRAMManager",
    "vram_guard",
    "cleanup_memory",
    "aggressive_cleanup",
    "force_unload_all_models",
    "purge_vram",
    "_deep_unload_model",
    "_print_vram",
    "_verify_vram_clear",
    "_VRAM_MGR",
]


class VRAMManager:
    """
    Auto-detects GPU type and selects optimal generation strategies.

    For T4 (< 16GB): forces tiled VAE, chunk FF, 3B model, frame cap 97, fp8 CLIP.
    For L4 (< 24GB): standard settings with optional tiled VAE.
    For A100+ (>= 24GB): full quality, no restrictions.
    """

    def __init__(self):
        """Initialize VRAMManager and detect GPU capabilities."""
        self.total_vram_gb = 0.0
        self.gpu_name = "unknown"
        self.is_t4 = False
        self.is_low_vram = False
        # Stage tracking: maps stage name -> estimated GB loaded (FEAT-002)
        self._loaded_stages = {}
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect GPU and set flags."""
        try:
            import torch as _torch
        except ImportError:
            self.total_vram_gb = 0.0
            self.gpu_name = "CPU"
            self.is_t4 = True
            self.is_low_vram = True
            return

        if _torch.cuda.is_available():
            self.gpu_name = _torch.cuda.get_device_name(0)
            self.total_vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.is_t4 = self.total_vram_gb < 16
            self.is_low_vram = self.total_vram_gb < 24
        else:
            self.total_vram_gb = 0.0
            self.gpu_name = "CPU"
            self.is_t4 = True
            self.is_low_vram = True

    def get_optimal_settings(self):
        """Return dict of optimal settings for detected GPU."""
        if self.is_t4:
            return {
                "use_tiled_vae": True,
                "use_chunk_ff": True,
                "llm_model": "3B",
                "max_frames": 97,
                "clip_precision": "fp4",
                "clip_name": "gemma_3_12B_it_fp4_mixed.safetensors",
                "width": 768,
                "height": 512,
            }
        elif self.is_low_vram:
            return {
                "use_tiled_vae": False,
                "use_chunk_ff": False,
                "llm_model": "8B",
                "max_frames": 161,
                "clip_precision": "fp4",
                "clip_name": "ggemma_3_12B_it_fp4_mixed.safetensors",
                "width": 1024,
                "height": 576,
            }
        else:
            return {
                "use_tiled_vae": False,
                "use_chunk_ff": False,
                "llm_model": "14B",
                "max_frames": 241,
                "clip_precision": "fp4",
                "clip_name": "gemma_3_12B_it_fp4_mixed.safetensors",
                "width": 1280,
                "height": 720,
            }

    def get_available_vram(self):
        """Return currently available VRAM in GB (driver-level free memory)."""
        try:
            import torch as _torch
        except ImportError:
            return 0.0
        if not _torch.cuda.is_available():
            return 0.0
        free_bytes, _ = _torch.cuda.mem_get_info()
        return free_bytes / 1024**3

    def can_fit(self, estimated_gb):
        """Check if an operation requiring estimated_gb will fit.

        Args:
            estimated_gb: Estimated VRAM requirement in gigabytes.

        Returns:
            True if available VRAM is at least 110% of estimated_gb.
        """
        available = self.get_available_vram()
        return available >= estimated_gb * 1.1  # 10% safety margin

    def print_status(self):
        """Print current VRAM status."""
        try:
            import torch as _torch
        except ImportError:
            print("   No GPU available")
            return
        if not _torch.cuda.is_available():
            print("   No GPU available")
            return
        used = _torch.cuda.memory_allocated() / 1024**3
        pct = used / self.total_vram_gb * 100
        print(f"   VRAM: {used:.1f}/{self.total_vram_gb:.1f} GB ({pct:.0f}%) [{self.gpu_name}]")

    def register_stage(self, name, estimated_gb):
        """Register a model stage as loaded in VRAM with its estimated size.

        Called when a model (LLM, Vision, CLIP, UNet, VAE) is loaded so the
        manager knows exactly what is consuming VRAM at any given time.

        Args:
            name: Stage identifier (e.g. 'LLM', 'Vision', 'CLIP', 'UNet', 'VAE')
            estimated_gb: Approximate VRAM footprint in gigabytes
        """
        self._loaded_stages[name] = estimated_gb
        print(f"   [VRAMManager] Registered stage: {name} (~{estimated_gb:.1f} GB)")

    def release_stage(self, name):
        """Mark a model stage as released/unloaded from VRAM.

        Called after a model is unloaded so the manager stops tracking it.

        Args:
            name: Stage identifier previously passed to register_stage()
        """
        if name in self._loaded_stages:
            del self._loaded_stages[name]
            print(f"   [VRAMManager] Released stage: {name}")

    def suggest_offload_order(self):
        """Return stages to unload based on priority (LLM first, Vision, CLIP, UNet last).

        The priority order ensures that the largest/least-needed models are
        freed first, keeping the UNet (most expensive to reload) loaded as
        long as possible.

        Returns:
            List of stage names in suggested unload order
        """
        # Priority: LLM > Vision > VAE > CLIP > UNet (UNet is last to free)
        priority = ["LLM", "Vision", "VAE", "CLIP", "UNet"]
        loaded = list(self._loaded_stages.keys())
        ordered = [s for s in priority if s in loaded]
        # Append any unknown stages at the end
        ordered += [s for s in loaded if s not in ordered]
        return ordered

    def enforce_sequential_loading(self, stage, required_gb):
        """Ensure enough VRAM is free for the next stage. Force-unload if needed.

        Args:
            stage: Name of the stage about to be loaded.
            required_gb: How much VRAM the stage needs in gigabytes.

        Raises:
            RuntimeError: If VRAM cannot be freed sufficiently after cleanup.
        """
        try:
            import torch as _torch
        except ImportError:
            return
        if not _torch.cuda.is_available():
            return
        available = self.get_available_vram()
        if available >= required_gb:
            return

        print(f"   [{stage}] Need {required_gb:.1f} GB, only {available:.1f} GB free. Force-unloading...")
        force_unload_all_models()
        aggressive_cleanup(f"{stage} pre-load")

        available = self.get_available_vram()
        if available < required_gb:
            # Second attempt
            time.sleep(0.5)
            gc.collect()
            gc.collect()
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            available = self.get_available_vram()

        if available < required_gb:
            raise RuntimeError(
                f"[{stage}] Cannot free enough VRAM. Need {required_gb:.1f} GB, "
                f"only {available:.1f} GB available after force cleanup. "
                f"Total VRAM: {self.total_vram_gb:.1f} GB [{self.gpu_name}].\n"
                f"  Fix: Ensure previous models are fully unloaded before loading {stage}."
            )
        print(f"   [{stage}] OK: {available:.1f} GB free (need {required_gb:.1f} GB)")


def vram_guard(func):
    """Decorator that catches OOM errors, clears cache, and retries once.

    Wraps a function so that if a CUDA OutOfMemoryError is raised, it
    performs garbage collection, empties the CUDA cache, and retries the
    function call exactly once.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch as _torch
        except ImportError:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except _torch.cuda.OutOfMemoryError:
            print(f"   [vram_guard] OOM in {func.__name__}, clearing cache and retrying...")
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            gc.collect()
            return func(*args, **kwargs)
    return wrapper


def cleanup_memory(verbose=False, force=False):
    """Enhanced memory cleanup including ipc_collect for fragmentation.

    Args:
        verbose: If True, print VRAM status after cleanup.
        force: If True, perform an extra sleep + gc + empty_cache pass.
    """
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
        if force:
            time.sleep(0.1)
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        if verbose:
            _print_vram()
    except ImportError:
        pass


def aggressive_cleanup(label=""):
    """Aggressive VRAM cleanup: triple gc + synchronize + empty_cache + ipc_collect.

    Args:
        label: Optional label printed alongside the VRAM status bar.
    """
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        import torch as _torch
    except ImportError:
        return

    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
        _torch.cuda.ipc_collect()
    # Only sleep if VRAM is still high after first gc pass
    if _torch.cuda.is_available() and _torch.cuda.memory_allocated() / 1024**3 > 1.0:
        time.sleep(0.1)
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    if label:
        _print_vram(label)
    # Warn if VRAM is still unexpectedly high
    if _torch.cuda.is_available():
        allocated_gb = _torch.cuda.memory_allocated() / 1024**3
        if allocated_gb > 2.0:
            print(f"   WARNING: {allocated_gb:.1f} GB still allocated after cleanup [{label}]")


def _deep_unload_model(model, label=""):
    """Shared helper: deeply unload a HuggingFace model from CUDA.

    Removes accelerate dispatch hooks, moves all parameters/buffers to CPU,
    runs triple gc, synchronize, empty_cache, ipc_collect, and prints verification.

    Args:
        model: A HuggingFace PreTrainedModel instance (or None).
        label: Optional label for logging the unload status.
    """
    if model is None:
        return

    # Remove accelerate dispatch hooks (they hold CUDA tensor references)
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass

    # Delete _hf_hook if present
    try:
        if hasattr(model, '_hf_hook'):
            del model._hf_hook
    except Exception:
        pass

    # Explicitly move all parameters and buffers to CPU
    try:
        for p in model.parameters():
            if p.device.type == 'cuda':
                p.data = p.data.cpu()
                if p.grad is not None:
                    p.grad = p.grad.cpu()
        for b in model.buffers():
            if b.device.type == 'cuda':
                b.data = b.data.cpu()
    except Exception:
        pass

    try:
        model.to("cpu")
    except Exception:
        pass

    # Triple gc + CUDA cleanup
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
            # Only sleep if VRAM is still high
            if _torch.cuda.memory_allocated() / 1024**3 > 1.0:
                time.sleep(0.1)
            gc.collect()
            _torch.cuda.empty_cache()
            allocated_mb = _torch.cuda.memory_allocated() / 1024**2
            if label:
                print(f"   [{label}] Unloaded. VRAM allocated: {allocated_mb:.0f} MB")
    except ImportError:
        pass


def force_unload_all_models():
    """Nuclear option: walk all Python objects and forcibly unload any HF model on CUDA.

    Iterates through all live Python objects, identifies HuggingFace
    PreTrainedModel instances with parameters on CUDA, removes their
    accelerate hooks, and moves all parameters/buffers to CPU.
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    try:
        import torch as _torch
    except ImportError:
        return

    unloaded = 0
    for obj in gc.get_objects():
        if isinstance(obj, PreTrainedModel):
            try:
                # Check if any parameter is on CUDA
                has_cuda = any(p.device.type == 'cuda' for p in obj.parameters())
                if not has_cuda:
                    continue
                # Remove accelerate hooks
                try:
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(obj, recurse=True)
                except Exception:
                    pass
                # Move all params to CPU
                for p in obj.parameters():
                    if p.device.type == 'cuda':
                        p.data = p.data.cpu()
                        if p.grad is not None:
                            p.grad = p.grad.cpu()
                for b in obj.buffers():
                    if b.device.type == 'cuda':
                        b.data = b.data.cpu()
                unloaded += 1
            except Exception:
                pass

    if unloaded > 0:
        print(f"   force_unload_all_models: moved {unloaded} model(s) to CPU")
    gc.collect()
    gc.collect()
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
        _torch.cuda.ipc_collect()


def _print_vram(label=""):
    """Print VRAM usage with a visual bar indicator.

    Args:
        label: Optional label appended to the output line.
    """
    try:
        import torch as _torch
    except ImportError:
        return
    if not _torch.cuda.is_available():
        return
    used = _torch.cuda.memory_allocated() / 1024**3
    total = _torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = used / total * 100 if total > 0 else 0
    filled = int(20 * used / total) if total > 0 else 0
    bar = "#" * filled + "." * (20 - filled)
    tag = f" [{label}]" if label else ""
    print(f"   VRAM [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%){tag}")


def purge_vram(label=""):
    """Purge VRAM after model loading phases.

    Tries LayerUtility: PurgeVRAM V2 ComfyUI node first, then falls back
    to torch.cuda.empty_cache().

    Args:
        label: Optional label for logging.
    """
    from ltx_pro.config import PURGE_VRAM_AFTER_MODELS
    if not PURGE_VRAM_AFTER_MODELS:
        return
    tag = f" [{label}]" if label else ""

    try:
        from nodes import NODE_CLASS_MAPPINGS
        if "LayerUtility: PurgeVRAM V2" in NODE_CLASS_MAPPINGS:
            try:
                node = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM V2"]()
                fn = getattr(node, node.FUNCTION)
                fn(anything="", purge_cache=True, purge_models=True)
                print(f"   Purged VRAM via PurgeVRAM V2{tag}")
                return
            except Exception as e:
                print(f"   PurgeVRAM V2 failed ({e}) - using torch fallback.")
    except ImportError:
        pass

    cleanup_memory()
    print(f"   VRAM cleared via torch.cuda.empty_cache{tag}")


def _verify_vram_clear(label, max_allowed_gb=1.0):
    """Verify that VRAM is sufficiently cleared between model loading phases.

    Called between model transitions in generate_pro() to ensure the previous
    model is fully unloaded before loading the next one.

    Args:
        label: Description of the checkpoint (e.g. 'post-Vision', 'pre-CLIP')
        max_allowed_gb: Maximum acceptable allocated VRAM in GB (default 1.0)

    Raises:
        RuntimeError: If allocated VRAM exceeds max_allowed_gb.
    """
    try:
        import torch as _torch
    except ImportError:
        return
    if not _torch.cuda.is_available():
        return
    _torch.cuda.synchronize()
    _torch.cuda.empty_cache()
    allocated_gb = _torch.cuda.memory_allocated() / 1024**3
    if allocated_gb > max_allowed_gb:
        raise RuntimeError(
            f"[VRAM CHECK FAILED] {label}: {allocated_gb:.2f} GB allocated, "
            f"max allowed is {max_allowed_gb:.1f} GB. "
            f"Previous model may not have been fully unloaded."
        )
    print(f"   [VRAM OK] {label}: {allocated_gb:.2f} GB allocated (limit: {max_allowed_gb:.1f} GB)")


# Module-level singleton instance
_VRAM_MGR = VRAMManager()
