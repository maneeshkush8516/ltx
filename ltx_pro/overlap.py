"""
Overlap and segment extension helpers for multi-segment video generation.

Contains frame blending functions (linear, crossfade, hard-cut), anchor frame
extraction, and segment seed computation. These implement the SVI-Pro-Workflow
techniques for seamless segment chaining.

All heavy imports (torch) are guarded inside function bodies so this module
passes py_compile without those packages.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "blend_overlap_frames",
    "extract_anchor_frame",
    "compute_segment_seeds",
]

def blend_overlap_frames(source_frames, new_frames,
                         overlap: int = 5, mode: str = "linear_blend",
                         side: str = "source") :
    """
    Blend overlapping frames between two video segments for seamless transitions.
    Mirrors ImageBatchExtendWithOverlap from comfyui-kjnodes (SVI-Pro-Workflow.json).
    
    Args:
        source_frames: Previous segment frames tensor (T, H, W, C) or (N, T, H, W, C)
        new_frames: New segment frames tensor (same format)
        overlap: Number of overlapping frames (SVI-Pro default: 5)
        mode: Blend mode - "linear_blend", "hard_cut", or "crossfade"
        side: Which segment contributes overlap - "source" or "target"
    
    Returns:
        Combined frames tensor with seamless blend at the junction
    """
    import torch
    # Ensure we're working with 4D tensors (T, H, W, C)
    if source_frames.ndim == 5:
        source_frames = source_frames.squeeze(0)
    if new_frames.ndim == 5:
        new_frames = new_frames.squeeze(0)
    
    if overlap <= 0 or overlap >= min(len(source_frames), len(new_frames)):
        # No valid overlap - just concatenate
        return torch.cat([source_frames, new_frames], dim=0)
    
    if mode == "hard_cut":
        # No blending - take source frames up to overlap, then new frames after
        if side == "source":
            return torch.cat([source_frames, new_frames[overlap:]], dim=0)
        else:
            return torch.cat([source_frames[:-overlap], new_frames], dim=0)
    
    elif mode == "linear_blend":
        # Linear blend in overlap region (SVI-Pro default)
        # Source provides the "tail" frames, new provides the "head" frames
        source_tail = source_frames[-overlap:]  # last N frames of source
        new_head = new_frames[:overlap]          # first N frames of new
        
        # Create linear blend weights
        weights = torch.linspace(1.0, 0.0, overlap, device=source_frames.device)
        weights = weights.view(-1, 1, 1, 1)  # (overlap, 1, 1, 1) for broadcasting
        
        # Blend: source_weight decreases, new_weight increases
        blended = source_tail * weights + new_head * (1.0 - weights)
        
        # Assemble: source (minus tail) + blended region + new (minus head)
        result = torch.cat([
            source_frames[:-overlap],
            blended,
            new_frames[overlap:]
        ], dim=0)
        return result
    
    elif mode == "crossfade":
        # Equal-weight crossfade (smoother than linear for some content)
        source_tail = source_frames[-overlap:]
        new_head = new_frames[:overlap]
        
        # Sigmoid-like weights for smoother transition
        t = torch.linspace(0.0, 1.0, overlap, device=source_frames.device)
        weights = 0.5 * (1.0 - torch.cos(t * 3.14159))  # cosine interpolation
        weights = weights.view(-1, 1, 1, 1)
        
        blended = source_tail * (1.0 - weights) + new_head * weights
        
        result = torch.cat([
            source_frames[:-overlap],
            blended,
            new_frames[overlap:]
        ], dim=0)
        return result
    
    else:
        # Unknown mode - fallback to linear_blend
        return blend_overlap_frames(source_frames, new_frames, overlap, "linear_blend", side)



def extract_anchor_frame(frames, position: str = "last"):
    """
    Extract a single frame from a video tensor for use as anchor/seed.
    Used by segment extension to chain segments with character consistency.
    
    Args:
        frames: Video frames tensor (T, H, W, C) or (N, T, H, W, C)
        position: "last", "first", or integer frame index
    
    Returns:
        Single frame tensor (1, H, W, C) suitable for I2V/anchor conditioning
    """
    import torch
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    
    if position == "last":
        return frames[-1:].clone()
    elif position == "first":
        return frames[:1].clone()
    elif isinstance(position, int):
        idx = min(position, len(frames) - 1)
        return frames[idx:idx+1].clone()
    else:
        return frames[-1:].clone()



def compute_segment_seeds(base_seed: int, num_segments: int,
                          mode: str = "fixed") -> list:
    """
    Compute seeds for each segment based on mode.
    SVI-Pro uses fixed seed=2025 for all segments.
    
    Args:
        base_seed: Starting seed value
        num_segments: Number of segments
        mode: "fixed", "increment", or "random"
    
    Returns:
        List of seed values, one per segment
    """
    if mode == "fixed":
        return [base_seed] * num_segments
    elif mode == "increment":
        return [base_seed + i for i in range(num_segments)]
    elif mode == "random":
        import random as _rng
        _rng.seed(base_seed)
        return [_rng.randint(0, 2**32 - 1) for _ in range(num_segments)]
    else:
        return [base_seed] * num_segments



