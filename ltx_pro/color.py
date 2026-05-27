"""
Style Transfer and Color Consistency utilities.

Provides LAB color histogram extraction, histogram matching for cross-segment
color consistency, and named color grading presets.
"""

from typing import Dict, List

__all__ = [
    "COLOR_GRADE_PRESETS",
    "extract_color_histogram",
    "match_color_histogram",
    "apply_color_grade",
]


COLOR_GRADE_PRESETS: Dict[str, Dict[str, List[int]]] = {
    'cinematic_warm': {
        'shadows': [10, -5, 15],
        'midtones': [5, 0, 10],
        'highlights': [-5, 5, 15],
    },
    'noir': {
        'shadows': [-10, -10, -10],
        'midtones': [0, 0, -5],
        'highlights': [5, 5, 0],
    },
    'cyberpunk': {
        'shadows': [0, -10, 20],
        'midtones': [-5, 5, 15],
        'highlights': [10, -5, 25],
    },
    'vintage': {
        'shadows': [15, 5, -10],
        'midtones': [10, 0, -5],
        'highlights': [5, 10, -15],
    },
    'cool_blue': {
        'shadows': [-10, 0, 15],
        'midtones': [-5, 0, 10],
        'highlights': [0, 5, 20],
    },
    'golden_hour': {
        'shadows': [15, 5, -5],
        'midtones': [10, 5, 0],
        'highlights': [20, 10, -10],
    },
}


def extract_color_histogram(frames_tensor):
    """
    Extract LAB color histogram from frames as reference palette.

    Args:
        frames_tensor: Video frames (T, H, W, C) in RGB float [0,1]

    Returns:
        Histogram array of shape (3, 256) for L, A, B channels
    """
    import numpy as np
    import cv2

    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    # Convert to uint8 numpy
    frames_np = (frames_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    histograms = np.zeros((3, 256), dtype=np.float64)
    for i in range(frames_np.shape[0]):
        lab = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2LAB)
        for c in range(3):
            hist = cv2.calcHist([lab], [c], None, [256], [0, 256])
            histograms[c] += hist.flatten()

    # Normalize
    total = frames_np.shape[0]
    if total > 0:
        histograms /= total

    return histograms


def match_color_histogram(source_frames, reference_histogram):
    """
    Apply LAB histogram matching to maintain color consistency across segments.

    Args:
        source_frames: Frames to adjust (T, H, W, C) in RGB float [0,1]
        reference_histogram: Target histogram from extract_color_histogram()

    Returns:
        Color-matched frames tensor (same shape as input)
    """
    import numpy as np
    import cv2
    import torch

    if source_frames.ndim == 5:
        source_frames = source_frames.squeeze(0)

    frames_np = (source_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    result_frames = np.zeros_like(frames_np)

    # Build reference CDF
    ref_cdfs = []
    for c in range(3):
        cdf = reference_histogram[c].cumsum()
        cdf_normalized = cdf / (cdf[-1] + 1e-8) * 255
        ref_cdfs.append(cdf_normalized)

    for i in range(frames_np.shape[0]):
        lab = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2LAB)

        for c in range(3):
            # Source CDF
            src_hist = cv2.calcHist([lab], [c], None, [256], [0, 256]).flatten()
            src_cdf = src_hist.cumsum()
            src_cdf_norm = src_cdf / (src_cdf[-1] + 1e-8) * 255

            # Build lookup table
            lut = np.zeros(256, dtype=np.uint8)
            for src_val in range(256):
                target_val = np.searchsorted(ref_cdfs[c], src_cdf_norm[src_val])
                lut[src_val] = min(255, target_val)

            lab[:, :, c] = lut[lab[:, :, c]]

        result_frames[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    result_tensor = torch.from_numpy(result_frames.astype(np.float32) / 255.0)
    return result_tensor


def apply_color_grade(frames_tensor, preset_name: str):
    """
    Apply color grading preset to video frames.

    Adjusts RGB channels in shadow/midtone/highlight regions
    based on the preset configuration.

    Args:
        frames_tensor: Video frames (T, H, W, C) in RGB float [0,1]
        preset_name: Key from COLOR_GRADE_PRESETS dict

    Returns:
        Color-graded frames tensor (same shape)
    """
    import torch

    if preset_name not in COLOR_GRADE_PRESETS:
        return frames_tensor

    preset = COLOR_GRADE_PRESETS[preset_name]
    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    result = frames_tensor.clone().float()

    shadows_adj = torch.tensor(preset['shadows'], dtype=torch.float32) / 255.0
    midtones_adj = torch.tensor(preset['midtones'], dtype=torch.float32) / 255.0
    highlights_adj = torch.tensor(preset['highlights'], dtype=torch.float32) / 255.0

    # Luminance for region masking
    lum = result.mean(dim=-1, keepdim=True)

    # Shadow mask (dark areas), midtone mask, highlight mask (bright areas)
    shadow_mask = (1.0 - lum * 3.0).clamp(0, 1)
    highlight_mask = ((lum - 0.67) * 3.0).clamp(0, 1)
    midtone_mask = (1.0 - shadow_mask - highlight_mask).clamp(0, 1)

    result = result + shadow_mask * shadows_adj
    result = result + midtone_mask * midtones_adj
    result = result + highlight_mask * highlights_adj

    return result.clamp(0, 1)
