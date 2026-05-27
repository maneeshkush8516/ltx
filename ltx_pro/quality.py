"""
Quality Gate and Multi-Resolution Strategy utilities.

Provides SSIM computation, histogram consistency analysis, artifact detection,
comprehensive segment quality scoring, quality gate pass/fail checks, and
shot type detection for multi-resolution strategy.
"""

from typing import Any, Dict, Optional, Tuple

__all__ = [
    "compute_frame_ssim",
    "compute_histogram_consistency",
    "compute_artifact_score",
    "compute_segment_quality",
    "quality_gate_check",
    "detect_shot_type",
    "get_resolution_for_shot",
]


def compute_frame_ssim(frame1, frame2) -> float:
    """
    Compute structural similarity between two frames (0-1 scale).

    Uses a simplified SSIM approximation without scipy dependency.

    Args:
        frame1: Frame tensor (H, W, C) or (1, H, W, C)
        frame2: Frame tensor (same shape)

    Returns:
        SSIM score between 0.0 and 1.0 (higher = more similar)
    """
    import torch  # noqa: F811

    if frame1.ndim == 4:
        frame1 = frame1.squeeze(0)
    if frame2.ndim == 4:
        frame2 = frame2.squeeze(0)

    f1 = frame1.float()
    f2 = frame2.float()

    mu1 = f1.mean()
    mu2 = f2.mean()
    sigma1_sq = ((f1 - mu1) ** 2).mean()
    sigma2_sq = ((f2 - mu2) ** 2).mean()
    sigma12 = ((f1 - mu1) * (f2 - mu2)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_val = (numerator / denominator).item()
    return max(0.0, min(1.0, ssim_val))


def compute_histogram_consistency(frames) -> float:
    """
    Check color histogram consistency across frames (0-1 scale).

    Compares the mean color distribution of each frame to the overall mean.
    Higher score means more consistent color across the segment.

    Args:
        frames: Video frames tensor (T, H, W, C)

    Returns:
        Consistency score between 0.0 and 1.0 (higher = more consistent)
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    if T < 2:
        return 1.0

    # Compute per-frame mean color
    frame_means = frames.float().mean(dim=(1, 2))  # (T, C)
    overall_mean = frame_means.mean(dim=0)  # (C,)

    # Compute deviation from overall mean
    deviations = (frame_means - overall_mean).abs().mean().item()

    # Map to 0-1 score (lower deviation = higher consistency)
    score = max(0.0, 1.0 - deviations * 10.0)
    return score


def compute_artifact_score(frames) -> float:
    """
    Detect artifacts via variance analysis (low = good, high = artifacts).

    Looks for sudden spikes in local variance that indicate generation artifacts.

    Args:
        frames: Video frames tensor (T, H, W, C)

    Returns:
        Artifact score between 0.0 and 1.0 (lower = fewer artifacts)
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    if T < 2:
        return 0.0

    # Compute per-frame variance
    variances = []
    for i in range(T):
        v = frames[i].float().var().item()
        variances.append(v)

    # Detect variance spikes (potential artifacts)
    mean_var = sum(variances) / len(variances)
    if mean_var < 1e-8:
        return 0.0

    max_deviation = max(abs(v - mean_var) for v in variances)
    score = min(1.0, max_deviation / (mean_var + 1e-8) * 0.5)
    return score


def compute_segment_quality(frames_tensor,
                            overlap_region=None) -> Dict[str, Any]:
    """
    Compute comprehensive quality metrics for a generated segment.

    Args:
        frames_tensor: Generated video frames (T, H, W, C)
        overlap_region: Optional overlap frames from previous segment for SSIM check

    Returns:
        Dict with keys: 'ssim', 'histogram', 'variance', 'passed' (bool)
    """
    if frames_tensor.ndim == 5:
        frames_tensor = frames_tensor.squeeze(0)

    # SSIM between overlap regions
    ssim_score = 1.0
    if overlap_region is not None and overlap_region.shape[0] > 0:
        n_overlap = min(overlap_region.shape[0], frames_tensor.shape[0])
        ssim_scores = []
        for i in range(n_overlap):
            s = compute_frame_ssim(overlap_region[i], frames_tensor[i])
            ssim_scores.append(s)
        ssim_score = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 1.0

    histogram_score = compute_histogram_consistency(frames_tensor)
    artifact_score = compute_artifact_score(frames_tensor)

    # Default thresholds
    passed = (ssim_score > 0.5 and histogram_score > 0.4 and artifact_score < 0.6)

    return {
        'ssim': ssim_score,
        'histogram': histogram_score,
        'variance': artifact_score,
        'passed': passed,
    }


def quality_gate_check(quality_scores: Dict[str, Any],
                       thresholds: Dict[str, float]) -> bool:
    """
    Return True if all quality metrics pass their thresholds.

    Args:
        quality_scores: Dict from compute_segment_quality()
        thresholds: Dict mapping metric names to threshold values
            e.g. {'ssim': 0.5, 'histogram': 0.4, 'variance': 0.6}

    Returns:
        True if all metrics pass, False otherwise
    """
    for metric, threshold in thresholds.items():
        score = quality_scores.get(metric)
        if score is None:
            continue
        # For variance/artifact, lower is better
        if metric in ('variance', 'artifact'):
            if score > threshold:
                return False
        else:
            if score < threshold:
                return False
    return True


def detect_shot_type(prompt: str) -> str:
    """
    Detect shot type from prompt keywords.

    Args:
        prompt: Text prompt for the scene

    Returns:
        One of: 'wide', 'closeup', 'transition', 'normal'
    """
    prompt_lower = prompt.lower()

    wide_keywords = ['wide', 'establishing', 'landscape', 'panorama', 'aerial',
                     'drone', 'vista', 'skyline', 'horizon']
    closeup_keywords = ['close', 'closeup', 'close-up', 'face', 'portrait',
                        'detail', 'macro', 'eye', 'lips', 'hands']
    transition_keywords = ['transition', 'blur', 'sweep', 'whip', 'flash',
                           'dissolve', 'wipe', 'fade']

    for kw in transition_keywords:
        if kw in prompt_lower:
            return 'transition'
    for kw in closeup_keywords:
        if kw in prompt_lower:
            return 'closeup'
    for kw in wide_keywords:
        if kw in prompt_lower:
            return 'wide'

    return 'normal'


def get_resolution_for_shot(shot_type: str, base_width: int,
                            base_height: int) -> Tuple[int, int, float]:
    """
    Get resolution and anchor weight for shot type.

    Args:
        shot_type: Output from detect_shot_type()
        base_width: Base generation width
        base_height: Base generation height

    Returns:
        Tuple of (width, height, anchor_weight)
    """
    if shot_type == 'wide':
        return base_width, base_height, 0.6
    elif shot_type == 'closeup':
        return base_width, base_height, 0.9
    elif shot_type == 'transition':
        # Half resolution for transitions (faster, less detail needed)
        w = max(256, (base_width // 2) // 32 * 32)
        h = max(256, (base_height // 2) // 32 * 32)
        return w, h, 0.3
    else:
        return base_width, base_height, 0.7
