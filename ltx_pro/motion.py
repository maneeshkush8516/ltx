"""
Motion Coherence System utilities.

Provides optical flow estimation, motion direction detection, camera LoRA
auto-selection, velocity latent computation, and adaptive overlap calculation.
"""

__all__ = [
    "estimate_optical_flow",
    "detect_motion_direction",
    "auto_select_camera_lora",
    "compute_velocity_latent",
    "compute_adaptive_overlap",
]


def estimate_optical_flow(frame1, frame2):
    """
    Estimate optical flow between two frames using Farneback method.

    Args:
        frame1: First frame as numpy array (H, W, 3) uint8 or float
        frame2: Second frame as numpy array (H, W, 3) uint8 or float

    Returns:
        Optical flow array of shape (H, W, 2) with (dx, dy) per pixel
    """
    import numpy as np
    import cv2

    if frame1.dtype == np.float32 or frame1.dtype == np.float64:
        frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
    if frame2.dtype == np.float32 or frame2.dtype == np.float64:
        frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def detect_motion_direction(flow) -> str:
    """
    Analyze optical flow to determine dominant motion direction.

    Args:
        flow: Optical flow array (H, W, 2)

    Returns:
        One of: 'left', 'right', 'up', 'down', 'zoom_in', 'zoom_out', 'static'
    """
    import numpy as np

    h, w = flow.shape[:2]
    mean_dx = flow[:, :, 0].mean()
    mean_dy = flow[:, :, 1].mean()

    # Check for zoom by comparing center vs edge flow magnitudes
    center_region = flow[h // 4:3 * h // 4, w // 4:3 * w // 4]
    edge_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).mean()
    center_mag = np.sqrt(center_region[:, :, 0] ** 2 + center_region[:, :, 1] ** 2).mean()

    # Threshold for considering motion significant
    threshold = 1.0

    if edge_mag < threshold and center_mag < threshold:
        return 'static'

    # Zoom detection: edges diverge from center
    if edge_mag > center_mag * 1.5 and edge_mag > threshold:
        return 'zoom_out'
    if center_mag > edge_mag * 1.5 and center_mag > threshold:
        return 'zoom_in'

    # Directional detection
    if abs(mean_dx) > abs(mean_dy):
        return 'right' if mean_dx > 0 else 'left'
    else:
        return 'down' if mean_dy > 0 else 'up'


def auto_select_camera_lora(motion_direction: str) -> str:
    """
    Map detected motion direction to appropriate camera LoRA name.

    Uses the _CAMERA_LORA_FILES dict to select a matching LoRA.

    Args:
        motion_direction: Output from detect_motion_direction()

    Returns:
        Camera LoRA key string (e.g. 'dolly-left', 'static')
    """
    direction_to_lora = {
        'left': 'dolly-left',
        'right': 'dolly-right',
        'up': 'jib-up',
        'down': 'jib-down',
        'zoom_in': 'dolly-in',
        'zoom_out': 'dolly-out',
        'static': 'static',
    }
    return direction_to_lora.get(motion_direction, 'static')


def compute_velocity_latent(frame_minus2, frame_minus1):
    """
    Compute velocity vector (frame[-1] - frame[-2]) for motion injection.

    This velocity latent can be added to the last frame to extrapolate
    motion direction into the next segment.

    Args:
        frame_minus2: Second-to-last frame tensor (1, H, W, C) or (H, W, C)
        frame_minus1: Last frame tensor (same shape)

    Returns:
        Velocity tensor (same shape as input) representing motion delta
    """
    return (frame_minus1.float() - frame_minus2.float())


def compute_adaptive_overlap(prev_frames,
                             min_overlap: int = 2,
                             max_overlap: int = 10) -> int:
    """
    Compute overlap frames based on motion magnitude.

    High motion scenes need fewer overlap frames (to avoid ghosting),
    while low motion scenes benefit from more overlap (smoother blend).

    Args:
        prev_frames: Previous segment frames (T, H, W, C)
        min_overlap: Minimum overlap frames (for high motion)
        max_overlap: Maximum overlap frames (for low/no motion)

    Returns:
        Recommended number of overlap frames
    """
    if prev_frames.ndim == 5:
        prev_frames = prev_frames.squeeze(0)

    T = prev_frames.shape[0]
    if T < 2:
        return max_overlap

    # Compute average frame-to-frame difference over last few frames
    num_check = min(5, T - 1)
    diffs = []
    for i in range(T - num_check, T):
        diff = (prev_frames[i].float() - prev_frames[i - 1].float()).abs().mean().item()
        diffs.append(diff)

    avg_motion = sum(diffs) / len(diffs) if diffs else 0.0

    # Map motion to overlap: high motion -> min_overlap, low motion -> max_overlap
    # Typical frame diff range: 0.0 (static) to ~0.15 (fast motion)
    motion_normalized = min(avg_motion / 0.10, 1.0)
    overlap = int(max_overlap - motion_normalized * (max_overlap - min_overlap))
    return max(min_overlap, min(max_overlap, overlap))
