"""
Export, Timeline, and Google Drive utilities.

Provides JSON timeline generation, EDL export for NLEs, Google Drive
mounting/sync/cache, thumbnail preview generation, and clip concatenation
and merging with various blend modes.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

__all__ = [
    "generate_timeline_json",
    "generate_edl",
    "mount_google_drive",
    "sync_to_drive",
    "check_drive_cache",
    "generate_thumbnail_frame",
    "display_thumbnail_grid",
    "concatenate_clips",
    "merge_clips_to_video",
    "auto_find_output_clips",
]


def generate_timeline_json(segments_info: List[Dict[str, Any]],
                           output_path: str) -> str:
    """
    Generate JSON timeline with timestamps, prompts, seeds per segment.

    Args:
        segments_info: List of segment dicts with keys like
            'index', 'prompt', 'seed', 'frames', 'fps', 'file_path'
        output_path: Path to write the JSON file

    Returns:
        Path to the written JSON file
    """
    timeline = {
        'version': '1.0',
        'generator': 'LTX-2-PRO',
        'segments': [],
    }

    current_time = 0.0
    for seg in segments_info:
        fps = seg.get('fps', 25)
        frames = seg.get('frames', 97)
        duration = frames / fps if fps > 0 else 0.0

        entry = {
            'index': seg.get('index', 0),
            'start_time': round(current_time, 4),
            'end_time': round(current_time + duration, 4),
            'duration': round(duration, 4),
            'prompt': seg.get('prompt', ''),
            'seed': seg.get('seed', 0),
            'frames': frames,
            'fps': fps,
            'file_path': seg.get('file_path', ''),
            'resolution': seg.get('resolution', ''),
        }
        timeline['segments'].append(entry)
        current_time += duration

    timeline['total_duration'] = round(current_time, 4)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, default=str)
    except Exception as e:
        print(f"   \u26a0\ufe0f  Could not write timeline JSON: {e}")

    return output_path


def generate_edl(segments_info: List[Dict[str, Any]], output_path: str,
                 fps: int = 25) -> str:
    """
    Generate EDL (Edit Decision List) compatible with NLEs.

    Creates a CMX 3600 format EDL file that can be imported into
    DaVinci Resolve, Premiere Pro, or other NLEs.

    Args:
        segments_info: List of segment dicts with 'frames', 'file_path', 'index'
        output_path: Path to write the EDL file
        fps: Frames per second for timecode calculation

    Returns:
        Path to the written EDL file
    """
    def _frames_to_tc(frame_num: int, rate: int) -> str:
        """Convert frame number to timecode HH:MM:SS:FF."""
        h = frame_num // (rate * 3600)
        remainder = frame_num % (rate * 3600)
        m = remainder // (rate * 60)
        remainder = remainder % (rate * 60)
        s = remainder // rate
        f = remainder % rate
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    lines = []
    lines.append("TITLE: LTX-2-PRO Timeline")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    current_frame = 0
    for i, seg in enumerate(segments_info):
        seg_frames = seg.get('frames', 97)
        src_in = "00:00:00:00"
        src_out = _frames_to_tc(seg_frames, fps)
        rec_in = _frames_to_tc(current_frame, fps)
        rec_out = _frames_to_tc(current_frame + seg_frames, fps)

        edit_num = f"{i + 1:03d}"
        reel = seg.get('file_path', f"SEG{i + 1:03d}").split('/')[-1][:8]

        lines.append(
            f"{edit_num}  {reel:8s} V     C        "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )
        # Comment with prompt
        prompt_short = seg.get('prompt', '')[:60]
        if prompt_short:
            lines.append(f"* COMMENT: {prompt_short}")
        lines.append("")

        current_frame += seg_frames

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        print(f"   \u26a0\ufe0f  Could not write EDL: {e}")

    return output_path


def mount_google_drive() -> bool:
    """
    Try to mount Google Drive in Colab. Returns True on success.

    Safe to call outside Colab - returns False without error.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"   \u26a0\ufe0f  Google Drive mount failed: {e}")
        return False


def sync_to_drive(local_path: str, gdrive_path: str) -> bool:
    """
    Copy completed segment to Google Drive for persistence.

    Args:
        local_path: Local file path to copy
        gdrive_path: Destination path on Google Drive (under /content/drive/)

    Returns:
        True if copy succeeded, False otherwise
    """
    import shutil

    if not os.path.exists(local_path):
        return False

    try:
        dest_dir = os.path.dirname(gdrive_path)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(local_path, gdrive_path)
        return True
    except Exception as e:
        print(f"   \u26a0\ufe0f  Drive sync failed: {e}")
        return False


def check_drive_cache(gdrive_path: str, segment_id: str) -> Optional[str]:
    """
    Check if segment is cached on Drive for resume. Returns path or None.

    Args:
        gdrive_path: Base Google Drive directory to search
        segment_id: Segment identifier to look for

    Returns:
        Full path to cached segment file, or None if not found
    """
    if not os.path.isdir(gdrive_path):
        return None

    # Look for segment file matching the ID
    for fname in os.listdir(gdrive_path):
        if segment_id in fname:
            full_path = os.path.join(gdrive_path, fname)
            if os.path.isfile(full_path):
                return full_path

    return None


def generate_thumbnail_frame(prompt: str, width: int, height: int,
                             seed: int):
    """
    Generate a single-frame thumbnail preview for a scene.

    Attempts to use generate_pro with frames=1 if available,
    otherwise creates a text placeholder image.

    Args:
        prompt: Scene prompt text
        width: Target width
        height: Target height
        seed: Random seed

    Returns:
        PIL Image of the thumbnail, or None on failure
    """
    from PIL import Image

    # Attempt real single-frame generation via generate_pro
    try:
        if callable(globals().get('generate_pro')):
            generate_pro = globals()['generate_pro']
            _thumb_output = generate_pro(
                user_input=prompt,
                image_path=None,
                frames=1,
                width=min(width, 512),
                height=min(height, 320),
                seed=seed,
                output_prefix="_thumb_preview",
            )
            if _thumb_output and os.path.exists(_thumb_output):
                import imageio as _iio
                _reader = _iio.get_reader(_thumb_output)
                _frame = next(iter(_reader))
                _reader.close()
                return Image.fromarray(_frame)
    except Exception:
        pass  # Fall through to placeholder

    # Fallback: create a placeholder thumbnail with text overlay
    try:
        thumb_w = min(width, 320)
        thumb_h = min(height, 192)
        img = Image.new('RGB', (thumb_w, thumb_h), color=(40, 40, 50))

        # Add simple text indicator (no font dependency)
        pixels = img.load()
        # Draw a simple border
        for x in range(thumb_w):
            pixels[x, 0] = (100, 100, 120)
            pixels[x, thumb_h - 1] = (100, 100, 120)
        for y in range(thumb_h):
            pixels[0, y] = (100, 100, 120)
            pixels[thumb_w - 1, y] = (100, 100, 120)

        return img
    except Exception:
        return None


def display_thumbnail_grid(thumbnails, cols: int = 3) -> None:
    """
    Display PIL images in a grid layout in Colab notebook.

    Args:
        thumbnails: List of PIL Image objects
        cols: Number of columns in the grid
    """
    from PIL import Image

    if not thumbnails:
        return

    rows = (len(thumbnails) + cols - 1) // cols
    thumb_w = thumbnails[0].width
    thumb_h = thumbnails[0].height

    grid_w = cols * thumb_w + (cols - 1) * 4
    grid_h = rows * thumb_h + (rows - 1) * 4
    grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))

    for i, thumb in enumerate(thumbnails):
        row = i // cols
        col = i % cols
        x = col * (thumb_w + 4)
        y = row * (thumb_h + 4)
        grid.paste(thumb, (x, y))

    try:
        display(grid)  # noqa: F821 - Colab built-in
    except Exception:
        # Fallback: save to file
        grid.save('/content/thumbnail_grid.png')
        print("   Thumbnail grid saved to /content/thumbnail_grid.png")


def concatenate_clips(clip_paths: List[str], output_path: str) -> Optional[str]:
    """Concatenate video clips using ffmpeg concat demuxer."""
    import subprocess

    valid_paths = [p for p in clip_paths if p and os.path.exists(p)]
    if len(valid_paths) < 2:
        return valid_paths[0] if valid_paths else None
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in valid_paths:
            f.write(f"file '{p}'\n")
    try:
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", list_file, "-c", "copy", output_path],
                       check=True, capture_output=True)
        print(f"   \u2713 Concatenated {len(valid_paths)} clips \u2192 {output_path}")
        return output_path
    except Exception as e:
        print(f"   \u26a0\ufe0f  Concatenation failed ({e}) \u2014 individual clips still available.")
        return None


def merge_clips_to_video(
    clip_paths: List[str],
    output_name: str = "Final_Merged",
    mode: str = "overlap_blend",
    overlap: int = 5,
    fps: int = 25,
) -> Optional[str]:
    """
    Merge multiple video clips into a single continuous video.

    Supports three modes:
    - "overlap_blend": SVI-Pro style linear blend at boundaries
    - "hard_concat": Simple ffmpeg concat (no blending, preserves exact frames)
    - "crossfade": Cosine-weighted crossfade for smooth transitions

    Args:
        clip_paths: List of video file paths to merge (in order)
        output_name: Output filename prefix
        mode: Merge strategy
        overlap: Number of overlap frames for blending modes
        fps: Output frame rate

    Returns:
        Path to the merged video file, or None on failure
    """
    import subprocess
    import imageio
    import numpy as np
    import torch

    # Validate inputs
    valid_paths = [p for p in clip_paths if p and os.path.exists(p)]
    if not valid_paths:
        print("\u274c No valid clip paths provided.")
        print("   Set CLIPS_TO_MERGE or enable AUTO_FIND_CLIPS.")
        return None

    if len(valid_paths) == 1:
        print(f"\u2139\ufe0f  Only one clip found \u2014 nothing to merge: {valid_paths[0]}")
        return valid_paths[0]

    print(f"\U0001f3ac Merging {len(valid_paths)} clips ({mode})...")
    for i, p in enumerate(valid_paths):
        print(f"   [{i+1}] {os.path.basename(p)}")

    output_dir = "/content/ComfyUI/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{output_name}_{int(time.time())}.mp4"

    # Mode: hard_concat (ffmpeg, fast)
    if mode == "hard_concat":
        return concatenate_clips(valid_paths, output_path)

    # Mode: overlap_blend or crossfade (frame-level processing)
    blend_mode = "linear_blend" if mode == "overlap_blend" else "crossfade"

    try:
        from ltx_pro.overlap import blend_overlap_frames
    except ImportError:
        blend_overlap_frames = None

    try:
        combined_frames = None
        total_input_frames = 0

        for idx, clip_path in enumerate(valid_paths):
            print(f"   Loading clip {idx+1}/{len(valid_paths)}: {os.path.basename(clip_path)}...")
            reader = imageio.get_reader(clip_path)
            frames_list = [frame for frame in reader]
            reader.close()
            total_input_frames += len(frames_list)

            seg_tensor = torch.from_numpy(np.stack(frames_list)).float() / 255.0

            if combined_frames is None:
                combined_frames = seg_tensor
            else:
                if blend_overlap_frames is not None:
                    combined_frames = blend_overlap_frames(
                        combined_frames, seg_tensor,
                        overlap=overlap,
                        mode=blend_mode,
                    )
                else:
                    combined_frames = torch.cat([combined_frames, seg_tensor], dim=0)

            print(f"      {len(frames_list)} frames loaded, running total: {len(combined_frames)}")

        if combined_frames is None:
            print("\u274c No frames loaded \u2014 merge failed.")
            return None

        # Save merged video
        print(f"   \U0001f4be Encoding {len(combined_frames)} frames @ {fps}fps...")
        final_np = (combined_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Step 1: Save video frames to a temporary file
        _temp_video = output_path.replace('.mp4', '_temp_noaudio.mp4')
        imageio.mimsave(_temp_video, [f for f in final_np], fps=fps, codec='libx264')

        # Step 2: Extract and concatenate audio from source clips, then mux
        _has_audio = False
        try:
            _probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "a",
                         "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                         valid_paths[0]]
            _probe_result = subprocess.run(_probe_cmd, capture_output=True, text=True)
            _has_audio = "audio" in _probe_result.stdout
        except Exception:
            _has_audio = False

        if _has_audio:
            print("   \U0001f50a Muxing audio from source clips...")
            try:
                _audio_list = "/tmp/audio_concat_list.txt"
                with open(_audio_list, "w") as f:
                    for p in valid_paths:
                        f.write(f"file '{p}'\n")

                _temp_audio = output_path.replace('.mp4', '_temp_audio.aac')
                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", _audio_list, "-vn", "-acodec", "aac",
                    "-b:a", "192k", _temp_audio
                ], check=True, capture_output=True)

                _video_duration = len(combined_frames) / fps
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", _temp_video,
                    "-i", _temp_audio,
                    "-c:v", "copy", "-c:a", "aac",
                    "-t", str(_video_duration),
                    "-shortest",
                    output_path
                ], check=True, capture_output=True)

                if os.path.exists(_temp_video):
                    os.remove(_temp_video)
                if os.path.exists(_temp_audio):
                    os.remove(_temp_audio)

                print("   \u2713 Audio muxed successfully")
            except Exception as audio_err:
                print(f"   \u26a0\ufe0f  Audio mux failed ({audio_err}) \u2014 video saved without audio.")
                if os.path.exists(_temp_video):
                    os.rename(_temp_video, output_path)
        else:
            os.rename(_temp_video, output_path)
            print("   \u2139\ufe0f  No audio detected in source clips \u2014 video-only output.")

        duration = len(combined_frames) / fps
        saved_frames = total_input_frames - len(combined_frames)

        print(f"\n{'=' * 60}")
        print("\u2705 MERGE COMPLETE!")
        print(f"   Output     : {output_path}")
        print(f"   Duration   : {duration:.1f}s ({len(combined_frames)} frames @ {fps}fps)")
        print(f"   Input clips: {len(valid_paths)}")
        print(f"   Overlap    : {overlap} frames x {len(valid_paths)-1} boundaries = {saved_frames} frames blended")
        print(f"   Mode       : {mode}")
        print(f"{'=' * 60}")

        return output_path

    except Exception as e:
        print(f"\u274c Merge failed: {type(e).__name__}: {e}")
        print("   Falling back to hard concatenation...")
        return concatenate_clips(valid_paths, output_path)


def auto_find_output_clips(
    prefix: str = None,
    output_dir: str = "/content/ComfyUI/output",
    pattern: str = None,
) -> List[str]:
    """
    Automatically find generated clips in the output directory.

    Args:
        prefix: Match clips starting with this prefix (e.g., "Story01")
        output_dir: Directory to search
        pattern: Glob pattern override (e.g., "Story*Character*.mp4")

    Returns:
        Sorted list of matching video paths
    """
    if not os.path.exists(output_dir):
        return []

    if pattern:
        import fnmatch
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if fnmatch.fnmatch(f, pattern) and f.endswith('.mp4')]
    elif prefix:
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.startswith(prefix) and f.endswith('.mp4')
                 and '_full' not in f and '_extended' not in f]
    else:
        clips = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.endswith('.mp4') and not f.startswith('.')
                 and '_full' not in f and '_extended' not in f
                 and '_segments' not in f and 'Final_Merged' not in f]

    clips.sort()
    return clips
