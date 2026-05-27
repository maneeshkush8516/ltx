"""
Audio Synchronization utilities.

Provides the AudioSyncEngine class for voice sync and beat-aligned generation,
beat detection from audio files, segment boundary computation aligned to beats,
and segment length adjustment for rhythmic alignment.
"""

import os
from typing import List, Optional

__all__ = [
    "VOICE_SYNC_ENABLED",
    "VOICE_AUDIO_PATH",
    "VOICE_TRANSCRIPT",
    "LIP_SYNC_STRENGTH",
    "AudioSyncEngine",
    "detect_beats",
    "compute_segment_boundaries",
    "adjust_segment_length",
]

# Voice sync configuration
VOICE_SYNC_ENABLED = False
VOICE_AUDIO_PATH = ""
VOICE_TRANSCRIPT = ""
LIP_SYNC_STRENGTH = 0.5


class AudioSyncEngine:
    """
    Audio synchronization engine for voice sync and beat-aligned generation.

    Provides placeholder methods for:
    - Audio loading and beat detection
    - Lip sync keyframe generation
    - Segment alignment to audio beats
    - TTS voice generation
    - Audio-reactive camera movement
    """

    def __init__(self):
        self.audio_data = None
        self.beat_map = None
        self.sample_rate = None
        self.duration = 0.0

    def load_audio(self, path):
        """
        Load an audio file for analysis.

        Args:
            path: Path to audio file (mp3, wav, etc.)

        Returns:
            True if loaded successfully, False otherwise
        """
        if not path or not os.path.exists(path):
            return False
        try:
            import librosa
            self.audio_data, self.sample_rate = librosa.load(path, sr=None)
            self.duration = len(self.audio_data) / self.sample_rate
            return True
        except ImportError:
            print("   [AudioSync] librosa not available - audio sync disabled")
            return False
        except Exception as e:
            print(f"   [AudioSync] Load failed: {e}")
            return False

    def extract_beat_map(self):
        """
        Extract beat timestamps from loaded audio.

        Returns:
            List of beat timestamps in seconds, or empty list
        """
        if self.audio_data is None:
            return []
        try:
            import librosa
            tempo, beat_frames = librosa.beat.beat_track(y=self.audio_data, sr=self.sample_rate)
            self.beat_map = librosa.frames_to_time(beat_frames, sr=self.sample_rate).tolist()
            return self.beat_map
        except Exception:
            return []

    def generate_lip_sync_keyframes(self, transcript):
        """
        Generate lip sync keyframes from a transcript.

        Args:
            transcript: Text transcript of speech

        Returns:
            List of dicts with timing and mouth shape info (placeholder)
        """
        if not transcript:
            return []
        # Placeholder: estimate timing from word count
        words = transcript.split()
        avg_word_duration = 0.4  # seconds per word
        keyframes = []
        current_time = 0.0
        for word in words:
            keyframes.append({
                "time": current_time,
                "word": word,
                "mouth_shape": "open" if word[-1:] in "aeiou" else "closed",
                "intensity": LIP_SYNC_STRENGTH,
            })
            current_time += avg_word_duration
        return keyframes

    def align_segments_to_beats(self, segment_count=8, fps=25, base_frames=97):
        """
        Align segment boundaries to detected audio beats.

        Args:
            segment_count: Number of segments to generate
            fps: Frame rate
            base_frames: Base segment length in frames

        Returns:
            List of frame counts per segment (aligned to beats)
        """
        if not self.beat_map:
            return [base_frames] * segment_count

        # Find beats that are close to segment boundaries
        segment_duration = base_frames / fps
        aligned_lengths = []
        current_time = 0.0

        for i in range(segment_count):
            target_end = current_time + segment_duration
            # Find nearest beat to target_end
            nearest_beat = min(self.beat_map, key=lambda b: abs(b - target_end),
                            default=target_end)
            if abs(nearest_beat - target_end) < segment_duration * 0.3:
                actual_duration = nearest_beat - current_time
            else:
                actual_duration = segment_duration
            frames = max(25, int(actual_duration * fps))
            aligned_lengths.append(frames)
            current_time += actual_duration

        return aligned_lengths

    def generate_voice(self, text, voice_preset="default"):
        """
        Generate voice audio from text (TTS placeholder).

        Args:
            text: Text to synthesize
            voice_preset: Voice preset name

        Returns:
            Path to generated audio file, or empty string
        """
        # Placeholder for TTS integration
        print(f"   [AudioSync] TTS placeholder: '{text[:50]}...' with voice={voice_preset}")
        return ""

    def get_camera_changes_on_beats(self, available_loras=None):
        """
        Suggest camera LoRA changes on beat boundaries.

        Args:
            available_loras: List of available camera LoRA names

        Returns:
            List of (beat_time, suggested_lora) tuples
        """
        if not self.beat_map or not available_loras:
            return []
        suggestions = []
        for i, beat_time in enumerate(self.beat_map):
            if i % 4 == 0:  # Change camera every 4 beats
                lora = available_loras[i // 4 % len(available_loras)]
                suggestions.append((beat_time, lora))
        return suggestions

    def get_segment_timing(self, total_duration: float, fps: int = 25,
                           segment_length: int = 81) -> list:
        """
        Return optimized segment frame counts aligned to audio beats.

        Computes segment boundaries that snap to detected beat positions,
        ensuring cuts happen on musically meaningful moments rather than
        at arbitrary frame counts.

        Args:
            total_duration: Total video duration in seconds
            fps: Frame rate for frame count calculation
            segment_length: Base segment length in frames (default 81 for SVI-Pro)

        Returns:
            List of frame counts per segment, aligned to beats where possible.
            If no beats are detected, returns uniform segment lengths.
        """
        total_frames = int(total_duration * fps)
        base_segment_duration = segment_length / fps
        num_segments = max(1, int(total_duration / base_segment_duration))

        if not self.beat_map:
            # No beats available - return uniform segments
            return [segment_length] * num_segments

        # Snap segment boundaries to nearest beats
        segment_frame_counts = []
        current_time = 0.0

        for i in range(num_segments):
            target_end = current_time + base_segment_duration
            # Find nearest beat to target boundary
            nearest_beat = min(self.beat_map, key=lambda b: abs(b - target_end),
                             default=target_end)
            # Only snap if the beat is within 30% of segment duration
            if abs(nearest_beat - target_end) < base_segment_duration * 0.3:
                actual_end = nearest_beat
            else:
                actual_end = target_end
            frames = max(25, min(segment_length * 2, int((actual_end - current_time) * fps)))
            segment_frame_counts.append(frames)
            current_time = actual_end

        return segment_frame_counts

    def sync_lip_keyframes_to_latent(self, latent, keyframes):
        """
        Inject lip sync keyframe data into a video latent tensor.

        This method modifies the latent space representation to encode
        mouth shape information at specific temporal positions, enabling
        the video model to generate synchronized lip movements.

        Args:
            latent: Video latent tensor (B, C, T, H, W) from the VAE encoder
            keyframes: List of dicts from generate_lip_sync_keyframes() containing
                       timing, word, mouth_shape, and intensity fields

        Returns:
            Modified latent tensor with lip sync conditioning applied.
            Currently returns the input latent unmodified (stub).
        """
        # Stub: future integration will modify latent channels at keyframe positions
        return latent

    def inject_audio_conditioning(self, conditioning, audio_features):
        """
        Inject extracted audio features into the text conditioning tensor.

        Blends audio-derived embeddings (energy, pitch, rhythm) into the
        CLIP conditioning so the video model can generate motion that
        matches the audio energy profile.

        Args:
            conditioning: CLIP text conditioning tensor from CLIPTextEncode
            audio_features: Dict with keys 'energy', 'pitch_contour', 'rhythm_mask'
                           extracted from the audio track

        Returns:
            Modified conditioning tensor with audio features blended in.
            Currently returns the input conditioning unmodified (stub).
        """
        # Stub: future integration will blend audio embeddings into conditioning
        return conditioning


def detect_beats(audio_path: str,
                 manual_bpm: Optional[int] = None) -> List[float]:
    """
    Detect beat timestamps from audio file.

    Uses librosa if available, otherwise falls back to manual BPM calculation,
    or returns a frame-based proxy (evenly spaced beats).

    Args:
        audio_path: Path to audio file
        manual_bpm: Optional manual BPM override

    Returns:
        List of beat timestamps in seconds
    """
    # Try librosa first
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times.tolist()
    except ImportError:
        pass
    except Exception as e:
        print(f"   \u26a0\ufe0f  librosa beat detection failed: {e}")

    # Fallback: manual BPM
    if manual_bpm and manual_bpm > 0:
        beat_interval = 60.0 / manual_bpm
        # Generate beats for up to 5 minutes
        max_duration = 300.0
        beats = []
        t = 0.0
        while t < max_duration:
            beats.append(t)
            t += beat_interval
        return beats

    # Final fallback: frame-based proxy (assume 120 BPM)
    default_bpm = 120
    beat_interval = 60.0 / default_bpm
    beats = []
    t = 0.0
    while t < 300.0:
        beats.append(t)
        t += beat_interval
    return beats


def compute_segment_boundaries(beats: List[float], target_fps: int,
                               base_segment_length: int = 97) -> List[int]:
    """
    Map beat times to segment frame boundaries.

    Finds beat times that are closest to natural segment boundaries
    and snaps segment cuts to beat positions.

    Args:
        beats: List of beat timestamps in seconds
        target_fps: Target frames per second
        base_segment_length: Default segment length in frames

    Returns:
        List of frame indices where segments should start
    """
    if not beats or target_fps <= 0:
        return [0]

    # Convert beats to frame numbers
    beat_frames = [int(b * target_fps) for b in beats]

    # Find beats closest to multiples of base_segment_length
    boundaries = [0]
    next_target = base_segment_length

    for bf in beat_frames:
        if bf >= next_target - target_fps and bf <= next_target + target_fps:
            boundaries.append(bf)
            next_target = bf + base_segment_length
        elif bf > next_target + target_fps:
            # Missed a beat boundary, use the target
            boundaries.append(next_target)
            next_target += base_segment_length

    return boundaries


def adjust_segment_length(base_length: int,
                          bpm: Optional[int] = None) -> int:
    """
    Adjust segment length based on tempo for rhythmic alignment.

    Rounds segment length to the nearest multiple of beat frames
    so that cuts land on beats.

    Args:
        base_length: Base segment length in frames
        bpm: Beats per minute (None to skip adjustment)

    Returns:
        Adjusted segment length in frames
    """
    if not bpm or bpm <= 0:
        return base_length

    # Assume 25fps default for calculation
    fps = 25
    frames_per_beat = (60.0 / bpm) * fps

    if frames_per_beat < 1:
        return base_length

    # Round base_length to nearest multiple of frames_per_beat
    n_beats = round(base_length / frames_per_beat)
    n_beats = max(1, n_beats)
    adjusted = int(n_beats * frames_per_beat)

    # Ensure minimum viable segment length
    return max(25, adjusted)
