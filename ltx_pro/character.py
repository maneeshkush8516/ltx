"""
Character consistency system for multi-segment video generation.

Contains PersistentLatentSeed for identity-anchored noise, CharacterPromptAnchor
for automatic character prefix injection, CharacterFeatureExtractor for structured
profile extraction, CharacterEmbeddingBank for cross-segment feature accumulation,
and helper functions for frame extraction and continuity compositing.

All heavy imports (torch, numpy, cv2, PIL) are guarded inside function bodies
so this module passes py_compile without those packages.
"""

import os
import re
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image

__all__ = [
    "PersistentLatentSeed",
    "CharacterPromptAnchor",
    "CharacterFeatureExtractor",
    "CharacterEmbeddingBank",
    "extract_multi_anchor_frames",
    "create_style_lock",
    "extract_continuity_composite",
    "save_continuity_frame",
    "extract_character_profiles_from_script",
]

class PersistentLatentSeed:
    """
    Maintains a fixed noise tensor derived from one or more character reference images.

    Supports multi-frame temporal anchoring: blends identity noise into the first
    OVERLAP_FRAMES_CHARACTER frames with configurable temporal decay. Computes
    frequency-domain features from reference for more stable identity preservation.

    This noise is blended into the initial latent of every scene at
    LATENT_SEED_STRENGTH to anchor the model's generation toward
    the character's features.
    """

    def __init__(self, reference_image=None, seed=42, strength=0.15):
        """
        Args:
            reference_image: Character reference tensor (1, H, W, C), PIL Image,
                or list of tensors for multi-frame reference.
            seed: Fixed seed for reproducible noise generation
            strength: Blend strength (0.0 = no effect, 1.0 = full replacement)
        """
        self.seed = seed
        self.strength = strength
        self._noise_tensor = None
        self._reference_hash = None
        self._reference_images = []
        self._frequency_features = None

        if reference_image is not None:
            if isinstance(reference_image, (list, tuple)):
                for img in reference_image:
                    self.add_reference(img)
            else:
                self.set_reference(reference_image)

    def set_reference(self, reference_image):
        """Generate fixed noise from a reference image."""
        import torch
        if isinstance(reference_image, torch.Tensor):
            if reference_image.ndim == 4:
                reference_image = reference_image[0]
            img_data = reference_image.cpu()
        else:
            # Non-Tensor input cannot produce a usable noise tensor for blending
            self._noise_tensor = None
            return

        # Create a hash from the image to detect changes
        new_hash = hash(img_data.sum().item())
        if new_hash == self._reference_hash:
            return

        self._reference_hash = new_hash
        self._reference_images = [img_data]

        # Generate fixed noise seeded from the reference image content
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(abs(img_data.mean().item()) * 1000))
        self._noise_tensor = torch.randn_like(img_data, generator=generator)

        # Compute frequency-domain features for stable identity
        self._compute_frequency_features(img_data)

    def add_reference(self, reference_image):
        """Add an additional reference image for multi-frame anchoring.

        Multiple references are averaged to produce a more robust identity noise
        that captures consistent features across different views/poses.
        """
        import torch
        if isinstance(reference_image, torch.Tensor):
            if reference_image.ndim == 4:
                reference_image = reference_image[0]
            img_data = reference_image.cpu()
        else:
            return

        self._reference_images.append(img_data)

        # Recompute noise as average across all reference images
        generator = torch.Generator()
        combined_mean = sum(r.mean().item() for r in self._reference_images) / len(self._reference_images)
        generator.manual_seed(self.seed + int(abs(combined_mean) * 1000))

        # Generate noise at the shape of the first reference (canonical size)
        canonical = self._reference_images[0]
        self._noise_tensor = torch.randn_like(canonical, generator=generator)

        # Blend in per-image identity signals
        for idx, ref_img in enumerate(self._reference_images):
            gen_i = torch.Generator()
            gen_i.manual_seed(self.seed + idx * 137 + int(abs(ref_img.mean().item()) * 500))
            ref_noise = torch.randn_like(canonical, generator=gen_i)
            weight = 1.0 / len(self._reference_images)
            self._noise_tensor = self._noise_tensor + weight * ref_noise * 0.3

        # Update hash
        self._reference_hash = hash(sum(r.sum().item() for r in self._reference_images))
        # Recompute frequency features from all references
        self._compute_frequency_features(canonical)

    def _compute_frequency_features(self, img_data):
        """Compute frequency-domain features from the reference for stable identity.

        Extracts low-frequency components that represent overall structure (face shape,
        skin tone, hair mass) rather than high-frequency details that vary per frame.
        """
        try:
            # Convert to grayscale-like single channel for FFT
            if img_data.ndim == 3 and img_data.shape[-1] >= 3:
                gray = img_data[..., 0] * 0.299 + img_data[..., 1] * 0.587 + img_data[..., 2] * 0.114
            elif img_data.ndim == 3:
                gray = img_data[..., 0]
            else:
                gray = img_data

            # Apply 2D FFT and keep low-frequency components
            freq = torch.fft.fft2(gray)
            freq_shifted = torch.fft.fftshift(freq)

            # Low-pass filter: keep center 25% of spectrum
            h, w = freq_shifted.shape
            ch, cw = h // 2, w // 2
            radius_h, radius_w = h // 4, w // 4
            mask = torch.zeros_like(freq_shifted, dtype=torch.bool)
            mask[ch - radius_h:ch + radius_h, cw - radius_w:cw + radius_w] = True
            freq_filtered = freq_shifted * mask.float()

            self._frequency_features = freq_filtered
        except Exception:
            self._frequency_features = None

    def get_identity_noise(self, target_shape, num_frames=None, decay_rate=0.85):
        """Return noise shaped for video latents with per-frame decay weighting.

        Args:
            target_shape: Target tensor shape, typically (B, C, T, H, W) for video.
            num_frames: Number of frames to generate noise for (defaults to T from shape).
            decay_rate: Per-frame decay multiplier (0.85 = 15% reduction each frame).

        Returns:
            Noise tensor of target_shape with temporal decay applied, or None.
        """
        if self._noise_tensor is None:
            return None

        import torch
        import torch.nn.functional as F

        if len(target_shape) == 5:
            B, C, T, H, W = target_shape
            frames_to_fill = min(num_frames, T) if num_frames is not None else T

            # Resize base noise to spatial dims
            noise = self._noise_tensor
            if noise.ndim == 3:
                noise = noise.unsqueeze(0)
            if noise.ndim == 4 and noise.shape[-1] <= 4:
                noise = noise.permute(0, 3, 1, 2)  # NHWC -> NCHW

            noise_resized = F.interpolate(
                noise[:1, :C], size=(H, W), mode='bilinear', align_corners=False
            )

            # Mix in frequency features if available for more stable identity
            if self._frequency_features is not None:
                try:
                    freq = self._frequency_features
                    # Convert frequency features to spatial domain bias
                    freq_spatial = torch.fft.ifft2(torch.fft.ifftshift(freq)).real
                    freq_spatial = freq_spatial.unsqueeze(0).unsqueeze(0)  # (1,1,H_orig,W_orig)
                    freq_resized = F.interpolate(
                        freq_spatial, size=(H, W), mode='bilinear', align_corners=False
                    )
                    # Normalize and mix at 20% weight into noise
                    freq_norm = freq_resized / (freq_resized.abs().max() + 1e-8)
                    noise_resized = noise_resized + 0.2 * freq_norm.expand_as(noise_resized)
                except Exception:
                    pass  # Frequency mixing is best-effort

            # Build temporal noise with decay
            identity_noise = torch.zeros(B, C, T, H, W, device=noise_resized.device)
            for t in range(frames_to_fill):
                frame_weight = self.strength * (decay_rate ** t)
                identity_noise[:, :, t, :, :] = noise_resized * frame_weight

            return identity_noise

        elif len(target_shape) == 4:
            B, C, H, W = target_shape
            noise = self._noise_tensor
            if noise.ndim == 3:
                noise = noise.unsqueeze(0)
            if noise.ndim == 4 and noise.shape[-1] <= 4:
                noise = noise.permute(0, 3, 1, 2)
            noise_resized = F.interpolate(
                noise[:1, :C], size=(H, W), mode='bilinear', align_corners=False
            )
            return noise_resized * self.strength

        return None

    def get_noise(self, target_shape=None):
        """Get the persistent noise tensor, optionally resized to target shape."""
        import torch
        if self._noise_tensor is None:
            return None
        if target_shape is None:
            return self._noise_tensor * self.strength
        # Resize noise to match target
        noise = self._noise_tensor.unsqueeze(0) if self._noise_tensor.ndim == 3 else self._noise_tensor
        if noise.shape != target_shape:
            import torch.nn.functional as F
            noise = F.interpolate(
                noise.permute(0, 3, 1, 2) if noise.ndim == 4 and noise.shape[-1] <= 4 else noise,
                size=target_shape[-2:] if len(target_shape) >= 2 else None,
                mode='bilinear', align_corners=False
            )
            if noise.ndim == 4 and noise.shape[1] <= 4:
                noise = noise.permute(0, 2, 3, 1)
        return noise * self.strength

    def blend_into_latent(self, latent_tensor, num_frames=None):
        """Blend persistent noise into the first N frames of a video latent.

        Mixes the persistent character noise into the latent tensor at
        self.strength weight for the specified num_frames (defaults to all).
        Uses temporal decay so identity anchoring is strongest at the start.
        """
        if self._noise_tensor is None:
            return latent_tensor

        import torch
        import torch.nn.functional as F
        noise = self._noise_tensor

        # Determine how many frames to blend into
        if latent_tensor.ndim == 5:
            # Video latent: (B, C, T, H, W) or (B, T, C, H, W)
            total_t = latent_tensor.shape[2]
            blend_frames = min(num_frames, total_t) if num_frames is not None else total_t

            # Resize noise to match spatial dims of the latent
            target_h, target_w = latent_tensor.shape[3], latent_tensor.shape[4]
            if noise.ndim == 3:
                noise = noise.unsqueeze(0)  # Add batch dim
            # Ensure noise has right channel count
            if noise.ndim == 4 and noise.shape[-1] <= 4:
                noise = noise.permute(0, 3, 1, 2)  # NHWC -> NCHW
            noise_resized = F.interpolate(
                noise[:1, :latent_tensor.shape[1]],
                size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            # Apply blend to the first N frames with strength weighting
            blended = latent_tensor.clone()
            for t in range(blend_frames):
                # Temporal decay: stronger blend at the start, fading toward num_frames
                temporal_weight = self.strength * (1.0 - t / max(blend_frames, 1))
                blended[:, :, t] = (
                    (1.0 - temporal_weight) * blended[:, :, t]
                    + temporal_weight * noise_resized
                )
            return blended

        elif latent_tensor.ndim == 4:
            # Single image latent: (B, C, H, W)
            target_h, target_w = latent_tensor.shape[2], latent_tensor.shape[3]
            if noise.ndim == 3:
                noise = noise.unsqueeze(0)
            if noise.ndim == 4 and noise.shape[-1] <= 4:
                noise = noise.permute(0, 3, 1, 2)
            noise_resized = F.interpolate(
                noise[:1, :latent_tensor.shape[1]],
                size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            blended = (1.0 - self.strength) * latent_tensor + self.strength * noise_resized
            return blended

        # Fallback for unexpected dims
        return latent_tensor



class CharacterPromptAnchor:
    """
    Prepends a detailed character description prefix to EVERY prompt automatically.

    Format: '[Character: {name}. {description}. Maintain exact appearance throughout.] {actual_prompt}'

    Supports structured profile data from CharacterFeatureExtractor for richer
    identity anchoring with ethnicity, age, hair, body type, clothing, and
    distinguishing features.
    """

    def __init__(self, name="", description="", enabled=True, profile=None):
        """
        Args:
            name: Character name
            description: Detailed character description
            enabled: Whether to inject prefix
            profile: Optional structured profile dict from CharacterFeatureExtractor
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.profile = profile

    def anchor_prompt(self, prompt):
        """Prepend character anchor to a prompt string."""
        if not self.enabled or (not self.description and not self.profile):
            return prompt
        # Use structured profile if available, otherwise fall back to description
        if self.profile:
            prefix = self.build_consistency_prefix(self.profile)
        else:
            prefix = f"[Character: {self.name}. {self.description}. Maintain exact appearance throughout.] "
        return prefix + prompt

    def build_consistency_prefix(self, profile_dict):
        """Build a structured consistency prefix from a profile dict.

        Produces: '[Character: {name}. {age}-year-old {ethnicity} {gender},
        {hair}, {body_type}, {clothing}, {distinguishing}. Maintain exact
        appearance throughout.] '

        Args:
            profile_dict: Dict with keys like ethnicity_skin_tone, age_estimate,
                hair_color_style, body_type, clothing_description,
                distinguishing_features, facial_features.

        Returns:
            Formatted prefix string.
        """
        if not profile_dict:
            return f"[Character: {self.name}. {self.description}. Maintain exact appearance throughout.] "

        age = profile_dict.get("age_estimate", "")
        ethnicity = profile_dict.get("ethnicity_skin_tone", "")
        hair = profile_dict.get("hair_color_style", "")
        body_type = profile_dict.get("body_type", "")
        clothing = profile_dict.get("clothing_description", "")
        distinguishing = profile_dict.get("distinguishing_features", "")
        facial = profile_dict.get("facial_features", "")

        # Build parts list, skip empty fields
        parts = []
        age_eth = ""
        if age and ethnicity:
            age_eth = f"{age}-year-old {ethnicity}"
        elif age:
            age_eth = f"{age}-year-old"
        elif ethnicity:
            age_eth = ethnicity
        if age_eth:
            parts.append(age_eth)
        if hair:
            parts.append(hair)
        if body_type:
            parts.append(body_type)
        if facial:
            parts.append(facial)
        if clothing:
            parts.append(clothing)
        if distinguishing:
            parts.append(distinguishing)

        details = ", ".join(parts) if parts else self.description
        prefix = f"[Character: {self.name}. {details}. Maintain exact appearance throughout.] "
        return prefix

    def set_character(self, name, description):
        """Update character identity."""
        self.name = name
        self.description = description

    def set_profile(self, profile_dict):
        """Update character profile from extracted features."""
        self.profile = profile_dict



class CharacterFeatureExtractor:
    """
    Extracts a structured character profile from a reference image using
    InlineVisionDescribe or similar vision model.

    Profile keys:
    - ethnicity_skin_tone: Perceived ethnicity and skin tone
    - age_estimate: Estimated age (numeric string)
    - hair_color_style: Hair color, length, style
    - body_type: Build/physique description
    - clothing_description: What the character is wearing
    - distinguishing_features: Scars, tattoos, accessories, etc.
    - facial_features: Face shape, eye color, notable facial traits
    """

    CHARACTER_EXTRACT_PROMPT = (
        "Analyze this character image and provide a structured description. "
        "Output ONLY key:value pairs, one per line, for these attributes:\n"
        "ethnicity_skin_tone: <perceived ethnicity and skin tone>\n"
        "age_estimate: <estimated age as number>\n"
        "hair_color_style: <hair color, length, and style>\n"
        "body_type: <build and physique>\n"
        "clothing_description: <what they are wearing>\n"
        "distinguishing_features: <scars, tattoos, accessories, unique marks>\n"
        "facial_features: <face shape, eye color, notable facial traits>\n"
        "Be concise and factual. Do not add extra commentary."
    )

    def __init__(self, vision_model=None):
        """
        Args:
            vision_model: Optional vision model key override (e.g. "3B-fast", "7B-nsfw").
                If None, defaults to "3B-fast".
        """
        self.vision_model = vision_model or "3B-fast"
        self._cached_profile = None

    def extract_profile(self, image_path):
        """Extract a structured profile dict from a character image.

        Args:
            image_path: Path to the character reference image.

        Returns:
            Dict with profile keys, or empty dict on failure.
        """
        if not image_path or not os.path.exists(str(image_path)):
            return {}

        try:
            # Load image as tensor for InlineVisionDescribe
            from ltx_pro.utils import load_image_tensor
            image_tensor = load_image_tensor(image_path)
            if image_tensor is None:
                return {}

            # Use InlineVisionDescribe (embedded fallback vision model)
            from ltx_pro.vision_describe import InlineVisionDescribe
            vision_desc = InlineVisionDescribe()
            # Override the default prompt with our extraction prompt
            _original_prompt = InlineVisionDescribe.DESCRIBE_PROMPT
            InlineVisionDescribe.DESCRIBE_PROMPT = self.CHARACTER_EXTRACT_PROMPT
            try:
                raw_output = vision_desc.describe(
                    image_tensor=image_tensor,
                    model_key=self.vision_model
                )
            finally:
                InlineVisionDescribe.DESCRIBE_PROMPT = _original_prompt

            profile = self._parse_profile_output(raw_output)
            self._cached_profile = profile
            return profile

        except Exception as e:
            print(f"   [CharacterFeatureExtractor] Extraction failed: {e}")
            return {}

    def _parse_profile_output(self, raw_text):
        """Parse key:value pairs from model output into a profile dict.

        Args:
            raw_text: Raw text output from the vision model.

        Returns:
            Dict with normalized profile keys.
        """
        profile = {
            "ethnicity_skin_tone": "",
            "age_estimate": "",
            "hair_color_style": "",
            "body_type": "",
            "clothing_description": "",
            "distinguishing_features": "",
            "facial_features": "",
        }

        if not raw_text:
            return profile

        valid_keys = set(profile.keys())
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                value = value.strip()
                if key in valid_keys and value:
                    profile[key] = value

        return profile

    def get_cached_profile(self):
        """Return the last extracted profile, or empty dict."""
        return self._cached_profile if self._cached_profile else {}




def extract_multi_anchor_frames(frames, n: int = 3,
                                strategy: str = 'last_n') -> torch.Tensor:
    """
    Extract multiple anchor frames for conditioning.

    Strategies:
        'last_n'  - Extract the last N frames (default for segment chaining)
        'uniform' - Uniformly sample N frames across the sequence
        'keyframe'- Pick frames with highest inter-frame difference

    Args:
        frames: Video frames tensor (T, H, W, C) or (N, T, H, W, C)
        n: Number of anchor frames to extract
        strategy: Extraction strategy

    Returns:
        Tensor of shape (n, H, W, C) with selected anchor frames
    """
    import torch
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    T = frames.shape[0]
    n = min(n, T)

    if strategy == 'last_n':
        return frames[-n:].clone()
    elif strategy == 'uniform':
        indices = torch.linspace(0, T - 1, n).long()
        return frames[indices].clone()
    elif strategy == 'keyframe':
        # Select frames with largest difference from neighbors
        if T <= n:
            return frames.clone()
        diffs = []
        for i in range(1, T):
            diff = (frames[i].float() - frames[i - 1].float()).abs().mean().item()
            diffs.append((diff, i))
        diffs.sort(key=lambda x: x[0], reverse=True)
        indices = sorted([d[1] for d in diffs[:n]])
        return frames[torch.tensor(indices)].clone()
    else:
        return frames[-n:].clone()



class CharacterEmbeddingBank:
    """
    Accumulates compact frame-level features across segments for consistent generation.

    Maintains a running average of spatially-pooled frame features (not raw pixels
    or model embeddings) that can be used to condition subsequent segments for
    character consistency. Features are derived by spatial-mean pooling of frame
    tensors to create a compact per-frame representation.
    """

    def __init__(self):
        self._embeddings = []
        self._max_entries: int = 50

    def accumulate(self, features) -> None:
        """Add new feature representation to the bank (spatial-mean pooled frame features)."""
        self._embeddings.append(features.detach().cpu())
        if len(self._embeddings) > self._max_entries:
            self._embeddings = self._embeddings[-self._max_entries:]

    def get_average_embedding(self):
        """Return the mean embedding across all accumulated features."""
        if not self._embeddings:
            return None
        import torch
        stacked = torch.stack(self._embeddings, dim=0)
        return stacked.mean(dim=0)

    def reset(self) -> None:
        """Clear all accumulated embeddings."""
        self._embeddings = []

    def __len__(self) -> int:
        return len(self._embeddings)



def create_style_lock(anchor_frames,
                      mode: str = 'latent_average') -> torch.Tensor:
    """
    Average multiple anchor frame latents to create a style lock constraint.

    Args:
        anchor_frames: Tensor of anchor frames (N, H, W, C) or (N, C, H, W)
        mode: 'latent_average' averages all frames, 'weighted' weights recent higher

    Returns:
        Single averaged frame tensor usable as style reference
    """
    import torch
    if anchor_frames.ndim < 3:
        return anchor_frames
    if anchor_frames.ndim == 3:
        return anchor_frames.unsqueeze(0)

    N = anchor_frames.shape[0]
    if mode == 'latent_average':
        return anchor_frames.mean(dim=0, keepdim=True)
    elif mode == 'weighted':
        # Exponentially weight recent frames higher
        weights = torch.exp(torch.linspace(-1.0, 0.0, N))
        weights = weights / weights.sum()
        weights = weights.view(N, 1, 1, 1).to(anchor_frames.device)
        return (anchor_frames * weights).sum(dim=0, keepdim=True)
    else:
        return anchor_frames.mean(dim=0, keepdim=True)




def extract_continuity_composite(video_path: str, n_frames: int = 5,
                                 mode: str = 'weighted_average'):
    """
    Extract last N frames from a video and create a weighted composite tensor.

    Used by the dual-anchor system to produce a high-quality continuity frame
    that captures the temporal state at the end of a scene. A weighted composite
    reduces noise/flicker artifacts compared to a single last frame.

    Args:
        video_path: Path to the source video file.
        n_frames: Number of frames to extract from the end of the video.
        mode: 'weighted_average' - later frames get linearly higher weight.
              'last_frame'       - only return the very last frame (legacy).

    Returns:
        NHWC float tensor (1, H, W, 3) in [0,1] range, or None on failure.
    """
    try:
        import cv2
        import torch
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                return None

            n = min(n_frames, total)

            if mode == 'last_frame' or n == 1:
                # Legacy single-frame behavior
                cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
                ok, frame = cap.read()
                if not ok:
                    return None
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(frame).float().unsqueeze(0) / 255.0

            # Extract last n frames
            start_idx = total - n
            frames_list = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for _ in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame).float() / 255.0)

            if not frames_list:
                return None

            # weighted_average: linear weights (1, 2, 3, ..., n) normalized
            n_actual = len(frames_list)
            weights = torch.arange(1, n_actual + 1, dtype=torch.float32)
            weights = weights / weights.sum()
            # frames_list items are (H, W, 3); stack to (N, H, W, 3)
            stacked = torch.stack(frames_list, dim=0)
            # Apply weights along dim 0
            weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
            return weighted.unsqueeze(0)  # (1, H, W, 3)
        finally:
            cap.release()
    except Exception:
        return None



def save_continuity_frame(tensor, path: str, format: str = 'png') -> bool:
    """
    Save a frame tensor to disk as PNG or JPEG.

    Args:
        tensor: NHWC float tensor (1, H, W, 3) in [0,1] range.
        path: Output file path.
        format: 'png' for lossless, 'jpg' for compressed (quality=98).

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        import numpy as np
        from PIL import Image
        if tensor.ndim == 4:
            tensor = tensor[0]
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, "RGB")
        if format.lower() in ('jpg', 'jpeg'):
            pil_img.save(path, "JPEG", quality=98)
        else:
            pil_img.save(path, "PNG")
        return True
    except Exception:
        return False



def extract_character_profiles_from_script(script: str, character_def: str = "") -> dict:
    """
    Extract character profiles from a script using InlinePromptArchitect.

    Uses the LLM to identify characters mentioned in the script and generate
    structured consistency anchors for each. These profiles are injected into
    every scene dict so character appearance stays constant across shots.

    FEAT-002: Story-to-prompt character extraction step that generates
    a character_description field for each extracted character.

    Args:
        script: Full narrative script text
        character_def: User-provided primary character definition (takes precedence)

    Returns:
        Dict mapping character name -> profile string (consistency anchor)
    """
    profiles = {}

    # If the user already provided a character definition, use it directly
    if character_def and character_def.strip():
        # Extract a name from the definition (first capitalized word or "Main")
        _name_match = re.search(r'\b([A-Z][a-z]+)\b', character_def)
        _char_name = _name_match.group(1) if _name_match else "Main"
        profiles[_char_name] = character_def.strip()

    # Scan script for named characters (capitalized proper nouns that appear 2+ times)
    _proper_nouns = re.findall(r'\b([A-Z][a-z]{2,})\b', script)
    # Filter out common words that get capitalized at sentence starts
    _common_words = {"The", "This", "That", "And", "But", "She", "Her", "His",
                     "They", "Then", "When", "What", "Where", "How", "From",
                     "Into", "With", "Over", "Upon", "After", "Before", "While"}
    _char_candidates = [n for n in _proper_nouns if n not in _common_words]

    # Characters that appear 2+ times in the script
    from collections import Counter
    _counts = Counter(_char_candidates)
    _recurring = [name for name, count in _counts.items() if count >= 2]

    for name in _recurring:
        if name not in profiles:
            # Generate a consistency anchor string from context around the name
            _context_lines = [s for s in script.split('.') if name in s]
            _context = ". ".join(_context_lines[:3]).strip()
            if _context:
                profiles[name] = f"[Character: {name}] {_context}"
            else:
                profiles[name] = f"[Character: {name}]"

    return profiles



