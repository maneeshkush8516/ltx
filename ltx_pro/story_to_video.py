"""
Story-to-Video pipeline for YouTube series production.

Provides the StoryToVideo class that takes a full narrative script and produces:
1. Character profiles (consistent across all episodes)
2. Scene breakdown with cinematic prompts
3. Camera direction per shot
4. Dialogue timing and language control
5. Visual continuity metadata
6. YouTube-optimized features (chapter markers, thumbnails, aspect ratios)

All heavy imports (torch, etc.) are guarded inside function bodies so this
module passes py_compile without those packages installed.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any

__all__ = ["StoryToVideo"]


# Aspect ratio dimension mappings
_ASPECT_RATIO_MAP = {
    "16:9": (768, 512),
    "9:16": (512, 768),
    "1:1": (512, 512),
}

# Keywords that indicate cliffhanger/tension in a scene
_CLIFFHANGER_KEYWORDS = [
    "suddenly", "but then", "who was", "what if", "to be continued",
    "the door opened", "a shadow", "gunshot", "scream", "vanished",
    "disappeared", "betrayal", "revealed", "secret", "truth",
    "confrontation", "stood face to face", "knife", "blood",
    "eyes widened", "heart stopped", "silence fell", "darkness",
    "final breath", "last chance", "no turning back", "cliff",
    "explosion", "collapsed", "never seen again", "who are you",
    "impossible", "can't be", "what have you done",
]

# Keywords indicating high visual interest for thumbnails
_THUMBNAIL_INTEREST_KEYWORDS = [
    "explosion", "reveal", "face", "close-up", "dramatic",
    "fire", "confrontation", "chase", "kiss", "fight",
    "surprise", "shock", "discovery", "transform", "magic",
    "sunset", "storm", "battle", "crowd", "weapon",
    "tears", "scream", "power", "glow", "epic",
]

# Suspense-enhancing quality keywords
_SUSPENSE_QUALITY_KEYWORDS = (
    "dramatic tension, suspenseful atmosphere, high contrast shadows, "
    "moody cinematic lighting, shallow depth of field, intense close-up, "
    "film noir undertones, visceral emotion"
)


class StoryToVideo:
    """
    Complete story-to-video pipeline for YouTube series production.

    Takes a full narrative script and produces:
    1. Character profiles (consistent across all episodes)
    2. Scene breakdown with cinematic prompts
    3. Camera direction per shot
    4. Dialogue timing and language control
    5. Visual continuity metadata
    """

    def __init__(self, config=None):
        """
        Initialize the StoryToVideo pipeline.

        Args:
            config: Optional configuration dict. If None, uses defaults
                from ltx_pro.config module constants.
        """
        self._story = None
        self._title = ""
        self._characters = {}
        self._series_characters = {}
        self._visual_style = "realistic"
        self._quality = "8K cinematic"
        self._aspect_ratio = "16:9"
        self._episode_history = []
        self._scenes = []
        self._config = config or {}

    def load_story(self, script: str, title: str = "") -> None:
        """
        Load a narrative script for processing.

        Stores the script text and automatically extracts character names
        using capitalized-word heuristics (proper nouns appearing multiple
        times are likely characters).

        Args:
            script: Full narrative script text.
            title: Optional episode/story title.
        """
        self._story = script
        self._title = title
        self._scenes = []

        # Auto-extract character names (capitalized words appearing 2+ times)
        words = re.findall(r'\b([A-Z][a-z]{2,})\b', script)
        # Filter common non-name words
        _common_words = {
            "The", "This", "That", "Then", "They", "There", "Their",
            "What", "When", "Where", "Which", "While", "With",
            "From", "Into", "After", "Before", "About", "Under",
            "Over", "Between", "Through", "During", "Against",
            "However", "Although", "Because", "Since", "Until",
            "Already", "Always", "Never", "Sometimes", "Often",
            "Scene", "Chapter", "Episode", "Part",
        }
        from collections import Counter
        name_counts = Counter(w for w in words if w not in _common_words)
        # Keep names appearing at least twice
        self._characters = {
            name: {} for name, count in name_counts.items() if count >= 2
        }

    def extract_characters(self) -> dict:
        """
        Extract all characters from the loaded story with structured profiles.

        Uses regex and NLP heuristics to find character descriptions including
        age patterns, appearance descriptions, clothing mentions, and
        distinguishing features.

        Returns:
            Dict mapping character name to profile dict with keys:
            age, ethnicity, hair, body_type, clothing, distinguishing_features.
        """
        if not self._story:
            return {}

        from ltx_pro.character import extract_character_profiles_from_script

        # Get basic profiles from the character module
        raw_profiles = extract_character_profiles_from_script(self._story, "")

        # Build structured profiles for each detected character
        profiles = {}
        for name in list(self._characters.keys()):
            profile = {
                "age": "",
                "ethnicity": "",
                "hair": "",
                "body_type": "",
                "clothing": "",
                "distinguishing_features": "",
            }

            # Search for age patterns near character name
            age_patterns = [
                rf'{name}[^.]*?(\d{{1,2}})[- ]?year[- ]?old',
                rf'(\d{{1,2}})[- ]?year[- ]?old[^.]*?{name}',
                rf'{name}[^.]*?aged?\s+(\d{{1,2}})',
            ]
            for pattern in age_patterns:
                match = re.search(pattern, self._story, re.IGNORECASE)
                if match:
                    profile["age"] = match.group(1)
                    break

            # Search for hair descriptions
            hair_patterns = [
                rf'{name}[^.]*?((?:long|short|curly|straight|wavy|blonde|brunette|red|black|brown|gray|silver|white)\s+hair)',
                rf'{name}[^.]*?hair\s+(?:was|is)\s+([^,.]+)',
            ]
            for pattern in hair_patterns:
                match = re.search(pattern, self._story, re.IGNORECASE)
                if match:
                    profile["hair"] = match.group(1).strip()
                    break

            # Search for clothing
            clothing_patterns = [
                rf'{name}[^.]*?(?:wore|wearing|dressed in|clad in)\s+([^,.]+)',
            ]
            for pattern in clothing_patterns:
                match = re.search(pattern, self._story, re.IGNORECASE)
                if match:
                    profile["clothing"] = match.group(1).strip()
                    break

            # Search for body type
            body_patterns = [
                rf'{name}[^.]*?(tall|short|slim|muscular|stocky|petite|athletic|thin|heavy|lean)',
            ]
            for pattern in body_patterns:
                match = re.search(pattern, self._story, re.IGNORECASE)
                if match:
                    profile["body_type"] = match.group(1).strip()
                    break

            # Search for distinguishing features
            feature_patterns = [
                rf'{name}[^.]*?(scar|tattoo|piercing|birthmark|glasses|eye patch|missing|prosthetic)[^,.]*',
            ]
            for pattern in feature_patterns:
                match = re.search(pattern, self._story, re.IGNORECASE)
                if match:
                    profile["distinguishing_features"] = match.group(0).strip()
                    break

            # Use raw profile string if available from character module
            if name in raw_profiles:
                profile["_raw_description"] = raw_profiles[name]

            profiles[name] = profile

        self._characters = profiles
        return profiles

    def set_series_characters(self, characters: dict) -> None:
        """
        Lock character profiles for a series (persist across episodes).

        These profiles will be used for character consistency across all
        episodes. Once set, they override per-episode extraction.

        Args:
            characters: Dict mapping character name to profile dict.
                Each profile should have keys: age, ethnicity, hair,
                body_type, clothing, distinguishing_features.
        """
        self._series_characters = dict(characters)

    def generate_episode_prompts(self, episode_number: int = 1) -> list:
        """
        Break story into scenes and generate LTX-2-ready prompts.

        Each prompt includes:
        - Character anchoring prefix (from series or episode characters)
        - Quality keywords (STORY_QUALITY_KEYWORDS from config)
        - Camera direction inferred from action verbs
        - Scene transition notes for continuity

        Args:
            episode_number: Episode number for series tracking. Used in
                output prefixes and for recap scene generation.

        Returns:
            List of scene dicts compatible with run_storyboard(). Each dict
            contains user_input, frames, seed, output_prefix, character
            metadata, and _metadata with timing/camera info.
        """
        if not self._story:
            return []

        from ltx_pro.storyboard import decompose_script_to_scenes
        from ltx_pro.config import (
            STORY_QUALITY_KEYWORDS, WIDTH, HEIGHT, FPS, SEED, FRAMES
        )

        # Determine active character profiles
        active_chars = self._series_characters if self._series_characters else self._characters

        # Build character definition string
        char_def = ""
        if active_chars:
            first_char_name = next(iter(active_chars))
            first_profile = active_chars[first_char_name]
            char_def = self._generate_character_consistency_prompt(first_profile)

        # Get resolution from aspect ratio
        width, height = _ASPECT_RATIO_MAP.get(self._aspect_ratio, (WIDTH, HEIGHT))

        # Decompose script to scenes
        scenes = decompose_script_to_scenes(
            script=self._story,
            target_duration=30,
            segment_duration=5,
            quality=self._quality,
            style=self._visual_style,
            character_def=char_def,
            fps=FPS,
        )

        # Enhance each scene with character anchoring and quality keywords
        enhanced_scenes = []
        for idx, scene in enumerate(scenes):
            # Build character anchor prefix
            anchor_prefix = ""
            if active_chars:
                for cname, cprofile in active_chars.items():
                    anchor_prefix += self._generate_character_consistency_prompt(cprofile)
                    break  # Primary character only for prefix

            # Inject quality keywords
            quality_suffix = f" {STORY_QUALITY_KEYWORDS}"

            # Update user_input with anchoring
            original_input = scene.get("user_input", "")
            scene["user_input"] = anchor_prefix + original_input + quality_suffix

            # Update output prefix with episode number
            scene["output_prefix"] = f"Ep{episode_number:02d}_Scene{idx+1:02d}"

            # Set resolution from aspect ratio
            scene["width"] = width
            scene["height"] = height

            # Add transition notes
            if idx > 0 and len(enhanced_scenes) > 0:
                transition = self._generate_scene_transition(
                    enhanced_scenes[-1], scene
                )
                meta = scene.get("_metadata", {})
                meta["transition_note"] = transition
                scene["_metadata"] = meta

            # Apply cliffhanger treatment to the last scene
            if idx == len(scenes) - 1:
                action_text = scene.get("_metadata", {}).get("action_text", "")
                if self._detect_cliffhanger(action_text):
                    scene = self._apply_cliffhanger_treatment(scene)

            enhanced_scenes.append(scene)

        self._scenes = enhanced_scenes

        # Track episode in history
        self._episode_history.append({
            "episode": episode_number,
            "title": self._title,
            "scene_count": len(enhanced_scenes),
            "characters": list(active_chars.keys()) if active_chars else [],
            "summary": self._story[:200] if self._story else "",
        })

        return enhanced_scenes

    def set_visual_style(self, style: str, quality: str = "8K cinematic") -> None:
        """
        Set visual style for the series.

        Args:
            style: Visual style name (e.g., "realistic", "cinematic noir",
                "anime", "documentary", "fantasy", "sci-fi").
            quality: Quality level (e.g., "8K cinematic", "4K professional",
                "HD broadcast", "social media").
        """
        self._visual_style = style
        self._quality = quality

    def generate_all_videos(self, scenes: list = None) -> list:
        """
        Generate all video clips for the episode.

        Calls run_storyboard() from ltx_pro.storyboard to process each
        scene sequentially with character continuity.

        Args:
            scenes: Optional list of scene dicts. If None, uses the scenes
                from the last generate_episode_prompts() call.

        Returns:
            List of output file paths (None for failed scenes).
        """
        from ltx_pro.storyboard import run_storyboard

        target_scenes = scenes if scenes is not None else self._scenes
        if not target_scenes:
            print("   No scenes to generate. Call generate_episode_prompts() first.")
            return []

        outputs = run_storyboard(
            scenes=target_scenes,
            use_continuity=True,
        )
        return outputs

    def export_production_package(self, output_dir: str) -> dict:
        """
        Export a complete production package for the episode.

        Creates a directory containing:
        - video_clips.json: Ordered list of video clip paths
        - timeline.json: JSON timeline with timestamps
        - timeline.edl: EDL file for NLE import
        - characters.json: Character reference sheets (profiles)
        - prompt_log.json: All prompts used for generation
        - continuity_notes.json: Scene transition and continuity notes
        - sora_export.json: Sora-style JSON export

        Args:
            output_dir: Directory path for the production package output.

        Returns:
            Dict with keys mapping to output file paths:
            {video_clips, timeline_json, timeline_edl, characters,
             prompt_log, continuity_notes, sora_json}
        """
        from ltx_pro.export import generate_timeline_json, generate_edl
        from ltx_pro.storyboard import generate_sora_json
        from ltx_pro.config import FPS

        os.makedirs(output_dir, exist_ok=True)
        result = {}

        # 1. Video clips list
        clips_path = os.path.join(output_dir, "video_clips.json")
        clips_data = {
            "title": self._title,
            "clips": [
                {
                    "scene": i + 1,
                    "output_prefix": s.get("output_prefix", f"Scene{i+1:02d}"),
                }
                for i, s in enumerate(self._scenes)
            ],
        }
        with open(clips_path, "w", encoding="utf-8") as f:
            json.dump(clips_data, f, indent=2)
        result["video_clips"] = clips_path

        # 2. Timeline JSON
        segments_info = []
        for i, scene in enumerate(self._scenes):
            meta = scene.get("_metadata", {})
            segments_info.append({
                "index": i,
                "prompt": scene.get("user_input", ""),
                "seed": scene.get("seed", 0),
                "frames": scene.get("frames", 121),
                "fps": FPS,
                "file_path": scene.get("output_prefix", ""),
                "resolution": f"{scene.get('width', 768)}x{scene.get('height', 512)}",
            })
        timeline_path = os.path.join(output_dir, "timeline.json")
        generate_timeline_json(segments_info, timeline_path)
        result["timeline_json"] = timeline_path

        # 3. Timeline EDL
        edl_path = os.path.join(output_dir, "timeline.edl")
        generate_edl(segments_info, edl_path, fps=FPS)
        result["timeline_edl"] = edl_path

        # 4. Character reference sheets
        chars_path = os.path.join(output_dir, "characters.json")
        active_chars = self._series_characters if self._series_characters else self._characters
        with open(chars_path, "w", encoding="utf-8") as f:
            json.dump(active_chars, f, indent=2, default=str)
        result["characters"] = chars_path

        # 5. Prompt log
        prompt_log_path = os.path.join(output_dir, "prompt_log.json")
        prompt_log = []
        for i, scene in enumerate(self._scenes):
            prompt_log.append({
                "scene": i + 1,
                "user_input": scene.get("user_input", ""),
                "output_prefix": scene.get("output_prefix", ""),
                "seed": scene.get("seed", 0),
                "frames": scene.get("frames", 0),
            })
        with open(prompt_log_path, "w", encoding="utf-8") as f:
            json.dump(prompt_log, f, indent=2)
        result["prompt_log"] = prompt_log_path

        # 6. Continuity notes
        continuity_path = os.path.join(output_dir, "continuity_notes.json")
        continuity_notes = []
        for i, scene in enumerate(self._scenes):
            meta = scene.get("_metadata", {})
            note = {
                "scene": i + 1,
                "camera": meta.get("camera_lora", "static"),
                "transition_note": meta.get("transition_note", ""),
                "has_dialogue": meta.get("has_dialogue", False),
                "timestamp_start": meta.get("timestamp_start", 0),
                "timestamp_end": meta.get("timestamp_end", 0),
            }
            continuity_notes.append(note)
        with open(continuity_path, "w", encoding="utf-8") as f:
            json.dump(continuity_notes, f, indent=2)
        result["continuity_notes"] = continuity_path

        # 7. Sora-style JSON export
        sora_path = os.path.join(output_dir, "sora_export.json")
        generate_sora_json(self._scenes, output_path=sora_path)
        result["sora_json"] = sora_path

        print(f"   Production package exported to: {output_dir}")
        print(f"   Files: {len(result)} artifacts created")

        return result

    def generate_chapter_markers(self, scenes: list = None) -> str:
        """
        Generate YouTube chapter markers from scenes.

        Outputs chapter markers in YouTube description format:
        "00:00 Scene Title\\n00:05 Next Scene..."

        Args:
            scenes: Optional list of scene dicts. If None, uses internal scenes.

        Returns:
            String with YouTube chapter markers, one per line.
        """
        target_scenes = scenes if scenes is not None else self._scenes
        if not target_scenes:
            return ""

        markers = []
        for i, scene in enumerate(target_scenes):
            meta = scene.get("_metadata", {})
            timestamp = meta.get("timestamp_start", i * 5)
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)

            # Generate scene title from action text or output prefix
            action = meta.get("action_text", "")
            if action:
                # Use first few words as title
                title_words = action.split()[:5]
                title = " ".join(title_words)
                if len(title) > 40:
                    title = title[:37] + "..."
            else:
                title = scene.get("output_prefix", f"Scene {i + 1}")

            markers.append(f"{minutes:02d}:{seconds:02d} {title}")

        return "\n".join(markers)

    def extract_thumbnail_candidates(self, scenes: list = None) -> list:
        """
        Pick the best thumbnail frame candidates per scene.

        Scores each scene based on the presence of high-visual-interest
        keywords (action, drama, close-ups, effects) and returns the
        top candidates sorted by visual interest score.

        Args:
            scenes: Optional list of scene dicts. If None, uses internal scenes.

        Returns:
            List of dicts with keys: scene_index, score, reason, prompt.
            Sorted by score descending (best thumbnail candidates first).
        """
        target_scenes = scenes if scenes is not None else self._scenes
        if not target_scenes:
            return []

        candidates = []
        for i, scene in enumerate(target_scenes):
            meta = scene.get("_metadata", {})
            action_text = meta.get("action_text", scene.get("user_input", ""))
            action_lower = action_text.lower()

            score = 0
            reasons = []
            for keyword in _THUMBNAIL_INTEREST_KEYWORDS:
                if keyword in action_lower:
                    score += 1
                    reasons.append(keyword)

            candidates.append({
                "scene_index": i,
                "score": score,
                "reason": ", ".join(reasons) if reasons else "general scene",
                "prompt": action_text[:100],
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def set_aspect_ratio(self, ratio: str) -> None:
        """
        Set the aspect ratio for video generation.

        Supported ratios:
        - '16:9': Landscape (768x512) - standard YouTube
        - '9:16': Portrait (512x768) - YouTube Shorts, TikTok
        - '1:1': Square (512x512) - social media posts

        Args:
            ratio: Aspect ratio string. Must be one of '16:9', '9:16', '1:1'.

        Raises:
            ValueError: If ratio is not a supported value.
        """
        if ratio not in _ASPECT_RATIO_MAP:
            supported = ", ".join(_ASPECT_RATIO_MAP.keys())
            raise ValueError(
                f"Unsupported aspect ratio '{ratio}'. Supported: {supported}"
            )
        self._aspect_ratio = ratio

    def generate_recap_scene(self, episode_number: int) -> dict:
        """
        Generate a 'Previously on...' recap scene from episode history.

        Creates a summary scene that references key events from previous
        episodes, suitable for opening the current episode.

        Args:
            episode_number: Current episode number. The recap will cover
                all episodes before this one.

        Returns:
            Scene dict compatible with run_storyboard(), containing a
            recap prompt with montage-style directions. Returns empty
            dict if no episode history is available.
        """
        from ltx_pro.config import STORY_QUALITY_KEYWORDS, FPS, SEED

        # Filter history for episodes before current
        prev_episodes = [
            ep for ep in self._episode_history
            if ep.get("episode", 0) < episode_number
        ]

        if not prev_episodes:
            return {}

        # Build recap text from episode summaries
        recap_parts = []
        for ep in prev_episodes[-3:]:  # Last 3 episodes max
            summary = ep.get("summary", "")
            if summary:
                # Take first sentence
                first_sentence = summary.split(".")[0] + "."
                recap_parts.append(first_sentence)

        recap_text = " ".join(recap_parts)
        if not recap_text:
            return {}

        # Get resolution
        width, height = _ASPECT_RATIO_MAP.get(self._aspect_ratio, (768, 512))

        recap_scene = {
            "user_input": (
                f"[Previously on {self._title}] "
                f"Quick montage recap: {recap_text} "
                f"Fast-paced editing, dramatic music cue, "
                f"desaturated flashback look. {STORY_QUALITY_KEYWORDS}"
            ),
            "image_path": None,
            "frames": 75,
            "seed": SEED,
            "output_prefix": f"Ep{episode_number:02d}_Recap",
            "width": width,
            "height": height,
            "character_mode": "anchor",
            "_metadata": {
                "timestamp_start": 0.0,
                "timestamp_end": 3.0,
                "camera_lora": "dolly-out",
                "has_dialogue": False,
                "segment_index": -1,
                "action_text": f"Recap of episodes 1-{episode_number - 1}",
                "is_recap": True,
            },
        }

        return recap_scene

    def _detect_cliffhanger(self, scene_text: str) -> bool:
        """
        Detect tension/suspense in a scene's text.

        Looks for indicators of unresolved tension: questions, unresolved
        action, dramatic pauses, confrontation, revelation keywords.

        Args:
            scene_text: The text content of a scene to analyze.

        Returns:
            True if cliffhanger indicators are detected, False otherwise.
        """
        if not scene_text:
            return False

        text_lower = scene_text.lower()

        # Check for cliffhanger keywords
        keyword_count = sum(
            1 for kw in _CLIFFHANGER_KEYWORDS if kw in text_lower
        )
        if keyword_count >= 2:
            return True

        # Check for questions (unresolved)
        if "?" in scene_text:
            return True

        # Check for ellipsis (dramatic pause)
        if "..." in scene_text:
            return True

        # Check for exclamation with short sentences (dramatic)
        sentences = scene_text.split(".")
        short_exclamations = [
            s for s in sentences
            if "!" in s and len(s.strip()) < 30
        ]
        if len(short_exclamations) >= 2:
            return True

        return False

    def _apply_cliffhanger_treatment(self, scene_dict: dict) -> dict:
        """
        Apply special cinematic treatment to a cliffhanger scene.

        Modifies the scene to use:
        - Slow dolly-in camera movement for tension
        - Dramatic lighting keywords
        - Suspense quality enhancement keywords

        Args:
            scene_dict: Scene dict to enhance with cliffhanger treatment.

        Returns:
            Modified scene dict with cliffhanger treatment applied.
        """
        scene = dict(scene_dict)
        meta = dict(scene.get("_metadata", {}))

        # Override camera to slow dolly-in for tension
        meta["camera_lora"] = "dolly-in"
        meta["is_cliffhanger"] = True

        # Add suspense quality keywords to the prompt
        user_input = scene.get("user_input", "")
        user_input += f" {_SUSPENSE_QUALITY_KEYWORDS}"
        scene["user_input"] = user_input

        scene["_metadata"] = meta
        return scene

    def _generate_character_consistency_prompt(self, character_profile: dict) -> str:
        """
        Build a detailed prompt prefix from a structured character profile.

        Constructs a character anchoring string that can be prepended to
        any scene prompt to maintain visual consistency.

        Args:
            character_profile: Dict with keys like age, ethnicity, hair,
                body_type, clothing, distinguishing_features.

        Returns:
            Formatted character consistency prefix string.
        """
        if not character_profile:
            return ""

        parts = []

        age = character_profile.get("age", "")
        ethnicity = character_profile.get("ethnicity", "")
        if age and ethnicity:
            parts.append(f"{age}-year-old {ethnicity}")
        elif age:
            parts.append(f"{age}-year-old")
        elif ethnicity:
            parts.append(ethnicity)

        hair = character_profile.get("hair", "")
        if hair:
            parts.append(hair)

        body_type = character_profile.get("body_type", "")
        if body_type:
            parts.append(body_type)

        clothing = character_profile.get("clothing", "")
        if clothing:
            parts.append(clothing)

        distinguishing = character_profile.get("distinguishing_features", "")
        if distinguishing:
            parts.append(distinguishing)

        if not parts:
            return ""

        details = ", ".join(parts)
        return f"[Character: {details}. Maintain exact appearance throughout.] "

    def _generate_scene_transition(self, prev_scene: dict, next_scene: dict) -> str:
        """
        Generate transition notes for smooth scene changes.

        Analyzes the camera movement and content of adjacent scenes to
        suggest appropriate transition techniques.

        Args:
            prev_scene: The preceding scene dict.
            next_scene: The following scene dict.

        Returns:
            Transition note string describing recommended transition.
        """
        prev_meta = prev_scene.get("_metadata", {})
        next_meta = next_scene.get("_metadata", {})

        prev_camera = prev_meta.get("camera_lora", "static")
        next_camera = next_meta.get("camera_lora", "static")

        # Determine transition type based on camera movement patterns
        if prev_camera == "dolly-out" and next_camera == "dolly-in":
            return "Match cut: outward to inward motion, maintain momentum"
        elif prev_camera == next_camera:
            return f"Continuous {prev_camera} movement, overlap blend 5 frames"
        elif prev_meta.get("has_dialogue") and not next_meta.get("has_dialogue"):
            return "Dialogue to action: quick cut with audio crossfade"
        elif prev_camera == "static" and next_camera != "static":
            return f"Static to {next_camera}: smooth ramp-up, 3-frame ease"
        else:
            return "Standard overlap blend, 5 frames linear crossfade"
