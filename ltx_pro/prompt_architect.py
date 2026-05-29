"""
Inline Prompt Architect for cinematic prompt expansion.

Contains InlinePromptArchitect class with full content-tier detection,
undressing segment enforcement, and LLM-based prompt expansion. Also provides
singleton accessor and the run_easy_prompt() wrapper for ComfyUI node calls.

All heavy imports (torch, transformers, huggingface_hub) are guarded inside
function bodies so this module passes py_compile without those packages.
"""

import re
import os
import json
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "InlinePromptArchitect",
    "_get_inline_prompt_architect",
    "run_easy_prompt",
    "_creativity_label",
]

class InlinePromptArchitect:
    """
    Embedded prompt expansion logic from LTX2EasyPromptLD.py.
    Used as fallback when the ComfyUI LTX2PromptArchitect node is not available.

    Loads LLM models (NeuralDaredevil 8B, Llama 3.2 3B, Qwen3 14B) with
    proper VRAM management (load -> generate -> unload pattern).
    """

    MODELS = {
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    # Full SYSTEM_PROMPT ported from LTX2EasyPromptLD.py (includes UNDRESSING RULE,
    # PRIORITY ORDER, WRITING RULES sections for complete tier detection support)
    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2, an AI video generation model. Your job is to expand a user's rough idea into a rich, detailed, video-ready prompt.

PRIORITY ORDER - always establish these first, in this order:
1. Video style & genre (e.g. slow-burn thriller, handheld documentary, high-fashion editorial, erotic cinema, action blockbuster)
2. Camera angle & shot type (e.g. low-angle close-up, bird's-eye wide shot, Dutch angle medium shot)
3. Character description (age MUST always be stated as a specific number e.g. "a 28-year-old woman" - never omit or approximate age; body type, hair, skin, clothing or nude state, distinguishing physical features - be extremely explicit and specific when instructed; name body parts directly using the exact words the user used, e.g. if user says "pussy" write "pussy" not "sex" or "womanhood")
4. Scene & environment (location, time of day, lighting, colour palette, textures, atmosphere)

THEN add:
5. Action & motion - describe the movement as a continuous present-tense sequence, start to finish. Structure the action in clear beats.

UNDRESSING RULE - this is mandatory when clothing removal is implied or stated:
If the subject is clothed at the start and the scene involves nudity, stripping, or explicit content, you MUST dedicate a full narrative segment to the undressing BEFORE any nudity or explicit act is described. This segment must:
  - Name each garment explicitly (shirt, bra, jeans, underwear, shoes, etc.)
  - Describe HOW it is removed - fingers on buttons, zipper pulled down, fabric sliding off shoulders, stepped out of, peeled away
  - Describe what is physically REVEALED at each step - the collarbone, the stomach, the curve of the hip, the chest, the bare legs
  - Follow the exact order the clothes come off - do not skip or compress steps
  - Use camera-aware language - the camera lingers, cuts to, tilts down to reveal, holds on
  - Do NOT jump from clothed to naked. The transition IS the content. Treat it as its own scene segment with the same density as any other beat.

6. Camera movement - describe camera motion as prose, not bracketed directions. Never write "(DOWN 10)", "(Pull back)", "(Fade to black)", "(HOLD)" or any screenplay-style bracketed camera instruction. Instead write it as description: "the camera slowly tilts down to reveal the wet pavement", "the shot pulls back to frame the empty street", "the scene fades to black as she disappears around the corner."
7. Audio - For each action beat, weave ambient sound naturally into the prose as a descriptive sentence or clause - never as a tag or label. Maximum 2 sounds active at any one time. The soundscape should evolve with the scene - each beat has its own sonic texture that matches its mood and energy. Do not stack more than 2 sounds at once or the audio will become overwhelming. Examples of correct format: "the refrigerator hums steadily in the background as she moves", "rain begins to tap softly against the window", "birdsong drifts through the gap in the curtains, barely audible over her breathing". Never write [AMBIENT: ...] tags. Sound is part of the prose, always.
   Dialogue - follow the DIALOGUE INSTRUCTION you are given exactly. When dialogue is included, write it as inline prose woven into the action - not as a labelled tag. The spoken words sit inside the sentence, attributed with delivery and physical action, exactly like a novel. Examples of correct format:
   'He leans back, satisfied, "I think I'll have to go back tomorrow for more," he chuckles, his eyes crinkling at the corners.'
   '"Don't stop," she breathes, gripping the sheets, her voice barely above a whisper.'
   'She turns to face him, "I've been waiting all day for this," her tone quiet and certain.'
   NEVER use [DIALOGUE: ...] tags. NEVER write dialogue as a separate bracketed block. Dialogue flows inside the prose as part of the action.

WRITING RULES:
- Use present tense throughout
- Be explicit and cinematic - LTX-2 responds well to dense, specific visual language
- Match detail level to shot scale: close-ups need more physical detail, wide shots need more environmental detail
- Do not use vague words like "beautiful" or "nice" - describe exactly what makes it visually striking
- Fill the full available length - do not stop early. Expand every section with rich, layered detail
- Aim for 8-12 sentences of dense, flowing prose - not a bullet list
- Write in sections separated by a single line break for clean model parsing

IMPORTANT: Output ONLY the expanded prompt. Do NOT include preamble, commentary, labels, or any explanation. Do NOT write "Sure!", "Here's your prompt:", or anything like that. Do NOT add a checklist, compliance summary, note, or any confirmation of instructions at the end - not in brackets, not as a "Note:", not in any form. Do NOT write token counts, word counts, action counts, or any meta-commentary about what you wrote. Do NOT ask for feedback or offer to revise. The output ends when the scene ends. Nothing after the last sentence of the scene. Begin immediately with the video style or shot description."""

    _PREAMBLE_RE = re.compile(
        r"^(Sure!?|Certainly!?|Absolutely!?|Of course!?|Here(?:'s| is).*?:|Great!?)[^\n]*\n?",
        re.IGNORECASE,
    )
    _ROLE_BLEED_RE = re.compile(
        r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$",
        re.IGNORECASE,
    )

    # ── Tier-detection regex patterns (compiled once at class level) ──────
    _EXPLICIT_RE = re.compile(
        r"\b(pussy|cock|dick|penis|vagina|clit|clitoris|anus|asshole|"
        r"tits|cum|jizz|squirt\w*|creampie|orgasm|fuck|fucking|"
        r"blowjob|handjob|bj|hj|breed\w*|bareback|raw\s+dog|"
        r"balls|ballsack|taint|penetrat\w*|thrust\w*)\b",
        re.IGNORECASE,
    )
    _SENSUAL_RE = re.compile(
        r"\b(naked|nude|topless|undress\w*|strip\w*|takes?\s+off|"
        r"removes?\s+(her|his|their|the)?\s*\w*\s*"
        r"(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
        r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
        r"shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress)|"
        r"titty\s+drop|titties\s+out|flash\w*\s+(her|his)?\s*(tits|titties|boobs|breasts)|"
        r"lift\w*\s+(her|his)?\s*(top|shirt)|show\w*\s+(her|his)?\s*(tits|titties|boobs)|"
        r"sensual|erotic|intimate|lingerie|bare\s+skin|bare\s+body|"
        r"braless|pantyless|commando|see.through|sheer|"
        r"bath\w*|shower\w*|changing|bikini|thong|g.string|"
        r"getting\s+(dressed|undressed|naked)|"
        r"body\s+paint\w*|titty|titties|titty\s+drop|boobs|"
        r"flash\w*\s+(her|his)?\s*(tits|titties|boobs|breasts)|"
        r"lift\w*\s+(her|his)?\s*(top|shirt)|show\w*\s+(her|his)?\s*(tits|titties|boobs))\b",
        re.IGNORECASE,
    )
    _UNDRESS_RE = re.compile(
        r"\b(undress\w*|strip\w*|takes?\s+off|"
        r"removes?\s+(her|his|their|the)?\s*\w*\s*"
        r"(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
        r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
        r"shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress)|"
        r"titty\s+drop|titties\s+out|flash\w*\s+(her|his)?\s*(tits|titties|boobs|breasts)|"
        r"lift\w*\s+(her|his)?\s*(top|shirt)|show\w*\s+(her|his)?\s*(tits|titties|boobs)|"
        r"slips?\s+out\s+of|shrugs?\s+off|steps?\s+out\s+of|"
        r"tears?\s+off|rips?\s+off|tugs?\s+down|pulls?\s+down|pushes?\s+down|"
        r"lifts?\s+(her|his)\s+(shirt|top|dress)|raises?\s+(her|his)\s+(dress|skirt)|"
        r"unhooks?|unclasps?|slides?\s+off|slips?\s+off|wriggles?\s+out\s+of|"
        r"buttons?\s+open|pops?\s+the\s+buttons?|rolls?\s+down|"
        r"still\s+dressed|fully\s+clothed|in\s+(her|his)\s+clothes|"
        r"gets?\s+undressed|gets?\s+naked|becomes?\s+naked)\b",
        re.IGNORECASE,
    )
    _ALREADY_NAKED_RE = re.compile(
        r"\b(naked|nude|topless|bare|undressed|"
        r"in\s+nothing\s+but|wearing\s+only|only\s+wearing|"
        r"just\s+out\s+of\s+the\s+shower|fresh\s+out\s+of\s+the\s+shower|"
        r"wrapped\s+in\s+a\s+towel|just\s+woke\s+up|waking\s+up)\b",
        re.IGNORECASE,
    )
    _CLOTHING_RE = re.compile(
        r"\b(wearing|dressed\s+in|clothed|shirt|dress|top|bra|pants|jeans|"
        r"skirt|blouse|jacket|coat|robe|lingerie|underwear|outfit|clothes|"
        r"gets?\s+naked|becomes?\s+naked|strip\w*|undress\w*|takes?\s+off)\b",
        re.IGNORECASE,
    )
    _MID_ACTION_RE = re.compile(
        r"\b(rubbing|touching|fingering|riding|sucking|licking|stroking|"
        r"grinding|bouncing|moaning|climax\w*|orgasm\w*|masturbat\w*|"
        r"already\s+naked|already\s+nude|already\s+undressed|"
        r"in\s+bed|on\s+the\s+bed|on\s+her\s+knees|on\s+his\s+knees|"
        r"spread\s+(her|his)\s+legs?|legs?\s+spread|her\s+legs\s+open|"
        r"sitting\s+on\s+(him|her|his|a)|"
        r"from\s+behind|doggy\s*style|doggy|"
        r"legs?\s+wrapped\s+around|wrapped\s+(her|his)\s+legs?|"
        r"on\s+top\s+of\s+(him|her)|between\s+(her|his)\s+legs?|"
        r"mid.sex|mid.act|mid.scene|after\s+sex|post.sex|"
        r"lying\s+(there|naked|nude)|bare\s+(back|chest|skin|legs?|arms?)|"
        r"exposed\s+(skin|body|chest|back)|"
        r"sunbath\w*|posing\s+(nude|naked)|"
        r"inside\s+(her|him)|penetrat\w*)\b",
        re.IGNORECASE,
    )
    _FLASH_RE = re.compile(
        r"\b(titty\s+drop|titties\s+out|"
        r"flash\w*\s+(her|his)?\s*(tits|titties|boobs|breasts)|"
        r"lift\w*\s+(her|his)?\s*(top|shirt)|"
        r"show\w*\s+(her|his)?\s*(tits|titties|boobs))\b",
        re.IGNORECASE,
    )
    _SEQUENCE_RE = re.compile(r"^\s*(\d+[\.\):])\s+.+", re.MULTILINE)
    _PERSON_RE = re.compile(
        r"\b(he|she|his|her|him|they|them|their|man|men|woman|women|girl|girls|boy|boys|guy|guys|"
        r"person|people|couple|figure|character|actress|actor|"
        r"someone|anybody|stranger|friend|lover|wife|husband|partner|spouse|"
        r"boyfriend|girlfriend|teenager|teenagers|adult|adults|female|male|"
        r"blonde|brunette|redhead|nude|naked|"
        r"singer|dancer|performer|athlete|soldier|worker|"
        r"player|nurse|doctor|student|teacher|child|children|kid|kids|"
        r"crowd|audience|escort|mistress|dominatrix|sub|submissive|"
        r"friends|friend|group|gang|party|crew|team|pair|duo)\b",
        re.IGNORECASE,
    )
    _MULTI_RE = re.compile(
        r"\b(two\s+(women|men|people|girls|guys|characters|figures)|"
        r"both\s+(of\s+them|women|men|girls|guys)|"
        r"(she|he)\s+and\s+(she|he|her|him)|"
        r"(a\s+man\s+and\s+a\s+woman|a\s+woman\s+and\s+a\s+man)|"
        r"couple|trio|they\s+(kiss|touch|embrace|undress|fuck|have))\b",
        re.IGNORECASE,
    )
    _STATIC_RE = re.compile(
        r"\b(static|locked.off|locked off|fixed|stationary|no camera movement|"
        r"camera still|still camera|camera locked|tripod shot|tripod|"
        r"fixed camera|fixed shot|static shot|static camera)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded_model_key = None

    def _load_model(self, model_key, offline_mode=False, local_path=""):
        """Load the specified LLM model."""
        if self.model is not None and self.loaded_model_key == model_key:
            return
        if self.model is not None:
            self._unload_model()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ltx_pro.vram import _VRAM_MGR

        hf_id = self.MODELS.get(model_key, self.MODELS["8B"])
        source = local_path.strip() if local_path and local_path.strip() else hf_id

        if not offline_mode and not (local_path and local_path.strip()):
            try:
                from huggingface_hub import snapshot_download
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=offline_mode)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            trust_remote_code=True, local_files_only=offline_mode)
        self.model.eval()
        self.loaded_model_key = model_key

        # Register LLM stage in VRAMManager for offload tracking
        try:
            _VRAM_MGR.register_stage("LLM", 8.0 if "14B" in model_key else 5.0)
        except Exception:
            pass  # VRAMManager may not be initialized yet

    def _unload_model(self):
        """Unload model and free VRAM."""
        if self.model is not None:
            from ltx_pro.vram import _deep_unload_model
            _deep_unload_model(self.model, label="InlinePromptArchitect")
            del self.model
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.loaded_model_key = None
        # Release LLM stage from VRAMManager
        try:
            from ltx_pro.vram import _VRAM_MGR
            _VRAM_MGR.release_stage("LLM")
        except Exception:
            pass

    @staticmethod
    def _clean_output(text):
        """Strip LLM preamble, role-token bleed, and compliance checklists."""
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = InlinePromptArchitect._PREAMBLE_RE.sub("", text)
        text = InlinePromptArchitect._ROLE_BLEED_RE.sub("", text)
        text = re.sub(r"\.(assistant|user|system|<\|[^|>]*\|>)\s*\n", ".\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\n+Note:.*$", "", text, flags=re.DOTALL).strip()

        # Strip everything AFTER the AMBIENT tag if one still appears
        ambient_match = re.search(r"\[AMBIENT:[^\]]*\]", text, flags=re.IGNORECASE)
        if ambient_match:
            text = text[:ambient_match.end()].strip()

        # Strip trailing (Lora: ...) tags the model echoes from the LoRA instruction
        text = re.sub(r"\s*\(Lora:[^)]*\)\s*$", "", text, flags=re.IGNORECASE).strip()

        # Strip trailing (Note: ...) blocks and everything after
        text = re.sub(r"\s*\(Note:.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Strip instruction labels that leaked into output
        text = re.sub(
            r"^(Action Beat \d+:|Undressing Segment:|Flash/Reveal Segment:|Titty Drop[^:]*:|Note:|Scene Instruction:|Pacing:|Dialogue Instruction:).*",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

        # Strip orphaned closing bracket spam: ) ) ) ) ) ...
        text = re.sub(r"[\s)]{3,}$", "", text).strip()

        # Catch the fake conversation / self-eval patterns
        text = re.sub(
            r"\s*\n+\d+\s+tokens[\s,].*$",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        text = re.sub(
            r"\s*\n+(Please let me know|Let me revise|No further revision|Confirmed\.|"
            r"Written to meet|The scene is now over|The output ends|The task is|The task was|"
            r"The goal was|Nothing more|No continuation|No additional|The response does not|"
            r"It does not continue|It ceases when|Any such statement|"
            r"Output length:|Action count:|Total time:|Last character:|I avoided|I wrote|"
            r"I adhered|I hope this|Thank you for your|Please confirm|I submitted|"
            r"I can revise|feel free to instruct).*$",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Strip model loop/panic patterns
        text = re.sub(
            r"\s*(Ended\.\s*\d+\s*actions|"
            r"\d+\s+actions[\.,]\s*\d+\s+tokens|"
            r"\d+\s+tokens[\.,]\s*Done|"
            r"Done\.\s+\d+\s+seconds|"
            r"Finished\.\s+\d+|"
            r"The end\.\s+\d+\s+seconds|"
            r"Fading to black\.\s+The end|"
            r"The model stops|The output ends here|The scene ends here|"
            r"It\'s complete now|All done\.|Stop now\.|"
            r"End of prompt|End of output|No more to add|Nothing to revise|"
            r"The work is (?:done|finished|complete)|The prompt is (?:done|finished|complete)|"
            r"No further writing|No more writing|Stop\.\s+Finish|Finished\.\s+Complete|"
            r"The scene is complete|The scene is over|Complete\.\s+Finished|"
            r"Done\.\s+No more|BorderSide:).*$",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Strip filler character spam (repeated single characters)
        text = re.sub(r"(\s*\b(\w)\b\s*){10,}", " ", text).strip()

        # Strip token+action count combos
        text = re.sub(r"\s*\(\d+\s+tokens?[^)]*\)", "", text, flags=re.IGNORECASE).strip()

        # Strip compliance checklist spam - 2+ consecutive parens after last sentence
        text = re.sub(r"\s*(\([^)]{5,120}\)\s*){2,}$", "", text, flags=re.DOTALL).strip()

        # Strip single trailing compliance paren with known instruction keywords
        text = re.sub(
            r"\s*\([^)]{0,200}(no setup|no resolution|action count|actions adhered|"
            r"token count|pacing|dialogue integrated|character age|inline prose|"
            r"no padding|no extraneous|exactly \d+ action|hard stop|BorderSide)[^)]{0,200}\)\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # Strip leaked pacing instruction echoes
        text = re.sub(r"\(Exact timing:.*?\)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Strip token/word count lines
        text = re.sub(r"\s*\n*(token|word)\s+count\s*:\s*\d+.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

        # Strip leaked internal pacing/time tags
        text = re.sub(r"\[TIME LIMIT[^\]]*\]", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\[PACING[^\]]*\]", "", text, flags=re.IGNORECASE).strip()

        # Strip leaked timestamp annotations
        text = re.sub(r"\s*\(\d+\s+seconds?\)\s*$", "", text).strip()
        text = re.sub(r"\s*\(\d+:\d+\s*[-\u2013]\s*\d+:\d+\)\s*", " ", text).strip()

        # Strip inline action-time annotations
        text = re.sub(r"\(The action takes up roughly[^\)]*\)", " ", text, flags=re.IGNORECASE).strip()

        # Strip screenplay-style bracketed camera directions
        text = re.sub(r"\((?:DOWN|UP|PULL|PUSH|ZOOM|HOLD|FADE|PAN|TILT|TRUCK|DOLLY|AMBIENT)[^\)]{0,80}\)", "", text, flags=re.IGNORECASE).strip()

        # Convert [AMBIENT: ...] tags to clean prose
        text = re.sub(r"\[AMBIENT:\s*([^\]]*)\]", r"\1", text, flags=re.IGNORECASE).strip()

        # Clean up any double blank lines left by removals
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @staticmethod
    def _build_negative_prompt(result, user_input):
        """Build a scene-aware negative prompt without a second LLM call."""
        _NEG_BASE = (
            "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
            "static, no motion, frozen, duplicate, watermark, text, signature, "
            "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
            "missing limbs, floating limbs, disconnected body parts, "
            "overexposed, underexposed, grainy, noise"
        )
        combined = (result + " " + user_input).lower()
        extras = []
        if any(w in combined for w in ["indoor", "room", "interior", "bedroom"]):
            extras.append("harsh outdoor lighting, direct sunlight")
        elif any(w in combined for w in ["outdoor", "street", "beach", "forest"]):
            extras.append("studio background, indoor lighting")
        if any(w in combined for w in ["night", "dark", "moonlight", "dimly lit"]):
            extras.append("overexposed, bright daylight, blown highlights")
        elif any(w in combined for w in ["daylight", "sunny", "golden hour"]):
            extras.append("underexposed, dark shadows, black crush")
        if any(w in combined for w in ["close-up", "portrait", "face shot"]):
            extras.append("wide angle distortion, fish eye, full body shot")
        # Explicit content negatives
        if any(w in combined for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "explicit", "nipple", "breast"]):
            extras.append("censored, mosaic, pixelated, black bar, blurred genitals")
        # Wide shot negatives
        if any(w in combined for w in ["wide shot", "wide angle", "aerial", "bird's-eye", "establishing"]):
            extras.append("close-up, portrait crop, tight frame")
        # Multi-character negatives
        if any(w in combined for w in ["two women", "two men", "two people", "both", "together", "couple", "they "]):
            extras.append("merged bodies, fused figures, incorrect number of people")
        parts = [_NEG_BASE] + extras
        return ", ".join(parts)

    def _build_stop_token_ids(self):
        """Build stop token IDs for generation."""
        delimiter_strings = [
            "assistant", "user", "system", "<|eot_id|>",
            "<|end_of_turn|>", "<|im_end|>", "<end_of_turn>",
            "[/INST]", "### Human", "### Assistant",
        ]
        stop_ids = [self.tokenizer.eos_token_id]
        for s in delimiter_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        seen = set()
        unique = []
        for tid in stop_ids:
            if tid is not None and tid not in seen:
                seen.add(tid)
                unique.append(tid)
        return unique

    def generate(self, user_input, frame_count=192, seed=-1, creativity=0.9,
                 model_key="8B", scene_context="", lora_triggers="",
                 offline_mode=False, local_path="", keep_loaded=False,
                 invent_dialogue=True):
        """
        Generate an expanded cinematic prompt from user input.

        Full content-tier detection ported from LTX2EasyPromptLD.py with all
        regex patterns (_explicit_re, _sensual_re, _undress_re, _already_naked_re,
        _clothing_re, _mid_action_re), undressing segment enforcement, flash/titty-drop
        detection, sequence detection, person detection, multi-subject detection,
        dialogue instruction variants, and static camera detection.

        Args:
            user_input: Simple scene description
            frame_count: Number of frames (controls pacing)
            seed: Random seed (-1 for random)
            creativity: Temperature (0.7, 0.9, or 1.1)
            model_key: "8B", "3B", or "14B"
            scene_context: Optional vision description context
            lora_triggers: LoRA trigger words to prepend
            offline_mode: Use cached models only
            local_path: Path to local model snapshot
            keep_loaded: Keep model in VRAM after generation
            invent_dialogue: Whether to invent dialogue for characters

        Returns:
            Tuple of (expanded_prompt, negative_prompt)
        """
        import torch
        self._load_model(model_key, offline_mode, local_path)

        # --- Timing & pacing (from LTX2EasyPromptLD.py) ---
        real_seconds = frame_count / 24.0
        action_count = max(1, min(10, round(real_seconds / 4)))

        if action_count == 1:
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY 1 action. One single moment. "
                f"Do not describe anything before or after it. No setup, no resolution. "
                f"HARD STOP after the 1st action. Do not continue."
            )
        else:
            ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY {action_count} distinct actions - NO MORE THAN {action_count}. "
                f"Each action takes roughly {real_seconds / action_count:.0f} seconds of screen time. "
                f"Do not add setup, backstory, or resolution beyond these {action_count} actions. "
                f"HARD STOP after the {ordinal} action is complete."
            )

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Dynamic token budget ---
        token_val = max(256, min(1200, action_count * 120))
        max_tokens_actual = int(token_val * 1.05)
        min_tokens = int(token_val * 0.75)

        # --- Build effective input with scene context ---
        if scene_context and scene_context.strip():
            effective_input = (
                f"[SCENE CONTEXT FROM IMAGE - use this as the authoritative description "
                f"of the subject and setting; do not invent or contradict it]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION - apply this as action, style, and mood over the above scene]\n"
                f"{user_input.strip()}"
            )
        else:
            effective_input = user_input.strip()

        # --- LoRA trigger injection ---
        lora_instruction = ""
        if lora_triggers and lora_triggers.strip():
            lora_instruction = (
                f"\n[LORA INSTRUCTION: You MUST begin the prompt output with these exact trigger words "
                f"before anything else: {lora_triggers.strip()} - place them as the very first words of your output, "
                f"then continue with the scene description immediately after.]"
            )

        # ══════════════════════════════════════════════════════════════════
        # FULL CONTENT-TIER DETECTION (ported from LTX2EasyPromptLD.py)
        # Includes all regex patterns and undressing segment enforcement
        # ══════════════════════════════════════════════════════════════════

        # Tier 3 triggers: direct anatomical / act terms (class-level compiled)
        _explicit_re = self._EXPLICIT_RE
        # Tier 2 triggers: nudity/sensuality implied but not explicit (class-level compiled)
        _sensual_re = self._SENSUAL_RE

        is_explicit = bool(_explicit_re.search(user_input))
        is_sensual = bool(_sensual_re.search(user_input)) and not is_explicit

        # Undressing detection for mandatory segment rule in tier 2 and tier 3
        # (class-level compiled patterns)
        has_undressing = bool(self._UNDRESS_RE.search(user_input))

        # Already-naked detection - subject starts the scene undressed
        is_already_naked = (
            bool(self._ALREADY_NAKED_RE.search(user_input)) and
            not bool(self._CLOTHING_RE.search(user_input))
        )

        # Mid-action detection - scene already in progress
        is_mid_action = bool(self._MID_ACTION_RE.search(user_input)) or is_already_naked

        # If mid-action or already naked, never fire the undressing segment
        if is_mid_action:
            has_undressing = False

        # Build tier-specific instruction with full undressing enforcement
        if is_explicit:
            undress_clause_t3 = (
                "UNDRESSING SEGMENT - MANDATORY: If the subject starts clothed, you MUST write a dedicated undressing segment "
                "before ANY explicit act or nudity. Go garment by garment: name it, describe how it is removed, "
                "describe what body part is revealed and how it looks. The camera must linger on each reveal. "
                "Do not compress. Do not skip to naked. The undressing IS the scene. "
            ) if not is_mid_action else (
                "The subject is already undressed or mid-act - do NOT write an undressing segment. "
                "Begin directly with the action as described. "
            )
            tier_instruction = (
                "\n[INSTRUCTION: This is an explicit adult scene. "
                "You MUST use direct anatomical terms - pussy, cock, anus, clit, ass - exactly as written. "
                "Do not replace them with euphemisms or metaphors. "
                + undress_clause_t3 +
                "Always state the character's age as a specific number. "
                "Describe body parts, positions, and physical actions with blunt, specific language.]"
            )
        elif is_sensual:
            # Detect flash/titty-drop specifically (class-level compiled)
            is_flash = bool(self._FLASH_RE.search(user_input))

            if is_flash:
                undress_clause = (
                    "TITTY DROP / FLASH SEGMENT - THIS IS THE ENTIRE SCENE. "
                    "Write ONLY: 1. Hands on fabric 2. Lift/pull motion 3. Breasts revealed with gravity drop 4. Hold pose. "
                    "Do NOT add sexual acts or describe genitals. Just the lift, the drop, the reveal. "
                )
            elif has_undressing:
                undress_clause = (
                    "UNDRESSING SEGMENT - MANDATORY: Write a dedicated undressing beat - garment by garment, "
                    "describing how each is removed and what is revealed. Camera lingers on each reveal. "
                )
            else:
                undress_clause = ""
            tier_instruction = (
                "\n[INSTRUCTION: This scene involves nudity or sensual content. "
                "Describing the bare body naturally is appropriate. "
                "Do NOT introduce sexual acts the user did not ask for. "
                "Keep the tone sensual and cinematic, not pornographic. "
                "Always state character age as a specific number. "
                + undress_clause + "]"
            )
        else:
            tier_instruction = (
                "\n[INSTRUCTION: Always state the character's age as a specific number.]"
            )

        # --- Sequence detection (class-level compiled) ---
        sequence_steps = self._SEQUENCE_RE.findall(user_input)
        if len(sequence_steps) >= 2:
            step_count = len(sequence_steps)
            sequence_instruction = (
                f"\n[SEQUENCE INSTRUCTION: The user has provided {step_count} numbered steps. "
                f"Follow them in exact order. Do not reorder, skip, or merge steps.]"
            )
        else:
            sequence_instruction = ""

        # --- Person detection (class-level compiled) ---
        has_person = bool(self._PERSON_RE.search(user_input + " " + scene_context))
        if not has_person:
            no_person_instruction = (
                "\n[SCENE INSTRUCTION: No person described. Do NOT invent characters. "
                "Pure environment/object scene only. No dialogue, no voices.]"
            )
        else:
            no_person_instruction = ""

        # --- Multi-subject detection (class-level compiled) ---
        has_multi_subject = bool(self._MULTI_RE.search(user_input + " " + scene_context))
        if has_multi_subject:
            multi_instruction = (
                "\n[MULTI-SUBJECT: Two or more people. Track each person's position "
                "and use consistent descriptors - not just 'she'/'he'.]"
            )
        else:
            multi_instruction = ""

        # --- Dialogue instruction (full variant from LTX2EasyPromptLD.py) ---
        if not has_person:
            dialogue_instruction = ""
        elif invent_dialogue:
            dialogue_instruction = (
                "\n\n[DIALOGUE INSTRUCTION: Invent natural dialogue woven into action as inline prose. "
                "Never use [DIALOGUE: ...] tags. Dialogue is part of the prose, always.]"
            )
        else:
            has_user_dialogue = bool(re.search(r'["\u201c\u201d]', user_input))
            if has_user_dialogue:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: Use ONLY the user's dialogue. "
                    "Place their exact words naturally as inline prose with attribution.]"
                )
            else:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: No dialogue. Weave ambient sound as prose instead.]"
                )

        # --- Static camera detection (class-level compiled) ---
        if self._STATIC_RE.search(user_input):
            camera_instruction = (
                "\n[CAMERA: Static locked-off shot. No camera movement whatsoever. "
                "All motion comes from the subject only.]"
            )
        else:
            camera_instruction = ""

        # --- Pacing enforcement ---
        length_instruction = (
            f"\n[PACING - MANDATORY: {pacing_hint} "
            f"Write approximately {token_val} tokens total. "
            f"Do not exceed the action count. "
            f"Do NOT write token count, word count, or any parenthetical summary at the end.]"
        )

        # --- Build messages with all instructions ---
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": effective_input + sequence_instruction +
             no_person_instruction + multi_instruction + dialogue_instruction +
             tier_instruction + lora_instruction + camera_instruction + length_instruction},
        ]

        is_qwen3 = "Qwen3" in (model_key or "")
        raw = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False if is_qwen3 else None)

        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self.model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self.model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self.model.device)
        else:
            input_ids = raw.to(self.model.device)

        input_length = input_ids.shape[1]
        stop_ids = self._build_stop_token_ids()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                min_new_tokens=min_tokens,
                max_new_tokens=max_tokens_actual,
                temperature=creativity,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.07,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )

        generated_tokens = output_ids[0][input_length:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        del output_ids, input_ids

        result = self._clean_output(result)
        # Strip any lone trailing bracket left by model
        result = re.sub(r'\s*[\(\[]\s*$', '', result).strip()
        neg_prompt = self._build_negative_prompt(result, user_input)

        if not keep_loaded:
            self._unload_model()

        return (result, neg_prompt)



# Global inline prompt architect instance (lazy-loaded)
_INLINE_PROMPT_ARCHITECT = None

def _get_inline_prompt_architect():
    """Get or create the singleton InlinePromptArchitect instance."""
    global _INLINE_PROMPT_ARCHITECT
    if _INLINE_PROMPT_ARCHITECT is None:
        _INLINE_PROMPT_ARCHITECT = InlinePromptArchitect()
    return _INLINE_PROMPT_ARCHITECT


_LLM_LABEL_MAP = {
    "8B":  "8B - NeuralDaredevil (High Quality)",
    "3B":  "3B - Llama-3.2 Abliterated (Low VRAM)",
    "14B": "14B - Qwen3 Abliterated (High VRAM)",
}
_VISION_LABEL_MAP = {
    "3B-fast": "Qwen2.5-VL-3B — Fast (huihui abliterated)",
    "7B-nsfw": "Qwen2.5-VL-7B — Better NSFW (prithiv caption)",
}
_CREATIVITY_MAP = {
    0.7: "0.7 - Literal & Grounded",
    0.9: "0.9 - Balanced Professional",
    1.1: "1.1 - Artistic Expansion",
}

def _creativity_label(c: float) -> str:
    """Map a numeric creativity value to its display label string."""
    closest = min(_CREATIVITY_MAP.keys(), key=lambda x: abs(x - c))
    return _CREATIVITY_MAP[closest]


def run_easy_prompt(user_input: str, frame_count: int, seed: int,
                    scene_context: str = "",
                    llm_model_override: str = None) -> Tuple[str, str]:
    """
    Calls LTX2PromptArchitect (node type: LTX2PromptArchitect from LTX2EasyPrompt-LD)
    to expand a simple story description into a dense cinematic prompt.

    Falls back to returning the raw input if the node is unavailable.
    LLM is loaded, run, then unloaded to free VRAM for the video model.

    Args:
        user_input: Simple scene description
        frame_count: Number of video frames
        seed: Random seed
        scene_context: Optional vision context
        llm_model_override: Override the LLM_MODEL global for this call

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}

    if "LTX2PromptArchitect" not in NODE_CLASS_MAPPINGS:
        print("   Warning: LTX2PromptArchitect not found - using raw user_input.")
        return user_input, ""

    from ltx_pro.vram import cleanup_memory

    LLM_MODEL = globals().get("LLM_MODEL", "3B")
    CREATIVITY = globals().get("CREATIVITY", 0.9)
    INVENT_DIALOGUE = globals().get("INVENT_DIALOGUE", True)
    LORA_TRIGGERS = globals().get("LORA_TRIGGERS", "")
    LLM_OFFLINE_MODE = globals().get("LLM_OFFLINE_MODE", False)
    LOCAL_PATH_3B = globals().get("LOCAL_PATH_3B", "")
    LOCAL_PATH_8B = globals().get("LOCAL_PATH_8B", "")
    LOCAL_PATH_14B = globals().get("LOCAL_PATH_14B", "")

    _model = llm_model_override if llm_model_override is not None else LLM_MODEL
    print(f"   [EasyPrompt] LLM={_model} | creativity={CREATIVITY} | frames={frame_count}")
    node = NODE_CLASS_MAPPINGS["LTX2PromptArchitect"]()
    _offline = LLM_OFFLINE_MODE
    _lp_3b = LOCAL_PATH_3B if _offline else ""
    _lp_8b = LOCAL_PATH_8B if _offline else ""
    _lp_14b = LOCAL_PATH_14B if _offline else ""
    if _offline:
        print("   [EasyPrompt] Offline mode: loading from local path...")
    result = node.generate(
        bypass=False,
        user_input=user_input,
        creativity=_creativity_label(CREATIVITY),
        seed=seed,
        invent_dialogue=INVENT_DIALOGUE,
        keep_model_loaded=False,
        offline_mode=_offline,
        frame_count=frame_count,
        model=_LLM_LABEL_MAP.get(_model, "8B - NeuralDaredevil (High Quality)"),
        local_path_8b=_lp_8b,
        local_path_3b=_lp_3b,
        local_path_14b=_lp_14b,
        scene_context=scene_context,
        lora_triggers=LORA_TRIGGERS,
    )
    prompt = result[0]
    neg_prompt = result[2]
    print(f"   [EasyPrompt] Done: {len(prompt.split())} words generated.")
    cleanup_memory()
    return prompt, neg_prompt



