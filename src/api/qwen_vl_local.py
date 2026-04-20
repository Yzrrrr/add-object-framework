"""
Qwen2.5-VL local backend via mlx-vlm for scene analysis.

Runs a vision-language model on Apple Silicon (MLX) to analyze an
input image and produce a full AnalysisResult (scene description,
object suggestion, position in % coords, positive/negative prompts).
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from PIL import Image

from ..utils import log
from ..modules.analysis import AnalysisResult, Position


DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


ANALYSIS_SYSTEM_PROMPT = """You are a professional photo-editing assistant. You analyze an input photograph and propose ONE realistic object to add that would enhance the scene while looking natural and physically plausible.

You must respond with STRICT JSON only (no prose, no markdown fences). Schema:
{
  "scene_description": string,            // 1-2 sentences describing the scene, subject, lighting, style
  "object_to_add": string,                // short noun phrase, ONE object, natural for this scene
  "style": string,                        // e.g. "photorealistic", "black and white photography"
  "lighting": string,                     // short lighting description matching the image
  "position": {
    "x": integer,                         // left edge of bbox, 0-100 (percent of image width)
    "y": integer,                         // top edge of bbox, 0-100 (percent of image height)
    "width": integer,                     // bbox width, 0-100 (percent of image width)
    "height": integer,                    // bbox height, 0-100 (percent of image height)
    "reason": string                      // short justification for this placement
  },
  "positive_prompt": string,              // rich inpainting prompt (50-80 words) describing the object, its integration, lighting, style
  "negative_prompt": string               // short comma-separated list of artifacts to avoid
}

Rules for position:
- Coordinates are PERCENTAGES of the image (0-100), with (0,0) at top-left.
- x + width <= 100 and y + height <= 100.
- Size must be reasonable: most added objects use width/height between 8 and 35.
- Place the object where it would be physically plausible (ground objects on the ground, birds in the sky, accessories on people, etc.).

Rules for prompts:
- positive_prompt must repeat the object, its appearance, the scene lighting, style, and include words like "photorealistic", "seamlessly integrated", "matching lighting".
- negative_prompt is short, comma-separated: e.g. "cartoon, blurry, distorted, AI art, wrong lighting, floating, deformed".

Respond with the JSON object only."""


USER_INSTRUCTION_TEMPLATE = """Analyze this image and propose ONE realistic object to add.

{extra}

Return the JSON object now."""


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


class QwenVLLocalBackend:
    """Local Qwen2.5-VL inference via mlx-vlm."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        max_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._processor = None
        self._config = None

    def load(self) -> None:
        """Explicit model load. Safe to call multiple times (idempotent)."""
        if self._model is not None:
            return
        log.info(f"[qwen-vl] Loading {self.model_id} (first call, may take ~10-20s)...")
        from mlx_vlm import load as mlx_load
        from mlx_vlm.utils import load_config

        self._model, self._processor = mlx_load(self.model_id)
        self._config = load_config(self.model_id)
        log.info("[qwen-vl] Model loaded.")

    def _ensure_loaded(self) -> None:
        self.load()

    def _generate(self, image_path: str, user_text: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = apply_chat_template(
            self._processor, self._config, messages, num_images=1
        )

        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=[image_path],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=False,
        )
        text = getattr(result, "text", None) or str(result)
        return text

    @staticmethod
    def _extract_json(text: str) -> dict:
        match = _JSON_OBJ_RE.search(text)
        if not match:
            raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
        raw = match.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            return json.loads(cleaned)

    @staticmethod
    def _clamp_bbox(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x = max(0, min(100, int(x)))
        y = max(0, min(100, int(y)))
        w = max(1, min(100, int(w)))
        h = max(1, min(100, int(h)))
        if x + w > 100:
            w = 100 - x
        if y + h > 100:
            h = 100 - y
        return x, y, w, h

    def analyze(
        self,
        image_path: str,
        instruction: Optional[str] = None,
    ) -> AnalysisResult:
        self._ensure_loaded()

        extra = ""
        if instruction:
            extra = f"Additional user instruction: {instruction}"

        pil = Image.open(image_path)
        log.info(f"[qwen-vl] Analyzing {image_path} ({pil.size[0]}x{pil.size[1]})")

        user_text = USER_INSTRUCTION_TEMPLATE.format(extra=extra)
        raw = self._generate(image_path, user_text)
        log.info(f"[qwen-vl] Raw output ({len(raw)} chars): {raw[:240]}...")

        data = self._extract_json(raw)

        pos = data.get("position", {}) or {}
        x, y, w, h = self._clamp_bbox(
            pos.get("x", 40),
            pos.get("y", 40),
            pos.get("width", 20),
            pos.get("height", 20),
        )

        result = AnalysisResult(
            scene_description=str(data.get("scene_description", "")).strip(),
            object_to_add=str(data.get("object_to_add", "a small object")).strip(),
            style=str(data.get("style", "photorealistic")).strip(),
            lighting=str(data.get("lighting", "natural lighting")).strip(),
            suggested_position=Position(
                x=x, y=y, width=w, height=h,
                reason=str(pos.get("reason", "")).strip(),
            ),
            positive_prompt=str(data.get("positive_prompt", "")).strip(),
            negative_prompt=str(
                data.get("negative_prompt", "blurry, low quality, AI art, distorted")
            ).strip(),
        )

        log.info(
            f"[qwen-vl] -> add '{result.object_to_add}' at "
            f"({result.suggested_position.x}%,{result.suggested_position.y}%) "
            f"size {result.suggested_position.width}x{result.suggested_position.height}"
        )
        return result
