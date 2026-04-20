"""
Analysis Module - Scene Understanding and Object Placement

Provides predefined scene analysis for demo images with natural language
edit instructions for wanx description_edit.
"""

from typing import Optional
from dataclasses import dataclass

from ..utils import log


@dataclass
class Position:
    """Suggested position for object placement (percentage coordinates 0-100)"""
    x: int
    y: int
    width: int
    height: int
    reason: str = ""


@dataclass
class AnalysisResult:
    """Complete scene analysis result"""
    scene_description: str
    object_to_add: str
    style: str
    lighting: str
    suggested_position: Position
    positive_prompt: str  # For inpainting (fallback)
    negative_prompt: str  # For inpainting (fallback)
    edit_instruction: str = ""  # Natural language instruction for description_edit


# Predefined analysis for demo images.
# edit_instruction is the key - it's a natural language description of WHERE and WHAT to add.
DEMO_ANALYSIS = {
    "demo1.jpg": AnalysisResult(
        scene_description="A blue tugboat moored on calm water with a grassy hill and trees in the background, overcast natural daylight",
        object_to_add="a seagull",
        style="photorealistic",
        lighting="soft overcast daylight, diffuse shadows",
        suggested_position=Position(x=25, y=12, width=16, height=12, reason="Open sky area above the hill, natural for a flying seagull"),
        positive_prompt="a single white seagull flying, photorealistic",
        negative_prompt="cartoon, blurry",
        edit_instruction="在画面左上方天空中添加一只飞翔的海鸥",
    ),
    "demo2.jpg": AnalysisResult(
        scene_description="Two tourists at a white temple with ornate red-gold doorway, warm natural light, one person taking a photo with a DSLR camera",
        object_to_add="a stray cat",
        style="photorealistic",
        lighting="warm afternoon sunlight, gentle shadows on white walls",
        suggested_position=Position(x=14, y=84, width=20, height=18, reason="Bottom-left stone floor near the red temple arch, where a stray would naturally rest"),
        positive_prompt="a stray cat sitting on stone floor, photorealistic",
        negative_prompt="cartoon, blurry",
        edit_instruction="在画面左下角石板地上添加一只坐着的流浪猫",
    ),
    "demo3.jpg": AnalysisResult(
        scene_description="A black kitten with a vibrant orange-and-green patterned collar and a purple bell, gray neutral background, soft window light",
        object_to_add="a small flower",
        style="photorealistic",
        lighting="soft diffused window light from the left",
        suggested_position=Position(x=36, y=70, width=14, height=14, reason="On the orange collar to the side of the purple bell"),
        positive_prompt="a small daisy flower tucked into collar, photorealistic",
        negative_prompt="cartoon, blurry",
        edit_instruction="在小猫的项圈上添加一朵小雏菊",
    ),
    "demo4.jpg": AnalysisResult(
        scene_description="Black and white intimate portrait of a woman with curly black hair and a man behind her, dramatic low-key lighting",
        object_to_add="a white flower",
        style="black and white photography",
        lighting="dramatic low-key studio lighting, strong contrast",
        suggested_position=Position(x=52, y=18, width=14, height=14, reason="Pinned into the curly hair above the woman's right temple"),
        positive_prompt="a white gardenia flower in hair, black and white",
        negative_prompt="color, cartoon",
        edit_instruction="在画面中间女性的卷发上添加一朵白色花朵",
    ),
    "demo5.jpg": AnalysisResult(
        scene_description="A hiker with turquoise backpack viewing snow-capped Dolomite mountains, bright daylight with blue sky",
        object_to_add="an eagle",
        style="photorealistic",
        lighting="bright alpine daylight, crisp blue sky",
        suggested_position=Position(x=78, y=15, width=14, height=12, reason="Upper-right sky area, natural soaring position for an eagle"),
        positive_prompt="a golden eagle soaring with spread wings, photorealistic",
        negative_prompt="cartoon, blurry",
        edit_instruction="在画面右上角天空中添加一只翱翔的老鹰",
    ),
}


class AnalysisModule:
    """
    Scene analysis module.
    
    Modes:
      - "predefined": use DEMO_ANALYSIS for known files
      - "vlm_local": use local Qwen2.5-VL (MLX) model
    """

    def __init__(
        self,
        mode: str = "predefined",
        vlm_model_id: Optional[str] = None,
    ):
        self.mode = mode
        self.vlm_model_id = vlm_model_id
        self._vlm = None
        log.info(f"AnalysisModule initialized in '{mode}' mode")

        if mode == "vlm_local":
            self._ensure_vlm()

    def _ensure_vlm(self):
        if self._vlm is not None:
            return self._vlm
        from ..api.qwen_vl_local import QwenVLLocalBackend, DEFAULT_MODEL
        model_id = self.vlm_model_id or DEFAULT_MODEL
        self._vlm = QwenVLLocalBackend(model_id=model_id)
        self._vlm.load()
        return self._vlm

    def analyze(self, image_path: str, instruction: Optional[str] = None) -> AnalysisResult:
        """Analyze an image and return placement suggestion."""
        from pathlib import Path
        image_name = Path(image_path).name

        if self.mode == "predefined":
            if image_name in DEMO_ANALYSIS:
                result = DEMO_ANALYSIS[image_name]
                log.info(f"Using predefined analysis for {image_name}: add '{result.object_to_add}'")
                return result
            log.warning(f"No predefined analysis for {image_name}, using generic default")
            return self._generic_default(instruction)

        if self.mode in ("vlm_local", "auto"):
            if image_name in DEMO_ANALYSIS:
                result = DEMO_ANALYSIS[image_name]
                log.info(f"Using predefined analysis for {image_name}: add '{result.object_to_add}'")
                return result
            
            vlm = self._ensure_vlm()
            try:
                return vlm.analyze(image_path, instruction=instruction)
            except Exception as e:
                log.error(f"VLM analysis failed ({e}); falling back to generic default")
                return self._generic_default(instruction)

        raise ValueError(f"Unknown analysis mode: {self.mode}")

    @staticmethod
    def _generic_default(instruction: Optional[str]) -> AnalysisResult:
        obj = instruction or "a small object"
        return AnalysisResult(
            scene_description="Unknown scene",
            object_to_add=obj,
            style="photorealistic",
            lighting="natural lighting",
            suggested_position=Position(x=50, y=50, width=20, height=20, reason="Default center placement"),
            positive_prompt=f"a realistic {obj}, photorealistic",
            negative_prompt="blurry, low quality, distorted",
            edit_instruction=f"在画面中间添加{obj}",
        )
