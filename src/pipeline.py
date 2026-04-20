"""
Add Object Pipeline - Main Orchestration

Cloud-only pipeline using:
  Analysis (Qwen2.5-VL) -> Edit (wanx description_edit) -> PixelAlignment -> Output
"""

import json
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

from .utils import log, load_image, setup_logging
from .modules.analysis import AnalysisModule
from .modules.planning import PlanningModule
from .modules.editor import EditModule
from .modules.blender import BlendModule, BlendMode, BlendConfig
from .modules.output import OutputModule, OutputResult


@dataclass
class PipelineConfig:
    """Configuration for the pipeline - Cloud-only version"""
    # Analysis settings
    analysis_mode: str = "predefined"  # "predefined" or "vlm_local"
    vlm_model_id: Optional[str] = None
    
    # Blend settings (PixelAlignment)
    blend_mode: str = "gaussian"
    blend_radius: int = 21
    dilation_kernel: int = 25
    
    # Output settings
    output_dir: str = "results"

    def __repr__(self):
        return (f"PipelineConfig(analysis={self.analysis_mode}, "
                f"blend={self.blend_mode})")


class AddObjectPipeline:
    """
    Complete Add Object pipeline.
    
    Uses wanx2.1-imageedit inpainting for object generation.
    Wanx API guarantees non-mask regions remain unchanged.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        setup_logging(log_level="INFO")
        
        log.info(f"Initializing pipeline: {self.config}")
        
        # Analysis: Image understanding
        self.analysis = AnalysisModule(
            mode=self.config.analysis_mode,
            vlm_model_id=self.config.vlm_model_id,
        )
        
        # Planning: Generate edit mask and prompts
        self.planning = PlanningModule(dilation_kernel_size=self.config.dilation_kernel)
        
        # Edit: wanx description_edit
        self.editor = EditModule()
        
        # Blend: PixelAlignment
        self.blender = BlendModule(
            BlendConfig(
                mode=BlendMode(self.config.blend_mode),
                blur_radius=self.config.blend_radius,
            )
        )
        
        # Output: Save and evaluate
        self.output = OutputModule()
        
        log.info("Pipeline ready (Cloud: wanx2.1-imageedit inpainting)")
    
    def run(
        self,
        image_path: str,
        instruction: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> OutputResult:
        """
        Run the pipeline on a single image.
        
        Args:
            image_path: Path to input image
            instruction: Optional override instruction
            output_name: If set, saves results
        
        Returns:
            OutputResult with final image and metrics
        """
        log.info(f"{'=' * 60}")
        log.info(f"Processing: {Path(image_path).name}")
        log.info(f"{'=' * 60}")
        
        # Step 1: Load image
        original = load_image(image_path)
        h, w = original.shape[:2]
        log.info(f"Image size: {w}x{h}")
        
        # Step 2: Analyze scene
        log.info("[Step 1] Scene analysis...")
        analysis = self.analysis.analyze(image_path, instruction)
        log.info(f"  Object to add: {analysis.object_to_add}")
        log.info(f"  Position: ({analysis.suggested_position.x}%, {analysis.suggested_position.y}%)")
        
        # Step 3: Create edit plan (mask for PixelAlignment)
        log.info("[Step 2] Creating edit plan...")
        plan = self.planning.create_plan(original.shape, analysis)
        
        # Build edit instruction
        edit_instruction = self._build_edit_instruction(analysis)
        log.info(f"  Inpainting prompt: {analysis.positive_prompt}")
        
        # Step 4: Run wanx-image-edit inpainting
        log.info(f"[Step 3] wanx-image-edit inpainting...")
        edit_result = self.editor.edit(
            image=original,
            mask=plan.mask,
            prompt=analysis.positive_prompt,
            negative_prompt=analysis.negative_prompt,
        )
        edited_raw = edit_result.image
        log.info(f"  Backend: {edit_result.backend}")
        
        # Step 5: Use Wanx output directly (Wanx API guarantees non-mask region unchanged)
        # Note: PixelAlignment skipped because Wanx already preserves non-edit regions
        result = edited_raw
        log.info("[Step 4] Using Wanx output directly (API guarantees alignment)")
        
        # Step 6: Package output
        log.info("[Step 5] Evaluating quality...")
        output_result = self.output.create_result(
            original=original,
            edited_raw=edited_raw,
            result=result,
            mask=plan.mask,
            image=Path(image_path).name,
            object_added=plan.object_to_add,
            backend=edit_result.backend,
            prompt=edit_instruction[:120],
        )
        
        if output_name:
            output_result.save(self.config.output_dir, output_name)
        
        return output_result
    
    def _build_edit_instruction(self, analysis) -> str:
        """
        Build edit instruction from analysis result.
        
        Prioritizes edit_instruction from analysis, with fallback to
        building from position and object.
        """
        # Use pre-defined edit instruction if available
        if analysis.edit_instruction:
            return analysis.edit_instruction
        
        # Otherwise build from position and object
        pos = analysis.suggested_position
        
        # Convert percentage to approximate location words
        x_word = self._percent_to_position_word(pos.x)
        y_word = self._percent_to_position_word(pos.y, is_y=True)
        
        # Build instruction
        instruction = f"在画面{y_word}{x_word}添加{analysis.object_to_add}"
        
        return instruction
    
    def _percent_to_position_word(self, pct: int, is_y: bool = False) -> str:
        """Convert percentage position to Chinese position word."""
        if is_y:
            if pct < 25:
                return "上方"
            elif pct < 50:
                return "中上"
            elif pct < 75:
                return "中间"
            else:
                return "下方"
        else:
            if pct < 25:
                return "左"
            elif pct < 50:
                return "偏左"
            elif pct < 75:
                return "偏右"
            else:
                return "右"
    
    def run_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[OutputResult]:
        """Run pipeline on multiple images."""
        out_dir = output_dir or self.config.output_dir
        results = []
        report = []
        
        for path in image_paths:
            name = Path(path).stem
            try:
                result = self.run(path, output_name=name)
                results.append(result)
                report.append({
                    "image": Path(path).name,
                    "object_added": result.metadata.get("object_added", ""),
                    "backend": result.metadata.get("backend", ""),
                    "pixel_alignment": float(result.metrics.pixel_alignment),
                    "boundary_quality": float(result.metrics.boundary_quality),
                    "overall_quality": float(result.metrics.overall_quality),
                    "passed": bool(float(result.metrics.pixel_alignment) >= 0.999),
                })
            except Exception as e:
                log.error(f"Failed on {path}: {e}")
                import traceback
                traceback.print_exc()
                report.append({"image": Path(path).name, "error": str(e)})
        
        # Save summary report
        report_path = Path(out_dir) / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Report saved: {report_path}")
        
        # Print summary
        log.info("")
        log.info(f"{'=' * 70}")
        log.info("SUMMARY")
        log.info(f"{'=' * 70}")
        for r in report:
            if "error" in r:
                log.info(f"  {r['image']:<15} ERROR: {r['error'][:40]}")
            else:
                status = "PASS" if r["passed"] else "WARN"
                log.info(
                    f"  {r['image']:<15} {r['object_added']:<20} "
                    f"align={r['pixel_alignment']:.6f}  {status}"
                )
        
        return results
