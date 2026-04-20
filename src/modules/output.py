"""
Output Module - Quality Evaluation, Comparison Images, and Result Export

Evaluates PixelAlignment score, generates side-by-side comparison images,
and saves all outputs in a structured format.
"""

import cv2
import json
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from ..utils import log, save_image


@dataclass
class QualityMetrics:
    """Quality evaluation metrics"""
    pixel_alignment: float
    boundary_quality: float
    overall_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_alignment": round(self.pixel_alignment, 6),
            "boundary_quality": round(self.boundary_quality, 4),
            "overall_quality": round(self.overall_quality, 4),
        }


@dataclass
class OutputResult:
    """Complete output with images, metrics, and metadata"""
    result_image: np.ndarray
    original_image: np.ndarray
    mask: np.ndarray
    edited_raw: np.ndarray
    metrics: QualityMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str, name: str) -> Dict[str, str]:
        """Save all outputs: result, mask, comparison, metrics."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        result_path = out / f"{name}_result.png"
        save_image(self.result_image, result_path)
        saved["result"] = str(result_path)

        mask_path = out / f"{name}_mask.png"
        save_image(self.mask, mask_path)
        saved["mask"] = str(mask_path)

        comp_path = out / f"{name}_comparison.png"
        comparison = self._create_comparison()
        save_image(comparison, comp_path)
        saved["comparison"] = str(comp_path)

        metrics_path = out / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {"metrics": self.metrics.to_dict(), "metadata": self.metadata,
                 "timestamp": datetime.now().isoformat()},
                f, indent=2, ensure_ascii=False,
            )
        saved["metrics"] = str(metrics_path)

        log.info(f"Results saved to {out}/{name}_*")
        return saved

    def _create_comparison(self) -> np.ndarray:
        """Create a 4-panel comparison: Original | Mask | Raw Edit | Final (Aligned)."""
        h, w = self.original_image.shape[:2]
        target_h = min(400, h)
        scale = target_h / h
        target_w = int(w * scale)

        panels = []
        mask_rgb = np.stack([self.mask] * 3, axis=-1) if self.mask.ndim == 2 else self.mask
        for img in [self.original_image, mask_rgb, self.edited_raw, self.result_image]:
            panels.append(cv2.resize(img, (target_w, target_h)))

        comparison = np.hstack(panels)

        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Original", "Mask", "Inpainted (raw)", f"PixelAligned ({self.metrics.pixel_alignment:.4f})"]
        for i, label in enumerate(labels):
            x_offset = i * target_w + 8
            cv2.putText(comparison, label, (x_offset, 26), font, 0.50, (255, 255, 255), 2)
            cv2.putText(comparison, label, (x_offset, 26), font, 0.50, (0, 0, 0), 1)

        return comparison


class OutputModule:
    """Quality evaluation and result packaging."""

    PIXEL_ALIGNMENT_THRESHOLD = 0.999

    def __init__(self):
        log.info("OutputModule initialized")

    def evaluate(
        self,
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
    ) -> QualityMetrics:
        pixel_alignment = self._pixel_alignment(original, result, mask)
        boundary_quality = self._boundary_quality(result, mask)
        overall = pixel_alignment * 0.7 + boundary_quality * 0.3

        log.info(f"Quality: PixelAlignment={pixel_alignment:.6f}, "
                 f"Boundary={boundary_quality:.4f}, Overall={overall:.4f}")
        if pixel_alignment < self.PIXEL_ALIGNMENT_THRESHOLD:
            log.warning(f"PixelAlignment below threshold: {pixel_alignment:.6f}")

        return QualityMetrics(
            pixel_alignment=pixel_alignment,
            boundary_quality=boundary_quality,
            overall_quality=min(overall, 1.0),
        )

    def create_result(
        self,
        original: np.ndarray,
        edited_raw: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        **extra_metadata,
    ) -> OutputResult:
        metrics = self.evaluate(original, result, mask)
        return OutputResult(
            result_image=result,
            original_image=original,
            mask=mask,
            edited_raw=edited_raw,
            metrics=metrics,
            metadata=extra_metadata,
        )

    @staticmethod
    def _pixel_alignment(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> float:
        non_edit = mask < 127
        if not np.any(non_edit):
            return 1.0
        diff = np.abs(original[non_edit].astype(float) - result[non_edit].astype(float))
        return max(0.0, 1.0 - np.mean(diff) / 255.0)

    @staticmethod
    def _boundary_quality(result: np.ndarray, mask: np.ndarray) -> float:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        boundary = dilated - eroded
        if not np.any(boundary):
            return 1.0
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        avg = np.mean(grad[boundary > 0])
        return max(0.0, 1.0 - min(avg / 50.0, 1.0))
