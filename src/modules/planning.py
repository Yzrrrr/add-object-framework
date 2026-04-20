"""
Planning Module - Mask Generation from Analysis Results

Converts analysis results (percentage coordinates) into pixel-accurate
elliptical masks with dilation for safe PixelAlignment blending.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils import log
from .analysis import Position, AnalysisResult


@dataclass
class EditPlan:
    """Complete plan for an edit operation"""
    mask: np.ndarray        # (H, W) uint8 mask, white=edit region
    prompt: str
    negative_prompt: str
    object_to_add: str
    position: Position


class PlanningModule:
    """
    Generates edit masks and prepares prompts from analysis results.

    Uses elliptical masks (more natural than rectangles) with
    morphological dilation to ensure the generated object isn't
    clipped by PixelAlignment blending.
    """

    def __init__(self, dilation_kernel_size: int = 25):
        self.dilation_kernel_size = dilation_kernel_size
        log.info("PlanningModule initialized")

    def create_plan(
        self,
        image_shape: Tuple[int, ...],
        analysis: AnalysisResult,
    ) -> EditPlan:
        """
        Create a complete edit plan from analysis results.

        Args:
            image_shape: (H, W, C) shape of the original image
            analysis: Scene analysis with position and prompts

        Returns:
            EditPlan with mask, prompts, and metadata
        """
        h, w = image_shape[:2]
        pos = analysis.suggested_position

        mask = self._create_elliptical_mask(h, w, pos)

        coverage = np.sum(mask > 0) / mask.size * 100
        log.info(f"Mask created: {w}x{h}, coverage={coverage:.1f}%, "
                 f"center=({pos.x}%,{pos.y}%), size=({pos.width}%x{pos.height}%)")

        return EditPlan(
            mask=mask,
            prompt=analysis.positive_prompt,
            negative_prompt=analysis.negative_prompt,
            object_to_add=analysis.object_to_add,
            position=pos,
        )

    def _create_elliptical_mask(
        self, height: int, width: int, position: Position
    ) -> np.ndarray:
        """Create an elliptical mask with dilation for safe blending."""
        cx = int(position.x * width / 100)
        cy = int(position.y * height / 100)
        rx = int(position.width * width / 100 / 2)
        ry = int(position.height * height / 100 / 2)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        if self.dilation_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel_size, self.dilation_kernel_size),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
