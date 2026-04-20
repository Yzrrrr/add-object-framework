"""
Edit Module - Cloud Inpainting with Wanx2.1-ImageEdit

Uses wanx2.1-imageedit for mask-based inpainting.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from ..utils import log
from ..api.wanx import WanxInpaintBackend, EditResult


class EditModule:
    """
    Main edit module using wanx2.1-imageedit inpainting.
    """
    
    def __init__(self, model: str = "wanx2.1-imageedit"):
        self.backend = WanxInpaintBackend(model=model)
        log.info(f"EditModule initialized (backend=wanx-image-edit, model={model})")
    
    def edit(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
    ) -> EditResult:
        """
        Run wanx inpainting.
        
        Args:
            image: RGB numpy array (H, W, 3), uint8
            mask: Grayscale numpy array (H, W), uint8, 255=edit region
            prompt: Inpainting prompt (e.g., "一只真实的流浪猫")
            negative_prompt: Negative prompt (optional, may be ignored by wanx)
        
        Returns:
            EditResult with the edited image
        """
        return self.backend.inpaint(image, mask, prompt, negative_prompt)
