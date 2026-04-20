"""
Blend Module - PixelAlignment Core Implementation

This module ensures PixelAlignment by blending edited and original images
in pixel space, guaranteeing that non-edited regions remain unchanged.

Mathematical Guarantee:
    For non-edited region (mask = 0): alpha = 0, result = original_image
    This guarantees 100% pixel alignment outside the edit region.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from ..utils import log


class BlendMode(Enum):
    """Available blend modes for PixelAlignment"""
    GAUSSIAN = "gaussian"      # Gaussian blur transition (recommended)
    ADAPTIVE = "adaptive"      # Adaptive boundary based on distance transform
    FEATHER = "feather"       # Feather edges with morphological ops
    HARD = "hard"             # Hard cut (no transition)


@dataclass
class BlendConfig:
    """Configuration for blend operations"""
    mode: BlendMode = BlendMode.GAUSSIAN
    blur_radius: int = 15     # Gaussian blur kernel size
    inner_radius: int = 5    # Inner boundary for adaptive mode
    outer_radius: int = 15   # Outer boundary for adaptive mode


class BlendModule:
    """
    PixelAlignment Core Module
    
    This module guarantees that non-edited regions remain pixel-perfect
    by performing blending in pixel space rather than latent space.
    
    Key Innovation:
    Traditional inpainting models modify pixels outside the mask region
    due to VAE encoding/decoding errors and latent space operations.
    Our approach directly blends in pixel space:
    
        result = original * (1 - alpha) + edited * alpha
    
    where alpha is a smoothly varying mask that transitions from 0 (outside)
    to 1 (inside) the edit region.
    
    Example:
        >>> blender = BlendModule()
        >>> result = blender.align(original, edited, mask)
        >>> # Non-edited region is guaranteed to be identical to original
    """
    
    def __init__(self, config: Optional[BlendConfig] = None):
        self.config = config or BlendConfig()
        log.info(f"BlendModule initialized with mode: {self.config.mode.value}")
    
    def align(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Align edited image with original to guarantee PixelAlignment.
        
        Args:
            original: Original image (H, W, 3), RGB format, uint8
            edited: Edited image from inpainting model (H, W, 3), RGB format, uint8
            mask: Edit region mask (H, W), uint8, white=edit region
            
        Returns:
            Blended result with PixelAlignment guarantee
            
        Note:
            Non-edited region (mask=0) will be EXACTLY identical to original.
            This is the mathematical guarantee of PixelAlignment.
        """
        # Validate inputs
        self._validate_inputs(original, edited, mask)
        
        # Select blend mode
        if self.config.mode == BlendMode.GAUSSIAN:
            return self._gaussian_blend(original, edited, mask)
        elif self.config.mode == BlendMode.ADAPTIVE:
            return self._adaptive_blend(original, edited, mask)
        elif self.config.mode == BlendMode.FEATHER:
            return self._feather_blend(original, edited, mask)
        else:
            return self._hard_blend(original, edited, mask)
    
    def _validate_inputs(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> None:
        """Validate input shapes and types."""
        assert original.shape == edited.shape, \
            f"Shape mismatch: original {original.shape} vs edited {edited.shape}"
        assert original.shape[:2] == mask.shape, \
            f"Mask shape {mask.shape} doesn't match image {original.shape[:2]}"
        assert original.dtype == np.uint8, "Images must be uint8"
    
    def _gaussian_blend(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Gaussian blur blend for smooth transitions.
        
        This method ensures PixelAlignment by:
        1. Computing alpha from blurred mask for smooth transitions
        2. BUT enforcing alpha=0 where original mask is exactly 0
        
        This guarantees non-edit regions remain pixel-perfect.
        """
        # Ensure mask is single channel grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Store original non-edit region for PixelAlignment enforcement
        original_non_edit = mask == 0
        
        # Normalize mask to [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to create smooth transition
        # Kernel size must be odd
        ksize = self.config.blur_radius
        if ksize % 2 == 0:
            ksize += 1
            
        alpha = cv2.GaussianBlur(mask_norm, (ksize, ksize), 0)
        
        # CRITICAL: Enforce alpha=0 where original mask is exactly 0
        # This guarantees PixelAlignment for non-edit regions
        alpha[original_non_edit] = 0.0
        
        # Expand dimensions for broadcasting (H, W) -> (H, W, 1)
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Perform pixel-space blending
        # result = original * (1 - alpha) + edited * alpha
        result = original.astype(np.float32) * (1 - alpha) + \
                 edited.astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _adaptive_blend(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Adaptive boundary blend based on distance transform.
        
        This mode creates sharper boundaries in the inner region
        while maintaining smooth transitions at the outer boundary.
        Useful when you want crisp object edges but smooth integration.
        
        PixelAlignment enforced: alpha=0 where original mask is exactly 0.
        """
        # Store original non-edit region for PixelAlignment
        original_non_edit = mask == 0
        
        # Ensure binary mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Distance transform: distance from each point to nearest zero
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 5)
        
        # Create gradient alpha based on distance
        inner_r = self.config.inner_radius
        outer_r = self.config.outer_radius
        transition = outer_r - inner_r
        
        # alpha = 1 inside, smoothly transitions to 0 outside
        alpha = np.clip(
            (dist_inside - dist_outside + outer_r) / (2 * transition),
            0, 1
        )
        
        # CRITICAL: Enforce PixelAlignment
        alpha[original_non_edit] = 0.0
        
        # Expand dimensions for broadcasting
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Blend
        result = original.astype(np.float32) * (1 - alpha) + \
                 edited.astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _feather_blend(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Feather blend using morphological operations.
        
        This mode creates a consistent transition band around the boundary.
        Good for objects with irregular shapes.
        
        PixelAlignment enforced: alpha=0 where original mask is exactly 0.
        """
        # Store original non-edit region for PixelAlignment
        original_non_edit = mask == 0
        
        # Create morphological kernel
        ksize = self.config.blur_radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (ksize, ksize)
        )
        
        # Erode and dilate to get boundary band
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Boundary band for transition
        boundary = mask_dilated.astype(np.float32) - mask_eroded.astype(np.float32)
        
        # Create alpha: 1 in eroded region, gradient in boundary, 0 outside
        alpha = np.zeros_like(mask, dtype=np.float32)
        alpha[mask_eroded > 0] = 1.0
        alpha[boundary > 0] = 0.5
        
        # Smooth the alpha
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
        
        # CRITICAL: Enforce PixelAlignment
        alpha[original_non_edit] = 0.0
        
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Blend
        result = original.astype(np.float32) * (1 - alpha) + \
                 edited.astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _hard_blend(
        self, 
        original: np.ndarray, 
        edited: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Hard blend - direct copy without transition.
        
        This mode copies the edited region directly.
        May create visible boundaries, but is fastest.
        """
        # Create binary mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        alpha = mask_binary.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Direct blend
        result = original.astype(np.float32) * (1 - alpha) + \
                 edited.astype(np.float32) * alpha
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def verify_alignment(
        self, 
        original: np.ndarray, 
        result: np.ndarray, 
        mask: np.ndarray,
        threshold: float = 0.999
    ) -> Tuple[bool, float]:
        """
        Verify that PixelAlignment is achieved.
        
        Args:
            original: Original image
            result: Result image after blending
            mask: Edit region mask
            threshold: Minimum alignment score (default 0.999 = 99.9% identical)
            
        Returns:
            Tuple of (is_aligned, alignment_score)
        """
        # Find non-edited region
        non_edit_mask = (mask < 127)
        
        if not np.any(non_edit_mask):
            log.warning("No non-edited region to verify")
            return True, 1.0
        
        # Calculate pixel difference in non-edited region
        original_region = original[non_edit_mask]
        result_region = result[non_edit_mask]
        
        # Calculate alignment as ratio of identical pixels
        diff = np.abs(original_region.astype(float) - result_region.astype(float))
        max_diff = np.max(diff) if len(diff) > 0 else 0
        
        # Calculate mean alignment
        alignment = 1.0 - (np.mean(diff) / 255.0)
        
        # With Gaussian blending, non-edited region should be EXACTLY identical
        # (within floating point precision)
        is_aligned = alignment >= threshold
        
        log.info(f"PixelAlignment verification: {alignment:.6f} (threshold: {threshold})")
        
        if not is_aligned:
            log.warning(f"PixelAlignment FAILED! Score: {alignment:.6f}")
        
        return is_aligned, alignment
