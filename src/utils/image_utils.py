"""
Image Processing Utilities
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path as RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save numpy array as image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    """Convert numpy image to bytes."""
    pil_img = Image.fromarray(image)
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> np.ndarray:
    """Convert bytes to numpy image."""
    import io
    pil_img = Image.open(io.BytesIO(data)).convert('RGB')
    return np.array(pil_img)


def resize_image(
    image: np.ndarray, 
    size: Tuple[int, int],
    keep_aspect_ratio: bool = False
) -> np.ndarray:
    """Resize image to target size."""
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return result
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def create_box_mask(
    image_shape: Tuple[int, int],
    box: Tuple[int, int, int, int],  # x, y, w, h
    padding: int = 10
) -> np.ndarray:
    """Create a rectangular mask from bounding box."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y, bw, bh = box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    mask[y1:y2, x1:x2] = 255
    return mask


def gaussian_blur_mask(
    mask: np.ndarray, 
    kernel_size: int = 15
) -> np.ndarray:
    """Apply Gaussian blur to mask for smooth transitions."""
    # Normalize to [0, 1]
    mask_norm = mask.astype(np.float32) / 255.0
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask_norm, (kernel_size, kernel_size), 0)
    # Denormalize back to [0, 255]
    return (blurred * 255).astype(np.uint8)


def compute_ssim(
    img1: np.ndarray, 
    img2: np.ndarray,
    win_size: int = 7
) -> float:
    """Compute SSIM between two images."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, win_size=win_size, channel_axis=2)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)
