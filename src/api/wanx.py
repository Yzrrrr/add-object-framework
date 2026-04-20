"""
Aliyun DashScope Wanxiang (通义万相) Image Edit Backend

Calls wanx2.1-imageedit / description_edit_with_mask for true mask-based
inpainting. Unlike local SD 1.5, this model actually generates the requested
object inside the mask region, not just "more background".

Pipeline:
    1. Resize image+mask to fit Wanx's [512, 4096] constraint.
    2. Encode both as base64 data URIs (no OSS needed).
    3. POST async task -> get task_id.
    4. Poll GET /tasks/{task_id} until SUCCEEDED.
    5. Download result URL, resize back to original resolution.

Docs: https://help.aliyun.com/zh/model-studio/developer-reference/wanx-image-edit-api-reference
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from PIL import Image

from ..utils import log


SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
POLL_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"


@dataclass
class EditResult:
    """Result from an inpainting operation (mirrors editor.EditResult)."""
    image: np.ndarray
    backend: str
    metadata: dict


class WanxInpaintBackend:
    """
    DashScope Wanxiang (wanx2.1-imageedit) inpainting backend.

    Uses the 'description_edit_with_mask' function which regenerates the
    masked region conditioned on a text prompt, keeping unmasked areas
    approximately intact (our PixelAlignment step guarantees exact intact).

    Env var: DASHSCOPE_API_KEY
    """

    def __init__(
        self,
        model: str = "wanx2.1-imageedit",
        max_edge: int = 1536,
        min_edge: int = 512,
        poll_interval: float = 2.0,
        poll_timeout: float = 180.0,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_edge = max_edge
        self.min_edge = min_edge
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY not set. Export it or add it to the .env file.\n"
                "Get one from https://bailian.console.aliyun.com/"
            )

        log.info(f"WanxInpaintBackend ready (model={self.model}, max_edge={self.max_edge})")

    def load(self):
        """No-op: API backend has nothing to load."""
        return

    def _compute_target_size(self, h: int, w: int) -> tuple[int, int]:
        """Pick a (new_w, new_h) inside Wanx's [min_edge, max_edge] range.

        Keep aspect ratio. Cap the longer edge at max_edge. Ensure the
        shorter edge >= min_edge. Round both to multiples of 8 for safety.
        """
        scale = 1.0
        long_edge = max(h, w)
        short_edge = min(h, w)

        if long_edge > self.max_edge:
            scale = self.max_edge / long_edge
        if short_edge * scale < self.min_edge:
            scale = self.min_edge / short_edge

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        new_w = max(self.min_edge, min(self.max_edge, (new_w // 8) * 8))
        new_h = max(self.min_edge, min(self.max_edge, (new_h // 8) * 8))
        return new_w, new_h

    @staticmethod
    def _to_data_uri(pil_img: Image.Image, fmt: str) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "jpeg" if fmt.lower() in ("jpg", "jpeg") else fmt.lower()
        return f"data:image/{mime};base64,{b64}"

    def _submit(self, image_uri: str, mask_uri: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "input": {
                "function": "description_edit_with_mask",
                "prompt": prompt,
                "base_image_url": image_uri,
                "mask_image_url": mask_uri,
            },
            "parameters": {"n": 1},
        }
        resp = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Wanx submit failed [{resp.status_code}]: {resp.text[:500]}")
        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Wanx submit returned no task_id: {data}")
        return task_id

    def _poll(self, task_id: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        deadline = time.time() + self.poll_timeout
        last_status = None
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            resp = requests.get(f"{POLL_URL}/{task_id}", headers=headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Wanx poll failed [{resp.status_code}]: {resp.text[:500]}")
            data = resp.json()
            output = data.get("output", {})
            status = output.get("task_status")
            if status != last_status:
                log.info(f"  Wanx task {task_id[:8]}... status={status}")
                last_status = status

            if status == "SUCCEEDED":
                for r in output.get("results", []):
                    if "url" in r:
                        return r["url"]
                raise RuntimeError(f"Wanx SUCCEEDED but no usable url: {output.get('results')}")
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise RuntimeError(f"Wanx task {status}: {output}")
        raise TimeoutError(f"Wanx task {task_id} timed out after {self.poll_timeout}s")

    @staticmethod
    def _download(url: str) -> Image.Image:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
    ) -> EditResult:
        """Run Wanx inpainting. `negative_prompt` is accepted for interface
        parity but ignored (Wanx2.1-imageedit doesn't expose it)."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"Expected grayscale mask (H,W), got shape {mask.shape}")

        orig_h, orig_w = image.shape[:2]
        new_w, new_h = self._compute_target_size(orig_h, orig_w)

        image_pil = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
        mask_pil = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

        image_uri = self._to_data_uri(image_pil, "jpeg")
        mask_uri = self._to_data_uri(mask_pil, "png")

        log.info(f"[wanx] Submitting task at {new_w}x{new_h} | prompt: {prompt[:80]}...")
        task_id = self._submit(image_uri, mask_uri, prompt)
        log.info(f"  task_id: {task_id}")

        result_url = self._poll(task_id)
        log.info(f"  result URL received, downloading...")

        result_pil = self._download(result_url)
        if result_pil.size != (orig_w, orig_h):
            result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)
        result_np = np.array(result_pil)

        return EditResult(
            image=result_np,
            backend="wanx",
            metadata={
                "task_id": task_id,
                "model": self.model,
                "submit_resolution": [new_w, new_h],
            },
        )
