#!/usr/bin/env python3
"""
Run the Add Object pipeline on demo images.

Cloud-only version: Uses wanx2.1-imageedit inpainting for object generation.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.pipeline import AddObjectPipeline, PipelineConfig


DEMO_IMAGES = [
    "demo/demo1.jpg",
    "demo/demo2.jpg",
    "demo/demo3.jpg",
    "demo/demo4.jpg",
    "demo/demo5.jpg",
]


def _load_dotenv(path: str = ".env") -> None:
    """Load .env file."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main():
    parser = argparse.ArgumentParser(description="Add Object Pipeline (Cloud)")
    parser.add_argument("--image", help="Process a single image")
    parser.add_argument("--analysis", default="predefined",
                        choices=["predefined", "vlm_local"],
                        help="Scene analysis backend")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    _load_dotenv()

    config = PipelineConfig(
        analysis_mode=args.analysis,
        output_dir=args.output,
    )

    pipeline = AddObjectPipeline(config)

    if args.image:
        images = [args.image]
    else:
        images = [p for p in DEMO_IMAGES if Path(p).exists()]

    if not images:
        print("No demo images found. Place images in demo/ directory.")
        sys.exit(1)

    print(f"\nProcessing {len(images)} images")
    print(f"Analysis: {args.analysis}")
    print(f"Backend: wanx2.1-imageedit (cloud)")
    print(f"Output: {args.output}/\n")

    results = pipeline.run_batch(images, output_dir=args.output)

    passed = sum(1 for r in results if r.metrics.pixel_alignment >= 0.999)
    print(f"\nDone! {passed}/{len(results)} passed PixelAlignment (>=0.999)")


if __name__ == "__main__":
    main()
