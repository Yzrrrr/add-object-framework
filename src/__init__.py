"""
Add Object Framework
A complete pipeline for adding objects to images with PixelAlignment guarantee.
"""

__version__ = "2.0.0"
__author__ = "Yi Zeren"

from .pipeline import AddObjectPipeline, PipelineConfig

__all__ = ["AddObjectPipeline", "PipelineConfig"]
