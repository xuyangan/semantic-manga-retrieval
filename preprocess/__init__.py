#!/usr/bin/env python3
"""
Preprocess Module

Provides image preprocessing utilities optimized for CLIP model input.
Uses letterbox method with margin trimming to maximize content size.

Key improvements over naive center-crop or pad-then-resize:
1. Trim white margins first (reduces "border feature" interference)
2. Resize long side to target FIRST (maximizes content size)
3. Pad short side SECOND (minimal padding)
"""

from .preprocess_images import (
    preprocess_image,
    preprocess_dataset,
    PreprocessConfig,
    trim_white_margins,
    letterbox_resize,
)

__all__ = [
    "preprocess_image",
    "preprocess_dataset",
    "PreprocessConfig",
    "trim_white_margins",
    "letterbox_resize",
]
