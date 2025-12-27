#!/usr/bin/env python3
"""
Manga Image Preprocessing for CLIP

Preprocesses manga images to be optimal for CLIP model input while preserving
all visual information and maximizing content size within the target resolution.

Strategy (Letterbox with Margin Trimming):
==========================================
1. **Trim White Margins**: Remove existing white/near-white borders from scans.
   This minimizes the amount of padding needed and prevents margins from becoming
   dominant CLIP features.

2. **Resize First (Long Side)**: Scale so the long side = target size (224/336).
   This MAXIMIZES content size within the frame.

3. **Pad Second (Short Side)**: Add minimal padding to the short side to make
   it square. Consistently use white (255) for manga.

4. **High-Quality Resize**: Uses LANCZOS interpolation for best quality,
   preserving fine linework details.

Why This Order Matters:
=======================
OLD (pad-then-resize): 750x1128 -> pad to 1128x1128 -> resize to 224x224
  - Content only occupies ~66% of the 224x224 frame
  - Fine details lost due to more aggressive downscaling

NEW (trim-resize-pad): 750x1128 -> trim margins -> resize to 149x224 -> pad to 224x224
  - Content occupies maximum possible area
  - Minimal padding = less "border feature" interference
  - Better preservation of author style and linework

Why Trim Margins:
=================
- Scanned manga often has variable white margins
- Large solid borders become dominant CLIP features (wrong clustering)
- Trimming first means padding is minimal and consistent

Usage:
======
    # Basic usage - preprocess entire dataset
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset

    # Skip margin trimming (if images are already cropped)
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --no-trim

    # With tiling for detailed preservation
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --tile
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Literal
from datetime import datetime
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing."""
    
    # Target size for CLIP ViT-L-14
    target_size: int = 224
    
    # Padding color: always white (255) for consistency in manga
    # This prevents border color from becoming a clustering feature
    pad_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Whether to trim white margins before processing
    # Highly recommended for scanned manga to minimize padding
    trim_margins: bool = True
    
    # Threshold for margin detection (0-255, pixels brighter than this are "white")
    trim_threshold: int = 250
    
    # Minimum content ratio after trimming (to prevent over-trimming)
    min_content_ratio: float = 0.3
    
    # Whether to create tiles for detailed preservation
    create_tiles: bool = False
    
    # Tile configuration (if create_tiles=True)
    tile_size: int = 224
    tile_overlap: float = 0.25  # 25% overlap between tiles
    
    # Resize interpolation
    interpolation: int = field(default_factory=lambda: Image.LANCZOS)
    
    # Output format
    output_format: str = 'PNG'  # PNG for lossless, JPEG for smaller files
    jpeg_quality: int = 95
    
    # Whether to keep original file extension
    keep_extension: bool = True


def trim_white_margins(
    image: Image.Image,
    threshold: int = 250,
    min_content_ratio: float = 0.3
) -> Image.Image:
    """
    Trim white/near-white margins from an image.
    
    This is crucial for manga scans which often have variable white borders.
    Removing them before processing prevents borders from becoming dominant
    CLIP features that cause wrong clustering.
    
    Args:
        image: Input image (RGB)
        threshold: Pixels brighter than this (all channels) are considered "white"
        min_content_ratio: Minimum ratio of content to keep (prevents over-trimming)
    
    Returns:
        Trimmed image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Find non-white pixels (any channel below threshold)
    # A pixel is "content" if ANY channel is below threshold
    non_white_mask = np.any(img_array < threshold, axis=2)
    
    # Find bounding box of non-white content
    rows = np.any(non_white_mask, axis=1)
    cols = np.any(non_white_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Image is entirely white/near-white, return as-is
        return image
    
    # Get bounding box
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Check if trimmed area is too small (prevent over-trimming)
    orig_h, orig_w = img_array.shape[:2]
    new_h = y_max - y_min + 1
    new_w = x_max - x_min + 1
    
    content_ratio = (new_h * new_w) / (orig_h * orig_w)
    
    if content_ratio < min_content_ratio:
        # Would trim too much, return original
        return image
    
    # Crop to bounding box
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))


def letterbox_resize(
    image: Image.Image,
    target_size: int,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
    interpolation: int = Image.LANCZOS
) -> Image.Image:
    """
    Resize image using letterbox method: resize long side to target, then pad short side.
    
    This MAXIMIZES content size within the target frame, unlike pad-then-resize
    which shrinks content unnecessarily.
    
    OLD (pad-then-resize): 750x1128 -> pad to 1128x1128 -> resize to 224x224
      Content occupies only ~66% of frame (750/1128 = 66%)
    
    NEW (resize-then-pad): 750x1128 -> resize to 149x224 -> pad to 224x224
      Content occupies ~66% of frame but at FULL resolution for that area
      (the content pixels themselves are larger/more detailed)
    
    Args:
        image: Input image
        target_size: Target size for output (will be square)
        pad_color: RGB tuple for padding color (default white for manga)
        interpolation: PIL interpolation method
    
    Returns:
        Square image with content maximized
    """
    width, height = image.size
    
    # Calculate scale factor to fit long side to target
    scale = target_size / max(width, height)
    
    # Compute new dimensions (maintaining aspect ratio)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image (long side = target_size)
    resized = image.resize((new_width, new_height), interpolation)
    
    # Create square canvas with padding color
    result = Image.new('RGB', (target_size, target_size), pad_color)
    
    # Calculate position to center the resized image
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    # Paste resized image onto canvas
    result.paste(resized, (x_offset, y_offset))
    
    return result


def create_tiles(
    image: Image.Image,
    tile_size: int,
    overlap: float
) -> List[Tuple[Image.Image, Tuple[int, int]]]:
    """
    Create overlapping tiles from an image.
    
    Args:
        image: Input image
        tile_size: Size of each tile
        overlap: Overlap ratio (0.0 to 1.0)
    
    Returns:
        List of (tile_image, (row, col)) tuples
    """
    width, height = image.size
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    row = 0
    y = 0
    
    while y < height:
        col = 0
        x = 0
        
        while x < width:
            # Extract tile (with boundary handling)
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            tile = image.crop((x, y, x_end, y_end))
            
            # Pad tile if it's smaller than tile_size
            if tile.size != (tile_size, tile_size):
                padded = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                padded.paste(tile, (0, 0))
                tile = padded
            
            tiles.append((tile, (row, col)))
            
            x += stride
            col += 1
            
            if x >= width:
                break
        
        y += stride
        row += 1
        
        if y >= height:
            break
    
    return tiles


def preprocess_image(
    image_path: Path,
    config: PreprocessConfig = None
) -> Image.Image:
    """
    Preprocess a single image for CLIP using letterbox method.
    
    Pipeline:
        1. Load image
        2. (Optional) Trim white margins to minimize padding
        3. Resize so long side = target size
        4. Pad short side to make square
    
    Args:
        image_path: Path to input image
        config: Preprocessing configuration
    
    Returns:
        Preprocessed image (224x224 or target_size x target_size)
    """
    if config is None:
        config = PreprocessConfig()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Step 1: Trim white margins (if enabled)
    if config.trim_margins:
        image = trim_white_margins(
            image,
            threshold=config.trim_threshold,
            min_content_ratio=config.min_content_ratio
        )
    
    # Step 2: Letterbox resize (resize long side first, then pad)
    result = letterbox_resize(
        image,
        target_size=config.target_size,
        pad_color=config.pad_color,
        interpolation=config.interpolation
    )
    
    return result


def preprocess_image_with_tiles(
    image_path: Path,
    config: PreprocessConfig = None
) -> Tuple[Image.Image, Optional[List[Tuple[Image.Image, Tuple[int, int]]]]]:
    """
    Preprocess image and optionally create tiles.
    
    Args:
        image_path: Path to input image
        config: Preprocessing configuration
    
    Returns:
        Tuple of (main_preprocessed_image, tiles_or_None)
    """
    if config is None:
        config = PreprocessConfig()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Step 1: Trim white margins (if enabled)
    if config.trim_margins:
        image = trim_white_margins(
            image,
            threshold=config.trim_threshold,
            min_content_ratio=config.min_content_ratio
        )
    
    original_size = image.size
    
    # Main preprocessed image using letterbox
    main_image = letterbox_resize(
        image,
        target_size=config.target_size,
        pad_color=config.pad_color,
        interpolation=config.interpolation
    )
    
    # Create tiles if enabled
    tiles = None
    if config.create_tiles:
        # Create tiles from trimmed image (before resize)
        tiles = create_tiles(image, config.tile_size, config.tile_overlap)
    
    return main_image, tiles


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    config: PreprocessConfig = None,
    verbose: bool = True
) -> dict:
    """
    Preprocess entire dataset directory.
    
    Maintains exact folder structure from input to output.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for preprocessed images
        config: Preprocessing configuration
        verbose: Print progress information
    
    Returns:
        Statistics dictionary
    """
    if config is None:
        config = PreprocessConfig()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f'*{ext}'))
        image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    if verbose:
        print(f"Found {len(image_files)} images in {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Configuration:")
        print(f"  - Target size: {config.target_size}x{config.target_size}")
        print(f"  - Trim margins: {config.trim_margins} (threshold={config.trim_threshold})")
        print(f"  - Padding color: RGB{config.pad_color}")
        print(f"  - Create tiles: {config.create_tiles}")
        print(f"  - Output format: {config.output_format}")
        print("=" * 60)
    
    # Statistics
    stats = {
        "total_images": len(image_files),
        "processed": 0,
        "failed": 0,
        "tiles_created": 0,
        "errors": [],
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": {
            "target_size": config.target_size,
            "trim_margins": config.trim_margins,
            "trim_threshold": config.trim_threshold,
            "pad_color": config.pad_color,
            "create_tiles": config.create_tiles,
            "output_format": config.output_format,
        }
    }
    
    # Process each image
    iterator = tqdm(image_files, desc="Preprocessing") if verbose else image_files
    
    for image_path in iterator:
        try:
            # Compute relative path
            rel_path = image_path.relative_to(input_dir)
            
            # Determine output path
            if config.keep_extension:
                output_path = output_dir / rel_path
            else:
                output_path = output_dir / rel_path.with_suffix(
                    '.png' if config.output_format == 'PNG' else '.jpg'
                )
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preprocess
            if config.create_tiles:
                main_image, tiles = preprocess_image_with_tiles(image_path, config)
                
                # Save main image
                if config.output_format == 'PNG':
                    main_image.save(output_path, 'PNG')
                else:
                    main_image.save(output_path, 'JPEG', quality=config.jpeg_quality)
                
                # Save tiles
                if tiles:
                    tiles_dir = output_path.parent / f"{output_path.stem}_tiles"
                    tiles_dir.mkdir(parents=True, exist_ok=True)
                    
                    for tile_img, (row, col) in tiles:
                        tile_path = tiles_dir / f"tile_{row:02d}_{col:02d}.png"
                        tile_img.save(tile_path, 'PNG')
                        stats["tiles_created"] += 1
            else:
                main_image = preprocess_image(image_path, config)
                
                # Save
                if config.output_format == 'PNG':
                    main_image.save(output_path, 'PNG')
                else:
                    main_image.save(output_path, 'JPEG', quality=config.jpeg_quality)
            
            stats["processed"] += 1
            
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({
                "file": str(image_path),
                "error": str(e)
            })
            if verbose:
                tqdm.write(f"Error processing {image_path}: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess manga images for CLIP model using letterbox method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic preprocessing (trim margins + letterbox, recommended)
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset

    # Skip margin trimming (if images are already cropped)
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --no-trim

    # Aggressive margin trimming (lower threshold)
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --trim-threshold 240

    # With tiling for detail preservation
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --tile

    # JPEG output for smaller files
    python -m preprocess.preprocess_images --input final_dataset --output preprocessed_final_dataset --format jpeg
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for preprocessed images"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target image size (default: 224 for CLIP ViT-L-14)"
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable white margin trimming (not recommended for scans)"
    )
    parser.add_argument(
        "--trim-threshold",
        type=int,
        default=250,
        help="Pixel brightness threshold for margin detection (0-255, default: 250)"
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Create overlapping tiles for detail preservation"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Tile size when tiling is enabled (default: 224)"
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Tile overlap ratio (default: 0.25)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['png', 'jpeg'],
        default='png',
        help="Output image format (default: png for lossless)"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if using jpeg format (default: 95)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PreprocessConfig(
        target_size=args.target_size,
        trim_margins=not args.no_trim,
        trim_threshold=args.trim_threshold,
        create_tiles=args.tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        output_format='PNG' if args.format == 'png' else 'JPEG',
        jpeg_quality=args.jpeg_quality,
    )
    
    # Run preprocessing
    print("=" * 60)
    print("MANGA IMAGE PREPROCESSING FOR CLIP")
    print("=" * 60)
    print(f"\nStrategy: Letterbox with margin trimming")
    print(f"  1. Trim white margins: {config.trim_margins} (threshold={config.trim_threshold})")
    print(f"  2. Resize long side to {config.target_size} (maximize content)")
    print(f"  3. Pad short side to {config.target_size} (minimal padding)")
    print(f"  - Padding color: RGB{config.pad_color}")
    if config.create_tiles:
        print(f"  - Creating {config.tile_size}x{config.tile_size} tiles with {config.tile_overlap*100:.0f}% overlap")
    print()
    
    start_time = datetime.now()
    
    stats = preprocess_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        config=config,
        verbose=not args.quiet
    )
    
    elapsed = datetime.now() - start_time
    
    # Save stats
    stats["timestamp"] = datetime.now().isoformat()
    stats["elapsed_seconds"] = elapsed.total_seconds()
    
    stats_file = Path(args.output) / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Total images: {stats['total_images']}")
    print(f"  - Successfully processed: {stats['processed']}")
    print(f"  - Failed: {stats['failed']}")
    if config.create_tiles:
        print(f"  - Tiles created: {stats['tiles_created']}")
    print(f"  - Time elapsed: {elapsed}")
    print(f"\nOutput saved to: {args.output}")
    print(f"Stats saved to: {stats_file}")
    
    if stats['failed'] > 0:
        print(f"\nWarning: {stats['failed']} images failed to process.")
        print("Check preprocessing_stats.json for details.")


if __name__ == "__main__":
    main()
