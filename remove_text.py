#!/usr/bin/env python3
"""
Text Removal Script for Manga Dataset

Removes text from manga images using EasyOCR and OpenCV inpainting.
Preserves the original directory structure and file names.

Usage:
    python remove_text.py --input Dataset/small --output Dataset/small_no_text
    python remove_text.py --input Dataset/large --output Dataset/large_no_text --method white --padding 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import sys

# Filter libpng ICC profile warnings from stderr
class FilterStderr:
    """Filter stderr to suppress libpng ICC profile warnings."""
    def __init__(self):
        self.original_stderr = sys.stderr
        self.filtered_lines = []
    
    def write(self, message):
        # Filter out libpng ICC profile warnings
        if 'libpng warning' in message.lower() and 'iccp' in message.lower():
            return  # Suppress this warning
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Install stderr filter
sys.stderr = FilterStderr()

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    import easyocr
    import torch
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr or torch not installed. Install with: pip install easyocr torch")


def create_text_mask(image_shape, bboxes, padding=5):
    """
    Create a binary mask for text regions.
    
    Args:
        image_shape: Shape of the image (height, width)
        bboxes: List of bounding boxes, each is a list of 4 points
        padding: Padding around text boxes in pixels
    
    Returns:
        Binary mask with text regions marked as 255
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.int32)
        
        if padding > 0:
            x, y, w, h = cv2.boundingRect(pts)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_shape[1] - x, w + 2 * padding)
            h = min(image_shape[0] - y, h + 2 * padding)
            pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        
        cv2.fillPoly(mask, [pts], 255)
    
    return mask


def remove_text_from_image(image_path, reader, padding=5, method='inpaint'):
    """
    Remove text from an image using EasyOCR detection and inpainting.
    
    Args:
        image_path: Path to input image
        reader: EasyOCR reader instance
        padding: Padding around detected text in pixels
        method: 'inpaint' (smart fill) or 'white' (fill with white)
    
    Returns:
        Cleaned image or None if error
    """
    # Read image (libpng warnings are suppressed globally)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    
    # Detect text in image
    try:
        results = reader.readtext(image)
    except Exception as e:
        print(f"\nWarning: EasyOCR error on {image_path.name}: {e}")
        return image  # Return original if OCR fails
    
    if len(results) == 0:
        return image  # No text detected, return original
    
    # Extract bounding boxes
    bboxes = [item[0] for item in results]
    mask = create_text_mask(image.shape, bboxes, padding)
    
    # Remove text using selected method
    if method == 'inpaint':
        cleaned_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:  # method == 'white'
        cleaned_image = image.copy()
        cleaned_image[mask == 255] = (255, 255, 255)
    
    return cleaned_image


def create_cleaned_dataset(
    input_dir,
    output_dir,
    method='inpaint',
    padding=5,
    preserve_structure=True,
    extensions=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    gpu=True,
):
    """
    Process all images in a dataset to remove text.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for cleaned images
        method: 'inpaint' (smart fill) or 'white' (fill with white)
        padding: Padding around detected text in pixels
        preserve_structure: If True, preserve directory structure
        extensions: List of image file extensions to process
        gpu: Use GPU if available
    
    Returns:
        Dictionary with processing statistics
    """
    if not EASYOCR_AVAILABLE:
        print("Error: easyocr is not installed. Install with: pip install easyocr")
        return None
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return None
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {input_dir}")
        return None
    
    print(f"Found {len(image_files)} images to process")
    print(f"Method: {method}")
    print(f"Padding: {padding}px")
    print(f"Output directory: {output_dir}")
    print(f"GPU: {gpu and torch.cuda.is_available()}")
    print("="*60)
    
    # Initialize EasyOCR reader (only once for efficiency)
    print("Initializing EasyOCR reader (this may take a moment on first run)...")
    use_gpu = gpu and torch.cuda.is_available()
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    print("EasyOCR ready!\n")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Determine output path
            if preserve_structure:
                relative_path = img_file.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = output_path / img_file.name
            
            # Skip if output already exists
            if output_file.exists():
                skipped_count += 1
                continue
            
            # Remove text from image
            cleaned_image = remove_text_from_image(
                img_file, reader, padding=padding, method=method
            )
            
            if cleaned_image is not None:
                # Preserve original file extension
                # libpng warnings are already filtered by FilterStderr
                cv2.imwrite(str(output_file), cleaned_image)
                success_count += 1
            else:
                print(f"\nWarning: Could not process {img_file.name}")
                error_count += 1
                
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            error_count += 1
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(image_files)}")
    print(f"Output saved to: {output_dir}")
    print("="*60)
    
    return {
        'success': success_count,
        'skipped': skipped_count,
        'errors': error_count,
        'total': len(image_files)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Remove text from manga images using EasyOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_text.py --input Dataset/small --output Dataset/small_no_text
  python remove_text.py --input Dataset/large --output Dataset/large_no_text --method white
  python remove_text.py --input Dataset/small --output Dataset/small_no_text --padding 10 --no-gpu
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for cleaned images",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["inpaint", "white"],
        default="inpaint",
        help="Text removal method: 'inpaint' (smart fill) or 'white' (fill with white) (default: inpaint)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around detected text in pixels (default: 5)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU even if available",
    )
    parser.add_argument(
        "--no-preserve-structure",
        action="store_true",
        help="Flatten output directory structure",
    )
    
    args = parser.parse_args()
    
    # Check if EasyOCR is available
    if not EASYOCR_AVAILABLE:
        print("Error: easyocr is not installed.")
        print("Install with: pip install easyocr")
        return
    
    # Process dataset
    result = create_cleaned_dataset(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        padding=args.padding,
        preserve_structure=not args.no_preserve_structure,
        gpu=not args.no_gpu,
    )
    
    if result:
        print(f"\n✓ Successfully processed {result['success']} images")
        if result['errors'] > 0:
            print(f"⚠ {result['errors']} errors occurred")


if __name__ == "__main__":
    main()


