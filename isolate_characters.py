#!/usr/bin/env python3
"""
Character Isolation Script for Manga Dataset

Uses the magi model to detect character bounding boxes and masks out the background.
Processes dataset with structure: dataset_name/manga/chapter/pages
Creates output dataset with same structure but only characters visible (background white).

Sample Execution:
    python isolate_characters.py --input Dataset/char --output Dataset/char_isolated
    python isolate_characters.py --input Dataset/char --output Dataset/char_isolated --confidence 0.5
    python isolate_characters.py --input Dataset/char/Jujutsu\ Kaisen\ 0 --output Dataset/char_isolated/Jujutsu\ Kaisen\ 0 --confidence 0.7
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import warnings
import os
import sys
import shutil

warnings.filterwarnings('ignore')

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")


def read_image_as_np_array(image_path):
    """Read image and convert to numpy array."""
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image


def save_bounding_boxes(bboxes_file, results, confidence_threshold=0.5):
    """
    Save bounding boxes with confidence scores to JSON file.
    
    Args:
        bboxes_file: Path to save bounding boxes JSON
        results: Results from model.predict_detections_and_associations
        confidence_threshold: Minimum confidence score to include
    """
    # Extract character bounding boxes from magi model results
    # The results structure can vary, but typically contains "characters" or "texts"
    bboxes_data = []
    
    for result in results:
        # Try to get character boxes - check multiple possible field names
        boxes = []
        scores = []
        
        # Check for characters field (most likely for character detection)
        if "characters" in result:
            boxes = result["characters"]
            # Check for corresponding scores
            if "character_scores" in result:
                scores = result["character_scores"]
            elif "characters_scores" in result:
                scores = result["characters_scores"]
        # Check for texts field (as shown in user's example code)
        elif "texts" in result:
            boxes = result["texts"]
            if "text_scores" in result:
                scores = result["text_scores"]
        # Check for other possible field names
        elif "detections" in result:
            boxes = result["detections"]
            if "detection_scores" in result:
                scores = result["detection_scores"]
        
        # Filter by confidence and format
        filtered_boxes = []
        for i, box in enumerate(boxes):
            confidence = None
            
            # Try to extract confidence from various formats
            if isinstance(box, dict):
                confidence = box.get("confidence") or box.get("score") or box.get("conf")
                bbox = box.get("bbox") or box.get("box") or [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
            elif isinstance(box, (list, tuple)):
                if len(box) >= 5:
                    # Format: [x1, y1, x2, y2, confidence]
                    bbox = box[:4]
                    confidence = float(box[4])
                elif len(box) == 4:
                    # Format: [x1, y1, x2, y2] - use score from separate list
                    bbox = box
                    if i < len(scores):
                        confidence = float(scores[i])
                else:
                    continue
            else:
                continue
            
            # Use confidence from scores list if available and not in box
            if confidence is None and i < len(scores):
                confidence = float(scores[i])
            
            # Default confidence to 1.0 if not found (include all boxes)
            if confidence is None:
                confidence = 1.0
            
            # Filter by confidence threshold
            if confidence >= confidence_threshold:
                filtered_boxes.append({
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "confidence": float(confidence)
                })
        
        bboxes_data.append(filtered_boxes)
    
    # Save to JSON (save as list of lists, one per image)
    with open(bboxes_file, 'w') as f:
        json.dump(bboxes_data, f, indent=2)
    
    return bboxes_data


def load_bounding_boxes(bboxes_file):
    """Load bounding boxes from JSON file."""
    if not bboxes_file.exists():
        return None
    
    with open(bboxes_file, 'r') as f:
        return json.load(f)


def create_character_mask(image_shape, bboxes_data):
    """
    Create a binary mask for character regions.
    
    Args:
        image_shape: Shape of the image (height, width)
        bboxes_data: List of bounding boxes, each is a dict with "bbox" and "confidence"
                     Can be:
                     - List of box dicts: [{"bbox": [...], "confidence": ...}, ...]
                     - Nested list (batch format): [[{"bbox": [...], ...}, ...], ...]
    
    Returns:
        Binary mask with character regions marked as 255
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Handle both single image result and batch result
    if not isinstance(bboxes_data, list) or len(bboxes_data) == 0:
        return mask
    
    # Check if it's a batch format (nested list) or single image format (list of boxes)
    # Batch format: [[box1, box2, ...], [box1, box2, ...], ...]
    # Single format: [box1, box2, ...]
    if len(bboxes_data) > 0 and isinstance(bboxes_data[0], list):
        # Batch format - take first image's boxes
        bboxes_list = bboxes_data[0]
    else:
        # Single image format - use directly
        bboxes_list = bboxes_data
    
    for box_data in bboxes_list:
        if isinstance(box_data, dict):
            bbox = box_data.get("bbox", [])
        elif isinstance(box_data, (list, tuple)):
            bbox = box_data
        else:
            continue
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image_shape[1], int(x2))
            y2 = min(image_shape[0], int(y2))
            
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    
    return mask


def isolate_characters_in_image(image_path, bboxes_data):
    """
    Mask out everything outside character bounding boxes (make background white).
    
    Args:
        image_path: Path to input image
        bboxes_data: List of bounding boxes for this image
    
    Returns:
        Image with background masked out (white) or None if error
    """
    # Read image
    image = read_image_as_np_array(image_path)
    if image is None:
        return None
    
    # Create mask for character regions
    mask = create_character_mask(image.shape, bboxes_data)
    
    # Create output image: white background, original where mask is 255
    output_image = np.ones_like(image) * 255  # White background
    output_image[mask == 255] = image[mask == 255]  # Copy character regions
    
    return output_image


def copy_metadata_files(input_dir, output_dir):
    """
    Copy metadata.json files from each manga folder to the output dataset.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
    
    Returns:
        Number of metadata files copied
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    metadata_files = list(input_path.rglob('metadata.json'))
    copied_count = 0
    
    for metadata_file in metadata_files:
        try:
            # Get relative path from input directory
            relative_path = metadata_file.relative_to(input_path)
            output_metadata_file = output_path / relative_path
            
            # Create parent directory if needed
            output_metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy metadata file
            shutil.copy2(metadata_file, output_metadata_file)
            copied_count += 1
        except Exception as e:
            print(f"\nWarning: Could not copy {metadata_file}: {e}")
    
    return copied_count


def process_dataset_folder(
    input_dir,
    output_dir,
    confidence_threshold=0.5,
    extensions=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    gpu=True,
    bboxes_cache_dir=None,
):
    """
    Process all images in a dataset folder with structure dataset_name/manga/chapter/pages.
    Creates output dataset with same structure but only characters visible (background white).
    
    Args:
        input_dir: Input dataset directory (dataset_name folder)
        output_dir: Output directory for isolated character images (will have same structure)
        confidence_threshold: Minimum confidence score for bounding boxes (default: 0.5)
        extensions: List of image file extensions to process
        gpu: Use GPU if available
        bboxes_cache_dir: Directory to cache bounding boxes (default: same as output_dir with _bboxes suffix)
    
    Returns:
        Dictionary with processing statistics
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers is not installed. Install with: pip install transformers")
        return None
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return None
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up bounding boxes cache directory
    if bboxes_cache_dir is None:
        bboxes_cache_dir = Path(str(output_dir) + "_bboxes")
    else:
        bboxes_cache_dir = Path(bboxes_cache_dir)
    bboxes_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {input_dir}")
        return None
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input structure: {input_dir}/manga/chapter/pages")
    print(f"Output structure: {output_dir}/manga/chapter/pages")
    print(f"Bounding boxes cache: {bboxes_cache_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"GPU: {gpu and torch.cuda.is_available()}")
    print("="*60)
    
    # Initialize model (only once for efficiency)
    print("Loading magi model (this may take a moment on first run)...")
    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    try:
        model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True)
        if device == "cuda":
            model = model.cuda()
        model.eval()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    cached_bboxes_count = 0
    
    # Process images in batches for efficiency
    batch_size = 8
    image_batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
    
    for batch in tqdm(image_batches, desc="Processing batches"):
        batch_images = []
        batch_paths = []
        batch_bboxes_files = []
        batch_output_files = []
        batch_needs_detection = []
        
        # Prepare batch
        for img_file in batch:
            relative_path = img_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if output already exists
            if output_file.exists():
                skipped_count += 1
                continue
            
            # Check for cached bounding boxes
            bboxes_file = bboxes_cache_dir / relative_path.with_suffix('.json')
            bboxes_file.parent.mkdir(parents=True, exist_ok=True)
            
            bboxes_data = load_bounding_boxes(bboxes_file)
            if bboxes_data is not None:
                # Use cached bounding boxes
                cached_bboxes_count += 1
                try:
                    # Process single image with cached boxes
                    # bboxes_data from JSON is always a list of lists (batch format)
                    # even for single image: [[box1, box2, ...]]
                    # So we take the first element to get the list of boxes for this image
                    if isinstance(bboxes_data, list) and len(bboxes_data) > 0:
                        boxes_for_image = bboxes_data[0] if isinstance(bboxes_data[0], list) else bboxes_data
                    else:
                        boxes_for_image = bboxes_data
                    
                    output_image = isolate_characters_in_image(img_file, boxes_for_image)
                    if output_image is not None:
                        Image.fromarray(output_image).save(output_file)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"\nError processing {img_file.name} with cached boxes: {e}")
                    error_count += 1
                continue
            
            # Need to run detection
            try:
                image_array = read_image_as_np_array(img_file)
                batch_images.append(image_array)
                batch_paths.append(img_file)
                batch_bboxes_files.append(bboxes_file)
                batch_output_files.append(output_file)
                batch_needs_detection.append(True)
            except Exception as e:
                print(f"\nError reading {img_file.name}: {e}")
                error_count += 1
        
        # Run detection on batch if needed
        if len(batch_images) > 0:
            try:
                with torch.no_grad():
                    results = model.predict_detections_and_associations(batch_images)
                
                # Process each result
                for i, (img_file, bboxes_file, output_file) in enumerate(zip(batch_paths, batch_bboxes_files, batch_output_files)):
                    try:
                        # Save bounding boxes
                        bboxes_data = save_bounding_boxes(bboxes_file, [results[i]], confidence_threshold)
                        
                        # Filter boxes by confidence for this image
                        filtered_boxes = bboxes_data[0] if bboxes_data else []
                        
                        # Isolate characters
                        output_image = isolate_characters_in_image(img_file, filtered_boxes)
                        
                        if output_image is not None:
                            Image.fromarray(output_image).save(output_file)
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        print(f"\nError processing {img_file.name}: {e}")
                        error_count += 1
                        
            except Exception as e:
                print(f"\nError in batch detection: {e}")
                for img_file in batch_paths:
                    error_count += 1
    
    # Copy metadata.json files from each manga folder
    print("\nCopying metadata.json files...")
    metadata_count = copy_metadata_files(input_dir, output_dir)
    if metadata_count > 0:
        print(f"Copied {metadata_count} metadata.json file(s)")
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Used cached bounding boxes: {cached_bboxes_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(image_files)}")
    print(f"Metadata files copied: {metadata_count}")
    print(f"Output saved to: {output_dir}")
    print(f"Bounding boxes cached to: {bboxes_cache_dir}")
    print("="*60)
    
    return {
        'success': success_count,
        'skipped': skipped_count,
        'cached_bboxes': cached_bboxes_count,
        'errors': error_count,
        'total': len(image_files),
        'metadata_copied': metadata_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="Isolate characters in manga images using magi model. Processes dataset with structure: dataset_name/manga/chapter/pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire character dataset
  python isolate_characters.py --input Dataset/char --output Dataset/char_isolated
  
  # Process with custom confidence threshold
  python isolate_characters.py --input Dataset/char --output Dataset/char_isolated --confidence 0.7
  
  # Process specific manga
  python isolate_characters.py --input Dataset/char/Jujutsu\\ Kaisen\\ 0 --output Dataset/char_isolated/Jujutsu\\ Kaisen\\ 0
  
  # Disable GPU
  python isolate_characters.py --input Dataset/char --output Dataset/char_isolated --no-gpu
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset directory (dataset_name folder with manga/chapter/pages structure)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for isolated character images (will preserve manga/chapter/pages structure)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for character bounding boxes (default: 0.5)",
    )
    parser.add_argument(
        "--bboxes-cache",
        type=str,
        default=None,
        help="Directory to cache bounding boxes (default: output_dir + '_bboxes')",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU even if available",
    )
    
    args = parser.parse_args()
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers is not installed.")
        print("Install with: pip install transformers")
        return
    
    # Process dataset
    result = process_dataset_folder(
        input_dir=args.input,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        gpu=not args.no_gpu,
        bboxes_cache_dir=args.bboxes_cache,
    )
    
    if result:
        print(f"\n✓ Successfully processed {result['success']} images")
        if result['cached_bboxes'] > 0:
            print(f"✓ Used {result['cached_bboxes']} cached bounding box files")
        if result['errors'] > 0:
            print(f"⚠ {result['errors']} errors occurred")


if __name__ == "__main__":
    main()

