#!/usr/bin/env python3
"""
CLIP Image Embedding Generator

Generates CLIP image embeddings from manga images.
Preserves folder structure: input_folder/manga/chapter/image.png -> input_folder_embeddings/manga/chapter/image.npy

Usage:
    python clip/image_embeddings.py final_dataset
    python clip/image_embeddings.py final_dataset --output-dir custom_embeddings
    python clip/image_embeddings.py final_dataset --batch-size 32
"""

import argparse
import torch
import open_clip
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import re


def get_device(prefer_cpu: bool = False) -> str:
    """Get the best available device: cuda, mps, or cpu."""
    if prefer_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip_model(device: str = None):
    """
    Load CLIP model and preprocessing function.
    
    Returns:
        tuple: (model, preprocess, device)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading CLIP model (ViT-L-14) on {device}...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    model = model.to(device).eval()
    
    return model, preprocess, device


def get_all_images(image_dir: Path) -> list[Path]:
    """Recursively find all images in a directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = []
    for ext in image_extensions:
        images.extend(image_dir.rglob(f"*{ext}"))
    return sorted(images)


def parse_image_metadata(path: Path, input_dir: Path) -> dict:
    """
    Parse metadata from image path.
    Expected structure: input_dir/manga/chapter_X/page_XXX.ext
    
    Returns:
        dict with manga, chapter, page, relative path
    """
    rel_path = path.relative_to(input_dir)
    parts = rel_path.parts
    
    page_name = path.stem
    chapter_dir = parts[-2] if len(parts) >= 2 else "unknown"
    manga = parts[-3] if len(parts) >= 3 else (parts[0] if len(parts) >= 1 else "unknown")
    
    # Extract page number
    page_match = re.search(r'(\d+)', page_name)
    page_num = int(page_match.group(1)) if page_match else 0
    
    # Extract chapter number
    chapter_match = re.search(r'(\d+)', chapter_dir)
    chapter_num = int(chapter_match.group(1)) if chapter_match else 0
    
    return {
        "manga": manga,
        "chapter": chapter_dir,
        "chapter_num": chapter_num,
        "page": page_name,
        "page_num": page_num,
        "path": str(path),
        "rel_path": str(rel_path),
    }


def extract_image_embeddings(
    model,
    preprocess,
    device: str,
    image_paths: list[Path],
    batch_size: int = 16,
    normalize: bool = True
) -> tuple[np.ndarray, list[Path]]:
    """
    Extract CLIP image embeddings.
    
    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device to run on
        image_paths: List of image paths
        batch_size: Batch size for processing
        normalize: If True, L2 normalize embeddings
    
    Returns:
        tuple: (embeddings array, valid paths list)
    """
    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []

        for path in batch_paths:
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                batch_images.append(img)
                batch_valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            
            if normalize:
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
            valid_paths.extend(batch_valid_paths)

    if all_embeddings:
        return np.vstack(all_embeddings).astype(np.float32), valid_paths
    return np.array([]).astype(np.float32), []


def process_images(
    input_dir: Path,
    output_dir: Path,
    model,
    preprocess,
    device: str,
    batch_size: int = 16
) -> dict:
    """
    Process all images and generate embeddings.
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for embeddings
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        Statistics dictionary
    """
    image_paths = get_all_images(input_dir)
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {"error": "No files found"}
    
    print(f"Found {len(image_paths)} images")
    
    stats = {
        "total_files": len(image_paths),
        "processed": 0,
        "failed": 0,
    }
    
    # Extract embeddings
    print(f"\nExtracting embeddings for {len(image_paths)} images...")
    embeddings, valid_paths = extract_image_embeddings(
        model, preprocess, device, image_paths, batch_size
    )
    
    stats["failed"] = len(image_paths) - len(valid_paths)
    
    if len(valid_paths) == 0:
        print("No valid images to process!")
        return stats
    
    # Save embeddings with same folder structure
    print(f"\nSaving embeddings to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_metadata = []
    
    for i, path in enumerate(tqdm(valid_paths, desc="Saving embeddings")):
        # Compute relative path from input_dir
        rel_path = path.relative_to(input_dir)
        
        # Create output path with .npy extension
        output_path = output_dir / rel_path.with_suffix('.npy')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual embedding
        np.save(output_path, embeddings[i])
        stats["processed"] += 1
        
        # Collect metadata
        meta = parse_image_metadata(path, input_dir)
        meta["embedding_path"] = str(output_path.relative_to(output_dir))
        file_metadata.append(meta)
    
    # Save all embeddings in a single file
    all_embeddings_path = output_dir / "all_embeddings.npy"
    np.save(all_embeddings_path, embeddings)
    
    # Save metadata
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "total_embeddings": len(embeddings),
        "files": file_metadata,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save a simple paths list for easy loading
    paths_list = [str(p.relative_to(input_dir)) for p in valid_paths]
    with open(output_dir / "paths.json", "w") as f:
        json.dump(paths_list, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP image embeddings from manga images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clip/clip.py final_dataset
    python clip/clip.py final_dataset --output-dir my_embeddings
    python clip/clip.py final_dataset --batch-size 32
        """
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing images (manga/chapter/image structure)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: input_dir_embeddings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation (default: 16)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Default output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_embeddings"
    
    print("=" * 60)
    print("  CLIP Image Embedding Generator")
    print("=" * 60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Load model
    device = "cpu" if args.cpu else get_device()
    model, preprocess, device = load_clip_model(device)
    
    # Process images
    stats = process_images(
        input_dir, output_dir,
        model, preprocess, device,
        batch_size=args.batch_size
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total files:  {stats.get('total_files', 0)}")
    print(f"  Processed:    {stats.get('processed', 0)}")
    print(f"  Failed:       {stats.get('failed', 0)}")
    print(f"\n  Saved to: {output_dir}")
    print(f"    - Individual .npy files per image")
    print(f"    - all_embeddings.npy (combined)")
    print(f"    - metadata.json")
    print(f"    - paths.json")


if __name__ == "__main__":
    main()
