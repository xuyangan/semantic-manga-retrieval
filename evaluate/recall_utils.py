#!/usr/bin/env python3
"""
Utility functions for recall evaluation scripts.

Common functions used across recall_image.py, recall_text.py, recall_fusion.py,
and recall_intervention.py.

Provides:
- FAISS index loading and searching
- Path normalization and ground truth matching
- Recall metric computation
- Visualization helpers
- Summary table generation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in path for imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================================================
# Device and Model Loading
# ============================================================================

def get_device(prefer_cpu: bool = False) -> str:
    """
    Get the best available device: cuda, mps, or cpu.
    
    Args:
        prefer_cpu: If True, always return 'cpu'
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    import torch
    if prefer_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_faiss_available() -> bool:
    """Check if FAISS is available and raise helpful error if not."""
    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss is not installed. Install with: pip install faiss-cpu or faiss-gpu"
        )
    return True


# ============================================================================
# FAISS Index Loading and Searching
# ============================================================================

def load_faiss_index(index_folder: Path) -> Tuple[faiss.Index, Dict[int, Dict], int]:
    """
    Load FAISS index and metadata from a folder.
    
    Args:
        index_folder: Path to folder containing faiss.index and metadata.json
    
    Returns:
        tuple: (faiss_index, id_to_meta mapping, dimension)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is not installed")
    
    # Load FAISS index
    index_file = index_folder / "faiss.index"
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}")
    
    index = faiss.read_index(str(index_file))
    dimension = index.d  # Get dimension from FAISS index
    print(f"Loaded FAISS index with {index.ntotal} vectors (dimension: {dimension})")
    
    # Load metadata
    metadata_file = index_folder / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)
    
    # Build id_to_meta mapping
    id_to_meta = {}
    if "metadata" in metadata_json:
        # Format: {"metadata": {"0": {...}, "1": {...}, ...}}
        for str_id, meta in metadata_json["metadata"].items():
            int_id = int(str_id)
            # For image indexes, extract specific fields; for text, keep all
            if "manga" in meta or "source_file" in meta:
                # Image format or text format - preserve all fields
                id_to_meta[int_id] = meta
            else:
                # Fallback: extract common fields
                id_to_meta[int_id] = {
                    "manga": meta.get("manga", ""),
                    "chapter": meta.get("chapter", ""),
                    "page": meta.get("page", ""),
                    "path": meta.get("path", ""),
                    "rel_path": meta.get("rel_path", ""),
                    "relative_path": meta.get("rel_path", ""),  # alias
                }
    else:
        # Direct format: {"0": {...}, "1": {...}, ...}
        for str_id, meta in metadata_json.items():
            if str_id.isdigit():
                int_id = int(str_id)
                id_to_meta[int_id] = meta
    
    print(f"Loaded metadata for {len(id_to_meta)} items")
    return index, id_to_meta, dimension


def load_faiss_index_direct(index_dir: Path) -> Tuple:
    """
    Load FAISS index directly from index directory (not from subfolder).
    
    This is used when the index files (faiss.index, metadata.json) are directly
    in the provided directory, not in a 'faiss_index' subfolder.
    
    Args:
        index_dir: Path to index directory containing faiss.index and metadata.json
    
    Returns:
        tuple: (faiss_index, id_to_meta dict, dimension)
    
    Raises:
        ImportError: If faiss is not installed
        FileNotFoundError: If index or metadata files don't exist
    """
    check_faiss_available()
    
    index_dir = Path(index_dir)
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path) as f:
        data = json.load(f)
    
    # Handle both metadata formats
    if "metadata" in data:
        id_to_meta = {int(k): v for k, v in data["metadata"].items()}
    else:
        id_to_meta = {int(k): v for k, v in data.items() if k.isdigit()}
    
    dimension = data.get("dimension", index.d)
    
    return index, id_to_meta, dimension


def search_faiss_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    id_to_meta: Dict[int, Dict],
    k: int = 100
) -> List[Dict]:
    """
    Search FAISS index for similar embeddings.
    
    Args:
        index: FAISS index
        query_embedding: Query embedding (D,) or (1, D)
        id_to_meta: Mapping from int ID to metadata
        k: Number of results to retrieve
    
    Returns:
        List of results with similarity scores and metadata
    """
    # Prepare query
    query = np.ascontiguousarray(
        query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding,
        dtype=np.float32
    )
    faiss.normalize_L2(query)
    
    # Search
    scores, indices = index.search(query, k)
    
    # Build results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1 and idx in id_to_meta:
            results.append({
                "rank": i + 1,
                "similarity": float(score),
                **id_to_meta[idx]
            })
    
    return results


# ============================================================================
# Path Normalization and Matching
# ============================================================================

def normalize_path_for_matching(path: str) -> str:
    """
    Normalize a path for matching with ground truth.
    Handles different path formats.
    
    Args:
        path: Path string to normalize
    
    Returns:
        Normalized path string (manga/chapter/page without extension)
    """
    # Remove file extension
    path = path.rsplit('.', 1)[0] if '.' in path else path
    # Normalize separators
    path = path.replace('\\', '/')
    # Remove leading/trailing slashes
    path = path.strip('/')
    return path


def extract_manga_chapter_page(path: str) -> str:
    """
    Extract manga/chapter/page from a path, taking the last 3 components.
    This handles paths that might have prefixes like 'final_dataset/' or dataset names.
    
    Args:
        path: Path string (can be relative_path, full path, etc.)
    
    Returns:
        Normalized manga/chapter/page string (without extension)
    """
    # Normalize the path first
    normalized = normalize_path_for_matching(path)
    
    # Split into parts
    parts = normalized.split('/')
    
    # Take last 3 parts: manga/chapter/page
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    elif len(parts) == 2:
        # If only 2 parts, assume chapter/page (missing manga)
        return "/".join(parts)
    else:
        # If only 1 part, return as is
        return parts[0] if parts else ""


def match_result_to_ground_truth(result: Dict, ground_truth: Set[str], include_page_key: bool = True, include_source_file: bool = True) -> bool:
    """
    Check if a search result matches any ground truth path.
    Ground truth labels are already in format: manga/chapter/page (normalized, without extension)
    We extract manga/chapter/page from the result and compare.
    
    Args:
        result: Search result dict with path information
        ground_truth: Set of ground truth paths (already normalized to manga/chapter/page format)
        include_page_key: Whether to check page_key field (for text/fusion results)
        include_source_file: Whether to check source_file field (for text results)
    
    Returns:
        True if result matches ground truth
    """
    # Ground truth is already normalized to manga/chapter/page format by load_ground_truth_labels
    # So we can use it directly (no need to normalize again)
    
    # Try different path formats from result and extract manga/chapter/page
    paths_to_check = []
    
    # Method 1: Use page_key if available (from deduplication or late fusion)
    if include_page_key and "page_key" in result and result["page_key"]:
        extracted = extract_manga_chapter_page(result["page_key"])
        if extracted:
            normalized = normalize_path_for_matching(extracted)
            paths_to_check.append(normalized)
    
    # Method 2: Use source_file (text embeddings format)
    if include_source_file and "source_file" in result and result["source_file"]:
        extracted = extract_manga_chapter_page(result["source_file"])
        if extracted:
            normalized = normalize_path_for_matching(extracted)
            paths_to_check.append(normalized)
    
    # Method 3: Use image_meta or text_meta (from late fusion)
    for meta_key in ["image_meta", "text_meta"]:
        if meta_key in result and result[meta_key]:
            meta = result[meta_key]
            # Try rel_path or relative_path
            for key in ["rel_path", "relative_path", "path"]:
                if key in meta and meta[key]:
                    extracted = extract_manga_chapter_page(meta[key])
                    if extracted:
                        normalized = normalize_path_for_matching(extracted)
                        paths_to_check.append(normalized)
            
            # Try constructing from manga/chapter/page fields
            if all(k in meta for k in ["manga", "chapter", "page"]):
                # Remove extension from page if present
                page = meta["page"]
                page_base = page.rsplit('.', 1)[0] if '.' in page else page
                constructed = f"{meta['manga']}/{meta['chapter']}/{page_base}"
                constructed = normalize_path_for_matching(constructed)
                paths_to_check.append(constructed)
    
    # Method 4: Use rel_path or relative_path (direct fields)
    for key in ["rel_path", "relative_path", "path"]:
        if key in result and result[key]:
            extracted = extract_manga_chapter_page(result[key])
            if extracted:
                normalized = normalize_path_for_matching(extracted)
                paths_to_check.append(normalized)
    
    # Method 5: Construct from manga/chapter/page fields (most reliable for image search)
    if all(k in result for k in ["manga", "chapter", "page"]):
        # Remove extension from page if present
        page = result["page"]
        page_base = page.rsplit('.', 1)[0] if '.' in page else page
        constructed = f"{result['manga']}/{result['chapter']}/{page_base}"
        constructed = normalize_path_for_matching(constructed)
        paths_to_check.append(constructed)
    
    # Check if any extracted path matches ground truth
    for path in paths_to_check:
        if path in ground_truth:
            return True
    
    return False


def find_image_path_from_result(result: Dict, image_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find image path from search result metadata.
    
    Tries multiple methods to locate the image file:
    1. Direct 'path' field
    2. From 'image_meta' nested dict
    3. Constructed from manga/chapter/page fields
    
    Args:
        result: Search result dictionary with metadata
        image_dir: Optional base directory for images
    
    Returns:
        Path to image if found, None otherwise
    """
    paths_to_try = []
    
    # Method 1: Direct path field
    if "path" in result and result["path"]:
        paths_to_try.append(Path(result["path"]))
    
    # Method 2: From image_meta
    if "image_meta" in result and result["image_meta"]:
        meta = result["image_meta"]
        if "path" in meta and meta["path"]:
            paths_to_try.append(Path(meta["path"]))
        elif "rel_path" in meta and meta["rel_path"]:
            paths_to_try.append(Path(meta["rel_path"]))
    
    # Method 3: Construct from manga/chapter/page
    meta = result.get("image_meta", result)
    if all(k in meta for k in ["manga", "chapter", "page"]):
        manga = meta["manga"]
        chapter = meta["chapter"]
        page = meta["page"]
        # Remove extension from page if present
        page_base = page.rsplit('.', 1)[0] if '.' in page else page
        
        if image_dir:
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                candidate = image_dir / manga / chapter / f"{page_base}{ext}"
                paths_to_try.append(candidate)
    
    # Try each path
    for path in paths_to_try:
        if path.exists():
            return path
    
    return None


def get_page_key_from_result(result: Dict) -> str:
    """
    Extract page key from result for deduplication and matching.
    
    Args:
        result: Search result dictionary
    
    Returns:
        Page key string (manga/chapter/page format)
    """
    # Try page_key field first
    if "page_key" in result and result["page_key"]:
        return result["page_key"]
    
    # Try source_file (text embeddings)
    if "source_file" in result and result["source_file"]:
        return str(Path(result["source_file"]).with_suffix(""))
    
    # Try constructing from fields
    if all(k in result for k in ["manga", "chapter", "page"]):
        page = result["page"]
        page_base = page.rsplit('.', 1)[0] if '.' in page else page
        return f"{result['manga']}/{result['chapter']}/{page_base}"
    
    return ""


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_ground_truth_labels(labels_file: Path) -> Set[str]:
    """
    Load ground truth labels from labels.txt file.
    Format: manga/chapter/page.ext or final_dataset/manga/chapter/page.ext (one per line)
    
    Args:
        labels_file: Path to labels.txt
    
    Returns:
        Set of ground truth paths (normalized to manga/chapter/page format, without extension)
    """
    if not labels_file.exists():
        return set()
    
    ground_truth = set()
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract manga/chapter/page from the path
            # This handles both "final_dataset/manga/chapter/page.ext" and "manga/chapter/page.ext"
            extracted = extract_manga_chapter_page(line)
            if extracted:
                # Normalize to ensure consistent format
                normalized = normalize_path_for_matching(extracted)
                ground_truth.add(normalized)
    
    return ground_truth


# ============================================================================
# Query Loading
# ============================================================================

def load_text_query(query_folder: Path) -> str:
    """
    Load text query from query folder.
    Tries text.txt first, then text_query.txt.
    
    Args:
        query_folder: Path to query folder
    
    Returns:
        Text query string or empty string if not found
    """
    # Try different possible file names
    for filename in ["text.txt", "text_query.txt", "query.txt"]:
        text_file = query_folder / filename
        if text_file.exists():
            return text_file.read_text(encoding="utf-8").strip()
    
    return ""


def load_llm_description(query_folder: Path) -> Optional[str]:
    """
    Load cached LLM description from query folder if it exists.
    
    Args:
        query_folder: Path to query folder
    
    Returns:
        LLM description string if file exists, None otherwise
    """
    llm_file = query_folder / "llm_description.txt"
    if llm_file.exists():
        return llm_file.read_text(encoding="utf-8").strip()
    return None


def save_llm_description(query_folder: Path, description: str) -> Path:
    """
    Save LLM description to query folder for future reuse.
    
    Args:
        query_folder: Path to query folder
        description: LLM-generated description to save
    
    Returns:
        Path to the saved file
    """
    llm_file = query_folder / "llm_description.txt"
    llm_file.write_text(description, encoding="utf-8")
    return llm_file


def find_query_image(query_folder: Path) -> Optional[Path]:
    """
    Find query image in query folder.
    Searches for any image file with common image extensions.
    
    Args:
        query_folder: Path to query folder
    
    Returns:
        Path to query image if found, None otherwise
    """
    # Search for any image file in the folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    # First, try to find any file with image extensions
    for ext in image_extensions:
        # Use glob to find files matching the pattern
        matches = list(query_folder.glob(f"*{ext}"))
        if matches:
            # Return the first match (prefer .png, then .jpg, etc.)
            return matches[0]
    
    # If no matches found with glob, try case-insensitive search
    for file_path in query_folder.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in image_extensions:
                return file_path
    
    return None


# ============================================================================
# Recall Computation
# ============================================================================

def compute_recall_at_k(
    search_results: List[Dict],
    ground_truth: Set[str],
    k: int,
    include_page_key: bool = True,
    include_source_file: bool = True
) -> Dict:
    """
    Compute recall@k for search results.
    
    Args:
        search_results: List of search results
        ground_truth: Set of ground truth paths
        k: Number of top results to consider
        include_page_key: Whether to check page_key in matching (for text/fusion)
        include_source_file: Whether to check source_file in matching (for text)
    
    Returns:
        Dictionary with recall metrics
    """
    if len(ground_truth) == 0:
        return {
            "k": k,
            "recall": 0.0,
            "relevant_retrieved": 0,
            "total_ground_truth": 0,
            "total_retrieved": min(k, len(search_results))
        }
    
    # Get top k results
    top_k = search_results[:k]
    
    # Count relevant results
    relevant_retrieved = []
    for result in top_k:
        if match_result_to_ground_truth(result, ground_truth, include_page_key, include_source_file):
            relevant_retrieved.append(result)
    
    recall = len(relevant_retrieved) / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    return {
        "k": k,
        "recall": float(recall),
        "relevant_retrieved": len(relevant_retrieved),
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(top_k),
        "relevant_results": relevant_retrieved
    }


def compute_map_at_k(
    search_results: List[Dict],
    ground_truth: Set[str],
    k: int,
    include_page_key: bool = True,
    include_source_file: bool = True
) -> Dict:
    """
    Compute Mean Average Precision (mAP) at k for search results.
    
    Average Precision (AP) at k is computed as:
    AP@k = (1 / min(k, |relevant|)) * sum(precision@i for each relevant item at position i)
    
    Args:
        search_results: List of search results (sorted by similarity)
        ground_truth: Set of ground truth paths
        k: Number of top results to consider
        include_page_key: Whether to check page_key in matching (for text/fusion)
        include_source_file: Whether to check source_file in matching (for text)
    
    Returns:
        Dictionary with mAP@k metrics
    """
    if len(ground_truth) == 0:
        return {
            "k": k,
            "map": 0.0,
            "ap": 0.0,
            "relevant_retrieved": 0,
            "total_ground_truth": 0,
            "total_retrieved": min(k, len(search_results))
        }
    
    # Get top k results
    top_k = search_results[:k]
    
    # Track which results are relevant and their positions
    relevant_positions = []
    for i, result in enumerate(top_k):
        if match_result_to_ground_truth(result, ground_truth, include_page_key, include_source_file):
            relevant_positions.append(i + 1)  # Position (1-indexed)
    
    if len(relevant_positions) == 0:
        return {
            "k": k,
            "map": 0.0,
            "ap": 0.0,
            "relevant_retrieved": 0,
            "total_ground_truth": len(ground_truth),
            "total_retrieved": len(top_k)
        }
    
    # Compute Average Precision (AP) at k
    # AP@k = (1 / |relevant|) * sum(precision@i for each relevant item found in top k)
    # This is the standard definition where we divide by total relevant items, not min(k, |relevant|)
    num_relevant = len(ground_truth)
    
    if num_relevant == 0:
        return {
            "k": k,
            "map": 0.0,
            "ap": 0.0,
            "relevant_retrieved": 0,
            "total_ground_truth": 0,
            "total_retrieved": len(top_k)
        }
    
    precision_sum = 0.0
    for pos in relevant_positions:
        # Precision at position i = (# relevant items up to position i) / i
        relevant_up_to_pos = len([p for p in relevant_positions if p <= pos])
        precision_at_pos = relevant_up_to_pos / pos
        precision_sum += precision_at_pos
    
    # Divide by total number of relevant items (standard AP@k definition)
    ap_at_k = precision_sum / num_relevant if num_relevant > 0 else 0.0
    
    return {
        "k": k,
        "map": float(ap_at_k),  # AP for this query (mAP is average across queries)
        "ap": float(ap_at_k),  # Alias for consistency
        "relevant_retrieved": len(relevant_positions),
        "total_ground_truth": num_relevant,
        "total_retrieved": len(top_k),
        "relevant_positions": relevant_positions
    }


def compute_average_metrics(all_results: List[Dict]) -> Dict:
    """
    Compute average recall@k and mAP@k metrics across all queries.
    
    Args:
        all_results: List of recall result dictionaries
    
    Returns:
        Dictionary with average metrics
    """
    # Filter out results with errors
    valid_results = [r for r in all_results if 'error' not in r and 'recall_metrics' in r]
    
    if len(valid_results) == 0:
        return {"error": "No valid results to average"}
    
    # Collect all recall metrics
    recall_scores = {}
    for result in valid_results:
        for metric_name, metric_data in result['recall_metrics'].items():
            if metric_name not in recall_scores:
                recall_scores[metric_name] = []
            recall_scores[metric_name].append(metric_data['recall'])
    
    # Collect all mAP metrics
    map_scores = {}
    for result in valid_results:
        if 'map_metrics' in result:
            for metric_name, metric_data in result['map_metrics'].items():
                if metric_name not in map_scores:
                    map_scores[metric_name] = []
                map_scores[metric_name].append(metric_data.get('map', metric_data.get('ap', 0.0)))
    
    # Compute averages for recall
    averages = {}
    for metric_name, scores in recall_scores.items():
        averages[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    
    # Compute averages for mAP
    for metric_name, scores in map_scores.items():
        averages[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    
    # Overall statistics
    total_ground_truth = sum(r.get('total_ground_truth', 0) for r in valid_results)
    total_retrieved = sum(r.get('total_retrieved', 0) for r in valid_results)
    total_relevant = sum(len(r.get('relevant_results', [])) for r in valid_results)
    
    return {
        "num_queries": len(valid_results),
        "num_failed": len(all_results) - len(valid_results),
        "average_metrics": averages,
        "overall_stats": {
            "total_ground_truth": int(total_ground_truth),
            "total_retrieved": int(total_retrieved),
            "total_relevant_retrieved": int(total_relevant),
            "overall_recall": float(total_relevant / total_ground_truth) if total_ground_truth > 0 else 0.0
        }
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def pad_image_to_square(img: Image.Image, fill_color: tuple = (255, 255, 255)) -> Image.Image:
    """
    Pad an image to make it square while preserving aspect ratio.
    
    Args:
        img: PIL Image to pad
        fill_color: RGB tuple for padding color (default: white)
    
    Returns:
        Square PIL Image with original image centered
    """
    width, height = img.size
    
    # If already square, return as is
    if width == height:
        return img
    
    # Determine the size of the square (use the larger dimension)
    size = max(width, height)
    
    # Create a new square image with fill color
    square_img = Image.new("RGB", (size, size), fill_color)
    
    # Calculate position to paste the original image (centered)
    paste_x = (size - width) // 2
    paste_y = (size - height) // 2
    
    # Paste the original image onto the square canvas
    square_img.paste(img, (paste_x, paste_y))
    
    return square_img


def add_border_to_image(img: Image.Image, border_width: int = 5, border_color: tuple = (0, 0, 0)) -> Image.Image:
    """
    Add a border around an image.
    
    Args:
        img: PIL Image to add border to
        border_width: Width of the border in pixels
        border_color: RGB tuple for border color
    
    Returns:
        PIL Image with border added
    """
    from PIL import ImageOps
    return ImageOps.expand(img, border=border_width, fill=border_color)


def plot_recall_curve(result: Dict, output_path: Path, color: str = 'blue', title_prefix: str = ""):
    """
    Plot recall@k and mAP@k curves for a single query, stacked vertically.
    
    Args:
        result: Result dictionary with recall_metrics and map_metrics
        output_path: Path to save the plot
        color: Color for the plot line
        title_prefix: Prefix for the title (e.g., "Late Fusion", "Text Query")
    """
    if 'error' in result or 'recall_metrics' not in result:
        return
    
    recall_metrics = result.get('recall_metrics', {})
    map_metrics = result.get('map_metrics', {})
    
    # Extract k values and recall scores, sort by k value
    recall_metrics_list = []
    for metric_name, metric_data in recall_metrics.items():
        k = metric_data.get('k', 0)
        recall = metric_data.get('recall', 0.0)
        recall_metrics_list.append((k, recall))
    
    recall_metrics_list.sort(key=lambda x: x[0])
    recall_k_values = [k for k, _ in recall_metrics_list]
    recall_scores = [recall for _, recall in recall_metrics_list]
    
    # Extract k values and mAP scores, sort by k value
    map_metrics_list = []
    for metric_name, metric_data in map_metrics.items():
        k = metric_data.get('k', 0)
        map_score = metric_data.get('map', metric_data.get('ap', 0.0))
        map_metrics_list.append((k, map_score))
    
    map_metrics_list.sort(key=lambda x: x[0])
    map_k_values = [k for k, _ in map_metrics_list]
    map_scores = [map_score for _, map_score in map_metrics_list]
    
    if len(recall_k_values) == 0 and len(map_k_values) == 0:
        return
    
    # Add origin point (k=0, value=0) for better visualization
    if recall_k_values and recall_k_values[0] > 0:
        recall_k_values = [0] + recall_k_values
        recall_scores = [0.0] + recall_scores
    
    if map_k_values and map_k_values[0] > 0:
        map_k_values = [0] + map_k_values
        map_scores = [0.0] + map_scores
    
    # Create subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    query_name = Path(result['query_folder']).name if 'query_folder' in result else "Query"
    query_text = result.get('user_query', result.get('query_text', ''))[:50]
    gt = result.get('total_ground_truth', 0)
    rel = len(result.get('relevant_results', []))
    
    title_prefix_str = f"{title_prefix} " if title_prefix else ""
    main_title = f"{title_prefix_str}Query: {query_name}\n\"{query_text}...\" | Ground Truth: {gt} | Relevant Retrieved: {rel}"
    fig.suptitle(main_title, fontsize=12, fontweight='bold')
    
    # Plot Recall@K on top
    if len(recall_k_values) > 0:
        # Plot line with markers only at actual data points (not at k=0)
        ax1.plot(recall_k_values, recall_scores, linewidth=2.5, color=color, label='Recall@K', zorder=5)
        # Add markers only at actual data points (skip k=0)
        actual_k = recall_k_values[1:] if recall_k_values[0] == 0 else recall_k_values
        actual_scores = recall_scores[1:] if recall_k_values[0] == 0 else recall_scores
        ax1.scatter(actual_k, actual_scores, s=80, color=color, zorder=10, edgecolors='white', linewidths=1.5)
        ax1.fill_between(recall_k_values, recall_scores, alpha=0.2, color=color)
        ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax1.set_ylabel("Recall@K", fontsize=11, fontweight='bold')
        ax1.set_title("Recall@K Curve", fontsize=11, fontweight='bold')
        ax1.set_xticks(recall_k_values)
        ax1.set_xlim(left=-2, right=max(recall_k_values) + 5)
        ax1.set_ylim(bottom=-0.02, top=1.05)
        ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels on actual data points (skip k=0)
        for k, recall in zip(actual_k, actual_scores):
            ax1.annotate(f'{recall:.3f}', (k, recall),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Plot mAP@K on bottom
    if len(map_k_values) > 0:
        # Plot line with markers only at actual data points (not at k=0)
        ax2.plot(map_k_values, map_scores, linewidth=2.5, color='green', label='mAP@K', zorder=5)
        # Add markers only at actual data points (skip k=0)
        actual_k = map_k_values[1:] if map_k_values[0] == 0 else map_k_values
        actual_scores = map_scores[1:] if map_k_values[0] == 0 else map_scores
        ax2.scatter(actual_k, actual_scores, s=80, color='green', zorder=10, marker='s', edgecolors='white', linewidths=1.5)
        ax2.fill_between(map_k_values, map_scores, alpha=0.2, color='green')
        ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax2.set_xlabel("k (Number of Retrieved Results)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("mAP@K", fontsize=11, fontweight='bold')
        ax2.set_title("mAP@K Curve", fontsize=11, fontweight='bold')
        ax2.set_xticks(map_k_values)
        ax2.set_xlim(left=-2, right=max(map_k_values) + 5)
        ax2.set_ylim(bottom=-0.02, top=1.05)
        ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add value labels on actual data points (skip k=0)
        for k, map_score in zip(actual_k, actual_scores):
            ax2.annotate(f'{map_score:.3f}', (k, map_score),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_aggregate_recall_curves(all_results: List[Dict], averages: Dict, output_path: Path, color: str = 'blue', title: str = "All Queries"):
    """
    Plot all recall@k and mAP@k curves together for comparison, stacked vertically.
    
    Args:
        all_results: List of result dictionaries
        averages: Average metrics dictionary
        output_path: Path to save the plot
        color: Color for the average curve
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Collect all recall curves
    all_recall_k_values = set()
    recall_curves_data = []
    
    # Collect all mAP curves
    all_map_k_values = set()
    map_curves_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
        
        # Process recall metrics
        if 'recall_metrics' in result:
            recall_metrics = result['recall_metrics']
            recall_k_values = []
            recall_scores = []
            
            metrics_list = []
            for metric_name, metric_data in recall_metrics.items():
                k = metric_data.get('k', 0)
                recall = metric_data.get('recall', 0.0)
                metrics_list.append((k, recall))
                all_recall_k_values.add(k)
            
            metrics_list.sort(key=lambda x: x[0])
            recall_k_values = [k for k, _ in metrics_list]
            recall_scores = [recall for _, recall in metrics_list]
            
            if len(recall_k_values) > 0:
                query_name = Path(result['query_folder']).name[:30] if 'query_folder' in result else "Query"
                recall_curves_data.append({
                    'k_values': recall_k_values,
                    'scores': recall_scores,
                    'label': query_name
                })
        
        # Process mAP metrics
        if 'map_metrics' in result:
            map_metrics = result['map_metrics']
            map_k_values = []
            map_scores = []
            
            metrics_list = []
            for metric_name, metric_data in map_metrics.items():
                k = metric_data.get('k', 0)
                map_score = metric_data.get('map', metric_data.get('ap', 0.0))
                metrics_list.append((k, map_score))
                all_map_k_values.add(k)
            
            metrics_list.sort(key=lambda x: x[0])
            map_k_values = [k for k, _ in metrics_list]
            map_scores = [map_score for _, map_score in metrics_list]
            
            if len(map_k_values) > 0:
                query_name = Path(result['query_folder']).name[:30] if 'query_folder' in result else "Query"
                map_curves_data.append({
                    'k_values': map_k_values,
                    'scores': map_scores,
                    'label': query_name
                })
    
    # Add origin point (k=0, value=0) for better visualization
    for curve in recall_curves_data:
        if curve['k_values'] and curve['k_values'][0] > 0:
            curve['k_values'] = [0] + curve['k_values']
            curve['scores'] = [0.0] + curve['scores']
    
    for curve in map_curves_data:
        if curve['k_values'] and curve['k_values'][0] > 0:
            curve['k_values'] = [0] + curve['k_values']
            curve['scores'] = [0.0] + curve['scores']
    
    # Add 0 to the k_value sets
    all_recall_k_values.add(0)
    all_map_k_values.add(0)
    
    # Plot Recall@K curves on top subplot
    # Plot individual recall curves (lighter colors)
    individual_labeled = False
    for curve in recall_curves_data:
        label = 'Individual Queries' if not individual_labeled else ''
        ax1.plot(curve['k_values'], curve['scores'],
               alpha=0.25, linewidth=1, color='steelblue', label=label)
        if not individual_labeled:
            individual_labeled = True
    
    # Plot average recall curve (bold)
    if 'error' not in averages and 'average_metrics' in averages:
        avg_metrics = averages['average_metrics']
        recall_avg_list = []
        
        for metric_name, stats in avg_metrics.items():
            if metric_name.startswith('recall@'):
                k = int(metric_name.split('@')[1])
                recall = stats.get('mean', 0.0)
                recall_avg_list.append((k, recall))
        
        recall_avg_list.sort(key=lambda x: x[0])
        recall_avg_k_values = [k for k, _ in recall_avg_list]
        recall_avg_scores = [recall for _, recall in recall_avg_list]
        
        # Add origin point to average curve
        if recall_avg_k_values and recall_avg_k_values[0] > 0:
            recall_avg_k_values = [0] + recall_avg_k_values
            recall_avg_scores = [0.0] + recall_avg_scores
        
        if len(recall_avg_k_values) > 0:
            ax1.plot(recall_avg_k_values, recall_avg_scores,
                   linewidth=3.5, color=color, label='Average Recall@K', zorder=10)
            # Add markers only at actual data points (skip k=0)
            actual_k = recall_avg_k_values[1:] if recall_avg_k_values[0] == 0 else recall_avg_k_values
            actual_scores = recall_avg_scores[1:] if recall_avg_k_values[0] == 0 else recall_avg_scores
            ax1.scatter(actual_k, actual_scores, s=120, color=color, zorder=15, 
                       edgecolors='white', linewidths=2)
            ax1.fill_between(recall_avg_k_values, recall_avg_scores, alpha=0.15, color=color)
            
            # Add value labels on average curve (skip k=0)
            for k, recall in zip(actual_k, actual_scores):
                ax1.annotate(f'{recall:.3f}', (k, recall),
                           textcoords="offset points", xytext=(0, 14),
                           ha='center', fontsize=10, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax1.set_ylabel("Recall@K", fontsize=12, fontweight='bold')
    ax1.set_title(f"Recall@K Curves - {title}", fontsize=14, fontweight='bold')
    ax1.set_xlim(left=-2)
    ax1.set_ylim(bottom=-0.02, top=1.05)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    if len(recall_curves_data) > 0:
        all_recall_k_sorted = sorted(all_recall_k_values)
        ax1.set_xticks(all_recall_k_sorted)
        ax1.set_xlim(right=max(all_recall_k_sorted) + 5)
    
    if 'error' not in averages:
        ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Plot mAP@K curves on bottom subplot
    # Plot individual mAP curves (lighter colors)
    individual_labeled = False
    for curve in map_curves_data:
        label = 'Individual Queries' if not individual_labeled else ''
        ax2.plot(curve['k_values'], curve['scores'],
               alpha=0.25, linewidth=1, color='lightgreen', label=label)
        if not individual_labeled:
            individual_labeled = True
    
    # Plot average mAP curve (bold)
    if 'error' not in averages and 'average_metrics' in averages:
        avg_metrics = averages['average_metrics']
        map_avg_list = []
        
        for metric_name, stats in avg_metrics.items():
            if metric_name.startswith('map@'):
                k = int(metric_name.split('@')[1])
                map_score = stats.get('mean', 0.0)
                map_avg_list.append((k, map_score))
        
        map_avg_list.sort(key=lambda x: x[0])
        map_avg_k_values = [k for k, _ in map_avg_list]
        map_avg_scores = [map_score for _, map_score in map_avg_list]
        
        # Add origin point to average curve
        if map_avg_k_values and map_avg_k_values[0] > 0:
            map_avg_k_values = [0] + map_avg_k_values
            map_avg_scores = [0.0] + map_avg_scores
        
        if len(map_avg_k_values) > 0:
            ax2.plot(map_avg_k_values, map_avg_scores,
                   linewidth=3.5, color='green', label='Average mAP@K', zorder=10)
            # Add markers only at actual data points (skip k=0)
            actual_k = map_avg_k_values[1:] if map_avg_k_values[0] == 0 else map_avg_k_values
            actual_scores = map_avg_scores[1:] if map_avg_k_values[0] == 0 else map_avg_scores
            ax2.scatter(actual_k, actual_scores, s=120, color='green', zorder=15, 
                       marker='s', edgecolors='white', linewidths=2)
            ax2.fill_between(map_avg_k_values, map_avg_scores, alpha=0.15, color='green')
            
            # Add value labels on average curve (skip k=0)
            for k, map_score in zip(actual_k, actual_scores):
                ax2.annotate(f'{map_score:.3f}', (k, map_score),
                           textcoords="offset points", xytext=(0, 14),
                           ha='center', fontsize=10, fontweight='bold', color='green',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax2.set_xlabel("k (Number of Retrieved Results)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("mAP@K", fontsize=12, fontweight='bold')
    ax2.set_title(f"mAP@K Curves - {title}", fontsize=14, fontweight='bold')
    ax2.set_xlim(left=-2)
    ax2.set_ylim(bottom=-0.02, top=1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    if len(map_curves_data) > 0:
        all_map_k_sorted = sorted(all_map_k_values)
        ax2.set_xticks(all_map_k_sorted)
        ax2.set_xlim(right=max(all_map_k_sorted) + 5)
    
    if 'error' not in averages:
        ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# Summary Table Creation
# ============================================================================

def create_summary_table(all_results: List[Dict], averages: Dict, output_path: Path, include_user_query: bool = False):
    """
    Create a summary table of all recall results.
    
    Args:
        all_results: List of result dictionaries
        averages: Average metrics dictionary
        output_path: Path to output directory
        include_user_query: Whether to include user_query column (for fusion)
    
    Returns:
        DataFrame if pandas available, None otherwise
    """
    rows = []
    
    for result in all_results:
        if 'error' in result:
            continue
        
        row = {
            'query_folder': Path(result['query_folder']).name,
            'ground_truth': result.get('total_ground_truth', 0),
            'retrieved': result.get('total_retrieved', 0),
            'relevant': len(result.get('relevant_results', [])),
            'query_time_ms': result.get('query_time_seconds', 0) * 1000,
        }
        
        # Add user_query if requested
        if include_user_query:
            row['user_query'] = result.get('user_query', '')[:50]
        
        # Add query_text for text queries
        if 'query_text' in result:
            row['query_text'] = result.get('query_text', '')[:50]
        
        # Add recall metrics
        for metric_name, metric_data in result.get('recall_metrics', {}).items():
            row[metric_name] = metric_data.get('recall', 0.0)
        
        # Add mAP metrics
        for metric_name, metric_data in result.get('map_metrics', {}).items():
            row[metric_name] = metric_data.get('map', metric_data.get('ap', 0.0))
        
        rows.append(row)
    
    if rows:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            csv_path = output_path / "recall_summary_table.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved summary table to: {csv_path}")
            return df
        else:
            csv_path = output_path / "recall_summary_table.csv"
            headers = ['query_folder', 'ground_truth', 'retrieved', 'relevant', 'query_time_ms']
            if include_user_query:
                headers.insert(1, 'user_query')
            if any('query_text' in row for row in rows):
                headers.insert(1, 'query_text')
            if rows:
                for key in rows[0].keys():
                    if key.startswith('recall@') or key.startswith('map@'):
                        headers.append(key)
            
            with open(csv_path, 'w') as f:
                f.write(','.join(headers) + '\n')
                
                for row in rows:
                    values = [str(row.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
            
            print(f"Saved summary table to: {csv_path}")
            return None
    return None

