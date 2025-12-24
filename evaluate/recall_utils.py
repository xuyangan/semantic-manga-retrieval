#!/usr/bin/env python3
"""
Utility functions for recall evaluation scripts.

Common functions used across recall_image.py, recall_text.py, and recall_fusion.py.
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def get_device(prefer_cpu: bool = False):
    """Get the best available device: cuda, mps, or cpu."""
    import torch
    if prefer_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
    Ground truth labels are in format: manga/chapter/page.ext
    We extract manga/chapter/page from the result and compare.
    
    Args:
        result: Search result dict with path information
        ground_truth: Set of ground truth paths (normalized, without extension)
        include_page_key: Whether to check page_key field (for text/fusion results)
        include_source_file: Whether to check source_file field (for text results)
    
    Returns:
        True if result matches ground truth
    """
    # Normalize ground truth set (remove extensions)
    normalized_ground_truth = {normalize_path_for_matching(gt) for gt in ground_truth}
    
    # Try different path formats from result and extract manga/chapter/page
    paths_to_check = []
    
    # Method 1: Use page_key if available (from deduplication or late fusion)
    if include_page_key and "page_key" in result and result["page_key"]:
        extracted = extract_manga_chapter_page(result["page_key"])
        if extracted:
            paths_to_check.append(extracted)
    
    # Method 2: Use source_file (text embeddings format)
    if include_source_file and "source_file" in result and result["source_file"]:
        extracted = extract_manga_chapter_page(result["source_file"])
        if extracted:
            paths_to_check.append(extracted)
    
    # Method 3: Use image_meta or text_meta (from late fusion)
    for meta_key in ["image_meta", "text_meta"]:
        if meta_key in result and result[meta_key]:
            meta = result[meta_key]
            # Try rel_path or relative_path
            for key in ["rel_path", "relative_path", "path"]:
                if key in meta and meta[key]:
                    extracted = extract_manga_chapter_page(meta[key])
                    if extracted:
                        paths_to_check.append(extracted)
            
            # Try constructing from manga/chapter/page fields
            if all(k in meta for k in ["manga", "chapter", "page"]):
                constructed = f"{meta['manga']}/{meta['chapter']}/{meta['page']}"
                constructed = normalize_path_for_matching(constructed)
                paths_to_check.append(constructed)
    
    # Method 4: Use rel_path or relative_path (direct fields)
    for key in ["rel_path", "relative_path", "path"]:
        if key in result and result[key]:
            extracted = extract_manga_chapter_page(result[key])
            if extracted:
                paths_to_check.append(extracted)
    
    # Method 5: Construct from manga/chapter/page fields
    if all(k in result for k in ["manga", "chapter", "page"]):
        constructed = f"{result['manga']}/{result['chapter']}/{result['page']}"
        constructed = normalize_path_for_matching(constructed)
        paths_to_check.append(constructed)
    
    # Check if any extracted path matches ground truth
    for path in paths_to_check:
        if path in normalized_ground_truth:
            return True
    
    return False


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_ground_truth_labels(labels_file: Path) -> Set[str]:
    """
    Load ground truth labels from labels.txt file.
    Format: manga/chapter/page.ext (one per line)
    
    Args:
        labels_file: Path to labels.txt
    
    Returns:
        Set of ground truth paths (normalized, without extension)
    """
    if not labels_file.exists():
        return set()
    
    ground_truth = set()
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Normalize path (remove extension, normalize separators)
            path = normalize_path_for_matching(line)
            ground_truth.add(path)
    
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


def compute_average_metrics(all_results: List[Dict]) -> Dict:
    """
    Compute average recall@k metrics across all queries.
    
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
    
    # Compute averages
    averages = {}
    for metric_name, scores in recall_scores.items():
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
    Plot recall@k curve for a single query.
    
    Args:
        result: Result dictionary with recall_metrics
        output_path: Path to save the plot
        color: Color for the plot line
        title_prefix: Prefix for the title (e.g., "Late Fusion", "Text Query")
    """
    if 'error' in result or 'recall_metrics' not in result:
        return
    
    recall_metrics = result['recall_metrics']
    
    # Extract k values and recall scores, sort by k value
    metrics_list = []
    for metric_name, metric_data in recall_metrics.items():
        k = metric_data.get('k', 0)
        recall = metric_data.get('recall', 0.0)
        metrics_list.append((k, recall))
    
    metrics_list.sort(key=lambda x: x[0])
    k_values = [k for k, _ in metrics_list]
    recall_scores = [recall for _, recall in metrics_list]
    
    if len(k_values) == 0:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, recall_scores, marker='o', linewidth=2, markersize=8, color=color)
    ax.fill_between(k_values, recall_scores, alpha=0.3, color=color)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    query_name = Path(result['query_folder']).name if 'query_folder' in result else "Query"
    query_text = result.get('user_query', result.get('query_text', ''))[:50]
    gt = result.get('total_ground_truth', 0)
    rel = len(result.get('relevant_results', []))
    
    title_prefix_str = f"{title_prefix} " if title_prefix else ""
    title = f"Recall@K Curve ({title_prefix_str}Query)\n{query_name}: \"{query_text}...\"\nGround Truth: {gt} | Relevant Retrieved: {rel}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("k (Number of Results)", fontsize=11)
    ax.set_ylabel("Recall@K", fontsize=11)
    
    ax.set_xticks(k_values)
    ax.set_xlim(left=0, right=max(k_values) * 1.1)
    ax.set_ylim(bottom=0, top=1.05)
    
    # Add value labels on points
    for k, recall in zip(k_values, recall_scores):
        ax.annotate(f'{recall:.3f}', (k, recall),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_recall_curves(all_results: List[Dict], averages: Dict, output_path: Path, color: str = 'blue', title: str = "All Queries"):
    """
    Plot all recall curves together for comparison.
    
    Args:
        all_results: List of result dictionaries
        averages: Average metrics dictionary
        output_path: Path to save the plot
        color: Color for the average curve
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all curves
    all_k_values = set()
    curves_data = []
    
    for result in all_results:
        if 'error' in result or 'recall_metrics' not in result:
            continue
        
        recall_metrics = result['recall_metrics']
        k_values = []
        recall_scores = []
        
        metrics_list = []
        for metric_name, metric_data in recall_metrics.items():
            k = metric_data.get('k', 0)
            recall = metric_data.get('recall', 0.0)
            metrics_list.append((k, recall))
            all_k_values.add(k)
        
        metrics_list.sort(key=lambda x: x[0])
        k_values = [k for k, _ in metrics_list]
        recall_scores = [recall for _, recall in metrics_list]
        
        if len(k_values) > 0:
            query_name = Path(result['query_folder']).name[:30] if 'query_folder' in result else "Query"
            curves_data.append({
                'k_values': k_values,
                'recall_scores': recall_scores,
                'label': query_name
            })
    
    # Plot individual curves (lighter colors)
    for curve in curves_data:
        ax.plot(curve['k_values'], curve['recall_scores'],
               alpha=0.3, linewidth=1, color='gray')
    
    # Plot average curve (bold)
    if 'error' not in averages and 'average_metrics' in averages:
        avg_metrics = averages['average_metrics']
        avg_list = []
        
        for metric_name, stats in avg_metrics.items():
            k = int(metric_name.split('@')[1])
            recall = stats.get('mean', 0.0)
            avg_list.append((k, recall))
        
        avg_list.sort(key=lambda x: x[0])
        avg_k_values = [k for k, _ in avg_list]
        avg_recall_scores = [recall for _, recall in avg_list]
        
        if len(avg_k_values) > 0:
            ax.plot(avg_k_values, avg_recall_scores,
                   marker='o', linewidth=3, markersize=10,
                   color=color, label='Average', zorder=10)
            
            # Add value labels on average curve
            for k, recall in zip(avg_k_values, avg_recall_scores):
                ax.annotate(f'{recall:.3f}', (k, recall),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=9, fontweight='bold', color=color)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel("k (Number of Results)", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title(f"Recall@K Curves - {title}", fontsize=14, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.05)
    
    if len(curves_data) > 0:
        all_k_sorted = sorted(all_k_values)
        ax.set_xticks(all_k_sorted)
    
    if 'error' not in averages:
        ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
                    if key.startswith('recall@'):
                        headers.append(key)
            
            with open(csv_path, 'w') as f:
                f.write(','.join(headers) + '\n')
                
                for row in rows:
                    values = [str(row.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
            
            print(f"Saved summary table to: {csv_path}")
            return None
    return None

