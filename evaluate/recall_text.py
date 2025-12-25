#!/usr/bin/env python3
"""
Recall Evaluation Script for LLM-Generated Text Queries

Evaluates recall@k and mAP@k using LLM-generated descriptions as text queries.
Similar to the text part of late fusion - generates descriptions from images via LLM.

Workflow:
    1. For each query folder, check for cached LLM description (llm_description.txt)
    2. If not cached, generate description from query image using LLM and cache it
    3. Use the LLM description as text query for CLIP text embedding
    4. Search FAISS text index with the embedding

Usage:
    python -m evaluate.recall_text --index final_dataset_text_embeddings/faiss_index \
        --queries queries --output results/recall_text
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torch
import open_clip

# Import utility functions (handles sys.path setup)
from evaluate.recall_utils import (
    get_device,
    check_faiss_available,
    load_faiss_index,
    search_faiss_index,
    load_ground_truth_labels,
    match_result_to_ground_truth,
    compute_recall_at_k,
    compute_map_at_k,
    compute_average_metrics,
    plot_recall_curve,
    plot_aggregate_recall_curves,
    add_border_to_image,
    create_summary_table,
    load_text_query,
    load_llm_description,
    save_llm_description,
    find_query_image,
    get_page_key_from_result,
    FAISS_AVAILABLE,
)

# Import LLM encoder for description generation
from late_fusion.llm_encoder import encode_query_llm

# Import faiss for type hints
try:
    import faiss
except ImportError:
    faiss = None


def load_clip_model(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k", device: str = None):
    """
    Load CLIP model and tokenizer for text encoding.
    
    Args:
        model_name: Name of the CLIP model
        pretrained: Pretrained weights to use
        device: Device to load model on
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = get_device()
    
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        model.eval()
        
        # Test the model with a dummy operation
        with torch.no_grad():
            dummy_tokens = tokenizer(["test"]).to(device)
            _ = model.encode_text(dummy_tokens)
        
        print(f"Loaded CLIP model: {model_name} ({pretrained}) on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        if device != "cpu":
            print(f"Error loading CLIP model on {device}: {e}")
            print("Falling back to CPU...")
            device = "cpu"
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model = model.to(device)
            model.eval()
            print(f"Loaded CLIP model: {model_name} ({pretrained}) on {device} (fallback)")
            return model, tokenizer, device
        else:
            raise




def deduplicate_by_page(results: List[Dict]) -> List[Dict]:
    """
    Deduplicate results by page, keeping only highest similarity per page.
    This is important for text search where multiple text lines per page exist.
    
    Args:
        results: List of search results (already sorted by similarity)
    
    Returns:
        Deduplicated results with updated ranks
    """
    seen_pages = {}
    
    for r in results:
        page_key = get_page_key_from_result(r)
        if not page_key:
            continue
        
        # Keep only the first (highest similarity) result per page
        if page_key not in seen_pages:
            seen_pages[page_key] = r.copy()
            seen_pages[page_key]["page_key"] = page_key
    
    # Update ranks
    deduped = list(seen_pages.values())
    for i, r in enumerate(deduped):
        r["rank"] = i + 1
    
    return deduped


def embed_text_query(model, tokenizer, device: str, query_text: str) -> np.ndarray:
    """
    Embed a text query using CLIP model.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: Device to use
        query_text: Text query string
    
    Returns:
        Embedding vector (D,)
    """
    tokens = tokenizer([query_text]).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()[0]
    
    return embedding




def evaluate_single_query(
    query_folder: Path,
    index: Any,  # faiss.Index
    id_to_meta: Dict[int, Dict],
    model,
    tokenizer,
    device: str,
    k_values: List[int] = [5, 10, 20, 30, 40, 50],
    map_k_values: List[int] = [5, 10, 20, 30, 40, 50],
    k_candidates: int = 100
) -> Dict:
    """
    Evaluate recall for a single query using LLM-generated text description.
    
    Workflow:
        1. Check for cached LLM description (llm_description.txt)
        2. If not cached, generate description from query image using LLM
        3. Cache the generated description for future runs
        4. Use the LLM description as text query for CLIP embedding
        5. Search FAISS text index with the embedding
    
    Args:
        query_folder: Path to query folder (contains query image, labels.txt)
        index: FAISS text index
        id_to_meta: Metadata mapping
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: Device to use
        k_values: List of k values for recall@k
        map_k_values: List of k values for mAP@k
        k_candidates: Number of candidates to retrieve before deduplication
    
    Returns:
        Dictionary with evaluation results
    """
    # Find query image (needed for LLM description generation)
    query_image = find_query_image(query_folder)
    
    # Load user-provided text query (used as hint for LLM)
    user_query = load_text_query(query_folder)
    
    # Try to load cached LLM description, or generate if not exists
    query_text = load_llm_description(query_folder)
    
    if query_text is None:
        # No cached LLM description - generate one
        if query_image is None:
            return {
                "query_folder": str(query_folder),
                "error": "No query image found for LLM description generation"
            }
        
        try:
            # Generate LLM description from the image
            query_text = encode_query_llm(query_image, user_query if user_query else "", verbose=False)
            # Cache the generated description
            save_llm_description(query_folder, query_text)
        except Exception as e:
            return {
                "query_folder": str(query_folder),
                "error": f"Error generating LLM description: {e}"
            }
    
    # Load ground truth
    labels_file = query_folder / "labels.txt"
    ground_truth = load_ground_truth_labels(labels_file)
    
    if len(ground_truth) == 0:
        return {
            "query_folder": str(query_folder),
            "query_text": query_text,
            "query_image": str(query_image) if query_image else None,
            "error": "No ground truth labels found",
            "total_ground_truth": 0
        }
    
    # Embed text query
    try:
        query_embedding = embed_text_query(model, tokenizer, device, query_text)
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_text": query_text,
            "query_image": str(query_image) if query_image else None,
            "error": f"Error embedding text query: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    # Search
    try:
        raw_results = search_faiss_index(index, query_embedding, id_to_meta, k=k_candidates)
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_text": query_text,
            "query_image": str(query_image) if query_image else None,
            "error": f"Error searching index: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    # Deduplicate by page (important for text search)
    search_results = deduplicate_by_page(raw_results)
    
    if len(search_results) == 0:
        return {
            "query_folder": str(query_folder),
            "query_text": query_text,
            "query_image": str(query_image) if query_image else None,
            "error": "No search results after deduplication",
            "total_ground_truth": len(ground_truth),
            "raw_results_count": len(raw_results)
        }
    
    # Compute recall@k for each k value
    recall_metrics = {}
    for k in k_values:
        metric_name = f"recall@{k}"
        recall_metrics[metric_name] = compute_recall_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=True)
    
    # Compute mAP@k for each k value
    map_metrics = {}
    for k in map_k_values:
        metric_name = f"map@{k}"
        map_metrics[metric_name] = compute_map_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=True)
    
    # Get all relevant retrieved results
    all_relevant = []
    for result in search_results:
        if match_result_to_ground_truth(result, ground_truth, include_page_key=True, include_source_file=True):
            all_relevant.append(result)
    
    return {
        "query_folder": str(query_folder),
        "query_text": query_text,
        "query_image": str(query_image) if query_image else None,
        "total_ground_truth": len(ground_truth),
        "raw_results_count": len(raw_results),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "map_metrics": map_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant
    }




def find_image_path(meta: dict, image_dir: Path) -> Path:
    """Find corresponding image file for a text result."""
    # Get page key
    page_key = get_page_key_from_result(meta)
    if not page_key:
        return None
    
    # Try common image extensions
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        img_path = image_dir / (page_key + ext)
        if img_path.exists():
            return img_path
    
    return None


def visualize_query_results(
    result: Dict,
    output_path: Path,
    image_dir: Path = None,
    num_images: int = 10,
    highlight_relevant: bool = True
):
    """
    Visualize search results for a text query.
    Shows query image (if available), query text, and retrieved images.
    """
    all_retrieved = result.get('all_retrieved', [])[:num_images]
    n_results = len(all_retrieved)
    
    if n_results == 0:
        print(f"Warning: No results to visualize")
        return
    
    # Determine if we have a query image
    query_image_path = result.get('query_image')
    has_query_image = query_image_path and Path(query_image_path).exists()
    
    # Create figure
    n_cols = n_results + (1 if has_query_image else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    col_offset = 0
    
    # Plot query image if available
    if has_query_image:
        try:
            query_img = Image.open(query_image_path).convert("RGB")
            query_img_bordered = add_border_to_image(query_img, border_width=8, border_color=(0, 128, 0))
            axes[0].imshow(query_img_bordered)
            axes[0].set_title("QUERY IMAGE", fontsize=12, fontweight='bold', color='green')
            axes[0].axis('off')
            col_offset = 1
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading query\n{e}", ha='center', va='center')
            axes[0].axis('off')
            col_offset = 1
    
    # Determine relevant paths for highlighting
    relevant_paths = set()
    for r in result.get('relevant_results', []):
        page_key = r.get('page_key', get_page_key_from_result(r))
        if page_key:
            relevant_paths.add(page_key)
    
    # Plot results
    for i, r in enumerate(all_retrieved):
        ax = axes[i + col_offset]
        
        # Find and display image
        img_loaded = False
        page_key = r.get('page_key', get_page_key_from_result(r))
        
        if image_dir:
            img_path = find_image_path(r, image_dir)
            if img_path and img_path.exists():
                try:
                    result_img = Image.open(img_path).convert("RGB")
                    
                    # Determine border color
                    is_relevant = page_key in relevant_paths
                    sim_pct = r.get('similarity', 0.0) * 100
                    
                    if is_relevant:
                        border_color = (0, 200, 0)  # Green for relevant
                    elif sim_pct >= 80:
                        border_color = (255, 165, 0)  # Orange for high similarity
                    else:
                        border_color = (100, 100, 100)  # Gray for others
                    
                    result_img_bordered = add_border_to_image(result_img, border_width=6, border_color=border_color)
                    ax.imshow(result_img_bordered)
                    img_loaded = True
                except Exception:
                    pass
        
        if not img_loaded:
            ax.text(0.5, 0.5, f"Image not found\n{page_key}", ha='center', va='center', fontsize=8)
            ax.set_facecolor('#f0f0f0')
        
        # Title with info
        sim_pct = r.get('similarity', 0.0) * 100
        parts = page_key.split('/') if page_key else []
        manga = parts[0][:20] + '...' if parts and len(parts[0]) > 20 else (parts[0] if parts else '?')
        rest = '/'.join(parts[1:]) if len(parts) > 1 else ''
        
        title = f"#{r.get('rank', i+1)} ({sim_pct:.1f}%)\n{manga}\n{rest}"
        
        is_relevant = page_key in relevant_paths
        if is_relevant:
            color = 'green'
            title += "\n✓ RELEVANT"
        elif sim_pct >= 80:
            color = 'orange'
        else:
            color = 'gray'
        
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    # Main title with LLM-generated description
    query_name = Path(result['query_folder']).name
    query_text = result.get('query_text', '')[:60]
    plt.suptitle(f"LLM Description: \"{query_text}...\"\n({query_name})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k for text-based queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate/recall_text.py --index index/text --queries a_queries/text --output results/recall_text
  python evaluate/recall_text.py --index index/text --queries a_queries/text --output results/recall_text --image-dir final_dataset
  python evaluate/recall_text.py --index index/text --queries a_queries/text --output results/recall_text --k-values 1 5 10 20
        """
    )
    parser.add_argument(
        "--index", "-i",
        type=str,
        required=True,
        help="Path to text index directory containing faiss.index and metadata.json",
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        required=True,
        help="Path to queries directory containing query folders with text.txt, query.png, labels.txt",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to image directory for visualization (e.g., final_dataset)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[5, 10, 20, 30, 40, 50],
        help="K values for recall@k evaluation (default: 5 10 20 30 40 50)",
    )
    parser.add_argument(
        "--map-k-values",
        type=int,
        nargs='+',
        default=[5, 10, 20, 30, 40, 50],
        help="K values for mAP@k evaluation (default: 5 10 20 30 40 50)",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=100,
        help="Number of candidates to retrieve before deduplication (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu, auto-detect if not specified)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip creating visualizations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="CLIP model to use (default: ViT-L-14)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="Pretrained weights to use (default: laion2b_s32b_b82k)",
    )
    
    args = parser.parse_args()
    
    if not FAISS_AVAILABLE:
        print("Error: faiss is not installed.")
        print("Install with: pip install faiss-cpu or faiss-gpu")
        return
    
    # Setup paths
    index_dir = Path(args.index)
    queries_dir = Path(args.queries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # Load FAISS index and metadata
    print(f"Loading FAISS index from: {index_dir}")
    try:
        index, id_to_meta, dimension = load_faiss_index(index_dir)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
    
    # Load CLIP model with tokenizer
    device = args.device if args.device else get_device()
    print(f"\nLoading CLIP model on {device}...")
    model, tokenizer, device = load_clip_model(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device
    )
    
    # Find query folders
    query_folders = sorted([d for d in queries_dir.iterdir() if d.is_dir()])
    if len(query_folders) == 0:
        print(f"Error: No query folders found in {queries_dir}")
        return
    
    print(f"\nFound {len(query_folders)} query folders")
    print(f"Using LLM-generated descriptions as text queries")
    print(f"Evaluating recall@k with k values: {args.k_values}")
    print(f"Evaluating mAP@k with k values: {args.map_k_values}")
    print(f"Retrieving {args.k_candidates} candidates before deduplication")
    print("="*60)
    
    # Evaluate each query with timing
    all_results = []
    query_times = []
    for query_folder in tqdm(query_folders, desc="Processing queries"):
        start_time = time.time()
        result = evaluate_single_query(
            query_folder=query_folder,
            index=index,
            id_to_meta=id_to_meta,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k_values=args.k_values,
            map_k_values=args.map_k_values,
            k_candidates=args.k_candidates
        )
        query_time = time.time() - start_time
        result['query_time_seconds'] = query_time
        query_times.append(query_time)
        all_results.append(result)
    
    # Compute averages
    print("\nComputing average metrics...")
    averages = compute_average_metrics(all_results)
    
    # Compute timing statistics
    timing_stats = {
        "total_time_seconds": float(sum(query_times)),
        "average_time_seconds": float(np.mean(query_times)),
        "std_time_seconds": float(np.std(query_times)),
        "min_time_seconds": float(np.min(query_times)),
        "max_time_seconds": float(np.max(query_times)),
        "median_time_seconds": float(np.median(query_times)),
    }
    
    # Save results
    output_data = {
        "config": {
            "index_dir": str(index_dir),
            "queries_dir": str(queries_dir),
            "image_dir": str(image_dir) if image_dir else None,
            "k_values": args.k_values,
            "map_k_values": args.map_k_values,
            "k_candidates": args.k_candidates,
            "num_queries": len(query_folders),
            "model": args.model,
            "pretrained": args.pretrained
        },
        "individual_results": all_results,
        "average_metrics": averages,
        "timing_stats": timing_stats,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = output_dir / "recall_results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(all_results, averages, output_dir, include_user_query=False)
    
    # Create visualizations
    if not args.no_viz:
        print("\nCreating visualizations...")
        
        # Create recall curves
        curves_dir = output_dir / "recall_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(all_results):
            if 'error' in result:
                continue
            
            query_name = Path(result['query_folder']).name if 'query_folder' in result else f"query_{i+1}"
            curve_file = curves_dir / f"query_{i+1:03d}_{query_name}_recall_curve.png"
            plot_recall_curve(result, curve_file, color='green', title_prefix="LLM Text Query")
        
        # Aggregate curve
        aggregate_file = curves_dir / "aggregate_all_queries.png"
        plot_aggregate_recall_curves(all_results, averages, aggregate_file, color='green', title="LLM Text Queries")
        print(f"Recall curves saved to: {curves_dir}")
        
        # Create image visualizations (if image_dir provided)
        if image_dir:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            for i, result in enumerate(all_results):
                if 'error' in result:
                    continue
                
                query_name = Path(result['query_folder']).name
                safe_name = "".join(c for c in query_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
                
                output_file = vis_dir / f"query_{i+1:03d}_{safe_name}.png"
                visualize_query_results(
                    result=result,
                    output_path=output_file,
                    image_dir=image_dir,
                    num_images=10,
                    highlight_relevant=True
                )
            
            print(f"Visualizations saved to: {vis_dir}")
        else:
            print("Note: Skipping image visualizations (--image-dir not provided)")
    
    # Print summary
    print("\n" + "="*60)
    print("RECALL EVALUATION SUMMARY (Text Queries)")
    print("="*60)
    if 'error' not in averages:
        print(f"Number of queries: {averages['num_queries']}")
        if averages['num_failed'] > 0:
            print(f"Failed queries: {averages['num_failed']}")
        
        print("\nAverage Recall@K Metrics:")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 70)
        
        # Print recall metrics
        for metric_name, stats in sorted(averages['average_metrics'].items()):
            if metric_name.startswith('recall@'):
                print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                      f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
        
        # Print mAP metrics
        print("\nAverage mAP@K Metrics:")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 70)
        
        for metric_name, stats in sorted(averages['average_metrics'].items()):
            if metric_name.startswith('map@'):
                print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                      f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
        
        print("\nOverall Statistics:")
        overall = averages['overall_stats']
        print(f"  Total ground truth images: {overall['total_ground_truth']}")
        print(f"  Total retrieved pages: {overall['total_retrieved']}")
        print(f"  Total relevant retrieved: {overall['total_relevant_retrieved']}")
        print(f"  Overall recall: {overall['overall_recall']:.4f}")
    
    print("\nQuery Timing Statistics:")
    print(f"  Total time: {timing_stats['total_time_seconds']:.3f} seconds")
    print(f"  Average query time: {timing_stats['average_time_seconds']*1000:.2f} ms")
    print(f"  Std dev: {timing_stats['std_time_seconds']*1000:.2f} ms")
    print(f"  Min: {timing_stats['min_time_seconds']*1000:.2f} ms")
    print(f"  Max: {timing_stats['max_time_seconds']*1000:.2f} ms")
    print(f"  Median: {timing_stats['median_time_seconds']*1000:.2f} ms")
    
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
