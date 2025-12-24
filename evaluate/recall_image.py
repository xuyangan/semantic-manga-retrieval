#!/usr/bin/env python3
"""
Recall Evaluation Script for Image-Based Queries

Evaluates recall@k for query images using FAISS index.
Processes query folders with query images and labels.txt files.

Sample Execution:
    python evaluate/recall_image.py --index index/image --queries query/image --output results/recall_image
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torch
import open_clip

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Import utility functions
from evaluate.recall_utils import (
    get_device,
    load_faiss_index,
    search_faiss_index,
    load_ground_truth_labels,
    normalize_path_for_matching,
    extract_manga_chapter_page,
    match_result_to_ground_truth,
    compute_recall_at_k,
    compute_average_metrics,
    plot_recall_curve,
    plot_aggregate_recall_curves,
    pad_image_to_square,
    add_border_to_image,
    create_summary_table,
)


def load_clip_model(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k", device: str = None):
    """
    Load CLIP model and preprocessing function.
    
    Args:
        model_name: Name of the CLIP model
        pretrained: Pretrained weights to use
        device: Device to load model on
    
    Returns:
        tuple: (model, preprocess, device)
    """
    if device is None:
        device = get_device()
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()
    
    print(f"Loaded CLIP model: {model_name} ({pretrained}) on {device}")
    return model, preprocess, device




def embed_query_image(model, preprocess, device: str, query_image_path: Path, use_padding: bool = False) -> np.ndarray:
    """
    Embed a query image using CLIP model.
    
    Args:
        model: CLIP model
        preprocess: Preprocessing function
        device: Device to use
        query_image_path: Path to query image
        use_padding: Whether to pad image to square (must match index building process)
    
    Returns:
        Embedding vector (D,)
    """
    img = Image.open(query_image_path).convert("RGB")
    
    # Pad image to square shape before preprocessing (only if index was built this way)
    if use_padding:
        img = pad_image_to_square(img, fill_color=(255, 255, 255))
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()[0]
    
    return embedding




def evaluate_single_query(
    query_folder: Path,
    index: faiss.Index,
    id_to_meta: Dict[int, Dict],
    model,
    preprocess,
    device: str,
    k_values: List[int] = [1, 5, 10, 20],
    use_padding: bool = False
) -> Dict:
    """
    Evaluate recall for a single query.
    
    Args:
        query_folder: Path to query folder (contains query.png and labels.txt)
        index: FAISS index
        id_to_meta: Metadata mapping
        model: CLIP model
        preprocess: Preprocessing function
        device: Device to use
        k_values: List of k values for recall@k
    
    Returns:
        Dictionary with evaluation results
    """
    # Find query image
    query_image = None
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        candidate = query_folder / f"query{ext}"
        if candidate.exists():
            query_image = candidate
            break
    
    if query_image is None:
        return {
            "query_folder": str(query_folder),
            "error": "Query image not found (expected query.png, query.jpg, etc.)"
        }
    
    # Load ground truth
    labels_file = query_folder / "labels.txt"
    ground_truth = load_ground_truth_labels(labels_file)
    
    if len(ground_truth) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "error": "No ground truth labels found",
            "total_ground_truth": 0
        }
    
    # Embed query image
    try:
        query_embedding = embed_query_image(model, preprocess, device, query_image, use_padding=use_padding)
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "error": f"Error embedding query image: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    # Search
    max_k = max(k_values)
    try:
        search_results = search_faiss_index(index, query_embedding, id_to_meta, k=max_k)
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "error": f"Error searching index: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    if len(search_results) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "error": "No search results",
            "total_ground_truth": len(ground_truth)
        }
    
    # Compute recall@k for each k value
    recall_metrics = {}
    for k in k_values:
        metric_name = f"recall@{k}"
        recall_metrics[metric_name] = compute_recall_at_k(search_results, ground_truth, k, include_page_key=False, include_source_file=False)
    
    # Get all relevant retrieved results
    all_relevant = []
    for result in search_results:
        if match_result_to_ground_truth(result, ground_truth, include_page_key=False, include_source_file=False):
            all_relevant.append(result)
    
    return {
        "query_folder": str(query_folder),
        "query_image": str(query_image),
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant
    }




def visualize_query_results(
    query_path: str,
    all_retrieved: List[Dict],
    output_path: Path,
    num_images: int = 10,
    highlight_relevant: bool = True,
    use_padding: bool = False
):
    """
    Visualize search results for a single query.
    Shows what CLIP actually sees - padded square images with borders.
    """
    query_path = Path(query_path)
    if not query_path.exists():
        print(f"Warning: Query image not found: {query_path}")
        return
    
    results = all_retrieved[:num_images]
    n_results = len(results)
    
    if n_results == 0:
        print(f"Warning: No results to visualize for {query_path.name}")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 5))
    
    if n_results == 0:
        axes = [axes]
    
    # Plot query image
    try:
        query_img = Image.open(query_path).convert("RGB")
        # Pad to square if needed (to show what CLIP actually sees)
        if use_padding:
            query_img_display = pad_image_to_square(query_img, fill_color=(255, 255, 255))
            title_suffix = "\n(padded to square)"
        else:
            query_img_display = query_img
            title_suffix = ""
        # Add blue border for query
        query_img_bordered = add_border_to_image(query_img_display, border_width=8, border_color=(0, 0, 255))
        axes[0].imshow(query_img_bordered)
        axes[0].set_title(f"QUERY{title_suffix}", fontsize=12, fontweight='bold', color='blue')
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error loading query\n{e}", ha='center', va='center')
        axes[0].axis('off')
    
    # Plot results
    for i, r in enumerate(results):
        ax = axes[i + 1]
        
        try:
            result_path = Path(r.get('path', ''))
            if result_path.exists():
                result_img = Image.open(result_path).convert("RGB")
                # Pad to square if needed (to show what CLIP sees)
                if use_padding:
                    result_img_display = pad_image_to_square(result_img, fill_color=(255, 255, 255))
                else:
                    result_img_display = result_img
                
                # Determine border color based on relevance
                is_relevant = r.get('is_relevant', False)
                sim_pct = r.get('similarity', 0.0) * 100
                
                if is_relevant:
                    border_color = (0, 200, 0)  # Green for relevant
                elif sim_pct >= 80:
                    border_color = (255, 165, 0)  # Orange for high similarity
                else:
                    border_color = (100, 100, 100)  # Gray for others
                
                # Add border
                result_img_bordered = add_border_to_image(result_img_display, border_width=6, border_color=border_color)
                ax.imshow(result_img_bordered)
            else:
                ax.text(0.5, 0.5, f"Image not found", ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{e}", ha='center', va='center')
        
        sim_pct = r.get('similarity', 0.0) * 100
        manga = r.get('manga', '')[:20] + '...' if len(r.get('manga', '')) > 20 else r.get('manga', '')
        chapter = r.get('chapter', '')
        page = r.get('page', '')
        
        chapter_num = chapter.replace('chapter_', '') if 'chapter_' in chapter else chapter
        page_num = page.replace('page_', '') if 'page_' in page else page
        
        title = f"#{r.get('rank', i+1)} ({sim_pct:.1f}%)\n{manga}\nCh.{chapter_num} Pg.{page_num}"
        
        is_relevant = r.get('is_relevant', False)
        if is_relevant:
            color = 'green'
            title += "\n✓ RELEVANT"
        elif sim_pct >= 80:
            color = 'orange'
        else:
            color = 'gray'
        
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    query_name = query_path.parent.name
    title_text = f"Query: {query_name}"
    if use_padding:
        title_text += " (CLIP view - padded to square)"
    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k for image-based queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics/recall_image.py --index index/image --queries query/image --output results/recall_image
  python metrics/recall_image.py --index index/image --queries query/image --output results/recall_image --k-values 1 5 10 20
        """
    )
    parser.add_argument(
        "--index", "-i",
        type=str,
        required=True,
        help="Path to index directory containing faiss.index and metadata.json (e.g., index/image)",
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        required=True,
        help="Path to queries directory containing query folders (e.g., query/image)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., results/recall_image)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[1, 5, 10, 20],
        help="K values for recall@k evaluation (default: 1 5 10 20)",
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
        help="CLIP model to use (default: ViT-L-14, 768 dimensions)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="Pretrained weights to use (default: laion2b_s32b_b82k)",
    )
    parser.add_argument(
        "--use-padding",
        action="store_true",
        help="Pad images to square before encoding (only if index was built with padding)",
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
    
    # Load FAISS index and metadata
    print(f"Loading FAISS index from: {index_dir}")
    try:
        index, id_to_meta, dimension = load_faiss_index(index_dir)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
    
    # Load CLIP model
    device = args.device if args.device else get_device()
    print(f"\nLoading CLIP model on {device}...")
    model, preprocess, device = load_clip_model(
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
    print(f"Evaluating recall@k with k values: {args.k_values}")
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
            preprocess=preprocess,
            device=device,
            k_values=args.k_values,
            use_padding=args.use_padding
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
            "k_values": args.k_values,
            "num_queries": len(query_folders),
            "model": args.model,
            "pretrained": args.pretrained,
            "use_padding": args.use_padding
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
            plot_recall_curve(result, curve_file, color='blue')
        
        # Aggregate curve
        aggregate_file = curves_dir / "aggregate_all_queries.png"
        plot_aggregate_recall_curves(all_results, averages, aggregate_file, color='red', title="All Image Queries")
        print(f"Recall curves saved to: {curves_dir}")
        
        # Create image visualizations
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(all_results):
            if 'error' in result or 'query_image' not in result:
                continue
            
            query_name = Path(result['query_folder']).name
            safe_name = "".join(c for c in query_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            
            # Mark relevant results
            all_retrieved = result.get('all_retrieved', [])
            relevant_paths = {r.get('path', '') for r in result.get('relevant_results', [])}
            for r in all_retrieved:
                r['is_relevant'] = r.get('path', '') in relevant_paths
            
            output_file = vis_dir / f"query_{i+1:03d}_{safe_name}.png"
            visualize_query_results(
                query_path=result['query_image'],
                all_retrieved=all_retrieved,
                output_path=output_file,
                num_images=10,
                highlight_relevant=True,
                use_padding=args.use_padding
            )
        
        print(f"Visualizations saved to: {vis_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("RECALL EVALUATION SUMMARY")
    print("="*60)
    if 'error' not in averages:
        print(f"Number of queries: {averages['num_queries']}")
        if averages['num_failed'] > 0:
            print(f"Failed queries: {averages['num_failed']}")
        
        print("\nAverage Recall@K Metrics:")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 70)
        
        for metric_name, stats in sorted(averages['average_metrics'].items()):
            print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                  f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
        
        print("\nOverall Statistics:")
        overall = averages['overall_stats']
        print(f"  Total ground truth images: {overall['total_ground_truth']}")
        print(f"  Total retrieved images: {overall['total_retrieved']}")
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

