#!/usr/bin/env python3
"""
Recall Evaluation Script for Late Fusion Queries

Evaluates recall@k for late fusion queries (image-first or text-first).
Supports grid search over multiple alpha and m values for hyperparameter exploration.

Usage:
    # Single alpha and m
    python -m evaluate.recall_fusion --queries queries --mode image --alpha 0.5 --m 50 \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/fusion_single

    # Grid search over multiple alpha and m values
    python -m evaluate.recall_fusion --queries queries --mode image \\
        --alphas 0.1 0.2 0.3 0.4 0.5 --m-values 50 60 70 \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/fusion_grid_search
        
    # Use defaults (alpha=0.1-0.5, m=50,60,70)
    python -m evaluate.recall_fusion --queries queries --mode image \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/fusion_grid_search_image_first
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
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
    load_faiss_index_direct,
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
    find_image_path_from_result,
    FAISS_AVAILABLE,
)

# Import late fusion modules
from late_fusion.faiss_search import get_page_key
from late_fusion.llm_encoder import encode_query_llm


def late_fusion_one_with_cached_description(
    image_path: Path,
    user_query: str,
    image_index_dir: Path,
    text_index_dir: Path,
    m: int,
    k: int,
    alpha: float,
    cached_description: Optional[str] = None,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    image_index=None,
    image_id_to_meta=None,
    text_index=None,
    text_id_to_meta=None,
    text_lookup=None,
    query_image_embedding=None,
    query_text_embedding=None,
    verbose: bool = False
) -> dict:
    """
    Image-first late fusion search with cached LLM description.
    Supports pre-computed embeddings for efficiency in grid search.
    """
    from late_fusion.clip_encoder import load_clip_model, encode_image, encode_text
    from late_fusion.faiss_search import (
        search_index,
        build_text_lookup,
        compute_text_similarities,
    )
    from late_fusion.fusion import late_fusion_rerank
    
    # Step 1: Generate or use cached LLM description
    if cached_description is not None:
        description = cached_description
        if verbose:
            print(f"Using cached LLM description: {description[:80]}...")
    else:
        if verbose:
            print("\n[1/6] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Use pre-computed embeddings or generate them
    if query_image_embedding is not None and query_text_embedding is not None:
        image_embedding = query_image_embedding
        text_embedding = query_text_embedding
        if verbose:
            print("\n[2/6] Using pre-computed embeddings...")
            print(f"  Image embedding: {image_embedding.shape}")
            print(f"  Text embedding: {text_embedding.shape}")
    else:
        # Load CLIP model (only if not provided)
        if model is None or preprocess is None or tokenizer is None:
            if verbose:
                print("\n[2/6] Loading CLIP model and generating embeddings...")
            model, preprocess, tokenizer, device = load_clip_model(device)
        elif verbose:
            print("\n[2/6] Using provided CLIP model for generating embeddings...")
        
        image_embedding = encode_image(model, preprocess, device, image_path)
        text_embedding = encode_text(model, tokenizer, device, description)
        if verbose:
            print(f"  Image embedding: {image_embedding.shape}")
            print(f"  Text embedding: {text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (only if not provided - allows reuse across queries)
    if image_index is None or image_id_to_meta is None:
        if verbose:
            print("\n[3/6] Loading FAISS indexes...")
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    elif verbose:
        print("\n[3/6] Using provided FAISS indexes...")
    
    if text_index is None or text_id_to_meta is None:
        if image_index is None:  # Only print if we haven't already
            if verbose:
                print("\n[3/6] Loading FAISS indexes...")
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    
    # Build text lookup (only if not provided)
    if text_lookup is None:
        from late_fusion.faiss_search import build_text_lookup
        text_lookup = build_text_lookup(text_id_to_meta)
        if verbose:
            print(f"  Text lookup: {len(text_lookup)} unique pages")
    
    # Step 4: Search image DB
    if verbose:
        print(f"\n[4/6] Searching image DB (m={m})...")
    image_candidates = search_index(image_index, image_id_to_meta, image_embedding, k=m)
    if verbose:
        print(f"  Found {len(image_candidates)} image candidates")
    
    # Step 5: Get text similarity scores
    if verbose:
        print("\n[5/6] Computing text similarity scores...")
    text_scores = compute_text_similarities(
        text_index, text_lookup, text_embedding, image_candidates
    )
    
    pages_with_text = sum(1 for v in text_scores.values() if v["num_texts"] > 0)
    if verbose:
        print(f"  {pages_with_text}/{len(text_scores)} candidates have text embeddings")
    
    # Step 6: Rerank
    if verbose:
        print(f"\n[6/6] Reranking with alpha={alpha}...")
    final_results = late_fusion_rerank(image_candidates, text_scores, alpha, k)
    if verbose:
        print(f"  Final results: {len(final_results)}")
    
    return {
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "alpha": alpha,
        "m": m,
        "k": k,
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
        "num_image_candidates": len(image_candidates),
        "num_pages_with_text": pages_with_text,
        "final_results": final_results,
    }


def late_fusion_two_with_cached_description(
    image_path: Path,
    user_query: str,
    image_index_dir: Path,
    text_index_dir: Path,
    m: int,
    k: int,
    alpha: float,
    cached_description: Optional[str] = None,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    image_index=None,
    image_id_to_meta=None,
    text_index=None,
    text_id_to_meta=None,
    image_lookup=None,
    query_image_embedding=None,
    query_text_embedding=None,
    verbose: bool = False
) -> dict:
    """
    Text-first late fusion search with cached LLM description.
    Supports pre-computed embeddings for efficiency in grid search.
    """
    from late_fusion.clip_encoder import load_clip_model, encode_image, encode_text
    from late_fusion.faiss_search import (
        search_index,
        build_image_lookup,
        compute_image_similarities,
        deduplicate_text_candidates,
    )
    from late_fusion.fusion import late_fusion_rerank_two
    
    # Step 1: Generate or use cached LLM description
    if cached_description is not None:
        description = cached_description
        if verbose:
            print(f"Using cached LLM description: {description[:80]}...")
    else:
        if verbose:
            print("\n[1/7] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Use pre-computed embeddings or generate them
    if query_image_embedding is not None and query_text_embedding is not None:
        image_embedding = query_image_embedding
        text_embedding = query_text_embedding
        if verbose:
            print("\n[2/7] Using pre-computed embeddings...")
            print(f"  Image embedding: {image_embedding.shape}")
            print(f"  Text embedding: {text_embedding.shape}")
    else:
        # Load CLIP model (only if not provided)
        if model is None or preprocess is None or tokenizer is None:
            if verbose:
                print("\n[2/7] Loading CLIP model and generating embeddings...")
            model, preprocess, tokenizer, device = load_clip_model(device)
        elif verbose:
            print("\n[2/7] Using provided CLIP model for generating embeddings...")
        
        image_embedding = encode_image(model, preprocess, device, image_path)
        text_embedding = encode_text(model, tokenizer, device, description)
        if verbose:
            print(f"  Image embedding: {image_embedding.shape}")
            print(f"  Text embedding: {text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (only if not provided - allows reuse across queries)
    if text_index is None or text_id_to_meta is None:
        if verbose:
            print("\n[3/7] Loading FAISS indexes...")
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    elif verbose:
        print("\n[3/7] Using provided FAISS indexes...")
    
    if image_index is None or image_id_to_meta is None:
        if text_index is None:  # Only print if we haven't already
            if verbose:
                print("\n[3/7] Loading FAISS indexes...")
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    
    # Build image lookup (only if not provided)
    if image_lookup is None:
        from late_fusion.faiss_search import build_image_lookup
        image_lookup = build_image_lookup(image_id_to_meta)
        if verbose:
            print(f"  Image lookup: {len(image_lookup)} unique pages")
    
    # Step 4: Search text DB
    if verbose:
        print(f"\n[4/7] Searching text DB (m={m})...")
    text_candidates_raw = search_index(text_index, text_id_to_meta, text_embedding, k=m)
    if verbose:
        print(f"  Found {len(text_candidates_raw)} text candidates")
    
    # Step 5: Deduplicate by page
    if verbose:
        print("\n[5/7] Deduplicating text candidates by page...")
    text_candidates = deduplicate_text_candidates(text_candidates_raw)
    if verbose:
        print(f"  Unique pages: {len(text_candidates)}")
    
    # Step 6: Get image similarity scores
    if verbose:
        print("\n[6/7] Computing image similarity scores...")
    image_scores = compute_image_similarities(
        image_index, image_lookup, image_embedding, text_candidates
    )
    
    pages_with_image = sum(1 for v in image_scores.values() if v["similarity"] > 0)
    if verbose:
        print(f"  {pages_with_image}/{len(image_scores)} candidates have image embeddings")
    
    # Step 7: Rerank
    if verbose:
        print(f"\n[7/7] Reranking with alpha={alpha}...")
    final_results = late_fusion_rerank_two(text_candidates, image_scores, alpha, k)
    if verbose:
        print(f"  Final results: {len(final_results)}")
    
    return {
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "alpha": alpha,
        "m": m,
        "k": k,
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
        "num_text_candidates_raw": len(text_candidates_raw),
        "num_text_candidates_dedup": len(text_candidates),
        "num_pages_with_image": pages_with_image,
        "final_results": final_results,
    }


def evaluate_single_query(
    query_folder: Path,
    image_index_dir: Path,
    text_index_dir: Path,
    mode: str,
    alpha: float,
    m: int,
    k_values: List[int],
    map_k_values: List[int] = [5, 10, 20, 30, 40, 50],
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    image_index=None,
    image_id_to_meta=None,
    text_index=None,
    text_id_to_meta=None,
    text_lookup=None,
    image_lookup=None,
    query_image_embedding=None,
    query_text_embedding=None,
    cached_description: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate recall for a single query using late fusion.
    
    Args:
        query_folder: Path to query folder (contains query.png, text.txt, labels.txt)
        image_index_dir: Path to image index directory (contains faiss.index and metadata.json)
        text_index_dir: Path to text index directory (contains faiss.index and metadata.json)
        mode: "image" for image-first, "text" for text-first
        alpha: Weight for image score in late fusion
        m: Number of initial candidates to retrieve
        k_values: List of k values for recall@k
        query_image_embedding: Pre-computed image embedding (for grid search efficiency)
        query_text_embedding: Pre-computed text embedding (for grid search efficiency)
        cached_description: Pre-loaded LLM description
        verbose: Verbose output
    
    Returns:
        Dictionary with evaluation results
    """
    # Find query image using utility function
    query_image = find_query_image(query_folder)
    
    if query_image is None:
        return {
            "query_folder": str(query_folder),
            "error": "Query image not found (expected query.png, query.jpg, etc.)"
        }
    
    # Load text query
    user_query = load_text_query(query_folder)
    if not user_query:
        # Use empty string if no text query found
        user_query = ""
    
    # Use provided cached_description or load from file
    if cached_description is None:
        cached_description = load_llm_description(query_folder)
    
    # Load ground truth
    labels_file = query_folder / "labels.txt"
    ground_truth = load_ground_truth_labels(labels_file)
    
    if len(ground_truth) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": "No ground truth labels found",
            "total_ground_truth": 0
        }
    
    # Run late fusion
    try:
        max_k = max(k_values)
        
        if mode == "image":
            # Image-first late fusion
            result = late_fusion_one_with_cached_description(
                image_path=query_image,
                user_query=user_query,
                image_index_dir=image_index_dir,
                text_index_dir=text_index_dir,
                m=m,
                k=max_k,
                alpha=alpha,
                cached_description=cached_description,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                image_index=image_index,
                image_id_to_meta=image_id_to_meta,
                text_index=text_index,
                text_id_to_meta=text_id_to_meta,
                text_lookup=text_lookup,
                query_image_embedding=query_image_embedding,
                query_text_embedding=query_text_embedding,
                verbose=verbose
            )
        else:  # mode == "text"
            # Text-first late fusion
            result = late_fusion_two_with_cached_description(
                image_path=query_image,
                user_query=user_query,
                image_index_dir=image_index_dir,
                text_index_dir=text_index_dir,
                m=m,
                k=max_k,
                alpha=alpha,
                cached_description=cached_description,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                image_index=image_index,
                image_id_to_meta=image_id_to_meta,
                text_index=text_index,
                text_id_to_meta=text_id_to_meta,
                image_lookup=image_lookup,
                query_image_embedding=query_image_embedding,
                query_text_embedding=query_text_embedding,
                verbose=verbose
            )
        
        search_results = result['final_results']
        
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": f"Error running late fusion: {e}",
            "total_ground_truth": len(ground_truth)
        }
    
    if len(search_results) == 0:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": "No search results",
            "total_ground_truth": len(ground_truth)
        }
    
    # Compute recall@k for each k value
    recall_metrics = {}
    for k in k_values:
        metric_name = f"recall@{k}"
        recall_metrics[metric_name] = compute_recall_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=False)
    
    # Compute mAP@k for each k value
    map_metrics = {}
    for k in map_k_values:
        metric_name = f"map@{k}"
        map_metrics[metric_name] = compute_map_at_k(search_results, ground_truth, k, include_page_key=True, include_source_file=False)
    
    # Get all relevant retrieved results
    all_relevant = []
    for result in search_results:
        if match_result_to_ground_truth(result, ground_truth, include_page_key=True, include_source_file=False):
            all_relevant.append(result)
    
    return {
        "query_folder": str(query_folder),
        "query_image": str(query_image),
        "user_query": user_query,
        "llm_description": result.get('llm_description', ''),
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "map_metrics": map_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant
    }


def visualize_query_results(
    query_path: str,
    all_retrieved: List[Dict],
    output_path: Path,
    num_images: int = 10,
    highlight_relevant: bool = True,
    image_dir: Path = None,
    title_prefix: str = "Late Fusion"
):
    """
    Visualize search results for a late fusion query.
    Similar to recall_image.py visualization.
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
        query_img_bordered = add_border_to_image(query_img, border_width=8, border_color=(0, 0, 255))
        axes[0].imshow(query_img_bordered)
        axes[0].set_title("QUERY", fontsize=12, fontweight='bold', color='blue')
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error loading query\n{e}", ha='center', va='center')
        axes[0].axis('off')
    
    # Get relevant paths for highlighting
    relevant_paths = set()
    for r in all_retrieved:
        if highlight_relevant:
            page_key = r.get('page_key', '')
            if page_key:
                relevant_paths.add(page_key)
    
    # Plot results
    for i, r in enumerate(results):
        ax = axes[i + 1]
        
        # Find image path
        img_path = find_image_path_from_result(r, image_dir)
        
        if img_path and img_path.exists():
            try:
                result_img = Image.open(img_path).convert("RGB")
                
                # Determine border color based on relevance
                page_key = r.get('page_key', '')
                is_relevant = page_key in relevant_paths if highlight_relevant else False
                combined_score = r.get('combined_score', r.get('similarity', 0.0))
                sim_pct = combined_score * 100
                
                if is_relevant:
                    border_color = (0, 200, 0)  # Green for relevant
                elif sim_pct >= 80:
                    border_color = (255, 165, 0)  # Orange for high similarity
                else:
                    border_color = (100, 100, 100)  # Gray for others
                
                result_img_bordered = add_border_to_image(result_img, border_width=6, border_color=border_color)
                ax.imshow(result_img_bordered)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{e}", ha='center', va='center', fontsize=8)
                ax.set_facecolor('#f0f0f0')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=8)
            ax.set_facecolor('#f0f0f0')
        
        # Title with scores and metadata
        meta = r.get('image_meta', r.get('text_meta', r))
        manga = meta.get('manga', '')[:20] + '...' if len(meta.get('manga', '')) > 20 else meta.get('manga', '')
        chapter = meta.get('chapter', '')
        page = meta.get('page', '')
        
        chapter_num = chapter.replace('chapter_', '') if 'chapter_' in chapter else chapter
        page_num = page.replace('page_', '') if 'page_' in page else page
        
        combined_score = r.get('combined_score', r.get('similarity', 0.0))
        img_sim = r.get('image_similarity', 0.0)
        txt_sim = r.get('text_similarity', 0.0)
        sim_pct = combined_score * 100
        
        title = f"#{r.get('rank', i+1)} ({sim_pct:.1f}%)\nC:{combined_score:.2f} I:{img_sim:.2f} T:{txt_sim:.2f}\n{manga}\nCh.{chapter_num} Pg.{page_num}"
        
        page_key = r.get('page_key', '')
        is_relevant = page_key in relevant_paths if highlight_relevant else False
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
    title_text = f"Query: {query_name} ({title_prefix})"
    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k for late fusion queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image-first late fusion
  python -m evaluate.recall_fusion --queries queries --mode image --alpha 0.5 \\
      --image-index indexes/image --text-index indexes/text --output results/fusion_image
  
  # Text-first late fusion
  python -m evaluate.recall_fusion --queries queries --mode text --alpha 0.5 \\
      --image-index indexes/image --text-index indexes/text --output results/fusion_text
  
  # Grid search over alpha and m values
  python -m evaluate.recall_fusion --queries queries --mode image \\
      --alphas 0.1 0.2 0.3 0.4 0.5 --m-values 50 60 70 \\
      --image-index indexes/image --text-index indexes/text --output results/fusion_grid
        """
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        required=True,
        help="Path to queries directory containing query folders",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        required=True,
        choices=["image", "text"],
        help="Late fusion mode: 'image' for image-first, 'text' for text-first",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Single alpha value (use --alphas for grid search)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=None,
        help="Multiple alpha values for grid search (e.g., 0.1 0.2 0.3 0.4 0.5)",
    )
    parser.add_argument(
        "--image-index",
        type=str,
        required=True,
        help="Path to image FAISS index directory (contains faiss.index and metadata.json)",
    )
    parser.add_argument(
        "--text-index",
        type=str,
        required=True,
        help="Path to text FAISS index directory (contains faiss.index and metadata.json)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., a_results/fusion_image)",
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
        "--m",
        type=int,
        default=None,
        help="Single m value (use --m-values for grid search)",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs='+',
        default=None,
        help="Multiple m values for grid search (e.g., 50 60 70)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to image directory for visualization (e.g., b_datasets/final_dataset)",
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
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if not FAISS_AVAILABLE:
        print("Error: faiss is not installed.")
        print("Install with: pip install faiss-cpu or faiss-gpu")
        return
    
    # Determine alpha and m values (grid search vs single values)
    if args.alphas:
        alpha_values = args.alphas
    elif args.alpha is not None:
        alpha_values = [args.alpha]
    else:
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Default grid search values
    
    if args.m_values:
        m_values = args.m_values
    elif args.m is not None:
        m_values = [args.m]
    else:
        m_values = [50, 60, 70, 80, 90, 100]  # Default grid search values
    
    # Validate alphas
    for alpha in alpha_values:
        if not 0 <= alpha <= 1:
            print(f"Error: alpha {alpha} must be between 0 and 1")
            return
    
    is_grid_search = len(alpha_values) > 1 or len(m_values) > 1
    
    # Setup paths
    queries_dir = Path(args.queries)
    image_index_dir = Path(args.image_index)
    text_index_dir = Path(args.text_index)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # Check if paths exist
    if not queries_dir.exists():
        print(f"Error: Queries directory not found: {queries_dir}")
        return
    
    if not image_index_dir.exists():
        print(f"Error: Image index directory not found: {image_index_dir}")
        return
    
    if not text_index_dir.exists():
        print(f"Error: Text index directory not found: {text_index_dir}")
        return
    
    # Check if faiss indexes exist
    image_faiss_index = image_index_dir / "faiss.index"
    if not image_faiss_index.exists():
        print(f"Error: FAISS index not found: {image_faiss_index}")
        return
    
    text_faiss_index = text_index_dir / "faiss.index"
    if not text_faiss_index.exists():
        print(f"Error: FAISS index not found: {text_faiss_index}")
        return
    
    # Find query folders
    query_folders = sorted([d for d in queries_dir.iterdir() if d.is_dir()])
    if len(query_folders) == 0:
        print(f"Error: No query folders found in {queries_dir}")
        return
    
    print(f"\nLate Fusion Recall Evaluation")
    print("="*60)
    print(f"Mode: {args.mode}-first")
    print(f"Alpha values: {alpha_values}")
    print(f"M values: {m_values}")
    print(f"Queries directory: {queries_dir}")
    print(f"Image index: {image_index_dir}")
    print(f"Text index: {text_index_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(query_folders)} query folders")
    print(f"Evaluating recall@k with k values: {args.k_values}")
    print(f"Evaluating mAP@k with k values: {args.map_k_values}")
    print("="*60)
    
    # Load CLIP model once before the loop (prevents segfault on macOS)
    # Try preferred device first, fall back to CPU if there's an issue
    print("\nLoading CLIP model (will be reused for all queries)...")
    device = args.device if args.device else get_device()
    
    try:
        print(f"Attempting to load CLIP model on {device}...")
        # Load model directly using open_clip (same as recall_image.py)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k",
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        
        # Test the model with a dummy operation to catch segfaults early
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, 3, 224, 224).to(device)
            _ = model.encode_image(dummy_tensor)
        
        print(f"CLIP model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading CLIP model on {device}: {e}")
        if device != "cpu":
            print("Falling back to CPU...")
            device = "cpu"
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14",
                    pretrained="laion2b_s32b_b82k",
                )
                model = model.to(device).eval()
                tokenizer = open_clip.get_tokenizer("ViT-L-14")
                print(f"CLIP model loaded on {device} (fallback)")
            except Exception as e2:
                print(f"Fatal error: Could not load CLIP model even on CPU: {e2}")
                return
        else:
            print(f"Fatal error: Could not load CLIP model: {e}")
            return
    
    # Load FAISS indexes once (reused for all queries - major performance improvement)
    print("\nLoading FAISS indexes (will be reused for all queries)...")
    image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
    text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
    print(f"Image index: {image_index.ntotal} vectors")
    print(f"Text index: {text_index.ntotal} vectors")
    
    # Build lookups once
    from late_fusion.faiss_search import build_text_lookup, build_image_lookup
    text_lookup = build_text_lookup(text_id_to_meta)
    image_lookup = build_image_lookup(image_id_to_meta)
    print(f"Text lookup: {len(text_lookup)} unique pages")
    print(f"Image lookup: {len(image_lookup)} unique pages")
    
    # Grid search storage
    grid_results = {}  # (alpha, m) -> results dict
    
    # Print grid search info
    if is_grid_search:
        print(f"\n{'='*60}")
        print("GRID SEARCH MODE")
        print(f"Alpha values: {alpha_values}")
        print(f"M values: {m_values}")
        print(f"Total combinations: {len(alpha_values) * len(m_values)}")
        print(f"{'='*60}")
    
    # Pre-compute embeddings for all queries ONCE (major performance optimization for grid search)
    print(f"\nPre-computing embeddings for {len(query_folders)} queries...")
    from late_fusion.clip_encoder import encode_image, encode_text
    
    query_embeddings = {}  # query_folder -> (image_embedding, text_embedding, cached_description, query_image, user_query, ground_truth)
    
    for query_folder in tqdm(query_folders, desc="Computing embeddings"):
        query_image = find_query_image(query_folder)
        if query_image is None:
            query_embeddings[str(query_folder)] = None
            continue
        
        # Load text query
        user_query = load_text_query(query_folder)
        if not user_query:
            user_query = ""
        
        # Load cached LLM description or generate and save it
        cached_description = load_llm_description(query_folder)
        if cached_description is None:
            # Generate LLM description if not cached
            try:
                cached_description = encode_query_llm(query_image, user_query, verbose=False)
                # Save the generated description for future reuse
                save_llm_description(query_folder, cached_description)
            except Exception as e:
                print(f"Warning: Could not generate LLM description for {query_folder}: {e}")
                cached_description = user_query if user_query else "manga image"
        
        # Load ground truth
        labels_file = query_folder / "labels.txt"
        ground_truth = load_ground_truth_labels(labels_file)
        
        # Compute embeddings
        try:
            image_embedding = encode_image(model, preprocess, device, query_image)
            text_embedding = encode_text(model, tokenizer, device, cached_description)
            
            query_embeddings[str(query_folder)] = {
                'image_embedding': image_embedding,
                'text_embedding': text_embedding,
                'cached_description': cached_description,
                'query_image': query_image,
                'user_query': user_query,
                'ground_truth': ground_truth
            }
        except Exception as e:
            print(f"Warning: Could not compute embeddings for {query_folder}: {e}")
            query_embeddings[str(query_folder)] = None
    
    valid_queries = sum(1 for v in query_embeddings.values() if v is not None)
    print(f"Successfully pre-computed embeddings for {valid_queries}/{len(query_folders)} queries")
    
    # Run grid search (now using pre-computed embeddings)
    for alpha in alpha_values:
        for m in m_values:
            combo_name = f"alpha_{alpha}_m_{m}"
            print(f"\n{'='*60}")
            print(f"Running combination: alpha={alpha}, m={m}")
            print(f"{'='*60}")
            
            # Evaluate each query with timing (using pre-computed embeddings)
            all_results = []
            query_times = []
            for query_folder in tqdm(query_folders, desc=f"Processing (α={alpha}, m={m})"):
                start_time = time.time()
                
                # Get pre-computed data
                precomputed = query_embeddings.get(str(query_folder))
                
                if precomputed is None:
                    result = {
                        "query_folder": str(query_folder),
                        "error": "Failed to pre-compute embeddings"
                    }
                else:
                    result = evaluate_single_query(
                        query_folder=query_folder,
                        image_index_dir=image_index_dir,
                        text_index_dir=text_index_dir,
                        mode=args.mode,
                        alpha=alpha,
                        m=m,
                        k_values=args.k_values,
                        map_k_values=args.map_k_values,
                        model=model,
                        preprocess=preprocess,
                        tokenizer=tokenizer,
                        device=device,
                        image_index=image_index,
                        image_id_to_meta=image_id_to_meta,
                        text_index=text_index,
                        text_id_to_meta=text_id_to_meta,
                        text_lookup=text_lookup,
                        image_lookup=image_lookup,
                        query_image_embedding=precomputed['image_embedding'],
                        query_text_embedding=precomputed['text_embedding'],
                        cached_description=precomputed['cached_description'],
                        verbose=args.verbose
                    )
                
                query_time = time.time() - start_time
                result['query_time_seconds'] = query_time
                query_times.append(query_time)
                all_results.append(result)
            
            # Compute averages for this combination
            averages = compute_average_metrics(all_results)
            
            # Store results
            grid_results[(alpha, m)] = {
                'all_results': all_results,
                'averages': averages,
                'query_times': query_times,
                'alpha': alpha,
                'm': m
            }
            
            # Create output subdirectory for this combination
            combo_output_dir = output_dir / combo_name
            combo_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute timing statistics
            timing_stats = {
                "total_time_seconds": float(sum(query_times)),
                "average_time_seconds": float(np.mean(query_times)),
                "std_time_seconds": float(np.std(query_times)),
                "min_time_seconds": float(np.min(query_times)),
                "max_time_seconds": float(np.max(query_times)),
                "median_time_seconds": float(np.median(query_times)),
            }
            
            # Save results for this combination
            output_data = {
                "config": {
                    "queries_dir": str(queries_dir),
                    "image_index_dir": str(image_index_dir),
                    "text_index_dir": str(text_index_dir),
                    "mode": args.mode,
                    "alpha": alpha,
                    "m": m,
                    "k_values": args.k_values,
                    "map_k_values": args.map_k_values,
                    "num_queries": len(query_folders),
                },
                "individual_results": all_results,
                "average_metrics": averages,
                "timing_stats": timing_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            results_file = combo_output_dir / "recall_results.json"
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Create summary table
            create_summary_table(all_results, averages, combo_output_dir, include_user_query=True)
            
            # Print quick summary for this combination
            if 'error' not in averages:
                recall_50 = averages['average_metrics'].get('recall@50', {}).get('mean', 0)
                map_50 = averages['average_metrics'].get('map@50', {}).get('mean', 0)
                print(f"  → Recall@50: {recall_50:.4f}, mAP@50: {map_50:.4f}")
    
    # Use the last combination's results for backward compatibility (if single values)
    last_alpha, last_m = alpha_values[-1], m_values[-1]
    all_results = grid_results[(last_alpha, last_m)]['all_results']
    averages = grid_results[(last_alpha, last_m)]['averages']
    query_times = grid_results[(last_alpha, last_m)]['query_times']
    
    # Initialize best_alpha and best_m (will be updated in grid search mode)
    best_alpha, best_m = None, None
    
    # Compute timing statistics for last combo
    timing_stats = {
        "total_time_seconds": float(sum(query_times)),
        "average_time_seconds": float(np.mean(query_times)),
        "std_time_seconds": float(np.std(query_times)),
        "min_time_seconds": float(np.min(query_times)),
        "max_time_seconds": float(np.max(query_times)),
        "median_time_seconds": float(np.median(query_times)),
    }
    
    # Create grid search comparison summary if multiple combinations
    if is_grid_search:
        print("\n" + "="*60)
        print("GRID SEARCH COMPARISON SUMMARY")
        print("="*60)
        
        # Build comparison table data
        comparison_data = []
        for (alpha, m), data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                row = {
                    'alpha': alpha,
                    'm': m,
                }
                # Add all recall and mAP metrics
                for metric_name, stats in avg['average_metrics'].items():
                    row[f'{metric_name}_mean'] = stats['mean']
                    row[f'{metric_name}_std'] = stats['std']
                comparison_data.append(row)
        
        # Save comparison CSV
        comparison_file = output_dir / "grid_search_comparison.csv"
        if comparison_data:
            import csv
            fieldnames = ['alpha', 'm'] + sorted([k for k in comparison_data[0].keys() if k not in ['alpha', 'm']])
            with open(comparison_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in sorted(comparison_data, key=lambda x: (x['alpha'], x['m'])):
                    writer.writerow(row)
            print(f"Grid search comparison saved to: {comparison_file}")
        
        # Save full grid results JSON
        grid_summary = {
            "alpha_values": alpha_values,
            "m_values": m_values,
            "mode": args.mode,
            "num_combinations": len(alpha_values) * len(m_values),
            "combinations": []
        }
        for (alpha, m), data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                grid_summary["combinations"].append({
                    "alpha": alpha,
                    "m": m,
                    "average_metrics": avg['average_metrics'],
                    "overall_stats": avg['overall_stats']
                })
        
        grid_json_file = output_dir / "grid_search_summary.json"
        with open(grid_json_file, 'w') as f:
            json.dump(grid_summary, f, indent=2)
        print(f"Grid search summary saved to: {grid_json_file}")
        
        # Print comparison table
        print(f"\n{'Alpha':<8} {'M':<6} {'Recall@20':<12} {'Recall@50':<12} {'mAP@20':<12} {'mAP@50':<12}")
        print("-" * 70)
        for row in sorted(comparison_data, key=lambda x: (x['alpha'], x['m'])):
            print(f"{row['alpha']:<8.1f} {row['m']:<6} "
                  f"{row.get('recall@20_mean', 0):<12.4f} "
                  f"{row.get('recall@50_mean', 0):<12.4f} "
                  f"{row.get('map@20_mean', 0):<12.4f} "
                  f"{row.get('map@50_mean', 0):<12.4f}")
        
        # Find best hyperparameter configuration (based on mAP@50, with recall@50 as tiebreaker)
        best_alpha, best_m = None, None
        best_score = -1
        best_recall = -1
        
        for (alpha, m), data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                map_50 = avg['average_metrics'].get('map@50', {}).get('mean', 0)
                recall_50 = avg['average_metrics'].get('recall@50', {}).get('mean', 0)
                # Primary: mAP@50, Secondary: recall@50
                if map_50 > best_score or (map_50 == best_score and recall_50 > best_recall):
                    best_score = map_50
                    best_recall = recall_50
                    best_alpha = alpha
                    best_m = m
        
        if best_alpha is not None and best_m is not None:
            best_data = grid_results[(best_alpha, best_m)]
            best_averages = best_data['averages']
            
            print("\n" + "="*60)
            print("BEST HYPERPARAMETER CONFIGURATION")
            print("="*60)
            print(f"Best Alpha: {best_alpha}")
            print(f"Best M: {best_m}")
            print(f"mAP@50: {best_score:.4f}")
            print(f"Recall@50: {best_recall:.4f}")
            
            print("\nAll metrics for best configuration:")
            print(f"{'Metric':<15} {'Mean':<10} {'Std':<10}")
            print("-" * 40)
            for metric_name, stats in sorted(best_averages['average_metrics'].items()):
                print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f}")
            
            # Save best configuration summary
            best_config = {
                "best_alpha": best_alpha,
                "best_m": best_m,
                "mode": args.mode,
                "primary_metric": "map@50",
                "primary_score": best_score,
                "secondary_metric": "recall@50", 
                "secondary_score": best_recall,
                "all_metrics": {k: v['mean'] for k, v in best_averages['average_metrics'].items()},
                "overall_stats": best_averages['overall_stats']
            }
            
            best_config_file = output_dir / "best_config.json"
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"\nBest configuration saved to: {best_config_file}")
            
            # Update variables to use best configuration for visualizations
            all_results = best_data['all_results']
            averages = best_averages
            query_times = best_data['query_times']
    
    # Create visualizations
    if not args.no_viz:
        print("\nCreating visualizations...")
        
        # Determine best config info for titles
        mode_name = "Image-First" if args.mode == "image" else "Text-First"
        if is_grid_search and best_alpha is not None:
            title_suffix = f" (Best: α={best_alpha}, m={best_m})"
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
        else:
            title_suffix = ""
            best_dir = output_dir
        
        # Create recall curves for best configuration
        curves_dir = best_dir / "recall_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(all_results):
            if 'error' in result:
                continue
            
            query_name = Path(result['query_folder']).name if 'query_folder' in result else f"query_{i+1}"
            curve_file = curves_dir / f"query_{i+1:03d}_{query_name}_recall_curve.png"
            plot_recall_curve(result, curve_file, color='purple', title_prefix=f"Late Fusion ({mode_name}){title_suffix}")
        
        # Aggregate curve
        aggregate_file = curves_dir / "aggregate_all_queries.png"
        plot_aggregate_recall_curves(all_results, averages, aggregate_file, color='purple', 
                                     title=f"Late Fusion ({mode_name}){title_suffix}")
        print(f"Recall curves saved to: {curves_dir}")
        
        # Create image visualizations for best configuration
        vis_dir = best_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(all_results):
            if 'error' in result or 'query_image' not in result:
                continue
            
            query_name = Path(result['query_folder']).name
            safe_name = "".join(c for c in query_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            
            # Mark relevant results
            all_retrieved = result.get('all_retrieved', [])
            relevant_paths = {r.get('page_key', '') for r in result.get('relevant_results', [])}
            for r in all_retrieved:
                r['is_relevant'] = r.get('page_key', '') in relevant_paths
            
            output_file = vis_dir / f"query_{i+1:03d}_{safe_name}.png"
            visualize_query_results(
                query_path=result['query_image'],
                all_retrieved=all_retrieved,
                output_path=output_file,
                num_images=10,
                highlight_relevant=True,
                image_dir=image_dir,
                title_prefix=f"Late Fusion ({mode_name}){title_suffix}"
            )
        
        print(f"Visualizations saved to: {vis_dir}")
        
        # Save best configuration results JSON
        if is_grid_search and best_alpha is not None:
            best_results_file = best_dir / "recall_results.json"
            best_output_data = {
                "config": {
                    "queries_dir": str(queries_dir),
                    "image_index_dir": str(image_index_dir),
                    "text_index_dir": str(text_index_dir),
                    "mode": args.mode,
                    "alpha": best_alpha,
                    "m": best_m,
                    "k_values": args.k_values,
                    "map_k_values": args.map_k_values,
                    "num_queries": len(query_folders),
                    "is_best_from_grid_search": True,
                },
                "individual_results": all_results,
                "average_metrics": averages,
                "timestamp": datetime.now().isoformat()
            }
            with open(best_results_file, 'w') as f:
                json.dump(best_output_data, f, indent=2)
            
            # Create summary table for best config
            create_summary_table(all_results, averages, best_dir, include_user_query=True)
            print(f"Best configuration results saved to: {best_dir}")
        
        # Create grid search comparison visualizations
        if is_grid_search:
            print("\nCreating grid search comparison visualizations...")
            comparison_dir = output_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comparison plots for each metric
            metrics_to_plot = ['recall@5', 'recall@10', 'recall@20', 'recall@30', 'recall@40', 'recall@50', 'map@5', 'map@10', 'map@20', 'map@30', 'map@40', 'map@50']
            
            # 1. Alpha comparison (fixed m)
            for m in m_values:
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                
                # Recall subplot
                ax1 = axes[0]
                for metric in ['recall@5', 'recall@10', 'recall@20', 'recall@30', 'recall@40', 'recall@50']:
                    x_vals = []
                    y_vals = []
                    for alpha in alpha_values:
                        if (alpha, m) in grid_results:
                            avg = grid_results[(alpha, m)]['averages']
                            if 'error' not in avg and metric in avg['average_metrics']:
                                x_vals.append(alpha)
                                y_vals.append(avg['average_metrics'][metric]['mean'])
                    if x_vals:
                        ax1.plot(x_vals, y_vals, marker='o', label=metric)
                ax1.set_xlabel('Alpha')
                ax1.set_ylabel('Recall')
                ax1.set_title(f'Recall@K vs Alpha (m={m})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # mAP subplot
                ax2 = axes[1]
                for metric in ['map@5', 'map@10', 'map@20', 'map@30', 'map@40', 'map@50']:
                    x_vals = []
                    y_vals = []
                    for alpha in alpha_values:
                        if (alpha, m) in grid_results:
                            avg = grid_results[(alpha, m)]['averages']
                            if 'error' not in avg and metric in avg['average_metrics']:
                                x_vals.append(alpha)
                                y_vals.append(avg['average_metrics'][metric]['mean'])
                    if x_vals:
                        ax2.plot(x_vals, y_vals, marker='s', label=metric)
                ax2.set_xlabel('Alpha')
                ax2.set_ylabel('mAP')
                ax2.set_title(f'mAP@K vs Alpha (m={m})')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(comparison_dir / f"alpha_comparison_m_{m}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 2. M comparison (fixed alpha)
            for alpha in alpha_values:
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                
                # Recall subplot
                ax1 = axes[0]
                for metric in ['recall@5', 'recall@10', 'recall@20', 'recall@30', 'recall@40', 'recall@50']:
                    x_vals = []
                    y_vals = []
                    for m in m_values:
                        if (alpha, m) in grid_results:
                            avg = grid_results[(alpha, m)]['averages']
                            if 'error' not in avg and metric in avg['average_metrics']:
                                x_vals.append(m)
                                y_vals.append(avg['average_metrics'][metric]['mean'])
                    if x_vals:
                        ax1.plot(x_vals, y_vals, marker='o', label=metric)
                ax1.set_xlabel('M (Initial Candidates)')
                ax1.set_ylabel('Recall')
                ax1.set_title(f'Recall@K vs M (alpha={alpha})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # mAP subplot
                ax2 = axes[1]
                for metric in ['map@5', 'map@10', 'map@20', 'map@30', 'map@40', 'map@50']:
                    x_vals = []
                    y_vals = []
                    for m in m_values:
                        if (alpha, m) in grid_results:
                            avg = grid_results[(alpha, m)]['averages']
                            if 'error' not in avg and metric in avg['average_metrics']:
                                x_vals.append(m)
                                y_vals.append(avg['average_metrics'][metric]['mean'])
                    if x_vals:
                        ax2.plot(x_vals, y_vals, marker='s', label=metric)
                ax2.set_xlabel('M (Initial Candidates)')
                ax2.set_ylabel('mAP')
                ax2.set_title(f'mAP@K vs M (alpha={alpha})')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(comparison_dir / f"m_comparison_alpha_{alpha}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 3. Create heatmap for recall@50 and mAP@50
            for metric in ['recall@50', 'map@50']:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Build heatmap data
                heatmap_data = np.zeros((len(m_values), len(alpha_values)))
                for i, m in enumerate(m_values):
                    for j, alpha in enumerate(alpha_values):
                        if (alpha, m) in grid_results:
                            avg = grid_results[(alpha, m)]['averages']
                            if 'error' not in avg and metric in avg['average_metrics']:
                                heatmap_data[i, j] = avg['average_metrics'][metric]['mean']
                
                im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(alpha_values)))
                ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
                ax.set_yticks(range(len(m_values)))
                ax.set_yticklabels(m_values)
                ax.set_xlabel('Alpha')
                ax.set_ylabel('M (Initial Candidates)')
                ax.set_title(f'{metric.upper()} Heatmap')
                
                # Add value annotations
                for i in range(len(m_values)):
                    for j in range(len(alpha_values)):
                        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                       ha='center', va='center', color='black', fontsize=9)
                
                plt.colorbar(im, ax=ax, label=metric.upper())
                plt.tight_layout()
                plt.savefig(comparison_dir / f"heatmap_{metric.replace('@', '_')}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"Comparison visualizations saved to: {comparison_dir}")
    
    # Print summary
    print("\n" + "="*60)
    if is_grid_search and best_alpha is not None:
        print(f"FINAL SUMMARY - BEST CONFIGURATION (α={best_alpha}, m={best_m})")
    else:
        print("RECALL EVALUATION SUMMARY (Late Fusion)")
    print("="*60)
    if 'error' not in averages:
        print(f"Mode: {args.mode}-first")
        if is_grid_search and best_alpha is not None:
            print(f"Best Alpha: {best_alpha}")
            print(f"Best M: {best_m}")
            print(f"(Selected from {len(alpha_values)} alphas × {len(m_values)} m values = {len(alpha_values) * len(m_values)} combinations)")
        else:
            print(f"Alpha: {alpha_values[0] if len(alpha_values) == 1 else alpha_values}")
            print(f"M: {m_values[0] if len(m_values) == 1 else m_values}")
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
        print(f"  Total ground truth pages: {overall['total_ground_truth']}")
        print(f"  Total retrieved pages: {overall['total_retrieved']}")
        print(f"  Total relevant retrieved: {overall['total_relevant_retrieved']}")
        print(f"  Overall recall: {overall['overall_recall']:.4f}")
    
    print("\nQuery Timing Statistics (per combination):")
    print(f"  Total time: {timing_stats['total_time_seconds']:.3f} seconds")
    print(f"  Average query time: {timing_stats['average_time_seconds']*1000:.2f} ms")
    print(f"  Std dev: {timing_stats['std_time_seconds']*1000:.2f} ms")
    print(f"  Min: {timing_stats['min_time_seconds']*1000:.2f} ms")
    print(f"  Max: {timing_stats['max_time_seconds']*1000:.2f} ms")
    print(f"  Median: {timing_stats['median_time_seconds']*1000:.2f} ms")
    
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")
    if is_grid_search and best_alpha is not None:
        print(f"Best configuration outputs saved to: {output_dir / 'best'}")


if __name__ == "__main__":
    main()

