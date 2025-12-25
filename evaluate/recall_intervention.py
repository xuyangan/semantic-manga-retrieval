#!/usr/bin/env python3
"""
Recall Evaluation Script for Intervention Queries

Evaluates recall@k for intervention queries using the gated search pipeline.
Processes query folders with query images, text queries, and labels.txt files.

Usage:
    python -m evaluate.recall_intervention --queries a_queries/text --beta 1.0 --gamma 1.0 \\
        --image-embedding b_datasets/final_dataset_embeddings \\
        --text-embedding b_datasets/final_dataset_text_embeddings \\
        --image-index a_indexes/image --text-index a_indexes/text --output a_results/intervention
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

# Import intervention modules
from intervention.gate import compute_gate_for_candidates
from intervention.transform import build_intervention_index, build_page_to_faiss_mapping
from late_fusion.clip_encoder import load_clip_model, encode_image, encode_text
from late_fusion.faiss_search import (
    search_index,
    deduplicate_text_candidates,
    get_page_key,
)
from late_fusion.llm_encoder import encode_query_llm


def intervention_search_with_cached_description(
    image_path: Path,
    user_query: str,
    image_db: Path,
    text_db: Path,
    image_index_dir: Path,
    text_index_dir: Path,
    m: int,
    k: int,
    gamma: float,
    beta: float,
    cached_description: Optional[str] = None,
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    text_index=None,
    text_id_to_meta=None,
    image_index=None,
    image_id_to_meta=None,
    verbose: bool = False
) -> dict:
    """
    Perform intervention search with gated image embedding modification.
    Uses cached LLM description and separate index directories.
    
    Pipeline:
    1. Generate LLM description from user query (image + text)
    2. Generate CLIP embeddings (image and text)
    3. Search text DB for m candidates
    4. Deduplicate by page (keep max similarity per page)
    5. Compute z-score: z = (s - mean) / std
    6. Compute gate factor: g = sigmoid(gamma * z)
    7. Apply intervention: v_i = normalize(v_i + β * g_i * t_q)
    8. Build temporary index with modified embeddings
    9. Search with query image embedding for top k results
    
    Args:
        image_path: Path to query image
        user_query: User's text query (e.g., "the male character")
        image_db: Path to image embeddings directory
        text_db: Path to text embeddings directory
        image_index_dir: Path to image index directory (contains faiss.index and metadata.json)
        text_index_dir: Path to text index directory (contains faiss.index and metadata.json)
        m: Number of initial text candidates to retrieve
        k: Number of final results to return
        gamma: Gate steepness parameter (higher = sharper transition)
        beta: Intervention strength scaling factor
        cached_description: Optional cached LLM description
        verbose: Print detailed info
    
    Returns:
        Dict with all results and intermediate data
    """
    # Step 1: Generate or use cached LLM description
    if cached_description is not None:
        description = cached_description
        if verbose:
            print(f"Using cached LLM description: {description[:80]}...")
    else:
        if verbose:
            print("\n[1/8] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (only if not provided)
    if model is None or preprocess is None or tokenizer is None:
        if verbose:
            print("\n[2/8] Loading CLIP model and generating embeddings...")
        model, preprocess, tokenizer, device = load_clip_model(device)
    elif verbose:
        print("\n[2/8] Using provided CLIP model for generating embeddings...")
    
    query_image_embedding = encode_image(model, preprocess, device, image_path)
    query_text_embedding = encode_text(model, tokenizer, device, description)
    if verbose:
        print(f"  Query image embedding: {query_image_embedding.shape}")
        print(f"  Query text embedding: {query_text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (only if not provided - allows reuse across queries)
    if text_index is None or text_id_to_meta is None:
        if verbose:
            print("\n[3/8] Loading FAISS indexes...")
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    elif verbose:
        print("\n[3/8] Using provided FAISS indexes...")
    
    if image_index is None or image_id_to_meta is None:
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    
    # Step 4: Search text DB
    if verbose:
        print(f"\n[4/8] Searching text DB (m={m})...")
    text_candidates_raw = search_index(text_index, text_id_to_meta, query_text_embedding, k=m)
    if verbose:
        print(f"  Found {len(text_candidates_raw)} text candidates")
    
    # Deduplicate by page (keep max similarity)
    text_candidates = deduplicate_text_candidates(text_candidates_raw)
    if verbose:
        print(f"  Unique pages: {len(text_candidates)}")
    
    # Step 5: Compute gate factors (using z-score normalization)
    if verbose:
        print(f"\n[5/8] Computing gate factors with z-score (gamma={gamma})...")
    candidates_with_gates = compute_gate_for_candidates(text_candidates, gamma)
    
    similarities = [c["similarity"] for c in candidates_with_gates]
    z_scores = [c["z_score"] for c in candidates_with_gates]
    gates = [c["gate_factor"] for c in candidates_with_gates]
    
    if verbose:
        print(f"  Similarity: mean={np.mean(similarities):.4f}, std={np.std(similarities):.4f}")
        print(f"  Z-score range: [{min(z_scores):.4f}, {max(z_scores):.4f}]")
        print(f"  Gate range: [{min(gates):.4f}, {max(gates):.4f}]")
        print(f"  Mean gate: {np.mean(gates):.4f}")
    
    # Step 6: Build page to FAISS index mapping
    if verbose:
        print("\n[6/8] Building page to index mapping...")
    page_to_faiss_idx = build_page_to_faiss_mapping(image_id_to_meta)
    if verbose:
        print(f"  Mapped {len(page_to_faiss_idx)} pages")
    
    # Step 7: Apply intervention and build new index
    if verbose:
        print(f"\n[7/8] Applying intervention transform (beta={beta})...")
    intervention_index, _ = build_intervention_index(
        image_index,
        image_id_to_meta,
        query_text_embedding,
        candidates_with_gates,
        page_to_faiss_idx,
        beta=beta
    )
    if verbose:
        print(f"  Created intervention index: {intervention_index.ntotal} vectors")
    
    # Step 8: Search with query image
    if verbose:
        print(f"\n[8/8] Searching intervention index (k={k})...")
    final_results = search_index(intervention_index, image_id_to_meta, query_image_embedding, k=k)
    if verbose:
        print(f"  Found {len(final_results)} results")
    
    # Add intervention info to results
    # Build lookup for faster matching
    gate_lookup = {c.get("page_key"): c for c in candidates_with_gates if c.get("page_key")}
    
    for result in final_results:
        page_key = get_page_key(result)
        result["page_key"] = page_key
        
        # Check if this result had intervention applied
        gate_info = gate_lookup.get(page_key)
        if gate_info:
            result["had_intervention"] = True
            result["gate_factor"] = gate_info["gate_factor"]
            result["z_score"] = gate_info["z_score"]
            result["text_similarity"] = gate_info["similarity"]
        else:
            result["had_intervention"] = False
            result["gate_factor"] = 0.0
            result["z_score"] = 0.0
            result["text_similarity"] = 0.0
    
    # Print gating info for results
    if verbose:
        intervened_count = sum(1 for r in final_results if r.get('had_intervention', False))
        print(f"\n  Gating summary: {intervened_count}/{len(final_results)} results had intervention")
        print("  Top 10 results with intervention status:")
        for r in final_results[:10]:
            status = "✓" if r.get('had_intervention') else "✗"
            gate = r.get('gate_factor', 0.0)
            page_key = r.get('page_key', '?')[:40]
            print(f"    Rank {r.get('rank', 0):2d}: {status} gate={gate:.3f} | {page_key}")
    
    return {
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "gamma": gamma,
        "beta": beta,
        "m": m,
        "k": k,
        "query_image_embedding": query_image_embedding,
        "query_text_embedding": query_text_embedding,
        "num_text_candidates_raw": len(text_candidates_raw),
        "num_text_candidates_dedup": len(text_candidates),
        "num_intervened": len(candidates_with_gates),
        "candidates_with_gates": candidates_with_gates,
        "final_results": final_results,
    }


def evaluate_single_query(
    query_folder: Path,
    image_db: Path,
    text_db: Path,
    image_index_dir: Path,
    text_index_dir: Path,
    gamma: float,
    beta: float,
    m: int,
    k_values: List[int],
    map_k_values: List[int] = [10, 20, 30, 50],
    model=None,
    preprocess=None,
    tokenizer=None,
    device: Optional[str] = None,
    text_index=None,
    text_id_to_meta=None,
    image_index=None,
    image_id_to_meta=None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate recall for a single query using intervention search.
    
    Args:
        query_folder: Path to query folder (contains query.png, text.txt, labels.txt)
        image_db: Path to image embeddings directory
        text_db: Path to text embeddings directory
        image_index_dir: Path to image index directory (contains faiss.index and metadata.json)
        text_index_dir: Path to text index directory (contains faiss.index and metadata.json)
        gamma: Gate steepness parameter
        beta: Intervention strength scaling factor
        m: Number of initial candidates to retrieve
        k_values: List of k values for recall@k
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
    
    # Load cached LLM description or generate and save it
    cached_description = load_llm_description(query_folder)
    if cached_description is None:
        # Generate LLM description if not cached
        try:
            cached_description = encode_query_llm(query_image, user_query, verbose=False)
            # Save the generated description for future reuse
            save_llm_description(query_folder, cached_description)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate LLM description: {e}")
            cached_description = user_query if user_query else "manga image"
    
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
    
    # Run intervention search
    try:
        max_k = max(k_values)
        
        result = intervention_search_with_cached_description(
            image_path=query_image,
            user_query=user_query,
            image_db=image_db,
            text_db=text_db,
            image_index_dir=image_index_dir,
            text_index_dir=text_index_dir,
            m=m,
            k=max_k,
            gamma=gamma,
            beta=beta,
            cached_description=cached_description,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            text_index=text_index,
            text_id_to_meta=text_id_to_meta,
            image_index=image_index,
            image_id_to_meta=image_id_to_meta,
            verbose=verbose
        )
        
        search_results = result['final_results']
        
    except Exception as e:
        return {
            "query_folder": str(query_folder),
            "query_image": str(query_image),
            "user_query": user_query,
            "error": f"Error running intervention search: {e}",
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
    
    # Compute gating statistics
    intervened_count = sum(1 for r in search_results if r.get('had_intervention', False))
    gate_factors = [r.get('gate_factor', 0.0) for r in search_results if r.get('had_intervention', False)]
    gating_stats = {
        "num_intervened": intervened_count,
        "num_not_intervened": len(search_results) - intervened_count,
        "intervened_percentage": intervened_count / len(search_results) if len(search_results) > 0 else 0.0,
        "mean_gate_factor": float(np.mean(gate_factors)) if len(gate_factors) > 0 else 0.0,
        "min_gate_factor": float(np.min(gate_factors)) if len(gate_factors) > 0 else 0.0,
        "max_gate_factor": float(np.max(gate_factors)) if len(gate_factors) > 0 else 0.0,
    }
    
    # Print gating summary for this query (only if not verbose to avoid duplication)
    if not verbose:
        query_name = query_folder.name
        print(f"  {query_name}: {intervened_count}/{len(search_results)} results had intervention (mean gate: {gating_stats['mean_gate_factor']:.3f})")
    
    return {
        "query_folder": str(query_folder),
        "query_image": str(query_image),
        "user_query": user_query,
        "llm_description": result.get('llm_description', ''),
        "gamma": gamma,
        "beta": beta,
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "map_metrics": map_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant,
        "gating_stats": gating_stats,
        "num_intervened_candidates": result.get('num_intervened', 0),
    }


def visualize_query_results(
    query_path: str,
    all_retrieved: List[Dict],
    output_path: Path,
    num_images: int = 10,
    highlight_relevant: bool = True,
    image_dir: Path = None
):
    """
    Visualize search results for an intervention query.
    Similar to recall_fusion.py visualization.
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
                sim = r.get('similarity', 0.0)
                sim_pct = sim * 100
                
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
        manga = r.get('manga', '')[:20] + '...' if len(r.get('manga', '')) > 20 else r.get('manga', '')
        chapter = r.get('chapter', '')
        page = r.get('page', '')
        
        chapter_num = chapter.replace('chapter_', '') if 'chapter_' in chapter else chapter
        page_num = page.replace('page_', '') if 'page_' in page else page
        
        sim = r.get('similarity', 0.0)
        gate = r.get('gate_factor', 0.0)
        intervened = "✓" if r.get('had_intervention') else "✗"
        sim_pct = sim * 100
        
        title = f"#{r.get('rank', i+1)} ({sim_pct:.1f}%)\nS:{sim:.2f} G:{gate:.2f} {intervened}\n{manga}\nCh.{chapter_num} Pg.{page_num}"
        
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
    title_text = f"Query: {query_name} (Intervention)"
    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall@k for intervention queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic intervention search
  python evaluate/recall_intervention.py --queries a_queries/text --beta 1.0 --gamma 1.0 --image-embedding b_datasets/final_dataset_embeddings --text-embedding b_datasets/final_dataset_text_embeddings --image-index a_indexes/image --text-index a_indexes/text --output a_results/intervention
  
  # With custom k values
  python evaluate/recall_intervention.py --queries a_queries/text --beta 1.5 --gamma 2.0 --image-embedding b_datasets/final_dataset_embeddings --text-embedding b_datasets/final_dataset_text_embeddings --image-index a_indexes/image --text-index a_indexes/text --output a_results/intervention --k-values 1 5 10 20 50
        """
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        required=True,
        help="Path to queries directory containing query folders (e.g., a_queries/text)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Beta parameter for intervention (intervention strength scaling factor)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="Gamma parameter for intervention (gate steepness parameter)",
    )
    parser.add_argument(
        "--image-embedding",
        type=str,
        required=True,
        help="Path to image embeddings directory (e.g., b_datasets/final_dataset_embeddings)",
    )
    parser.add_argument(
        "--text-embedding",
        type=str,
        required=True,
        help="Path to text embeddings directory (e.g., b_datasets/final_dataset_text_embeddings)",
    )
    parser.add_argument(
        "--image-index",
        type=str,
        required=True,
        help="Path to image index directory containing faiss.index and metadata.json (e.g., a_indexes/image)",
    )
    parser.add_argument(
        "--text-index",
        type=str,
        required=True,
        help="Path to text index directory containing faiss.index and metadata.json (e.g., a_indexes/text)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., a_results/intervention)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[10, 20, 30, 50],
        help="K values for recall@k evaluation (default: 10 20 30 50)",
    )
    parser.add_argument(
        "--map-k-values",
        type=int,
        nargs='+',
        default=[10, 20, 30, 50],
        help="K values for mAP@k evaluation (default: 10 20 30 50)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=50,
        help="Number of initial text candidates to retrieve before intervention (default: 50)",
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
    
    # Setup paths
    queries_dir = Path(args.queries)
    image_embedding_dir = Path(args.image_embedding)
    text_embedding_dir = Path(args.text_embedding)
    image_index_dir = Path(args.image_index)
    text_index_dir = Path(args.text_index)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # Check if paths exist
    if not queries_dir.exists():
        print(f"Error: Queries directory not found: {queries_dir}")
        return
    
    if not image_embedding_dir.exists():
        print(f"Error: Image embeddings directory not found: {image_embedding_dir}")
        return
    
    if not text_embedding_dir.exists():
        print(f"Error: Text embeddings directory not found: {text_embedding_dir}")
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
    
    print(f"\nIntervention Recall Evaluation")
    print("="*60)
    print(f"Beta: {args.beta} (intervention strength)")
    print(f"Gamma: {args.gamma} (gate steepness)")
    print(f"Initial candidates (m): {args.m}")
    print(f"Queries directory: {queries_dir}")
    print(f"Image embeddings: {image_embedding_dir}")
    print(f"Text embeddings: {text_embedding_dir}")
    print(f"Image index: {image_index_dir}")
    print(f"Text index: {text_index_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(query_folders)} query folders")
    print(f"Evaluating recall@k with k values: {args.k_values}")
    print(f"Evaluating mAP@k with k values: {args.map_k_values}")
    print("="*60)
    
    # Load CLIP model once before the loop (prevents segfault on macOS)
    print("\nLoading CLIP model (will be reused for all queries)...")
    device = args.device if args.device else get_device()
    
    try:
        print(f"Attempting to load CLIP model on {device}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k",
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        
        # Test the model with a dummy operation
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
    
    # Evaluate each query with timing
    all_results = []
    query_times = []
    for query_folder in tqdm(query_folders, desc="Processing queries"):
        start_time = time.time()
        result = evaluate_single_query(
            query_folder=query_folder,
            image_db=image_embedding_dir,
            text_db=text_embedding_dir,
            image_index_dir=image_index_dir,
            text_index_dir=text_index_dir,
            gamma=args.gamma,
            beta=args.beta,
            m=args.m,
            k_values=args.k_values,
            map_k_values=args.map_k_values,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            text_index=text_index,
            text_id_to_meta=text_id_to_meta,
            image_index=image_index,
            image_id_to_meta=image_id_to_meta,
            verbose=args.verbose
        )
        query_time = time.time() - start_time
        result['query_time_seconds'] = query_time
        query_times.append(query_time)
        all_results.append(result)
    
    # Compute averages
    print("\nComputing average metrics...")
    averages = compute_average_metrics(all_results)
    
    # Compute aggregate gating statistics
    print("\nComputing gating statistics...")
    valid_results = [r for r in all_results if 'error' not in r and 'gating_stats' in r]
    if valid_results:
        all_intervened_counts = [r['gating_stats']['num_intervened'] for r in valid_results]
        all_not_intervened_counts = [r['gating_stats']['num_not_intervened'] for r in valid_results]
        all_gate_factors = []
        for r in valid_results:
            all_gate_factors.extend([res.get('gate_factor', 0.0) for res in r.get('all_retrieved', []) if res.get('had_intervention', False)])
        
        aggregate_gating = {
            "total_intervened": int(sum(all_intervened_counts)),
            "total_not_intervened": int(sum(all_not_intervened_counts)),
            "mean_intervened_per_query": float(np.mean(all_intervened_counts)) if all_intervened_counts else 0.0,
            "mean_not_intervened_per_query": float(np.mean(all_not_intervened_counts)) if all_not_intervened_counts else 0.0,
            "overall_intervened_percentage": float(sum(all_intervened_counts) / (sum(all_intervened_counts) + sum(all_not_intervened_counts))) if (sum(all_intervened_counts) + sum(all_not_intervened_counts)) > 0 else 0.0,
            "mean_gate_factor": float(np.mean(all_gate_factors)) if all_gate_factors else 0.0,
            "min_gate_factor": float(np.min(all_gate_factors)) if all_gate_factors else 0.0,
            "max_gate_factor": float(np.max(all_gate_factors)) if all_gate_factors else 0.0,
            "std_gate_factor": float(np.std(all_gate_factors)) if all_gate_factors else 0.0,
        }
    else:
        aggregate_gating = {}
    
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
            "queries_dir": str(queries_dir),
            "image_embedding_dir": str(image_embedding_dir),
            "text_embedding_dir": str(text_embedding_dir),
            "image_index_dir": str(image_index_dir),
            "text_index_dir": str(text_index_dir),
            "beta": args.beta,
            "gamma": args.gamma,
            "m": args.m,
            "k_values": args.k_values,
            "map_k_values": args.map_k_values,
            "num_queries": len(query_folders),
        },
        "individual_results": all_results,
        "average_metrics": averages,
        "aggregate_gating_stats": aggregate_gating,
        "timing_stats": timing_stats,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = output_dir / "recall_results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(all_results, averages, output_dir, include_user_query=True)
    
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
            plot_recall_curve(result, curve_file, color='red', title_prefix="Intervention")
        
        # Aggregate curve
        aggregate_file = curves_dir / "aggregate_all_queries.png"
        plot_aggregate_recall_curves(all_results, averages, aggregate_file, color='red', title="All Intervention Queries")
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
                image_dir=image_dir
            )
        
        print(f"Visualizations saved to: {vis_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("RECALL EVALUATION SUMMARY (Intervention)")
    print("="*60)
    if 'error' not in averages:
        print(f"Beta: {args.beta}")
        print(f"Gamma: {args.gamma}")
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
    
    # Print gating statistics
    if aggregate_gating:
        print("\nGating Statistics:")
        print(f"  Total intervened results: {aggregate_gating['total_intervened']}")
        print(f"  Total not intervened results: {aggregate_gating['total_not_intervened']}")
        print(f"  Overall intervened percentage: {aggregate_gating['overall_intervened_percentage']:.2%}")
        print(f"  Mean intervened per query: {aggregate_gating['mean_intervened_per_query']:.1f}")
        print(f"  Mean not intervened per query: {aggregate_gating['mean_not_intervened_per_query']:.1f}")
        print(f"  Mean gate factor: {aggregate_gating['mean_gate_factor']:.4f}")
        print(f"  Gate factor range: [{aggregate_gating['min_gate_factor']:.4f}, {aggregate_gating['max_gate_factor']:.4f}]")
        print(f"  Gate factor std: {aggregate_gating['std_gate_factor']:.4f}")
    
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

