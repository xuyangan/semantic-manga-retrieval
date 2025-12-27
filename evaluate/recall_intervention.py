#!/usr/bin/env python3
"""
Recall Evaluation Script for Intervention/QCFR Queries

Evaluates recall@k using two search modes:
1. Gated Intervention: Modifies image embeddings using gate-weighted text embeddings
2. QCFR: Two-pass retrieval with hybrid scoring and Rocchio-style query refinement

Supports grid search over hyperparameters for finding optimal configurations.

Usage (Gated Intervention - single run):
    python -m evaluate.recall_intervention --queries queries --mode gated \\
        --beta 1.0 --gamma 1.0 \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/intervention

Usage (Gated Intervention - grid search):
    python -m evaluate.recall_intervention --queries queries --mode gated \\
        --betas 0.5 1.0 1.5 2.0 --gammas 0.5 1.0 2.0 \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/intervention_grid

Usage (QCFR - single run):
    python -m evaluate.recall_intervention --queries queries --mode qcfr \\
        --alpha 0.5 --m-img 800 --l-pos 30 --b 0.35 --c 0.14 \\
        --image-index indexes/image --text-index indexes/text \\
        --output results/qcfr

Usage (QCFR - grid search):
    python -m evaluate.recall_intervention --queries queries --mode qcfr \
        --alphas 0.3 0.5 0.8\
        --m-imgs 100 200 300 \
        --l-pos-values 20 30 50 \
        --bs 0.2 0.35 0.5 \
        --cs 0.1 0.2 0.3 \
        --image-index final_dataset_embeddings/faiss_index --text-index final_dataset_text_embeddings/faiss_index \
        --output results/qcfr_grid

Grid Search Parameters:
    Gated Mode:
        --beta/--betas    Intervention strength (single or multiple values)
        --gamma/--gammas  Gate steepness (single or multiple values)
    
    QCFR Mode (5 tunable parameters):
        --alpha/--alphas      Hybrid score weight (default: 0.5)
        --m-img/--m-imgs      Image candidates in Pass-1 (default: 800, m_txt = 3*m_img)
        --l-pos/--l-pos-values  Pseudo-positive/negative count (default: 30, l_neg = l_pos)
        --b/--bs              Rocchio positive feedback weight (default: 0.35)
        --c/--cs              Rocchio negative feedback weight (default: 0.14)

QCFR Fixed Parameters:
    --a           Rocchio original query weight (default: 1.0)
    --d           Rocchio text query weight (default: 0.21)
    --no-true-max-pooling  Disable true max over all text embeddings per page (faster)
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
from intervention.qcfr import qcfr_search, qcfr_search_with_description
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
    Perform gated intervention search with image embedding modification.
    
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
        image_index_dir: Path to image FAISS index directory
        text_index_dir: Path to text FAISS index directory
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


def qcfr_search_with_cached_description(
    image_path: Path,
    user_query: str,
    image_index_dir: Path,
    text_index_dir: Path,
    m_img: int = 800,
    m_txt: int = 2000,
    l_pos: int = 30,
    l_neg: int = 30,
    alpha: float = 0.5,
    a: float = 1.0,
    b: float = 0.35,
    c: float = 0.14,
    d: float = 0.21,
    k: int = 50,
    true_max_pooling: bool = True,
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
    Perform QCFR search with two-pass retrieval and Rocchio-style query refinement.
    
    Pipeline:
    1. Generate LLM description from user query (image + text)
    2. Generate CLIP embeddings (image and text)
    3. Pass-1: Retrieve candidates from both image and text channels
    4. Compute hybrid scores with max-pooled text per page
    5. Select pseudo-positives (top) and pseudo-negatives (bottom)
    6. Compute feedback centroids using softmax weights
    7. Refine query using Rocchio-style update: q' = normalize(a*q + b*pos - c*neg + d*tq)
    8. Pass-2: Full re-search with refined query
    
    Args:
        image_path: Path to query image
        user_query: User's text query (e.g., "the male character")
        image_index_dir: Path to image FAISS index directory
        text_index_dir: Path to text FAISS index directory
        m_img: Number of image candidates in Pass-1
        m_txt: Number of text candidates in Pass-1 (before page aggregation)
        l_pos: Number of pseudo-positives
        l_neg: Number of pseudo-negatives
        alpha: Hybrid score weight (0=image only, 1=text only)
        a, b, c, d: Rocchio parameters
        k: Number of final results to return
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
            print("\n[1/6] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (only if not provided)
    if model is None or preprocess is None or tokenizer is None:
        if verbose:
            print("\n[2/6] Loading CLIP model and generating embeddings...")
        model, preprocess, tokenizer, device = load_clip_model(device)
    elif verbose:
        print("\n[2/6] Using provided CLIP model for generating embeddings...")
    
    query_image_embedding = encode_image(model, preprocess, device, image_path)
    query_text_embedding = encode_text(model, tokenizer, device, description)
    if verbose:
        print(f"  Query image embedding: {query_image_embedding.shape}")
        print(f"  Query text embedding: {query_text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (only if not provided)
    if image_index is None or image_id_to_meta is None:
        if verbose:
            print("\n[3/6] Loading FAISS indexes...")
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    elif verbose:
        print("\n[3/6] Using provided image index...")
    
    if text_index is None or text_id_to_meta is None:
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    elif verbose:
        print("  Using provided text index...")
    
    # Step 4-8: Run QCFR search
    if verbose:
        print("\n[4-6/6] Running QCFR search...")
    
    qcfr_result = qcfr_search(
        query_image_embedding=query_image_embedding,
        query_text_embedding=query_text_embedding,
        image_index=image_index,
        image_id_to_meta=image_id_to_meta,
        text_index=text_index,
        text_id_to_meta=text_id_to_meta,
        m_img=m_img,
        m_txt=m_txt,
        l_pos=l_pos,
        l_neg=l_neg,
        alpha=alpha,
        a=a,
        b=b,
        c=c,
        d=d,
        k=k,
        true_max_pooling=true_max_pooling,
        verbose=verbose
    )
    
    final_results = qcfr_result['final_results']
    
    if verbose:
        pos_in_results = sum(1 for r in final_results if r.get('is_pseudo_positive', False))
        neg_in_results = sum(1 for r in final_results if r.get('is_pseudo_negative', False))
        print(f"\n  QCFR summary:")
        print(f"    Union candidates: {qcfr_result['num_union_candidates']}")
        print(f"    Pseudo-positives in results: {pos_in_results}/{len(final_results)}")
        print(f"    Pseudo-negatives in results: {neg_in_results}/{len(final_results)}")
    
    return {
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "params": {
            "m_img": m_img,
            "m_txt": m_txt,
            "l_pos": l_pos,
            "l_neg": l_neg,
            "alpha": alpha,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "true_max_pooling": true_max_pooling,
        },
        "k": k,
        "query_image_embedding": query_image_embedding,
        "query_text_embedding": query_text_embedding,
        "num_image_candidates": qcfr_result['num_image_candidates'],
        "num_text_candidates_raw": qcfr_result['num_text_candidates_raw'],
        "num_text_pages": qcfr_result['num_text_pages'],
        "num_union_candidates": qcfr_result['num_union_candidates'],
        "num_pseudo_positives": len(qcfr_result['positive_pages']),
        "num_pseudo_negatives": len(qcfr_result['negative_pages']),
        "positive_pages": qcfr_result['positive_pages'],
        "negative_pages": qcfr_result['negative_pages'],
        "final_results": final_results,
    }


def evaluate_single_query_qcfr(
    query_folder: Path,
    image_index_dir: Path,
    text_index_dir: Path,
    m_img: int,
    m_txt: int,
    l_pos: int,
    l_neg: int,
    alpha: float,
    a: float,
    b: float,
    c: float,
    d: float,
    k_values: List[int],
    map_k_values: List[int] = [10, 20, 30, 50],
    true_max_pooling: bool = True,
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
    Evaluate recall for a single query using QCFR search.
    
    Args:
        query_folder: Path to query folder (contains query.png, text.txt, labels.txt)
        image_index_dir: Path to image FAISS index directory
        text_index_dir: Path to text FAISS index directory
        m_img: Number of image candidates in Pass-1
        m_txt: Number of text candidates in Pass-1
        l_pos: Number of pseudo-positives
        l_neg: Number of pseudo-negatives
        alpha: Hybrid score weight
        a, b, c, d: Rocchio parameters
        k_values: List of k values for recall@k
        map_k_values: List of k values for mAP@k
        true_max_pooling: If True, compute true max over ALL text embeddings per page
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
        user_query = ""
    
    # Load cached LLM description or generate and save it
    cached_description = load_llm_description(query_folder)
    if cached_description is None:
        try:
            cached_description = encode_query_llm(query_image, user_query, verbose=False)
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
    
    # Run QCFR search
    try:
        max_k = max(k_values)
        
        result = qcfr_search_with_cached_description(
            image_path=query_image,
            user_query=user_query,
            image_index_dir=image_index_dir,
            text_index_dir=text_index_dir,
            m_img=m_img,
            m_txt=m_txt,
            l_pos=l_pos,
            l_neg=l_neg,
            alpha=alpha,
            a=a,
            b=b,
            c=c,
            d=d,
            k=max_k,
            true_max_pooling=true_max_pooling,
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
            "error": f"Error running QCFR search: {e}",
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
    for res in search_results:
        if match_result_to_ground_truth(res, ground_truth, include_page_key=True, include_source_file=False):
            all_relevant.append(res)
    
    # Compute QCFR statistics
    pos_count = sum(1 for r in search_results if r.get('is_pseudo_positive', False))
    neg_count = sum(1 for r in search_results if r.get('is_pseudo_negative', False))
    qcfr_stats = {
        "num_pseudo_positives_in_results": pos_count,
        "num_pseudo_negatives_in_results": neg_count,
        "pseudo_positive_percentage": pos_count / len(search_results) if len(search_results) > 0 else 0.0,
        "num_union_candidates": result.get('num_union_candidates', 0),
    }
    
    # Print QCFR summary for this query (use tqdm.write to avoid progress bar interference)
    if not verbose:
        query_name = query_folder.name
        tqdm.write(f"  {query_name}: {pos_count}/{len(search_results)} pseudo-positives in results")
    
    return {
        "query_folder": str(query_folder),
        "query_image": str(query_image),
        "user_query": user_query,
        "llm_description": result.get('llm_description', ''),
        "params": result.get('params', {}),
        "total_ground_truth": len(ground_truth),
        "total_retrieved": len(search_results),
        "relevant_retrieved": len(all_relevant),
        "recall_metrics": recall_metrics,
        "map_metrics": map_metrics,
        "all_retrieved": search_results,
        "relevant_results": all_relevant,
        "qcfr_stats": qcfr_stats,
        "num_image_candidates": result.get('num_image_candidates', 0),
        "num_text_candidates_raw": result.get('num_text_candidates_raw', 0),
        "num_text_pages": result.get('num_text_pages', 0),
    }


def evaluate_single_query(
    query_folder: Path,
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
    Evaluate recall for a single query using gated intervention search.
    
    Args:
        query_folder: Path to query folder (contains query.png, text.txt, labels.txt)
        image_index_dir: Path to image FAISS index directory
        text_index_dir: Path to text FAISS index directory
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
        description="Evaluate recall@k for intervention/QCFR queries with grid search support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gated intervention - single run
  python -m evaluate.recall_intervention --queries queries --mode gated \\
      --beta 1.0 --gamma 1.0 \\
      --image-index indexes/image --text-index indexes/text --output results/gated
  
  # Gated intervention - grid search
  python -m evaluate.recall_intervention --queries queries --mode gated \\
      --betas 0.5 1.0 1.5 2.0 --gammas 0.5 1.0 2.0 \\
      --image-index indexes/image --text-index indexes/text --output results/gated_grid
  
  # QCFR - single run
  python -m evaluate.recall_intervention --queries queries --mode qcfr \\
      --alpha 0.5 --m-img 800 --l-pos 30 --b 0.35 --c 0.14 \\
      --image-index indexes/image --text-index indexes/text --output results/qcfr
  
  # QCFR - grid search (5 parameters: alpha, m_img, l_pos, b, c)
  python -m evaluate.recall_intervention --queries queries --mode qcfr \\
      --alphas 0.3 0.5 0.7 \\
      --m-imgs 300 500 800 \\
      --l-pos-values 20 30 50 \\
      --bs 0.2 0.35 0.5 \\
      --cs 0.1 0.14 0.2 \\
      --image-index indexes/image --text-index indexes/text --output results/qcfr_grid
        """
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        required=True,
        help="Path to queries directory containing query folders (e.g., a_queries/text)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gated", "qcfr"],
        default="gated",
        help="Search mode: 'gated' (original intervention) or 'qcfr' (two-pass Rocchio) (default: gated)",
    )
    
    # Gated intervention parameters (single or grid search)
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Single beta value for gated intervention (use --betas for grid search)",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs='+',
        default=None,
        help="Multiple beta values for grid search (e.g., 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Single gamma value for gated intervention (use --gammas for grid search)",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs='+',
        default=None,
        help="Multiple gamma values for grid search (e.g., 0.5 1.0 2.0)",
    )
    
    # QCFR parameters (single or grid search)
    parser.add_argument(
        "--m-img",
        type=int,
        default=None,
        help="Single m_img value (use --m-imgs for grid search, default: 800)",
    )
    parser.add_argument(
        "--m-imgs",
        type=int,
        nargs='+',
        default=None,
        help="Multiple m_img values for grid search (e.g., 300 500 800). m_txt = 3 * m_img",
    )
    parser.add_argument(
        "--l-pos",
        type=int,
        default=None,
        help="Single l_pos value (use --l-pos-values for grid search, default: 30). l_neg = l_pos",
    )
    parser.add_argument(
        "--l-pos-values",
        type=int,
        nargs='+',
        default=None,
        help="Multiple l_pos values for grid search (e.g., 20 30 50). l_neg = l_pos by default",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Single alpha value for QCFR (use --alphas for grid search, default: 0.5)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=None,
        help="Multiple alpha values for grid search (e.g., 0.3 0.5 0.7)",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=1.0,
        help="QCFR: Rocchio parameter for original query weight (default: 1.0)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Single b value for QCFR positive feedback (use --bs for grid search, default: 0.35)",
    )
    parser.add_argument(
        "--bs",
        type=float,
        nargs='+',
        default=None,
        help="Multiple b values for grid search (e.g., 0.2 0.35 0.5)",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Single c value for QCFR negative feedback (use --cs for grid search, default: 0.14)",
    )
    parser.add_argument(
        "--cs",
        type=float,
        nargs='+',
        default=None,
        help="Multiple c values for grid search (e.g., 0.1 0.14 0.2)",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=0.21,
        help="QCFR: Rocchio parameter for text query weight (default: 0.21)",
    )
    parser.add_argument(
        "--no-true-max-pooling",
        action="store_true",
        help="QCFR: Disable true max-pooling over all text embeddings per page (faster but less accurate)",
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
    
    # Determine parameter values (grid search vs single values)
    if args.mode == "gated":
        # Gated mode: beta and gamma
        if args.betas:
            beta_values = args.betas
        elif args.beta is not None:
            beta_values = [args.beta]
        else:
            beta_values = [1.0]  # Default
        
        if args.gammas:
            gamma_values = args.gammas
        elif args.gamma is not None:
            gamma_values = [args.gamma]
        else:
            gamma_values = [1.0]  # Default
        
        is_grid_search = len(beta_values) > 1 or len(gamma_values) > 1
        num_combinations = len(beta_values) * len(gamma_values)
    else:  # QCFR mode
        # QCFR mode: alpha, m_img, l_pos, b, c
        if args.alphas:
            alpha_values = args.alphas
        elif args.alpha is not None:
            alpha_values = [args.alpha]
        else:
            alpha_values = [0.5]  # Default
        
        if args.m_imgs:
            m_img_values = args.m_imgs
        elif args.m_img is not None:
            m_img_values = [args.m_img]
        else:
            m_img_values = [800]  # Default
        
        if args.l_pos_values:
            l_pos_values = args.l_pos_values
        elif args.l_pos is not None:
            l_pos_values = [args.l_pos]
        else:
            l_pos_values = [30]  # Default
        
        if args.bs:
            b_values = args.bs
        elif args.b is not None:
            b_values = [args.b]
        else:
            b_values = [0.35]  # Default
        
        if args.cs:
            c_values = args.cs
        elif args.c is not None:
            c_values = [args.c]
        else:
            c_values = [0.14]  # Default
        
        # Validate alphas
        for alpha in alpha_values:
            if not 0 <= alpha <= 1:
                print(f"Error: alpha {alpha} must be between 0 and 1")
                return
        
        is_grid_search = (len(alpha_values) > 1 or len(m_img_values) > 1 or 
                         len(l_pos_values) > 1 or len(b_values) > 1 or len(c_values) > 1)
        num_combinations = len(alpha_values) * len(m_img_values) * len(l_pos_values) * len(b_values) * len(c_values)
    
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
    
    # Print header based on mode
    if args.mode == "qcfr":
        print(f"\nQCFR Recall Evaluation (Two-Pass Rocchio)")
        print("="*60)
        print(f"Mode: QCFR (Query-Conditioned Feedback Re-ranking)")
        if is_grid_search:
            print(f"Grid Search Parameters:")
            print(f"  alpha: {alpha_values}")
            print(f"  m_img: {m_img_values} (m_txt = 3 * m_img)")
            print(f"  l_pos: {l_pos_values} (l_neg = l_pos)")
            print(f"  b: {b_values}")
            print(f"  c: {c_values}")
            print(f"Total combinations: {num_combinations}")
        else:
            print(f"Hybrid weight: alpha={alpha_values[0]}")
            print(f"Pass-1 candidates: m_img={m_img_values[0]}, m_txt={3*m_img_values[0]}")
            print(f"Pseudo-labels: l_pos=l_neg={l_pos_values[0]}")
            print(f"Rocchio b (positive): {b_values[0]}, c (negative): {c_values[0]}")
        print(f"Fixed params: a={args.a}, d={args.d}")
        print(f"True max-pooling: {not args.no_true_max_pooling}")
    else:
        print(f"\nGated Intervention Recall Evaluation")
        print("="*60)
        print(f"Mode: Gated Intervention")
        if is_grid_search:
            print(f"Grid Search: beta={beta_values}, gamma={gamma_values}")
            print(f"Total combinations: {num_combinations}")
        else:
            print(f"Beta: {beta_values[0]} (intervention strength)")
            print(f"Gamma: {gamma_values[0]} (gate steepness)")
        print(f"Initial candidates (m): {args.m}")
    
    print(f"Queries directory: {queries_dir}")
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
    
    # Grid search storage
    grid_results = {}  # param_tuple -> results dict
    
    # Print grid search info if applicable
    if is_grid_search:
        print(f"\n{'='*60}")
        print("GRID SEARCH MODE")
        if args.mode == "gated":
            print(f"Beta values: {beta_values}")
            print(f"Gamma values: {gamma_values}")
        else:
            print(f"Alpha values: {alpha_values}")
            print(f"M_img values: {m_img_values} (m_txt = 3*m_img)")
            print(f"L_pos values: {l_pos_values} (l_neg = l_pos)")
            print(f"B values: {b_values}")
            print(f"C values: {c_values}")
        print(f"Total combinations: {num_combinations}")
        print(f"{'='*60}")
    
    # Build parameter combinations
    if args.mode == "gated":
        param_combinations = [(beta, gamma) for beta in beta_values for gamma in gamma_values]
        param_names = ("beta", "gamma")
    else:
        # QCFR: 5 parameters - alpha, m_img, l_pos, b, c
        from itertools import product
        param_combinations = list(product(alpha_values, m_img_values, l_pos_values, b_values, c_values))
        param_names = ("alpha", "m_img", "l_pos", "b", "c")
    
    # Run grid search (or single evaluation)
    combo_idx = 0
    for params in param_combinations:
        combo_idx += 1
        
        if args.mode == "gated":
            param1, param2 = params
            combo_name = f"beta_{param1}_gamma_{param2}"
            combo_desc = f"beta={param1}, gamma={param2}"
        else:
            alpha, m_img, l_pos, b, c = params
            combo_name = f"a{alpha}_m{m_img}_l{l_pos}_b{b}_c{c}"
            combo_desc = f"α={alpha}, m_img={m_img}, l_pos={l_pos}, b={b}, c={c}"
        
        if is_grid_search:
            print(f"\n{'='*60}")
            print(f"Running combination {combo_idx}/{num_combinations}: {combo_desc}")
            print(f"{'='*60}")
        
        # Evaluate each query with timing
        all_results = []
        query_times = []
        mode_desc = "QCFR queries" if args.mode == "qcfr" else "Gated queries"
        
        progress_desc = f"[{combo_idx}/{num_combinations}]" if is_grid_search else mode_desc
        for query_folder in tqdm(query_folders, desc=progress_desc):
            start_time = time.time()
            
            if args.mode == "qcfr":
                # m_txt = 3 * m_img, l_neg = l_pos
                result = evaluate_single_query_qcfr(
                    query_folder=query_folder,
                    image_index_dir=image_index_dir,
                    text_index_dir=text_index_dir,
                    m_img=m_img,
                    m_txt=3 * m_img,  # m_txt = 3 * m_img
                    l_pos=l_pos,
                    l_neg=l_pos,  # l_neg = l_pos
                    alpha=alpha,
                    a=args.a,
                    b=b,
                    c=c,
                    d=args.d,
                    k_values=args.k_values,
                    map_k_values=args.map_k_values,
                    true_max_pooling=not args.no_true_max_pooling,
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
            else:
                param1, param2 = params
                result = evaluate_single_query(
                    query_folder=query_folder,
                    image_index_dir=image_index_dir,
                    text_index_dir=text_index_dir,
                    gamma=param2,  # From grid search
                    beta=param1,   # From grid search
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
    
        # Compute averages for this combination
        averages = compute_average_metrics(all_results)
        
        # Compute aggregate statistics based on mode
        aggregate_stats = {}
        if args.mode == "qcfr":
            valid_results = [r for r in all_results if 'error' not in r and 'qcfr_stats' in r]
            if valid_results:
                all_pos_counts = [r['qcfr_stats']['num_pseudo_positives_in_results'] for r in valid_results]
                all_neg_counts = [r['qcfr_stats']['num_pseudo_negatives_in_results'] for r in valid_results]
                all_union_counts = [r['qcfr_stats']['num_union_candidates'] for r in valid_results]
                
                aggregate_stats = {
                    "mode": "qcfr",
                    "total_pseudo_positives_in_results": int(sum(all_pos_counts)),
                    "mean_pseudo_positives_per_query": float(np.mean(all_pos_counts)) if all_pos_counts else 0.0,
                    "total_pseudo_negatives_in_results": int(sum(all_neg_counts)),
                    "mean_pseudo_negatives_per_query": float(np.mean(all_neg_counts)) if all_neg_counts else 0.0,
                    "mean_union_candidates": float(np.mean(all_union_counts)) if all_union_counts else 0.0,
                }
        else:
            valid_results = [r for r in all_results if 'error' not in r and 'gating_stats' in r]
            if valid_results:
                all_intervened_counts = [r['gating_stats']['num_intervened'] for r in valid_results]
                all_not_intervened_counts = [r['gating_stats']['num_not_intervened'] for r in valid_results]
                all_gate_factors = []
                for r in valid_results:
                    all_gate_factors.extend([res.get('gate_factor', 0.0) for res in r.get('all_retrieved', []) if res.get('had_intervention', False)])
                
                aggregate_stats = {
                    "mode": "gated",
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
        
        # Store results for grid search
        if args.mode == "qcfr":
            grid_results[params] = {
                'all_results': all_results,
                'averages': averages,
                'query_times': query_times,
                'aggregate_stats': aggregate_stats,
                'alpha': alpha,
                'm_img': m_img,
                'l_pos': l_pos,
                'b': b,
                'c': c,
            }
        else:
            grid_results[params] = {
                'all_results': all_results,
                'averages': averages,
                'query_times': query_times,
                'aggregate_stats': aggregate_stats,
                'beta': params[0],
                'gamma': params[1],
            }
        
        # Create output subdirectory for this combination (if grid search)
        if is_grid_search:
            combo_output_dir = output_dir / combo_name
        else:
            combo_output_dir = output_dir
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
        if args.mode == "qcfr":
            config = {
                "queries_dir": str(queries_dir),
                "image_index_dir": str(image_index_dir),
                "text_index_dir": str(text_index_dir),
                "mode": "qcfr",
                "m_img": m_img,
                "m_txt": 3 * m_img,
                "l_pos": l_pos,
                "l_neg": l_pos,
                "alpha": alpha,
                "a": args.a,
                "b": b,
                "c": c,
                "d": args.d,
                "k_values": args.k_values,
                "map_k_values": args.map_k_values,
                "num_queries": len(query_folders),
            }
        else:
            config = {
                "queries_dir": str(queries_dir),
                "image_index_dir": str(image_index_dir),
                "text_index_dir": str(text_index_dir),
                "mode": "gated",
                "beta": params[0],
                "gamma": params[1],
                "m": args.m,
                "k_values": args.k_values,
                "map_k_values": args.map_k_values,
                "num_queries": len(query_folders),
            }
        
        output_data = {
            "config": config,
            "individual_results": all_results,
            "average_metrics": averages,
            "aggregate_stats": aggregate_stats,
            "timing_stats": timing_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = combo_output_dir / "recall_results.json"
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if is_grid_search:
            # Print quick summary for this combination
            if 'error' not in averages:
                recall_50 = averages['average_metrics'].get('recall@50', {}).get('mean', 0)
                map_50 = averages['average_metrics'].get('map@50', {}).get('mean', 0)
                print(f"  Recall@50: {recall_50:.4f}, mAP@50: {map_50:.4f}")
        else:
            print(f"\nResults saved to: {results_file}")
        
        # Create summary table for this combination
        create_summary_table(all_results, averages, combo_output_dir, include_user_query=True)
    
    # Grid search comparison summary
    best_params = None
    if is_grid_search:
        print(f"\n{'='*60}")
        print("GRID SEARCH COMPARISON SUMMARY")
        print("="*60)
        
        # Build comparison table data
        comparison_data = []
        for params_key, data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                row = {name: params_key[i] for i, name in enumerate(param_names)}
                # Add all recall and mAP metrics
                for metric_name, stats in avg['average_metrics'].items():
                    row[f'{metric_name}_mean'] = stats['mean']
                    row[f'{metric_name}_std'] = stats['std']
                comparison_data.append(row)
        
        # Save comparison CSV
        comparison_file = output_dir / "grid_search_comparison.csv"
        if comparison_data:
            import csv
            fieldnames = list(param_names) + sorted([k for k in comparison_data[0].keys() if k not in param_names])
            with open(comparison_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in sorted(comparison_data, key=lambda x: tuple(x[n] for n in param_names)):
                    writer.writerow(row)
            print(f"Grid search comparison saved to: {comparison_file}")
        
        # Save full grid results JSON
        if args.mode == "gated":
            grid_summary = {
                "beta_values": beta_values,
                "gamma_values": gamma_values,
                "mode": args.mode,
                "num_combinations": num_combinations,
                "combinations": []
            }
        else:
            grid_summary = {
                "alpha_values": alpha_values,
                "m_img_values": m_img_values,
                "l_pos_values": l_pos_values,
                "b_values": b_values,
                "c_values": c_values,
                "mode": args.mode,
                "num_combinations": num_combinations,
                "combinations": []
            }
        
        for params_key, data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                combo_entry = {name: params_key[i] for i, name in enumerate(param_names)}
                combo_entry["average_metrics"] = avg['average_metrics']
                combo_entry["overall_stats"] = avg['overall_stats']
                grid_summary["combinations"].append(combo_entry)
        
        grid_json_file = output_dir / "grid_search_summary.json"
        with open(grid_json_file, 'w') as f:
            json.dump(grid_summary, f, indent=2)
        print(f"Grid search summary saved to: {grid_json_file}")
        
        # Print comparison table (top 20 by mAP@50)
        sorted_data = sorted(comparison_data, key=lambda x: x.get('map@50_mean', 0), reverse=True)
        print(f"\nTop configurations by mAP@50:")
        if args.mode == "gated":
            print(f"{'beta':<8} {'gamma':<8} {'Recall@50':<12} {'mAP@50':<12}")
            print("-" * 50)
            for row in sorted_data[:20]:
                print(f"{row['beta']:<8.2f} {row['gamma']:<8.2f} "
                      f"{row.get('recall@50_mean', 0):<12.4f} "
                      f"{row.get('map@50_mean', 0):<12.4f}")
        else:
            print(f"{'alpha':<7} {'m_img':<6} {'l_pos':<6} {'b':<6} {'c':<6} {'R@50':<10} {'mAP@50':<10}")
            print("-" * 70)
            for row in sorted_data[:20]:
                print(f"{row['alpha']:<7.2f} {row['m_img']:<6} {row['l_pos']:<6} "
                      f"{row['b']:<6.2f} {row['c']:<6.2f} "
                      f"{row.get('recall@50_mean', 0):<10.4f} "
                      f"{row.get('map@50_mean', 0):<10.4f}")
        
        # Find best hyperparameter configuration (based on mAP@50, with recall@50 as tiebreaker)
        best_score = -1
        best_recall = -1
        
        for params_key, data in grid_results.items():
            avg = data['averages']
            if 'error' not in avg:
                map_50 = avg['average_metrics'].get('map@50', {}).get('mean', 0)
                recall_50 = avg['average_metrics'].get('recall@50', {}).get('mean', 0)
                # Primary: mAP@50, Secondary: recall@50
                if map_50 > best_score or (map_50 == best_score and recall_50 > best_recall):
                    best_score = map_50
                    best_recall = recall_50
                    best_params = params_key
        
        if best_params is not None:
            best_data = grid_results[best_params]
            best_averages = best_data['averages']
            
            print("\n" + "="*60)
            print("BEST HYPERPARAMETER CONFIGURATION")
            print("="*60)
            for i, name in enumerate(param_names):
                print(f"Best {name}: {best_params[i]}")
            print(f"mAP@50: {best_score:.4f}")
            print(f"Recall@50: {best_recall:.4f}")
            
            print("\nAll metrics for best configuration:")
            print(f"{'Metric':<15} {'Mean':<10} {'Std':<10}")
            print("-" * 40)
            for metric_name, stats in sorted(best_averages['average_metrics'].items()):
                print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f}")
            
            # Save best configuration summary
            best_config = {f"best_{name}": best_params[i] for i, name in enumerate(param_names)}
            best_config.update({
                "mode": args.mode,
                "primary_metric": "map@50",
                "primary_score": best_score,
                "secondary_metric": "recall@50", 
                "secondary_score": best_recall,
                "all_metrics": {k: v['mean'] for k, v in best_averages['average_metrics'].items()},
                "overall_stats": best_averages['overall_stats']
            })
            
            best_config_file = output_dir / "best_config.json"
            with open(best_config_file, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"\nBest configuration saved to: {best_config_file}")
            
            # Update variables to use best configuration for visualizations
            all_results = best_data['all_results']
            averages = best_averages
            query_times = best_data['query_times']
            aggregate_stats = best_data['aggregate_stats']
    
    # Create visualizations
    if not args.no_viz:
        print("\nCreating visualizations...")
        
        # Determine best config info for titles
        mode_prefix = "QCFR" if args.mode == "qcfr" else "Intervention"
        mode_color = 'purple' if args.mode == "qcfr" else 'red'
        if is_grid_search and best_params is not None:
            if args.mode == "gated":
                title_suffix = f" (Best: β={best_params[0]}, γ={best_params[1]})"
            else:
                title_suffix = f" (Best: α={best_params[0]}, m={best_params[1]}, l={best_params[2]}, b={best_params[3]}, c={best_params[4]})"
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
            plot_recall_curve(result, curve_file, color=mode_color, title_prefix=f"{mode_prefix}{title_suffix}")
        
        # Aggregate curve
        aggregate_file = curves_dir / "aggregate_all_queries.png"
        plot_aggregate_recall_curves(all_results, averages, aggregate_file, color=mode_color, 
                                     title=f"All {mode_prefix} Queries{title_suffix}")
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
                image_dir=image_dir
            )
        
        print(f"Visualizations saved to: {vis_dir}")
        
        # Save best configuration results JSON (if grid search)
        if is_grid_search and best_params is not None:
            best_results_file = best_dir / "recall_results.json"
            if args.mode == "qcfr":
                best_alpha, best_m_img, best_l_pos, best_b, best_c = best_params
                best_config_data = {
                    "queries_dir": str(queries_dir),
                    "image_index_dir": str(image_index_dir),
                    "text_index_dir": str(text_index_dir),
                    "mode": "qcfr",
                    "m_img": best_m_img,
                    "m_txt": 3 * best_m_img,
                    "l_pos": best_l_pos,
                    "l_neg": best_l_pos,
                    "alpha": best_alpha,
                    "a": args.a,
                    "b": best_b,
                    "c": best_c,
                    "d": args.d,
                    "k_values": args.k_values,
                    "map_k_values": args.map_k_values,
                    "num_queries": len(query_folders),
                    "is_best_from_grid_search": True,
                }
            else:
                best_beta, best_gamma = best_params
                best_config_data = {
                    "queries_dir": str(queries_dir),
                    "image_index_dir": str(image_index_dir),
                    "text_index_dir": str(text_index_dir),
                    "mode": "gated",
                    "beta": best_beta,
                    "gamma": best_gamma,
                    "m": args.m,
                    "k_values": args.k_values,
                    "map_k_values": args.map_k_values,
                    "num_queries": len(query_folders),
                    "is_best_from_grid_search": True,
                }
            
            best_output_data = {
                "config": best_config_data,
                "individual_results": all_results,
                "average_metrics": averages,
                "aggregate_stats": aggregate_stats,
                "timestamp": datetime.now().isoformat()
            }
            with open(best_results_file, 'w') as f:
                json.dump(best_output_data, f, indent=2)
            
            # Create summary table for best config
            create_summary_table(all_results, averages, best_dir, include_user_query=True)
            print(f"Best configuration results saved to: {best_dir}")
    
    # Print final summary
    print("\n" + "="*60)
    if args.mode == "qcfr":
        print("RECALL EVALUATION SUMMARY (QCFR)")
    else:
        print("RECALL EVALUATION SUMMARY (Gated Intervention)")
    print("="*60)
    
    if is_grid_search and best_params is not None:
        for i, name in enumerate(param_names):
            print(f"Best {name}: {best_params[i]}")
    
    if 'error' not in averages:
        print(f"Number of queries: {averages['num_queries']}")
        if averages['num_failed'] > 0:
            print(f"Failed queries: {averages['num_failed']}")
        
        print("\nAverage Recall@K Metrics (Best Config):")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 70)
        
        # Print recall metrics
        for metric_name, stats in sorted(averages['average_metrics'].items()):
            if metric_name.startswith('recall@'):
                print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                      f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
        
        # Print mAP metrics
        print("\nAverage mAP@K Metrics (Best Config):")
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
    
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

