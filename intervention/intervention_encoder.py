#!/usr/bin/env python3
"""
Intervention Encoder

Implements gated intervention search: modifies image embeddings using
gate-weighted text embeddings for non-linear transformation.
Uses z-score normalized similarity for gate computation.

Formula: v_i = normalize(v_i + β * g_i * t_q)
where g_i = sigmoid(γ * z_i) and z_i = (s_i - mean) / std

Usage:
    python -m intervention.intervention_encoder --image query.jpg --text "the male character" \\
        --text-db final_dataset_text_embeddings --image-db final_dataset_embeddings \\
        -m 50 -k 10 --gamma 1.0 --beta 1.0
"""

import argparse
from pathlib import Path

import numpy as np

# Reuse components from late_fusion
from late_fusion.llm_encoder import encode_query_llm
from late_fusion.clip_encoder import load_clip_model, encode_image, encode_text
from late_fusion.faiss_search import (
    load_faiss_index,
    search_index,
    deduplicate_text_candidates,
)

from .gate import compute_gate_for_candidates
from .transform import build_intervention_index, build_page_to_faiss_mapping


def visualize_intervention_results(
    query_image_path: Path,
    results: list[dict],
    title: str = "Intervention Search Results",
    output_path: Path = None,
    show: bool = True
):
    """
    Visualize intervention search results with query image and similar images.
    
    Args:
        query_image_path: Path to query image
        results: List of search results with metadata
        title: Title for the figure
        output_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    n_results = len(results)
    if n_results == 0:
        print("No results to visualize")
        return
    
    # Create figure: query on left, results on right
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4.5))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Plot query image
    query_img = Image.open(query_image_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Plot result images
    for i, r in enumerate(results):
        ax = axes[i + 1]
        
        # Try to find and load result image
        img_path = r.get('path')
        if not img_path:
            # Try to construct path from metadata
            manga = r.get('manga', '')
            chapter = r.get('chapter', '')
            page = r.get('page', '')
            if manga and chapter and page:
                possible_paths = [
                    Path(f"final_dataset/{manga}/{chapter}/{page}.png"),
                    Path(f"final_dataset/{manga}/{chapter}/{page}.jpg"),
                ]
                for p in possible_paths:
                    if p.exists():
                        img_path = str(p)
                        break
        
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=8)
                ax.set_facecolor('#f0f0f0')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=8)
            ax.set_facecolor('#f0f0f0')
        
        # Title with scores and metadata
        manga = r.get('manga', r.get('page_key', 'unknown'))
        if isinstance(manga, str) and len(manga) > 12:
            manga = manga[:9] + "..."
        
        sim = r.get('similarity', 0)
        gate = r.get('gate_factor', 0)
        intervened = "✓" if r.get('had_intervention') else "✗"
        
        ax.set_title(f"#{r['rank']} S:{sim:.2f}\nG:{gate:.2f} {intervened}\n{manga}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def intervention_search(
    image_path: Path,
    user_query: str,
    text_db: Path,
    image_db: Path,
    m: int = 50,
    k: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
    verbose: bool = False
) -> dict:
    """
    Perform intervention search with gated image embedding modification.
    
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
        text_db: Path to text embeddings directory
        image_db: Path to image embeddings directory
        m: Number of initial text candidates to retrieve
        k: Number of final results to return
        gamma: Gate steepness parameter (higher = sharper transition)
        beta: Intervention strength scaling factor
        verbose: Print detailed info
    
    Returns:
        Dict with all results and intermediate data
    """
    print("\n" + "=" * 60)
    print("  Intervention Search")
    print("=" * 60)
    print(f"  Params: gamma={gamma}, beta={beta}")
    print(f"  Retrieve m={m} text candidates, return k={k} results")
    
    # Step 1: Generate LLM description (same as late_fusion)
    print("\n[1/8] Generating LLM description...")
    description = encode_query_llm(image_path, user_query, verbose)
    print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (before FAISS to avoid macOS segfault)
    print("\n[2/8] Loading CLIP model and generating embeddings...")
    model, preprocess, tokenizer, device = load_clip_model()
    
    query_image_embedding = encode_image(model, preprocess, device, image_path)
    query_text_embedding = encode_text(model, tokenizer, device, description)
    print(f"  Query image embedding: {query_image_embedding.shape}")
    print(f"  Query text embedding: {query_text_embedding.shape}")
    
    # Step 3: Load FAISS indexes
    print("\n[3/8] Loading FAISS indexes...")
    text_index, text_id_to_meta, _ = load_faiss_index(text_db)
    print(f"  Text index: {text_index.ntotal} vectors")
    
    image_index, image_id_to_meta, _ = load_faiss_index(image_db)
    print(f"  Image index: {image_index.ntotal} vectors")
    
    # Step 4: Search text DB
    print(f"\n[4/8] Searching text DB (m={m})...")
    text_candidates_raw = search_index(text_index, text_id_to_meta, query_text_embedding, k=m)
    print(f"  Found {len(text_candidates_raw)} text candidates")
    
    # Deduplicate by page (keep max similarity)
    text_candidates = deduplicate_text_candidates(text_candidates_raw)
    print(f"  Unique pages: {len(text_candidates)}")
    
    # Step 5: Compute gate factors (using z-score normalization)
    print(f"\n[5/8] Computing gate factors with z-score (gamma={gamma})...")
    candidates_with_gates = compute_gate_for_candidates(text_candidates, gamma)
    
    similarities = [c["similarity"] for c in candidates_with_gates]
    z_scores = [c["z_score"] for c in candidates_with_gates]
    gates = [c["gate_factor"] for c in candidates_with_gates]
    
    print(f"  Similarity: mean={np.mean(similarities):.4f}, std={np.std(similarities):.4f}")
    print(f"  Z-score range: [{min(z_scores):.4f}, {max(z_scores):.4f}]")
    print(f"  Gate range: [{min(gates):.4f}, {max(gates):.4f}]")
    print(f"  Mean gate: {np.mean(gates):.4f}")
    
    if verbose:
        print("\n  Top 10 intervened pages:")
        for c in candidates_with_gates[:10]:
            print(f"    {c.get('page_key', '?')}: sim={c['similarity']:.4f}, gate={c['gate_factor']:.4f}")
    
    # Step 6: Build page to FAISS index mapping
    print("\n[6/8] Building page to index mapping...")
    page_to_faiss_idx = build_page_to_faiss_mapping(image_id_to_meta)
    print(f"  Mapped {len(page_to_faiss_idx)} pages")
    
    # Step 7: Apply intervention and build new index
    print(f"\n[7/8] Applying intervention transform (beta={beta})...")
    intervention_index, _ = build_intervention_index(
        image_index,
        image_id_to_meta,
        query_text_embedding,
        candidates_with_gates,
        page_to_faiss_idx,
        beta=beta
    )
    print(f"  Created intervention index: {intervention_index.ntotal} vectors")
    
    # Step 8: Search with query image
    print(f"\n[8/8] Searching intervention index (k={k})...")
    final_results = search_index(intervention_index, image_id_to_meta, query_image_embedding, k=k)
    print(f"  Found {len(final_results)} results")
    
    # Add intervention info to results
    # Build lookup for faster matching
    gate_lookup = {c.get("page_key"): c for c in candidates_with_gates if c.get("page_key")}
    
    if verbose:
        print("\n  Sample gate_lookup keys:", list(gate_lookup.keys())[:5])
    
    for result in final_results:
        from late_fusion.faiss_search import get_page_key
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
    
    if verbose:
        print("  Sample result page_keys:", [r["page_key"] for r in final_results[:5]])
    
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


def print_intervention_results(results: list[dict], top_k: int = None):
    """Print intervention search results."""
    display = results[:top_k] if top_k else results
    
    print(f"\nFinal Results ({len(display)} shown):")
    print("-" * 100)
    print(f"{'Rank':<5} {'Score':<10} {'Z-score':<10} {'Gate':<8} {'Intervened':<12} {'Page'}")
    print("-" * 100)
    
    for r in display:
        page_key = r.get("page_key", "?")
        if len(page_key) > 35:
            page_key = "..." + page_key[-32:]
        
        intervened = "Yes" if r.get("had_intervention") else "No"
        gate = r.get("gate_factor", 0.0)
        z_score = r.get("z_score", 0.0)
        
        print(f"{r['rank']:<5} {r['similarity']:<10.4f} {z_score:<10.4f} {gate:<8.4f} {intervened:<12} {page_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Intervention Encoder - Gated Image Embedding Modification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic search
    python -m intervention.intervention_encoder --image page.jpg \\
        --text "find similar with the male character" \\
        --text-db final_dataset_text_embeddings --image-db final_dataset_embeddings \\
        -m 50 -k 10 --gamma 1.0 --beta 1.0
    
    # With visualization
    python -m intervention.intervention_encoder --image page.jpg \\
        --text "find similar with the male character" \\
        --text-db final_dataset_text_embeddings --image-db final_dataset_embeddings \\
        -m 50 -k 10 --gamma 1.0 --beta 1.0 --visualize
    
    # Save visualization to file
    python -m intervention.intervention_encoder --image page.jpg \\
        --text "find similar with the male character" \\
        --text-db final_dataset_text_embeddings --image-db final_dataset_embeddings \\
        -m 50 -k 10 --gamma 1.0 --beta 1.0 --save-viz results.png
        """
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to query image")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text query")
    parser.add_argument("--text-db", type=str, required=True, help="Path to text embeddings (with faiss_index)")
    parser.add_argument("--image-db", type=str, required=True, help="Path to image embeddings (with faiss_index)")
    parser.add_argument("-m", type=int, default=50, help="Number of text candidates (default: 50)")
    parser.add_argument("-k", type=int, default=10, help="Number of final results (default: 10)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gate steepness (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0, help="Intervention strength (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Display visualization of results")
    parser.add_argument("--save-viz", type=str, help="Save visualization to file")
    
    args = parser.parse_args()
    image_path = Path(args.image)
    text_db = Path(args.text_db)
    image_db = Path(args.image_db)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    if not text_db.exists():
        print(f"Error: Text DB not found: {text_db}")
        return 1
    if not image_db.exists():
        print(f"Error: Image DB not found: {image_db}")
        return 1
    
    try:
        result = intervention_search(
            image_path,
            args.text,
            text_db,
            image_db,
            m=args.m,
            k=args.k,
            gamma=args.gamma,
            beta=args.beta,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("  Results Summary")
        print("=" * 60)
        print(f"\nLLM Description: {result['llm_description']}")
        print(f"\nParams: gamma={result['gamma']}, beta={result['beta']}")
        print(f"Text candidates: {result['num_text_candidates_raw']} raw -> {result['num_text_candidates_dedup']} unique")
        print(f"Images with intervention: {result['num_intervened']}")
        
        print_intervention_results(result['final_results'])
        
        # Show how many results had intervention
        intervened_count = sum(1 for r in result['final_results'] if r.get('had_intervention'))
        print(f"\n{intervened_count}/{len(result['final_results'])} results had intervention applied")
        
        # Visualization
        if args.visualize or args.save_viz:
            visualize_intervention_results(
                image_path,
                result['final_results'],
                title=f"Intervention Search (γ={result['gamma']}, β={result['beta']})",
                output_path=Path(args.save_viz) if args.save_viz else None,
                show=args.visualize
            )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
