#!/usr/bin/env python3
"""
Late Fusion Query Encoder

Main entry point for late fusion search. Combines LLM-generated descriptions
with CLIP embeddings to search and rerank results.

Usage:
    # Basic: Generate description only
    python late_fusion/late_fusion_encoder.py --image query.jpg --text "the character at the left"
    
    # Late Fusion: Full pipeline with reranking
    python late_fusion/late_fusion_encoder.py --image query.jpg --text "the tall warrior" \\
        --late-fusion-one --image-db final_dataset_embeddings --text-db final_dataset_text_embeddings \\
        -m 50 -k 10 --alpha 0.5
"""

import argparse
from pathlib import Path

from .llm_encoder import encode_query_llm
from .clip_encoder import load_clip_model, encode_image, encode_text
from .faiss_search import (
    load_faiss_index,
    search_index,
    build_text_lookup,
    build_image_lookup,
    compute_text_similarities,
    compute_image_similarities,
    deduplicate_text_candidates,
)
from .fusion import late_fusion_rerank, late_fusion_rerank_two, print_fusion_results


def visualize_fusion_results(
    query_image_path: Path,
    results: list[dict],
    title: str = "Late Fusion Results",
    output_path: Path = None,
    show: bool = True
):
    """
    Visualize late fusion results with query image and similar images.
    
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
        
        # For fusion_two: image metadata is nested in 'image_meta'
        if not img_path and r.get('image_meta'):
            img_path = r['image_meta'].get('path')
        
        if not img_path:
            # Try to construct path from metadata (direct or from image_meta)
            meta = r.get('image_meta', r)
            manga = meta.get('manga', '')
            chapter = meta.get('chapter', '')
            page = meta.get('page', '')
            if manga and chapter and page:
                # Try common locations
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
        meta = r.get('image_meta', r)
        manga = meta.get('manga', r.get('page_key', 'unknown'))
        if isinstance(manga, str) and len(manga) > 15:
            manga = manga[:12] + "..."
        
        combined = r.get('combined_score', r.get('similarity', 0))
        img_sim = r.get('image_similarity', 0)
        txt_sim = r.get('text_similarity', 0)
        
        ax.set_title(f"#{r['rank']} C:{combined:.2f}\nI:{img_sim:.2f} T:{txt_sim:.2f}\n{manga}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def late_fusion_one(
    image_path: Path,
    user_query: str,
    image_db: Path,
    text_db: Path,
    m: int = 50,
    k: int = 10,
    alpha: float = 0.5,
    verbose: bool = False
) -> dict:
    """
    Perform late fusion search with reranking.
    
    Pipeline:
    1. Generate LLM description from user query (image + text)
    2. Generate CLIP embeddings (image and text)
    3. Search image DB for m candidates
    4. For each candidate, find corresponding text embeddings
    5. Compute text similarity scores (max per page)
    6. Rerank using alpha-weighted combination
    7. Return top k results
    
    Args:
        image_path: Path to query image
        user_query: User's text query
        image_db: Path to image embeddings directory
        text_db: Path to text embeddings directory
        m: Number of initial image candidates to retrieve
        k: Number of final results after reranking
        alpha: Weight for image score (0=text only, 1=image only, 0.5=equal)
        verbose: Print detailed info
    
    Returns:
        Dict with all results and intermediate data
    """
    print("\n" + "=" * 60)
    print("  Late Fusion One")
    print("=" * 60)
    print(f"  Alpha: {alpha} (image={alpha:.1%}, text={1-alpha:.1%})")
    print(f"  Retrieve m={m} candidates, rerank to k={k}")
    
    # Step 1: Generate LLM description
    print("\n[1/6] Generating LLM description...")
    description = encode_query_llm(image_path, user_query, verbose)
    print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (before FAISS to avoid macOS segfault)
    print("\n[2/6] Loading CLIP model and generating embeddings...")
    model, preprocess, tokenizer, device = load_clip_model()
    
    image_embedding = encode_image(model, preprocess, device, image_path)
    text_embedding = encode_text(model, tokenizer, device, description)
    print(f"  Image embedding: {image_embedding.shape}")
    print(f"  Text embedding: {text_embedding.shape}")
    
    # Step 3: Load FAISS indexes
    print("\n[3/6] Loading FAISS indexes...")
    image_index, image_id_to_meta, _ = load_faiss_index(image_db)
    print(f"  Image index: {image_index.ntotal} vectors")
    
    text_index, text_id_to_meta, _ = load_faiss_index(text_db)
    print(f"  Text index: {text_index.ntotal} vectors")
    
    # Build text lookup
    text_lookup = build_text_lookup(text_id_to_meta)
    print(f"  Text lookup: {len(text_lookup)} unique pages")
    
    # Step 4: Search image DB
    print(f"\n[4/6] Searching image DB (m={m})...")
    image_candidates = search_index(image_index, image_id_to_meta, image_embedding, k=m)
    print(f"  Found {len(image_candidates)} image candidates")
    
    # Step 5: Get text similarity scores
    print("\n[5/6] Computing text similarity scores...")
    text_scores = compute_text_similarities(
        text_index, text_lookup, text_embedding, image_candidates
    )
    
    pages_with_text = sum(1 for v in text_scores.values() if v["num_texts"] > 0)
    print(f"  {pages_with_text}/{len(text_scores)} candidates have text embeddings")
    
    # Step 6: Rerank
    print(f"\n[6/6] Reranking with alpha={alpha}...")
    final_results = late_fusion_rerank(image_candidates, text_scores, alpha, k)
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


def late_fusion_two(
    image_path: Path,
    user_query: str,
    image_db: Path,
    text_db: Path,
    m: int = 50,
    k: int = 10,
    alpha: float = 0.5,
    verbose: bool = False
) -> dict:
    """
    Perform late fusion search with TEXT-FIRST retrieval, then IMAGE reranking.
    
    Pipeline:
    1. Generate LLM description from user query (image + text)
    2. Generate CLIP embeddings (image and text)
    3. Search text DB for m candidates
    4. Deduplicate text results by page (keep max similarity)
    5. For each candidate page, find corresponding image embedding
    6. Compute image similarity scores
    7. Rerank using alpha-weighted combination
    8. Return top k results
    
    Args:
        image_path: Path to query image
        user_query: User's text query
        image_db: Path to image embeddings directory
        text_db: Path to text embeddings directory
        m: Number of initial text candidates to retrieve
        k: Number of final results after reranking
        alpha: Weight for image score (0=text only, 1=image only, 0.5=equal)
        verbose: Print detailed info
    
    Returns:
        Dict with all results and intermediate data
    """
    print("\n" + "=" * 60)
    print("  Late Fusion Two (Text-First)")
    print("=" * 60)
    print(f"  Alpha: {alpha} (image={alpha:.1%}, text={1-alpha:.1%})")
    print(f"  Retrieve m={m} text candidates, rerank to k={k}")
    
    # Step 1: Generate LLM description
    print("\n[1/7] Generating LLM description...")
    description = encode_query_llm(image_path, user_query, verbose)
    print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (before FAISS to avoid macOS segfault)
    print("\n[2/7] Loading CLIP model and generating embeddings...")
    model, preprocess, tokenizer, device = load_clip_model()
    
    image_embedding = encode_image(model, preprocess, device, image_path)
    text_embedding = encode_text(model, tokenizer, device, description)
    print(f"  Image embedding: {image_embedding.shape}")
    print(f"  Text embedding: {text_embedding.shape}")
    
    # Step 3: Load FAISS indexes
    print("\n[3/7] Loading FAISS indexes...")
    text_index, text_id_to_meta, _ = load_faiss_index(text_db)
    print(f"  Text index: {text_index.ntotal} vectors")
    
    image_index, image_id_to_meta, _ = load_faiss_index(image_db)
    print(f"  Image index: {image_index.ntotal} vectors")
    
    # Build image lookup
    image_lookup = build_image_lookup(image_id_to_meta)
    print(f"  Image lookup: {len(image_lookup)} unique pages")
    
    # Step 4: Search text DB
    print(f"\n[4/7] Searching text DB (m={m})...")
    text_candidates_raw = search_index(text_index, text_id_to_meta, text_embedding, k=m)
    print(f"  Found {len(text_candidates_raw)} text candidates")
    
    # Step 5: Deduplicate by page
    print("\n[5/7] Deduplicating text candidates by page...")
    text_candidates = deduplicate_text_candidates(text_candidates_raw)
    print(f"  Unique pages: {len(text_candidates)}")
    
    # Step 6: Get image similarity scores
    print("\n[6/7] Computing image similarity scores...")
    image_scores = compute_image_similarities(
        image_index, image_lookup, image_embedding, text_candidates
    )
    
    pages_with_image = sum(1 for v in image_scores.values() if v["similarity"] > 0)
    print(f"  {pages_with_image}/{len(image_scores)} candidates have image embeddings")
    
    # Step 7: Rerank
    print(f"\n[7/7] Reranking with alpha={alpha}...")
    final_results = late_fusion_rerank_two(text_candidates, image_scores, alpha, k)
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


def main():
    parser = argparse.ArgumentParser(
        description="Late Fusion Query Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic: Generate description only
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "the character at the left"
    
    # Late Fusion One: Image-first, rerank with text
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "the tall warrior" \\
        --late-fusion-one --image-db final_dataset_embeddings --text-db final_dataset_text_embeddings \\
        -m 50 -k 10 --alpha 0.5
    
    # Late Fusion Two: Text-first, rerank with image
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "the tall warrior" \\
        --late-fusion-two --image-db final_dataset_embeddings --text-db final_dataset_text_embeddings \\
        -m 50 -k 10 --alpha 0.5
    
    # With visualization
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "the tall warrior" \\
        --late-fusion-one --image-db final_dataset_embeddings --text-db final_dataset_text_embeddings \\
        -m 50 -k 10 --alpha 0.5 --visualize
    
    # Save visualization to file
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "the tall warrior" \\
        --late-fusion-one --image-db final_dataset_embeddings --text-db final_dataset_text_embeddings \\
        -m 50 -k 10 --alpha 0.5 --save-viz results.png
        """
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to query image")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text query")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Late Fusion options
    parser.add_argument("--late-fusion-one", action="store_true", help="Image-first, rerank with text")
    parser.add_argument("--late-fusion-two", action="store_true", help="Text-first, rerank with image")
    parser.add_argument("--image-db", type=str, help="Path to image embeddings (with faiss_index)")
    parser.add_argument("--text-db", type=str, help="Path to text embeddings (with faiss_index)")
    parser.add_argument("-m", type=int, default=50, help="Initial candidates (default: 50)")
    parser.add_argument("-k", type=int, default=10, help="Final results after reranking (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Image weight (0-1, default: 0.5)")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Display visualization of results")
    parser.add_argument("--save-viz", type=str, help="Save visualization to file")
    
    args = parser.parse_args()
    image_path = Path(args.image)
    
    try:
        if args.late_fusion_one or args.late_fusion_two:
            if not args.image_db or not args.text_db:
                print("Error: --image-db and --text-db required for late fusion")
                return 1
            
            image_db = Path(args.image_db)
            text_db = Path(args.text_db)
            
            if not image_db.exists():
                print(f"Error: Image DB not found: {image_db}")
                return 1
            if not text_db.exists():
                print(f"Error: Text DB not found: {text_db}")
                return 1
            
            if args.late_fusion_one:
                result = late_fusion_one(
                    image_path,
                    args.text,
                    image_db,
                    text_db,
                    m=args.m,
                    k=args.k,
                    alpha=args.alpha,
                    verbose=args.verbose
                )
                
                print("\n" + "=" * 60)
                print("  Results Summary (Fusion One: Image-First)")
                print("=" * 60)
                print(f"\nLLM Description: {result['llm_description']}")
                print(f"\nAlpha: {result['alpha']} | Candidates: {result['num_image_candidates']} | With text: {result['num_pages_with_text']}")
                
            else:  # late_fusion_two
                result = late_fusion_two(
                    image_path,
                    args.text,
                    image_db,
                    text_db,
                    m=args.m,
                    k=args.k,
                    alpha=args.alpha,
                    verbose=args.verbose
                )
                
                print("\n" + "=" * 60)
                print("  Results Summary (Fusion Two: Text-First)")
                print("=" * 60)
                print(f"\nLLM Description: {result['llm_description']}")
                print(f"\nAlpha: {result['alpha']} | Raw candidates: {result['num_text_candidates_raw']} | Unique pages: {result['num_text_candidates_dedup']} | With image: {result['num_pages_with_image']}")
            
            print_fusion_results(result['final_results'])
            
            # Visualization
            if args.visualize or args.save_viz:
                title = "Late Fusion One (Image-First)" if args.late_fusion_one else "Late Fusion Two (Text-First)"
                visualize_fusion_results(
                    image_path,
                    result['final_results'],
                    title=title,
                    output_path=Path(args.save_viz) if args.save_viz else None,
                    show=args.visualize
                )
            
        else:
            description = encode_query_llm(image_path, args.text, args.verbose)
            print("\nGenerated Description:")
            print("=" * 60)
            print(description)
            print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
