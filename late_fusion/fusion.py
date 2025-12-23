#!/usr/bin/env python3
"""
Late Fusion Module

Combines image and text similarity scores for reranking.
"""

from .faiss_search import get_page_key


def late_fusion_rerank(
    image_candidates: list[dict],
    text_scores: dict,
    alpha: float,
    k: int
) -> list[dict]:
    """
    Rerank candidates using late fusion.
    
    Formula: combined_score = alpha * image_score + (1 - alpha) * text_score
    
    Args:
        image_candidates: List of image search results with similarity scores
        text_scores: Dict mapping page_key -> {max_similarity, text_meta, num_texts}
        alpha: Weight for image score (0-1). Higher = more weight on image.
        k: Number of top results to return
    
    Returns:
        Reranked list of top k results with combined scores
    """
    reranked = []
    
    for candidate in image_candidates:
        page_key = get_page_key(candidate)
        
        image_sim = candidate["similarity"]
        text_info = text_scores.get(page_key, {"max_similarity": 0.0, "text_meta": None})
        text_sim = text_info["max_similarity"]
        
        # Late fusion score
        combined_score = alpha * image_sim + (1 - alpha) * text_sim
        
        reranked.append({
            **candidate,
            "image_similarity": image_sim,
            "text_similarity": text_sim,
            "combined_score": combined_score,
            "text_meta": text_info.get("text_meta"),
            "num_text_embeddings": text_info.get("num_texts", 0),
            "page_key": page_key,
        })
    
    # Sort by combined score
    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Update ranks and return top k
    for i, r in enumerate(reranked[:k]):
        r["rank"] = i + 1
    
    return reranked[:k]


def late_fusion_rerank_two(
    text_candidates: list[dict],
    image_scores: dict,
    alpha: float,
    k: int
) -> list[dict]:
    """
    Rerank text candidates using late fusion with image scores.
    
    Formula: combined_score = alpha * image_score + (1 - alpha) * text_score
    
    Args:
        text_candidates: List of text search results (deduplicated by page)
        image_scores: Dict mapping page_key -> {similarity, image_meta}
        alpha: Weight for image score (0-1). Higher = more weight on image.
        k: Number of top results to return
    
    Returns:
        Reranked list of top k results with combined scores
    """
    reranked = []
    
    for candidate in text_candidates:
        page_key = candidate.get("page_key") or get_page_key(candidate)
        
        text_sim = candidate["similarity"]
        image_info = image_scores.get(page_key, {"similarity": 0.0, "image_meta": None})
        image_sim = image_info["similarity"]
        
        # Late fusion score
        combined_score = alpha * image_sim + (1 - alpha) * text_sim
        
        reranked.append({
            **candidate,
            "image_similarity": image_sim,
            "text_similarity": text_sim,
            "combined_score": combined_score,
            "image_meta": image_info.get("image_meta"),
            "page_key": page_key,
        })
    
    # Sort by combined score
    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Update ranks and return top k
    for i, r in enumerate(reranked[:k]):
        r["rank"] = i + 1
    
    return reranked[:k]


def print_fusion_results(results: list[dict]):
    """Print late fusion results with scores breakdown."""
    print("\nFinal Ranked Results:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Combined':<10} {'Image':<10} {'Text':<10} {'Page'}")
    print("-" * 80)
    
    for r in results:
        page_key = r.get("page_key", "?")
        # Truncate page_key if too long
        if len(page_key) > 40:
            page_key = "..." + page_key[-37:]
        
        print(f"{r['rank']:<5} {r['combined_score']:<10.4f} {r['image_similarity']:<10.4f} {r['text_similarity']:<10.4f} {page_key}")
        
        # Show matched text if available
        if r.get("text_meta") and r["text_meta"].get("text"):
            text_preview = r["text_meta"]["text"][:60]
            if len(r["text_meta"]["text"]) > 60:
                text_preview += "..."
            print(f"      Text: {text_preview}")
