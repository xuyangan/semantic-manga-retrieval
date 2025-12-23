#!/usr/bin/env python3
"""
FAISS Search Module

Load and search FAISS indexes for image and text embeddings.
"""

import json
from pathlib import Path
import numpy as np


def load_faiss_index(db_path: Path, index_name: str = "faiss_index"):
    """
    Load FAISS index from directory.
    
    Args:
        db_path: Path to embeddings directory
        index_name: Name of index subfolder (default: faiss_index)
    
    Returns:
        tuple: (faiss_index, id_to_meta dict, dimension)
    """
    import faiss
    
    index_dir = db_path / index_name
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_dir}")
    
    index_path = index_dir / "faiss.index"
    index = faiss.read_index(str(index_path))
    
    metadata_path = index_dir / "metadata.json"
    with open(metadata_path) as f:
        data = json.load(f)
    
    id_to_meta = {int(k): v for k, v in data["metadata"].items()}
    dimension = data.get("dimension", index.d)
    
    return index, id_to_meta, dimension


def search_index(index, id_to_meta: dict, query_embedding: np.ndarray, k: int) -> list[dict]:
    """
    Search FAISS index with query embedding.
    
    Args:
        index: FAISS index
        id_to_meta: Mapping from index ID to metadata
        query_embedding: Query embedding (1, D) or (D,)
        k: Number of results
    
    Returns:
        List of results with similarity and metadata
    """
    import faiss
    
    query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(query)
    
    scores, indices = index.search(query, k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1 and idx in id_to_meta:
            results.append({
                "rank": i + 1,
                "similarity": float(score),
                "faiss_idx": int(idx),
                **id_to_meta[idx]
            })
    
    return results


def get_page_key(meta: dict) -> str:
    """
    Extract page key from metadata for matching image and text.
    
    Args:
        meta: Metadata dictionary
    
    Returns:
        Page key string (e.g., "manga/chapter/page")
    """
    # For image: manga/chapter/page (from manga, chapter, page fields)
    if "manga" in meta and "chapter" in meta and "page" in meta:
        return f"{meta['manga']}/{meta['chapter']}/{meta['page']}"
    
    # For text: source_file without extension
    if "source_file" in meta:
        return str(Path(meta["source_file"]).with_suffix(""))
    
    # Fallback
    return meta.get("rel_path", meta.get("path", "unknown"))


def build_text_lookup(text_id_to_meta: dict) -> dict:
    """
    Build a lookup from page_key to list of (faiss_idx, metadata).
    
    Args:
        text_id_to_meta: Mapping from FAISS index ID to text metadata
    
    Returns:
        Dict mapping page_key -> list of (idx, meta) tuples
    """
    lookup = {}
    
    for idx, meta in text_id_to_meta.items():
        page_key = get_page_key(meta)
        if page_key not in lookup:
            lookup[page_key] = []
        lookup[page_key].append((idx, meta))
    
    return lookup


def build_image_lookup(image_id_to_meta: dict) -> dict:
    """
    Build a lookup from page_key to (faiss_idx, metadata).
    
    Note: Unlike text, each page has only one image embedding.
    
    Args:
        image_id_to_meta: Mapping from FAISS index ID to image metadata
    
    Returns:
        Dict mapping page_key -> (idx, meta) tuple
    """
    lookup = {}
    
    for idx, meta in image_id_to_meta.items():
        page_key = get_page_key(meta)
        lookup[page_key] = (idx, meta)
    
    return lookup


def deduplicate_text_candidates(text_candidates: list[dict]) -> list[dict]:
    """
    Deduplicate text search results by page, keeping max similarity per page.
    
    Args:
        text_candidates: List of text search results
    
    Returns:
        Deduplicated list with one result per page (highest similarity)
    """
    page_best = {}
    
    for candidate in text_candidates:
        page_key = get_page_key(candidate)
        
        if page_key not in page_best or candidate["similarity"] > page_best[page_key]["similarity"]:
            page_best[page_key] = {
                **candidate,
                "page_key": page_key,
            }
    
    # Sort by similarity and re-rank
    results = sorted(page_best.values(), key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    
    return results


def compute_text_similarities(
    text_index,
    text_lookup: dict,
    query_text_embedding: np.ndarray,
    image_candidates: list[dict]
) -> dict:
    """
    For each image candidate, find corresponding text embeddings and compute max similarity.
    
    Args:
        text_index: FAISS text index
        text_lookup: page_key -> list of (idx, meta)
        query_text_embedding: Query text embedding (1, D)
        image_candidates: List of image search results
    
    Returns:
        Dict mapping page_key -> {max_similarity, text_meta, num_texts}
    """
    import faiss
    
    # Normalize query
    query = np.ascontiguousarray(query_text_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(query)
    
    text_scores = {}
    
    for candidate in image_candidates:
        page_key = get_page_key(candidate)
        
        if page_key in text_scores:
            continue  # Already computed
        
        if page_key not in text_lookup:
            # No text embeddings for this page
            text_scores[page_key] = {
                "max_similarity": 0.0,
                "text_meta": None,
                "num_texts": 0
            }
            continue
        
        # Get all text embeddings for this page
        text_entries = text_lookup[page_key]
        
        max_sim = -1.0
        best_meta = None
        
        for idx, meta in text_entries:
            # Reconstruct embedding from index
            text_emb = text_index.reconstruct(idx)
            text_emb = text_emb.reshape(1, -1)
            faiss.normalize_L2(text_emb)
            
            # Compute similarity (dot product of normalized vectors = cosine similarity)
            sim = float(np.dot(query.flatten(), text_emb.flatten()))
            
            if sim > max_sim:
                max_sim = sim
                best_meta = meta
        
        text_scores[page_key] = {
            "max_similarity": max_sim,
            "text_meta": best_meta,
            "num_texts": len(text_entries)
        }
    
    return text_scores


def compute_image_similarities(
    image_index,
    image_lookup: dict,
    query_image_embedding: np.ndarray,
    text_candidates: list[dict]
) -> dict:
    """
    For each text candidate, find corresponding image embedding and compute similarity.
    
    Args:
        image_index: FAISS image index
        image_lookup: page_key -> (idx, meta) tuple
        query_image_embedding: Query image embedding (1, D)
        text_candidates: List of text search results (deduplicated by page)
    
    Returns:
        Dict mapping page_key -> {similarity, image_meta}
    """
    import faiss
    
    # Normalize query
    query = np.ascontiguousarray(query_image_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(query)
    
    image_scores = {}
    
    for candidate in text_candidates:
        page_key = candidate.get("page_key") or get_page_key(candidate)
        
        if page_key in image_scores:
            continue  # Already computed
        
        if page_key not in image_lookup:
            # No image embedding for this page
            image_scores[page_key] = {
                "similarity": 0.0,
                "image_meta": None
            }
            continue
        
        # Get image embedding
        idx, meta = image_lookup[page_key]
        
        image_emb = image_index.reconstruct(idx)
        image_emb = image_emb.reshape(1, -1)
        faiss.normalize_L2(image_emb)
        
        # Compute similarity
        sim = float(np.dot(query.flatten(), image_emb.flatten()))
        
        image_scores[page_key] = {
            "similarity": sim,
            "image_meta": meta
        }
    
    return image_scores
