#!/usr/bin/env python3
"""
Intervention Transform Module

Applies gate-weighted text embedding intervention to image embeddings.
"""

import numpy as np


def apply_intervention(
    image_embeddings: np.ndarray,
    query_text_embedding: np.ndarray,
    gate_factors: np.ndarray,
    indices: np.ndarray,
    beta: float = 1.0
) -> np.ndarray:
    """
    Apply gate-weighted text intervention to image embeddings.
    
    Formula: v_i = normalize(v_i + β * g_i * t_q)
    
    For images without gate factors (not in indices), they remain unchanged.
    
    Args:
        image_embeddings: All image embeddings (N, D)
        query_text_embedding: Query text embedding t_q (1, D) or (D,)
        gate_factors: Gate factors g_i for selected images (M,)
        indices: FAISS indices of images to apply intervention (M,)
        beta: Scaling factor for intervention strength
    
    Returns:
        Modified embeddings (N, D), normalized
    """
    # Copy to avoid modifying original
    modified = image_embeddings.copy()
    
    # Flatten text embedding
    text_emb = query_text_embedding.flatten()
    
    # Apply intervention to selected images
    for i, (idx, gate) in enumerate(zip(indices, gate_factors)):
        if idx < len(modified):
            # v_i = v_i + β * g_i * t_q
            modified[idx] = modified[idx] + beta * gate * text_emb
    
    # Normalize all embeddings (including unmodified ones for consistency)
    norms = np.linalg.norm(modified, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    modified = modified / norms
    
    return modified.astype(np.float32)


def build_intervention_index(
    image_index,
    image_id_to_meta: dict,
    query_text_embedding: np.ndarray,
    candidates_with_gates: list[dict],
    page_to_faiss_idx: dict,
    beta: float = 1.0
):
    """
    Build a temporary FAISS index with intervention-modified embeddings.
    
    Formula: v_i = normalize(v_i + β * g_i * t_q)
    
    Args:
        image_index: Original FAISS image index
        image_id_to_meta: Metadata mapping for image index
        query_text_embedding: Query text embedding t_q (1, D)
        candidates_with_gates: List of candidates with gate_factor and page_key
        page_to_faiss_idx: Mapping from page_key to FAISS index
        beta: Scaling factor for intervention strength
    
    Returns:
        tuple: (new_faiss_index, id_to_meta)
    """
    import faiss
    
    n_vectors = image_index.ntotal
    dim = image_index.d
    
    # Reconstruct all image embeddings
    print(f"    Reconstructing {n_vectors} image embeddings...")
    all_embeddings = np.zeros((n_vectors, dim), dtype=np.float32)
    for i in range(n_vectors):
        all_embeddings[i] = image_index.reconstruct(i)
    
    # Build gate mapping: faiss_idx -> gate_factor
    gate_map = {}
    for candidate in candidates_with_gates:
        page_key = candidate.get("page_key")
        if page_key and page_key in page_to_faiss_idx:
            faiss_idx = page_to_faiss_idx[page_key]
            gate_map[faiss_idx] = candidate["gate_factor"]
    
    print(f"    Applying intervention (beta={beta}) to {len(gate_map)} images...")
    
    # Extract indices and gates
    indices = np.array(list(gate_map.keys()), dtype=np.int64)
    gate_factors = np.array([gate_map[i] for i in indices], dtype=np.float32)
    
    # Apply intervention
    modified_embeddings = apply_intervention(
        all_embeddings, query_text_embedding, gate_factors, indices, beta
    )
    
    # Create new FAISS index
    new_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(modified_embeddings)
    new_index.add(modified_embeddings)
    
    return new_index, image_id_to_meta


def build_page_to_faiss_mapping(image_id_to_meta: dict) -> dict:
    """
    Build mapping from page_key to FAISS index.
    
    Args:
        image_id_to_meta: Mapping from FAISS index to metadata
    
    Returns:
        Dict mapping page_key -> faiss_idx
    """
    from late_fusion.faiss_search import get_page_key
    
    mapping = {}
    for idx, meta in image_id_to_meta.items():
        page_key = get_page_key(meta)
        mapping[page_key] = idx
    
    return mapping
