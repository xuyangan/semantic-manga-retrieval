#!/usr/bin/env python3
"""
QCFR (Query-Conditioned Feedback Retrieval) with Max-Pooled Text

Two-pass retrieval algorithm that uses hybrid scoring and Rocchio-style query refinement.

Data structure:
- Image index: one embedding per page (v_p)
- Text index: multiple text embeddings per page (T_p = {t_{p,1}, ..., t_{p,m_p}})

Algorithm:
1. Pass-1: Retrieve candidates from both image and text channels
2. Hybrid scoring with max-pooled text per page
3. Select pseudo-positives (top) and pseudo-negatives (bottom)
4. Compute feedback centroids using softmax weights
5. Refine query using Rocchio-style update
6. Pass-2: Full re-search with refined query

Reference hyperparameters:
- M_img = 800, M_txt = 2000
- L_pos = 30, L_neg = 30
- alpha = 0.5
- a=1.0, b=0.35, c=0.14, d=0.21
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax of input array.
    
    Args:
        x: Input array
    
    Returns:
        Softmax normalized array
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def build_page_to_text_indices(text_id_to_meta: Dict, get_page_key_fn) -> Dict[str, List[int]]:
    """
    Build mapping from page_key to list of text FAISS indices for that page.
    
    This enables computing true max-pooled text scores over ALL text embeddings
    for a page, not just those that appeared in top-k search results.
    
    Args:
        text_id_to_meta: Mapping from FAISS index -> text metadata
        get_page_key_fn: Function to extract page key from metadata
    
    Returns:
        Dict mapping page_key -> list of FAISS indices for that page's text embeddings
    """
    page_to_text_indices = {}
    
    for idx, meta in text_id_to_meta.items():
        page_key = get_page_key_fn(meta)
        if page_key not in page_to_text_indices:
            page_to_text_indices[page_key] = []
        page_to_text_indices[page_key].append(idx)
    
    return page_to_text_indices


def compute_true_max_text_score(
    page_key: str,
    query_text_embedding: np.ndarray,
    text_index,
    page_to_text_indices: Dict[str, List[int]]
) -> float:
    """
    Compute true max text similarity for a page over ALL its text embeddings.
    
    Args:
        page_key: Page identifier
        query_text_embedding: Query text embedding (1, D) or (D,)
        text_index: FAISS text index
        page_to_text_indices: Mapping from page_key to list of text FAISS indices
    
    Returns:
        Max similarity score, or -inf if page has no text embeddings
    """
    if page_key not in page_to_text_indices:
        return -np.inf
    
    text_indices = page_to_text_indices[page_key]
    if not text_indices:
        return -np.inf
    
    # Normalize query
    tq = query_text_embedding.flatten()
    tq_norm = np.linalg.norm(tq)
    if tq_norm > 1e-8:
        tq = tq / tq_norm
    
    max_sim = -np.inf
    for idx in text_indices:
        text_emb = text_index.reconstruct(idx)
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        sim = float(np.dot(tq, text_emb))
        if sim > max_sim:
            max_sim = sim
    
    return max_sim


def aggregate_text_scores_by_page(
    text_candidates: List[Dict],
    get_page_key_fn
) -> Dict[str, Dict]:
    """
    Aggregate text search results by page using max-pooling.
    
    For each page p, computes:
    - s_txt[p] = max score among text rows belonging to page p
    - argmax text_id (which text matched best)
    
    Note: This only considers text candidates from the initial search.
    For true max-pooling over ALL text embeddings, use compute_true_max_text_score().
    
    Args:
        text_candidates: Raw text search results with 'similarity' and metadata
        get_page_key_fn: Function to extract page key from result
    
    Returns:
        Dict mapping page_key -> {
            'max_similarity': float,
            'best_text_meta': dict,
            'text_count': int
        }
    """
    page_scores = {}
    
    for candidate in text_candidates:
        page_key = get_page_key_fn(candidate)
        sim = candidate.get('similarity', 0.0)
        
        if page_key not in page_scores or sim > page_scores[page_key]['max_similarity']:
            page_scores[page_key] = {
                'max_similarity': sim,
                'best_text_meta': candidate,
                'text_count': page_scores.get(page_key, {}).get('text_count', 0) + 1
            }
        else:
            page_scores[page_key]['text_count'] += 1
    
    return page_scores


def compute_hybrid_scores(
    image_scores: Dict[str, float],
    text_scores: Dict[str, float],
    alpha: float = 0.5,
    default_text_score: float = -np.inf
) -> Dict[str, float]:
    """
    Compute hybrid scores for candidate pages.
    
    Formula: s[p] = (1 - alpha) * s_img[p] + alpha * s_txt[p]
    
    Args:
        image_scores: Dict mapping page_key -> image similarity score
        text_scores: Dict mapping page_key -> text similarity score (max-pooled)
        alpha: Weight for text score (0=image only, 1=text only)
        default_text_score: Default score for pages without text match
    
    Returns:
        Dict mapping page_key -> hybrid score
    """
    # Union of all candidate pages
    all_pages = set(image_scores.keys()) | set(text_scores.keys())
    
    hybrid_scores = {}
    
    for page in all_pages:
        s_img = image_scores.get(page, 0.0)  # If not in image candidates, score = 0
        s_txt = text_scores.get(page, default_text_score)  # Default to very low if not in text
        
        # Handle -inf case: if text score is -inf, hybrid favors image
        if s_txt == -np.inf:
            # Just use image score (text not available for this page)
            hybrid_scores[page] = s_img
        else:
            hybrid_scores[page] = (1 - alpha) * s_img + alpha * s_txt
    
    return hybrid_scores


def select_pseudo_labels(
    hybrid_scores: Dict[str, float],
    l_pos: int = 30,
    l_neg: int = 30
) -> Tuple[List[str], List[str]]:
    """
    Select pseudo-positive and pseudo-negative pages.
    
    Args:
        hybrid_scores: Dict mapping page_key -> hybrid score
        l_pos: Number of pseudo-positives (top pages)
        l_neg: Number of pseudo-negatives (bottom pages)
    
    Returns:
        Tuple of (positive_pages, negative_pages)
    """
    # Sort pages by hybrid score
    sorted_pages = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Top L_pos as pseudo-positives
    positives = [page for page, _ in sorted_pages[:l_pos]]
    
    # Bottom L_neg as pseudo-negatives
    negatives = [page for page, _ in sorted_pages[-l_neg:]]
    
    return positives, negatives


def compute_feedback_centroids(
    positive_pages: List[str],
    negative_pages: List[str],
    hybrid_scores: Dict[str, float],
    page_to_embedding: Dict[str, np.ndarray],
    embedding_dim: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feedback centroids using softmax-weighted embeddings.
    
    Formula:
    - w_pos[p] = softmax(s[p] for p in P)
    - w_neg[p] = softmax(-s[p] for p in N)  # Note: negate for negatives
    - pos = normalize(sum(w_pos[p] * v_p))
    - neg = normalize(sum(w_neg[p] * v_p))
    
    Args:
        positive_pages: List of pseudo-positive page keys
        negative_pages: List of pseudo-negative page keys
        hybrid_scores: Dict mapping page_key -> hybrid score
        page_to_embedding: Dict mapping page_key -> image embedding (D,)
        embedding_dim: Embedding dimension (inferred if not provided)
    
    Returns:
        Tuple of (pos_centroid, neg_centroid) both normalized (D,)
    """
    # Infer embedding dimension from page_to_embedding if not provided
    if embedding_dim is None:
        if page_to_embedding:
            first_emb = next(iter(page_to_embedding.values()))
            embedding_dim = first_emb.shape[-1]
        else:
            embedding_dim = 768  # Fallback for ViT-L-14
    
    # Compute positive centroid
    pos_scores = np.array([hybrid_scores[p] for p in positive_pages if p in page_to_embedding])
    pos_embeddings = np.array([page_to_embedding[p] for p in positive_pages if p in page_to_embedding])
    
    if len(pos_scores) > 0:
        pos_weights = softmax(pos_scores)
        pos_centroid = np.sum(pos_weights[:, np.newaxis] * pos_embeddings, axis=0)
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-8)
    else:
        pos_centroid = np.zeros(embedding_dim)
    
    # Compute negative centroid (negate scores for softmax)
    neg_scores = np.array([hybrid_scores[p] for p in negative_pages if p in page_to_embedding])
    neg_embeddings = np.array([page_to_embedding[p] for p in negative_pages if p in page_to_embedding])
    
    if len(neg_scores) > 0:
        neg_weights = softmax(-neg_scores)  # Negate for negatives
        neg_centroid = np.sum(neg_weights[:, np.newaxis] * neg_embeddings, axis=0)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-8)
    else:
        neg_centroid = np.zeros(embedding_dim)
    
    return pos_centroid.astype(np.float32), neg_centroid.astype(np.float32)


def refine_query_rocchio(
    q: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    tq: np.ndarray,
    a: float = 1.0,
    b: float = 0.35,
    c: float = 0.14,
    d: float = 0.21
) -> np.ndarray:
    """
    Refine query using Rocchio-style update.
    
    Formula: q' = normalize(a*q + b*pos - c*neg + d*tq)
    
    Args:
        q: Original image query embedding (D,) or (1, D)
        pos: Positive feedback centroid (D,)
        neg: Negative feedback centroid (D,)
        tq: Text query embedding (D,) or (1, D)
        a: Weight for original query
        b: Weight for positive feedback
        c: Weight for negative feedback
        d: Weight for text query
    
    Returns:
        Refined query embedding, normalized (1, D)
    """
    # Flatten inputs
    q = q.flatten()
    pos = pos.flatten()
    neg = neg.flatten()
    tq = tq.flatten()
    
    # Rocchio update
    q_prime = a * q + b * pos - c * neg + d * tq
    
    # Normalize
    norm = np.linalg.norm(q_prime)
    if norm > 1e-8:
        q_prime = q_prime / norm
    
    return q_prime.reshape(1, -1).astype(np.float32)


def qcfr_search(
    query_image_embedding: np.ndarray,
    query_text_embedding: np.ndarray,
    image_index,
    image_id_to_meta: Dict,
    text_index,
    text_id_to_meta: Dict,
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
    verbose: bool = False
) -> Dict:
    """
    QCFR (Query-Conditioned Feedback Re-ranking) search.
    
    Two-pass retrieval with hybrid scoring and Rocchio-style query refinement.
    
    Args:
        query_image_embedding: Query image embedding q (1, D) - normalized
        query_text_embedding: Query text embedding tq (1, D) - normalized
        image_index: FAISS index for page images (one per page)
        image_id_to_meta: Metadata mapping for image index
        text_index: FAISS index for text embeddings (many per page)
        text_id_to_meta: Metadata mapping for text index
        m_img: Number of image candidates in Pass-1
        m_txt: Number of text candidates in Pass-1 (before aggregation)
        l_pos: Number of pseudo-positive pages
        l_neg: Number of pseudo-negative pages
        alpha: Hybrid score weight (0=image, 1=text)
        a, b, c, d: Rocchio parameters
        k: Number of final results
        true_max_pooling: If True, compute true max over ALL text embeddings per page
                         (more accurate but slower). If False, use max from top m_txt only.
        verbose: Print debug info
    
    Returns:
        Dict with search results and intermediate data
    """
    import faiss
    from late_fusion.faiss_search import search_index, get_page_key
    
    # Infer embedding dimension from query
    embedding_dim = query_image_embedding.shape[-1]
    
    if verbose:
        print(f"\n{'='*60}")
        print("  QCFR Search")
        print(f"{'='*60}")
        print(f"  Params: alpha={alpha}, a={a}, b={b}, c={c}, d={d}")
        print(f"  M_img={m_img}, M_txt={m_txt}, L_pos={l_pos}, L_neg={l_neg}, k={k}")
        print(f"  True max-pooling: {true_max_pooling}, Embedding dim: {embedding_dim}")
    
    # =====================================================================
    # Step 1: Pass-1 retrieval (candidates from both channels)
    # =====================================================================
    if verbose:
        print("\n[1/6] Pass-1 retrieval...")
    
    # Image channel: top M_img pages
    image_candidates = search_index(image_index, image_id_to_meta, query_image_embedding, k=m_img)
    
    # Build image scores dict
    image_scores = {}  # page_key -> score
    for candidate in image_candidates:
        page_key = get_page_key(candidate)
        image_scores[page_key] = candidate['similarity']
    
    if verbose:
        print(f"  Image channel: {len(image_candidates)} pages")
    
    # Text channel: top M_txt text rows
    text_candidates_raw = search_index(text_index, text_id_to_meta, query_text_embedding, k=m_txt)
    
    # Aggregate to pages by max-pooling (over returned results)
    text_page_scores = aggregate_text_scores_by_page(text_candidates_raw, get_page_key)
    
    # Build text scores dict (initial from search results)
    text_scores = {page: info['max_similarity'] for page, info in text_page_scores.items()}
    
    if verbose:
        print(f"  Text channel: {len(text_candidates_raw)} text rows -> {len(text_page_scores)} unique pages")
    
    # Union of candidate pages
    c_pages = set(image_scores.keys()) | set(text_scores.keys())
    if verbose:
        print(f"  Union candidates: {len(c_pages)} pages")
    
    # =====================================================================
    # Step 2: Hybrid scoring at page level
    # =====================================================================
    if verbose:
        print("\n[2/6] Computing hybrid scores...")
    
    # Build page_to_faiss_idx mapping for images
    from intervention.transform import build_page_to_faiss_mapping
    page_to_faiss_idx = build_page_to_faiss_mapping(image_id_to_meta)
    
    # Build page_to_text_indices mapping for true max-pooling
    page_to_text_indices = None
    if true_max_pooling:
        page_to_text_indices = build_page_to_text_indices(text_id_to_meta, get_page_key)
        if verbose:
            print(f"  Built page->text mapping: {len(page_to_text_indices)} pages with text")
    
    # For pages missing image scores, compute them
    missing_image_pages = set(text_scores.keys()) - set(image_scores.keys())
    if missing_image_pages and verbose:
        print(f"  Computing image scores for {len(missing_image_pages)} additional pages...")
    
    # Normalize query for manual similarity computation
    q = np.ascontiguousarray(query_image_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(q)
    q_flat = q.flatten()
    
    # Compute missing image scores
    for page in missing_image_pages:
        if page in page_to_faiss_idx:
            idx = page_to_faiss_idx[page]
            v_p = image_index.reconstruct(idx)
            v_p = v_p / (np.linalg.norm(v_p) + 1e-8)
            image_scores[page] = float(np.dot(q_flat, v_p))
        else:
            image_scores[page] = 0.0  # Page not in image index
    
    # For true max-pooling: compute true max text scores for pages from image channel
    # that weren't in text search results (their best text might not be in top m_txt)
    if true_max_pooling and page_to_text_indices:
        pages_needing_true_text_score = set(image_scores.keys()) - set(text_scores.keys())
        if pages_needing_true_text_score and verbose:
            print(f"  Computing true max text scores for {len(pages_needing_true_text_score)} image-only pages...")
        
        for page in pages_needing_true_text_score:
            text_scores[page] = compute_true_max_text_score(
                page, query_text_embedding, text_index, page_to_text_indices
            )
    
    # Compute hybrid scores
    hybrid_scores = compute_hybrid_scores(
        image_scores, text_scores, alpha=alpha, default_text_score=-np.inf
    )
    
    if verbose:
        scores_list = sorted(hybrid_scores.values(), reverse=True)
        print(f"  Hybrid scores: min={min(scores_list):.4f}, max={max(scores_list):.4f}, mean={np.mean(scores_list):.4f}")
    
    # =====================================================================
    # Step 3: Select pseudo-positives and pseudo-negatives
    # =====================================================================
    if verbose:
        print("\n[3/6] Selecting pseudo-positives and pseudo-negatives...")
    
    positives, negatives = select_pseudo_labels(hybrid_scores, l_pos, l_neg)
    
    if verbose:
        print(f"  Pseudo-positives: {len(positives)} pages")
        print(f"  Pseudo-negatives: {len(negatives)} pages")
        if positives:
            print(f"    Top positive: {positives[0]} (score={hybrid_scores[positives[0]]:.4f})")
        if negatives:
            print(f"    Top negative: {negatives[0]} (score={hybrid_scores[negatives[0]]:.4f})")
    
    # =====================================================================
    # Step 4: Compute feedback centroids
    # =====================================================================
    if verbose:
        print("\n[4/6] Computing feedback centroids...")
    
    # Build page_to_embedding mapping for relevant pages
    page_to_embedding = {}
    relevant_pages = set(positives) | set(negatives)
    
    for page in relevant_pages:
        if page in page_to_faiss_idx:
            idx = page_to_faiss_idx[page]
            emb = image_index.reconstruct(idx)
            emb = emb / (np.linalg.norm(emb) + 1e-8)  # Normalize
            page_to_embedding[page] = emb
    
    pos_centroid, neg_centroid = compute_feedback_centroids(
        positives, negatives, hybrid_scores, page_to_embedding,
        embedding_dim=embedding_dim
    )
    
    if verbose:
        pos_norm = np.linalg.norm(pos_centroid)
        neg_norm = np.linalg.norm(neg_centroid)
        print(f"  Positive centroid norm: {pos_norm:.4f}")
        print(f"  Negative centroid norm: {neg_norm:.4f}")
    
    # =====================================================================
    # Step 5: Refine query (Rocchio-style)
    # =====================================================================
    if verbose:
        print("\n[5/6] Refining query (Rocchio-style)...")
    
    q_refined = refine_query_rocchio(
        query_image_embedding, pos_centroid, neg_centroid, query_text_embedding,
        a=a, b=b, c=c, d=d
    )
    
    if verbose:
        # Compute similarity between original and refined query
        q_orig = query_image_embedding.flatten()
        q_ref = q_refined.flatten()
        sim = float(np.dot(q_orig, q_ref))
        print(f"  Query refinement similarity (q vs q'): {sim:.4f}")
    
    # =====================================================================
    # Step 6: Pass-2 retrieval (full re-search)
    # =====================================================================
    if verbose:
        print(f"\n[6/6] Pass-2 retrieval (k={k})...")
    
    final_results = search_index(image_index, image_id_to_meta, q_refined, k=k)
    
    # Add additional info to results
    for result in final_results:
        page_key = get_page_key(result)
        result['page_key'] = page_key
        result['is_pseudo_positive'] = page_key in positives
        result['is_pseudo_negative'] = page_key in negatives
        result['image_score_pass1'] = image_scores.get(page_key, 0.0)
        result['text_score_pass1'] = text_scores.get(page_key, 0.0)
        result['hybrid_score'] = hybrid_scores.get(page_key, 0.0)
    
    if verbose:
        pos_in_results = sum(1 for r in final_results if r['is_pseudo_positive'])
        neg_in_results = sum(1 for r in final_results if r['is_pseudo_negative'])
        print(f"  Found {len(final_results)} results")
        print(f"  Pseudo-positives in results: {pos_in_results}")
        print(f"  Pseudo-negatives in results: {neg_in_results}")
    
    return {
        "final_results": final_results,
        "refined_query": q_refined,
        "positive_pages": positives,
        "negative_pages": negatives,
        "pos_centroid": pos_centroid,
        "neg_centroid": neg_centroid,
        "hybrid_scores": hybrid_scores,
        "image_scores": image_scores,
        "text_scores": text_scores,
        "num_image_candidates": len(image_candidates),
        "num_text_candidates_raw": len(text_candidates_raw),
        "num_text_pages": len(text_page_scores),
        "num_union_candidates": len(c_pages),
    }


def qcfr_search_with_description(
    image_path: Path,
    user_query: str,
    image_db: Path,
    text_db: Path,
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
    image_index=None,
    image_id_to_meta=None,
    text_index=None,
    text_id_to_meta=None,
    verbose: bool = False
) -> Dict:
    """
    QCFR search with LLM description generation.
    
    Full pipeline including:
    1. Generate LLM description (or use cached)
    2. Generate CLIP embeddings
    3. Run QCFR search
    
    Args:
        image_path: Path to query image
        user_query: User's text query
        image_db: Path to image embeddings directory
        text_db: Path to text embeddings directory
        image_index_dir: Path to image FAISS index
        text_index_dir: Path to text FAISS index
        m_img: Number of image candidates in Pass-1
        m_txt: Number of text candidates in Pass-1
        l_pos: Number of pseudo-positives
        l_neg: Number of pseudo-negatives
        alpha: Hybrid score weight
        a, b, c, d: Rocchio parameters
        k: Number of final results
        cached_description: Optional pre-computed LLM description
        model, preprocess, tokenizer, device: CLIP model components
        image_index, image_id_to_meta: Pre-loaded image index
        text_index, text_id_to_meta: Pre-loaded text index
        verbose: Print debug info
    
    Returns:
        Dict with all results and intermediate data
    """
    from late_fusion.llm_encoder import encode_query_llm
    from late_fusion.clip_encoder import load_clip_model, encode_image, encode_text
    from evaluate.recall_utils import load_faiss_index_direct
    
    # Step 1: Generate or use cached LLM description
    if cached_description is not None:
        description = cached_description
        if verbose:
            print(f"Using cached LLM description: {description[:80]}...")
    else:
        if verbose:
            print("\n[1/4] Generating LLM description...")
        description = encode_query_llm(image_path, user_query, verbose)
        if verbose:
            print(f"  Description: {description[:80]}...")
    
    # Step 2: Load CLIP model (if not provided)
    if model is None or preprocess is None or tokenizer is None:
        if verbose:
            print("\n[2/4] Loading CLIP model...")
        model, preprocess, tokenizer, device = load_clip_model(device)
    elif verbose:
        print("\n[2/4] Using provided CLIP model...")
    
    # Generate embeddings
    query_image_embedding = encode_image(model, preprocess, device, image_path)
    query_text_embedding = encode_text(model, tokenizer, device, description)
    
    if verbose:
        print(f"  Query image embedding: {query_image_embedding.shape}")
        print(f"  Query text embedding: {query_text_embedding.shape}")
    
    # Step 3: Load FAISS indexes (if not provided)
    if image_index is None or image_id_to_meta is None:
        if verbose:
            print("\n[3/4] Loading FAISS indexes...")
        image_index, image_id_to_meta, _ = load_faiss_index_direct(image_index_dir)
        if verbose:
            print(f"  Image index: {image_index.ntotal} vectors")
    elif verbose:
        print("\n[3/4] Using provided image index...")
    
    if text_index is None or text_id_to_meta is None:
        text_index, text_id_to_meta, _ = load_faiss_index_direct(text_index_dir)
        if verbose:
            print(f"  Text index: {text_index.ntotal} vectors")
    elif verbose:
        print("  Using provided text index...")
    
    # Step 4: Run QCFR search
    if verbose:
        print("\n[4/4] Running QCFR search...")
    
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
    
    # Add metadata to result
    qcfr_result.update({
        "query_image": str(image_path),
        "user_query": user_query,
        "llm_description": description,
        "query_image_embedding": query_image_embedding,
        "query_text_embedding": query_text_embedding,
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
            "k": k,
            "true_max_pooling": true_max_pooling,
        }
    })
    
    return qcfr_result
