"""
Late Fusion Package

Combines LLM-generated descriptions with CLIP embeddings for similarity search.

Usage as module:
    python -m late_fusion.late_fusion_encoder --image page.jpg --text "query" ...

Usage as library:
    from late_fusion import encode_query_llm, load_clip_model
    from late_fusion.late_fusion_encoder import late_fusion_one, late_fusion_two
"""

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
    get_page_key,
)
from .fusion import late_fusion_rerank, late_fusion_rerank_two, print_fusion_results

# Note: late_fusion_one, late_fusion_two not imported here to avoid circular import
# when running as module. Import directly: from late_fusion.late_fusion_encoder import late_fusion_one

__all__ = [
    # LLM
    "encode_query_llm",
    # CLIP
    "load_clip_model",
    "encode_image",
    "encode_text",
    # FAISS
    "load_faiss_index",
    "search_index",
    "build_text_lookup",
    "build_image_lookup",
    "compute_text_similarities",
    "compute_image_similarities",
    "deduplicate_text_candidates",
    "get_page_key",
    # Fusion
    "late_fusion_rerank",
    "late_fusion_rerank_two",
    "print_fusion_results",
]
