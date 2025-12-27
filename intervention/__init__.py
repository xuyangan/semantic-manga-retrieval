"""
Intervention Package

Implements two approaches for query-conditioned retrieval:

1. Gated Intervention: modifies image embeddings using gate-weighted text embeddings
2. QCFR (Query-Conditioned Feedback Re-ranking): two-pass retrieval with Rocchio-style refinement

Usage as module:
    python -m intervention.intervention_encoder --image page.jpg --text "query" ...

Usage as library:
    # Gated intervention
    from intervention import compute_gate_factors, apply_intervention
    from intervention.intervention_encoder import intervention_search
    
    # QCFR
    from intervention import qcfr_search, qcfr_search_with_description
"""

from .gate import compute_gate_factors, compute_gate_for_candidates, compute_z_scores, sigmoid
from .transform import (
    apply_intervention,
    build_intervention_index,
    build_page_to_faiss_mapping,
)
from .qcfr import (
    qcfr_search,
    qcfr_search_with_description,
    aggregate_text_scores_by_page,
    compute_hybrid_scores,
    select_pseudo_labels,
    compute_feedback_centroids,
    refine_query_rocchio,
    softmax,
    build_page_to_text_indices,
    compute_true_max_text_score,
)

# Note: intervention_search not imported here to avoid circular import
# when running as module. Import directly: from intervention.intervention_encoder import intervention_search

__all__ = [
    # Gate computation
    "compute_gate_factors",
    "compute_gate_for_candidates",
    "compute_z_scores",
    "sigmoid",
    # Transform
    "apply_intervention",
    "build_intervention_index",
    "build_page_to_faiss_mapping",
    # QCFR
    "qcfr_search",
    "qcfr_search_with_description",
    "aggregate_text_scores_by_page",
    "compute_hybrid_scores",
    "select_pseudo_labels",
    "compute_feedback_centroids",
    "refine_query_rocchio",
    "softmax",
    "build_page_to_text_indices",
    "compute_true_max_text_score",
]
