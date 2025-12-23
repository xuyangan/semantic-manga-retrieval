"""
Intervention Package

Implements gated intervention: modifies image embeddings using
gate-weighted text embeddings for non-linear transformation.

Usage as module:
    python -m intervention.intervention_encoder --image page.jpg --text "query" ...

Usage as library:
    from intervention import compute_gate_factors, apply_intervention
    from intervention.intervention_encoder import intervention_search
"""

from .gate import compute_gate_factors, compute_gate_for_candidates, compute_z_scores, sigmoid
from .transform import (
    apply_intervention,
    build_intervention_index,
    build_page_to_faiss_mapping,
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
]
