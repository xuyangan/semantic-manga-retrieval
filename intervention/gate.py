#!/usr/bin/env python3
"""
Gate Computation Module

Computes gate factors for intervention-based fusion using z-score normalized similarity.
"""

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Compute sigmoid function.
    
    Args:
        x: Input value(s)
    
    Returns:
        Sigmoid of input: 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_z_scores(similarities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute z-score normalization of similarity scores.
    
    Formula: z = (s - μ) / (σ + ε)
    
    Args:
        similarities: Raw similarity scores
        eps: Small epsilon for numerical stability
    
    Returns:
        Z-score normalized similarities
    """
    mean = np.mean(similarities)
    std = np.std(similarities, ddof=0)
    
    return (similarities - mean) / (std + eps)


def compute_gate_factors(
    similarities: list[float] | np.ndarray,
    gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gate factors from z-score normalized similarity scores.
    
    Formula: g = sigmoid(gamma * z)
    where z = (s - mean(s)) / std(s)
    
    Args:
        similarities: Text similarity scores (one per candidate)
        gamma: Steepness parameter (higher = sharper transition)
    
    Returns:
        Tuple of (gate_factors, z_scores)
    
    Example:
        - gamma=1: z-scores around 0 (mean) get gate ~0.5
        - z > 0: gate approaches 1 (above-average similarity)
        - z < 0: gate approaches 0 (below-average similarity)
    """
    s = np.asarray(similarities, dtype=np.float32)
    z = compute_z_scores(s)
    gates = sigmoid(gamma * z)
    return gates, z


def compute_gate_for_candidates(
    candidates: list[dict],
    gamma: float
) -> list[dict]:
    """
    Compute gate factors for a list of candidates using z-score normalization.
    
    Args:
        candidates: List of candidate dicts with 'similarity' key
        gamma: Steepness parameter
    
    Returns:
        Same candidates list with 'gate_factor' and 'z_score' added to each
    """
    if not candidates:
        return candidates
    
    # Extract all similarities
    similarities = np.array([c.get("similarity", 0.0) for c in candidates], dtype=np.float32)
    
    # Compute z-scores and gates
    gates, z_scores = compute_gate_factors(similarities, gamma)
    
    # Add to candidates
    for i, candidate in enumerate(candidates):
        candidate["z_score"] = float(z_scores[i])
        candidate["gate_factor"] = float(gates[i])
    
    return candidates
