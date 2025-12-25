#!/usr/bin/env python3
"""
Cluster and Evaluate Manga Embeddings with t-SNE Visualization

This script analyzes embeddings from final_dataset_embeddings, computes clustering metrics
(Silhouette, Davies-Bouldin, Calinski-Harabasz) and creates t-SNE visualizations grouped by manga.

Sample Execution:
    # Basic clustering with visualization
    python metrics/cluster.py --input final_dataset_embeddings --output clusters/image_embeddings
    
    # With custom t-SNE parameters
    python metrics/cluster.py --input final_dataset_embeddings --output clusters/image_embeddings --perplexity 50 --n-iter 2000
    
    # Text embeddings clustering
    python metrics/cluster.py --input final_dataset_text_embeddings --output clusters/text_embeddings
    
    # Skip visualization (metrics only)
    python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --no-viz
    
    # With outlier detection
    python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --detect-outliers
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_score, silhouette_samples, 
        davies_bouldin_score, calinski_harabasz_score
    )
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. Clustering features will be unavailable.")


def load_embeddings_from_dataset(dataset_dir: Path):
    """
    Load embeddings and metadata from final_dataset_embeddings directory.
    
    Args:
        dataset_dir: Directory containing all_embeddings.npy and paths.json (or metadata.json)
    
    Returns:
        tuple: (embeddings array, metadata list) or None if not found
    """
    dataset_dir = Path(dataset_dir)
    
    embeddings_file = dataset_dir / "all_embeddings.npy"
    paths_file = dataset_dir / "paths.json"
    metadata_file = dataset_dir / "metadata.json"
    
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}")
        return None
    
    print(f"Loading embeddings from: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    # Try to load paths from paths.json first, then from metadata.json
    paths = None
    if paths_file.exists():
        print(f"Loading paths from: {paths_file}")
        with open(paths_file, 'r') as f:
            paths = json.load(f)
    elif metadata_file.exists():
        print(f"Paths file not found, loading from: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata_json = json.load(f)
        
        # Extract paths from metadata.json
        if 'files' in metadata_json and isinstance(metadata_json['files'], list):
            # Use embedding_file as the path (for text embeddings with multiple lines per file)
            # or source_file if embedding_file is not available
            paths = []
            for file_entry in metadata_json['files']:
                if 'embedding_file' in file_entry:
                    # Remove the .npy extension and use the embedding file path
                    path = file_entry['embedding_file'].replace('.npy', '')
                    # Convert path like "Manga/chapter_X/page_Y_line_Z.npy" to remove _line_Z
                    # But keep it for differentiation if it exists
                    paths.append(path)
                elif 'source_file' in file_entry:
                    path = file_entry['source_file'].replace('.txt', '').replace('.jpg', '').replace('.png', '')
                    paths.append(path)
                else:
                    paths.append('unknown')
            print(f"Extracted {len(paths)} paths from metadata.json")
        else:
            print(f"Error: metadata.json does not contain 'files' array")
            return None
    else:
        print(f"Error: Neither paths.json nor metadata.json found in {dataset_dir}")
        return None
    
    if len(embeddings) != len(paths):
        print(f"Warning: Embeddings ({len(embeddings)}) and paths ({len(paths)}) count mismatch!")
    
    # Parse paths to create metadata
    metadata = []
    for path in paths:
        # Parse path: "Manga Name/chapter_X/page_Y.ext" or "Manga Name/chapter_X/page_Y_line_Z"
        parts = path.split('/')
        if len(parts) >= 3:
            manga = parts[0]
            chapter = parts[1]
            page = parts[2]
        else:
            manga = "unknown"
            chapter = "unknown"
            page = "unknown"
        
        metadata.append({
            'path': path,
            'manga': manga,
            'chapter': chapter,
            'page': page
        })
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    print(f"Created metadata for {len(metadata)} items")
    
    # Print manga distribution
    manga_counts = defaultdict(int)
    for m in metadata:
        manga_counts[m['manga']] += 1
    
    print(f"\nDataset contains {len(manga_counts)} unique manga:")
    for manga, count in sorted(manga_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {manga}: {count} items")
    
    return embeddings, metadata


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE projection of embeddings."""
    perplexity = min(perplexity, len(embeddings) - 1)
    
    print(f"\nComputing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    
    coords = tsne.fit_transform(embeddings)
    print("t-SNE computation complete!")
    return coords


def create_manga_labels(metadata: list[dict]) -> tuple[np.ndarray, dict]:
    """
    Create numeric labels from metadata based on manga.
    
    Args:
        metadata: List of metadata dictionaries
    
    Returns:
        tuple: (labels array, id_to_manga mapping)
    """
    categories = [m.get("manga", "unknown") for m in metadata]
    
    unique_categories = sorted(set(categories))
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    
    labels = np.array([category_to_id[cat] for cat in categories])
    id_to_category = {i: cat for cat, i in category_to_id.items()}
    
    return labels, id_to_category


def compute_clustering_metrics(embeddings: np.ndarray, metadata: list[dict]):
    """
    Compute clustering metrics for manga grouping.
    
    Args:
        embeddings: Embedding vectors (N x D)
        metadata: List of metadata dictionaries
    
    Returns:
        dict: Dictionary containing all metrics
    """
    manga_labels, manga_names = create_manga_labels(metadata)
    n_mangas = len(set(manga_labels))
    
    results = {
        'n_mangas': n_mangas,
        'manga_names': manga_names,
        'total_samples': len(embeddings)
    }
    
    if n_mangas > 1:
        print(f"\nComputing clustering metrics for {n_mangas} mangas...")
        results['manga_silhouette'] = silhouette_score(embeddings, manga_labels)
        results['manga_db'] = davies_bouldin_score(embeddings, manga_labels)
        results['manga_ch'] = calinski_harabasz_score(embeddings, manga_labels)
    else:
        print(f"\nOnly {n_mangas} manga in dataset - cannot compute separation metrics")
        results['manga_silhouette'] = None
        results['manga_db'] = None
        results['manga_ch'] = None
    
    return results


def evaluate_clustering_quality(embeddings: np.ndarray, metadata: list[dict]):
    """
    Evaluate and print clustering quality metrics.
    
    Args:
        embeddings: Embedding vectors
        metadata: List of metadata dictionaries
    
    Returns:
        dict: Metrics dictionary
    """
    print("\n" + "="*80)
    print("CLUSTERING QUALITY EVALUATION")
    print("="*80)
    
    metrics = compute_clustering_metrics(embeddings, metadata)
    
    # Evaluate by manga
    print("\n--- Grouping by MANGA ---")
    print(f"Number of unique manga: {metrics['n_mangas']}")
    print(f"Total samples: {metrics['total_samples']}")
    
    if metrics['n_mangas'] > 1:
        print(f"\nSilhouette Score:        {metrics['manga_silhouette']:.4f} (higher is better, range: -1 to 1)")
        print(f"Davies-Bouldin Index:    {metrics['manga_db']:.4f} (lower is better)")
        print(f"Calinski-Harabasz Index: {metrics['manga_ch']:.2f} (higher is better)")
        
        sil = metrics['manga_silhouette']
        if sil > 0.5:
            print("\n✓ Excellent: Embeddings strongly separate by manga")
        elif sil > 0.3:
            print("\n✓ Good: Embeddings moderately separate by manga")
        elif sil > 0.2:
            print("\n~ Fair: Embeddings weakly separate by manga")
        else:
            print("\n✗ Poor: Embeddings don't separate well by manga")
    else:
        print("Only one manga in dataset - cannot compute separation metrics")
    
    print("\n" + "="*80)
    
    return metrics


def plot_tsne_with_metrics(
    coords: np.ndarray,
    metadata: list[dict],
    metrics: dict,
    title: str = "t-SNE Visualization with Clustering Metrics",
    figsize: tuple = (16, 10),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization colored by manga with metrics displayed.
    
    Args:
        coords: t-SNE coordinates (N x 2)
        metadata: List of metadata dictionaries
        metrics: Metrics dictionary
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get manga categories
    categories = [m.get("manga", "unknown") for m in metadata]
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)
    
    # Color map
    cmap = plt.colormaps.get_cmap("tab20" if n_cats <= 20 else "hsv")
    colors = {cat: cmap(i / n_cats) for i, cat in enumerate(unique_cats)}
    
    # Plot each manga
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        
        ax.scatter(
            cat_coords[:, 0],
            cat_coords[:, 1],
            c=[colors[cat]],
            label=cat[:40],  # Truncate long names
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
        )
    
    # Build metrics text box
    metrics_text = "Colored by: MANGA\n"
    metrics_text += "="*40 + "\n"
    
    if metrics['n_mangas'] > 1:
        metrics_text += f"MANGA METRICS:\n"
        metrics_text += f"  Silhouette:        {metrics['manga_silhouette']:.4f}\n"
        metrics_text += f"  Davies-Bouldin:    {metrics['manga_db']:.4f}\n"
        metrics_text += f"  Calinski-Harabasz: {metrics['manga_ch']:.1f}\n"
        metrics_text += f"\nNumber of mangas: {metrics['n_mangas']}\n"
        metrics_text += f"Total samples: {metrics['total_samples']}\n"
    
    # Add metrics box to plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    
    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title="Manga",
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics(metrics: dict, output_path: Path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save metrics JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if key == 'manga_names':
            metrics_serializable[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif value is None:
            metrics_serializable[key] = None
        else:
            metrics_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Saved metrics to: {output_path}")


def detect_outliers_silhouette(
    embeddings: np.ndarray, 
    labels: np.ndarray,
    threshold_percentile: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using per-sample silhouette scores.
    
    Args:
        embeddings: Embedding vectors
        labels: Cluster labels
        threshold_percentile: Percentile below which samples are considered outliers
    
    Returns:
        tuple: (outlier boolean array, silhouette scores array)
    """
    if len(set(labels)) < 2:
        # Need at least 2 clusters
        return np.zeros(len(embeddings), dtype=bool), np.zeros(len(embeddings), dtype=float)
    
    # Compute per-sample silhouette scores
    sample_scores = silhouette_samples(embeddings, labels)
    
    # Threshold: samples below this percentile are outliers
    threshold = np.percentile(sample_scores, threshold_percentile)
    outliers = sample_scores < threshold
    
    return outliers, sample_scores


def detect_outliers_dbscan(
    embeddings: np.ndarray,
    labels: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> dict:
    """
    Detect outliers using DBSCAN within each cluster.
    
    Args:
        embeddings: Embedding vectors
        labels: Cluster labels
        eps: Maximum distance for DBSCAN
        min_samples: Minimum samples for DBSCAN
    
    Returns:
        Dictionary with outlier masks per cluster and overall
    """
    unique_labels = np.unique(labels)
    outlier_mask = np.zeros(len(embeddings), dtype=bool)
    cluster_outliers = {}
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_embeddings) < min_samples:
            # Too small for DBSCAN, mark all as outliers
            cluster_outliers[label] = np.ones(len(cluster_embeddings), dtype=bool)
            outlier_mask[cluster_indices] = True
            continue
        
        # Run DBSCAN on this cluster
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels_db = dbscan.fit_predict(cluster_embeddings)
        
        # Outliers are those labeled as -1 by DBSCAN
        is_outlier = cluster_labels_db == -1
        cluster_outliers[label] = is_outlier
        outlier_mask[cluster_indices[is_outlier]] = True
    
    return {
        'outlier_mask': outlier_mask,
        'cluster_outliers': cluster_outliers
    }


def detect_outliers_isolation_forest(
    embeddings: np.ndarray,
    labels: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42
) -> dict:
    """
    Detect outliers using Isolation Forest within each cluster.
    
    Args:
        embeddings: Embedding vectors
        labels: Cluster labels
        contamination: Expected proportion of outliers
        random_state: Random seed
    
    Returns:
        Dictionary with outlier masks per cluster and overall
    """
    unique_labels = np.unique(labels)
    outlier_mask = np.zeros(len(embeddings), dtype=bool)
    cluster_outliers = {}
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_embeddings) < 10:
            # Too small for Isolation Forest, skip
            cluster_outliers[label] = np.zeros(len(cluster_embeddings), dtype=bool)
            continue
        
        # Run Isolation Forest on this cluster
        iso_forest = IsolationForest(
            contamination=min(contamination, 0.5),
            random_state=random_state,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(cluster_embeddings)
        
        # Outliers are those labeled as -1
        is_outlier = predictions == -1
        cluster_outliers[label] = is_outlier
        outlier_mask[cluster_indices[is_outlier]] = True
    
    return {
        'outlier_mask': outlier_mask,
        'cluster_outliers': cluster_outliers
    }


def detect_all_outliers(
    embeddings: np.ndarray,
    metadata: list[dict],
    silhouette_threshold: float = 10.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    iso_contamination: float = 0.1
) -> dict:
    """
    Detect outliers using all three methods for manga clusters.
    
    Args:
        embeddings: Embedding vectors
        metadata: Metadata list
        silhouette_threshold: Percentile threshold for silhouette method
        dbscan_eps: Epsilon for DBSCAN
        dbscan_min_samples: Min samples for DBSCAN
        iso_contamination: Contamination for Isolation Forest
    
    Returns:
        Dictionary with all outlier detection results
    """
    labels, id_to_category = create_manga_labels(metadata)
    
    print(f"\nDetecting outliers for manga clusters...")
    print(f"Number of mangas: {len(set(labels))}")
    
    # Method 1: Silhouette Score per Sample
    print("  Method 1: Silhouette Score per Sample...")
    silhouette_outliers, silhouette_scores = detect_outliers_silhouette(
        embeddings, labels, threshold_percentile=silhouette_threshold
    )
    
    # Method 2: DBSCAN
    print("  Method 2: DBSCAN...")
    dbscan_results = detect_outliers_dbscan(
        embeddings, labels, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    
    # Method 3: Isolation Forest
    print("  Method 3: Isolation Forest...")
    iso_results = detect_outliers_isolation_forest(
        embeddings, labels, contamination=iso_contamination
    )
    
    # Combine results
    combined_outliers = (
        silhouette_outliers | 
        dbscan_results['outlier_mask'] | 
        iso_results['outlier_mask']
    )
    
    return {
        'labels': labels,
        'id_to_category': id_to_category,
        'silhouette': {
            'outliers': silhouette_outliers,
            'scores': silhouette_scores
        },
        'dbscan': dbscan_results,
        'isolation_forest': iso_results,
        'combined': combined_outliers
    }


def print_outlier_statistics(outlier_results: dict, metadata: list[dict]):
    """Print statistics about detected outliers."""
    labels = outlier_results['labels']
    id_to_category = outlier_results['id_to_category']
    
    print("\n" + "="*80)
    print("OUTLIER DETECTION STATISTICS - MANGA")
    print("="*80)
    
    # Overall statistics
    total = len(labels)
    sil_outliers = np.sum(outlier_results['silhouette']['outliers'])
    dbscan_outliers = np.sum(outlier_results['dbscan']['outlier_mask'])
    iso_outliers = np.sum(outlier_results['isolation_forest']['outlier_mask'])
    combined_outliers = np.sum(outlier_results['combined'])
    
    print(f"\nTotal samples: {total}")
    print(f"\nOutliers detected:")
    print(f"  Silhouette Score:     {sil_outliers:4d} ({100*sil_outliers/total:5.2f}%)")
    print(f"  DBSCAN:                {dbscan_outliers:4d} ({100*dbscan_outliers/total:5.2f}%)")
    print(f"  Isolation Forest:     {iso_outliers:4d} ({100*iso_outliers/total:5.2f}%)")
    print(f"  Combined (any method): {combined_outliers:4d} ({100*combined_outliers/total:5.2f}%)")
    
    # Per-cluster statistics
    print(f"\nPer-manga breakdown:")
    unique_labels = np.unique(labels)
    
    for label in sorted(unique_labels):
        cluster_mask = labels == label
        cluster_name = id_to_category[label]
        cluster_size = np.sum(cluster_mask)
        
        sil_count = np.sum(outlier_results['silhouette']['outliers'][cluster_mask])
        dbscan_count = np.sum(outlier_results['dbscan']['outlier_mask'][cluster_mask])
        iso_count = np.sum(outlier_results['isolation_forest']['outlier_mask'][cluster_mask])
        combined_count = np.sum(outlier_results['combined'][cluster_mask])
        
        print(f"\n  {cluster_name}:")
        print(f"    Total: {cluster_size:4d} | Outliers: {combined_count:4d} ({100*combined_count/cluster_size:5.2f}%)")
        print(f"      Silhouette: {sil_count:4d} | DBSCAN: {dbscan_count:4d} | Isolation Forest: {iso_count:4d}")
    
    print("\n" + "="*80)


def plot_outliers_tsne(
    coords: np.ndarray,
    metadata: list[dict],
    outlier_results: dict,
    metrics: dict,
    title: str = "t-SNE Visualization with Outliers",
    figsize: tuple = (18, 12),
    output_path: Path | None = None,
) -> None:
    """Plot t-SNE visualization with outliers highlighted."""
    labels = outlier_results['labels']
    id_to_category = outlier_results['id_to_category']
    
    # Get manga categories
    categories = [m.get("manga", "unknown") for m in metadata]
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)
    
    # Color map for clusters
    cmap = plt.colormaps.get_cmap("tab20" if n_cats <= 20 else "hsv")
    colors = {cat: cmap(i / n_cats) for i, cat in enumerate(unique_cats)}
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: All points colored by cluster
    ax1 = fig.add_subplot(gs[0, 0])
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        ax1.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c=[colors[cat]], label=cat[:30],
            s=50, alpha=0.6, edgecolors='white', linewidths=0.5
        )
    ax1.set_title('All Points (Colored by Manga)', fontsize=12, fontweight='bold')
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Silhouette outliers
    ax2 = fig.add_subplot(gs[0, 1])
    sil_outliers = outlier_results['silhouette']['outliers']
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        ax2.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c=[colors[cat]], label=cat[:30],
            s=50, alpha=0.3, edgecolors='white', linewidths=0.5
        )
    # Highlight outliers
    outlier_coords = coords[sil_outliers]
    ax2.scatter(
        outlier_coords[:, 0], outlier_coords[:, 1],
        c='red', marker='X', s=200, alpha=0.8,
        edgecolors='black', linewidths=1.5, label='Outliers (Silhouette)'
    )
    ax2.set_title('Silhouette Score Outliers', fontsize=12, fontweight='bold')
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DBSCAN outliers
    ax3 = fig.add_subplot(gs[1, 0])
    dbscan_outliers = outlier_results['dbscan']['outlier_mask']
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        ax3.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c=[colors[cat]], label=cat[:30],
            s=50, alpha=0.3, edgecolors='white', linewidths=0.5
        )
    # Highlight outliers
    outlier_coords = coords[dbscan_outliers]
    ax3.scatter(
        outlier_coords[:, 0], outlier_coords[:, 1],
        c='orange', marker='X', s=200, alpha=0.8,
        edgecolors='black', linewidths=1.5, label='Outliers (DBSCAN)'
    )
    ax3.set_title('DBSCAN Outliers', fontsize=12, fontweight='bold')
    ax3.set_xlabel("t-SNE 1")
    ax3.set_ylabel("t-SNE 2")
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined outliers
    ax4 = fig.add_subplot(gs[1, 1])
    combined_outliers = outlier_results['combined']
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        ax4.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c=[colors[cat]], label=cat[:30],
            s=50, alpha=0.3, edgecolors='white', linewidths=0.5
        )
    # Highlight outliers
    outlier_coords = coords[combined_outliers]
    ax4.scatter(
        outlier_coords[:, 0], outlier_coords[:, 1],
        c='purple', marker='X', s=200, alpha=0.8,
        edgecolors='black', linewidths=1.5, label='Outliers (Any Method)'
    )
    ax4.set_title('Combined Outliers (Any Method)', fontsize=12, fontweight='bold')
    ax4.set_xlabel("t-SNE 1")
    ax4.set_ylabel("t-SNE 2")
    ax4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = "Colored by: MANGA\n"
    metrics_text += "="*40 + "\n"
    
    if metrics['n_mangas'] > 1:
        metrics_text += f"MANGA METRICS:\n"
        metrics_text += f"  Silhouette:        {metrics['manga_silhouette']:.4f}\n"
        metrics_text += f"  Davies-Bouldin:    {metrics['manga_db']:.4f}\n"
        metrics_text += f"  Calinski-Harabasz: {metrics['manga_ch']:.1f}\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, metrics_text, fontsize=9,
             verticalalignment='bottom', bbox=props, family='monospace')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved outlier visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_outlier_results(
    outlier_results: dict,
    metadata: list[dict],
    output_path: Path
):
    """Save outlier detection results to JSON file."""
    results_dict = {
        'total_samples': len(outlier_results['labels']),
        'outlier_counts': {
            'silhouette': int(np.sum(outlier_results['silhouette']['outliers'])),
            'dbscan': int(np.sum(outlier_results['dbscan']['outlier_mask'])),
            'isolation_forest': int(np.sum(outlier_results['isolation_forest']['outlier_mask'])),
            'combined': int(np.sum(outlier_results['combined']))
        },
        'outliers': []
    }
    
    # Add outlier information for each sample
    for i, meta in enumerate(metadata):
        is_sil = bool(outlier_results['silhouette']['outliers'][i])
        is_dbscan = bool(outlier_results['dbscan']['outlier_mask'][i])
        is_iso = bool(outlier_results['isolation_forest']['outlier_mask'][i])
        is_combined = bool(outlier_results['combined'][i])
        
        if is_combined:
            results_dict['outliers'].append({
                'index': i,
                'path': meta.get('path', ''),
                'manga': meta.get('manga', ''),
                'chapter': meta.get('chapter', ''),
                'page': meta.get('page', ''),
                'silhouette_outlier': is_sil,
                'dbscan_outlier': is_dbscan,
                'isolation_forest_outlier': is_iso,
                'silhouette_score': float(outlier_results['silhouette']['scores'][i])
            })
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Saved outlier results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster and evaluate manga embeddings by manga with t-SNE visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic clustering with visualization (image embeddings)
  python metrics/cluster.py --input final_dataset_embeddings --output clusters/image_embeddings
  
  # Text embeddings clustering
  python metrics/cluster.py --input final_dataset_text_embeddings --output clusters/text_embeddings
  
  # Custom t-SNE parameters
  python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --perplexity 50 --n-iter 2000
  
  # Skip visualization (metrics only)
  python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --no-viz
  
  # Detect outliers in clusters
  python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --detect-outliers
  
  # Detect outliers with custom parameters
  python metrics/cluster.py --input final_dataset_embeddings --output clusters/char --detect-outliers --silhouette-threshold 5.0
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to embeddings dataset directory (e.g., final_dataset_embeddings)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., clusters/image_embeddings)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (default: 30)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="t-SNE iterations (default: 1000)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip t-SNE visualization (only compute metrics)",
    )
    parser.add_argument(
        "--detect-outliers",
        action="store_true",
        help="Detect outliers in each manga cluster using multiple methods",
    )
    parser.add_argument(
        "--silhouette-threshold",
        type=float,
        default=10.0,
        help="Percentile threshold for silhouette outliers (default: 10.0)",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter (default: 0.5)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter (default: 5)",
    )
    parser.add_argument(
        "--iso-contamination",
        type=float,
        default=0.1,
        help="Isolation Forest contamination (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    # Check sklearn availability
    if not SKLEARN_AVAILABLE:
        print("Error: sklearn is required for clustering. Install with: pip install scikit-learn")
        return
    
    # Load embeddings
    dataset_dir = Path(args.input)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Loading embeddings from: {dataset_dir}")
    loaded = load_embeddings_from_dataset(dataset_dir)
    
    if loaded is None:
        print("Error: Failed to load embeddings")
        return
    
    embeddings, metadata = loaded
    
    if len(embeddings) == 0:
        print("Error: No embeddings found")
        return
    
    # Compute metrics
    metrics = evaluate_clustering_quality(embeddings, metadata)
    
    # Save metrics
    output_dir = Path(args.output)
    metrics_file = output_dir / "metrics.json"
    save_metrics(metrics, metrics_file)
    
    # Compute t-SNE if needed for visualization or outlier detection
    coords = None
    if not args.no_viz or args.detect_outliers:
        print("\nGenerating t-SNE visualization...")
        coords = compute_tsne(
            embeddings,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
        )
    
    # Create visualization if requested
    if not args.no_viz:
        viz_file = output_dir / "tsne_visualization.png"
        plot_tsne_with_metrics(
            coords, metadata, metrics,
            title="Manga Embeddings Clustering by Manga",
            output_path=viz_file,
        )
    
    # Detect outliers if requested
    if args.detect_outliers:
        outlier_results = detect_all_outliers(
            embeddings,
            metadata,
            silhouette_threshold=args.silhouette_threshold,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            iso_contamination=args.iso_contamination
        )
        
        # Print statistics
        print_outlier_statistics(outlier_results, metadata)
        
        # Generate outlier visualization
        if coords is not None:
            outlier_viz_file = output_dir / "outliers_tsne.png"
            plot_outliers_tsne(
                coords, metadata, outlier_results, metrics,
                title="Outlier Detection - Manga Clusters",
                output_path=outlier_viz_file
            )
        
        # Save outlier results
        outlier_results_file = output_dir / "outliers.json"
        save_outlier_results(outlier_results, metadata, outlier_results_file)
    
    print(f"\n{'='*80}")
    print("CLUSTERING COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics saved to: {metrics_file}")
    if not args.no_viz:
        print(f"Visualization saved to: {output_dir / 'tsne_visualization.png'}")
    if args.detect_outliers:
        print(f"Outlier visualization saved to: {output_dir / 'outliers_tsne.png'}")
        print(f"Outlier results saved to: {output_dir / 'outliers.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

