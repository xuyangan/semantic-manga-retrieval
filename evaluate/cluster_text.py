#!/usr/bin/env python3
"""
Text Embeddings Character Clustering

This script analyzes text embeddings of character descriptions and clusters them
by character attributes (currently gender) rather than by manga.

Sample Execution:
    # Basic gender clustering
    python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender
    
    # With custom t-SNE parameters
    python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --perplexity 50 --n-iter 2000
    
    # Skip visualization (metrics only)
    python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --no-viz
    
    # With outlier detection
    python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --detect-outliers
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. Clustering features will be unavailable.")


def load_text_embeddings_and_descriptions(dataset_dir: Path):
    """
    Load text embeddings and descriptions from metadata.json.
    
    Args:
        dataset_dir: Directory containing all_embeddings.npy and metadata.json
    
    Returns:
        tuple: (embeddings array, descriptions list, metadata list) or None if not found
    """
    dataset_dir = Path(dataset_dir)
    
    embeddings_file = dataset_dir / "all_embeddings.npy"
    metadata_file = dataset_dir / "metadata.json"
    
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}")
        return None
    
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return None
    
    print(f"Loading embeddings from: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata_json = json.load(f)
    
    if 'files' not in metadata_json:
        print("Error: metadata.json does not contain 'files' array")
        return None
    
    # Extract text descriptions and create metadata
    descriptions = []
    metadata = []
    
    for file_entry in metadata_json['files']:
        # Get text description
        text = file_entry.get('text', '')
        descriptions.append(text)
        
        # Get source file path
        source_file = file_entry.get('source_file', '')
        embedding_file = file_entry.get('embedding_file', '')
        
        # Parse manga/chapter/page from source file
        parts = source_file.split('/')
        if len(parts) >= 3:
            manga = parts[0]
            chapter = parts[1]
            page = parts[2].replace('.txt', '')
        else:
            manga = "unknown"
            chapter = "unknown"
            page = "unknown"
        
        metadata.append({
            'text': text,
            'source_file': source_file,
            'embedding_file': embedding_file,
            'manga': manga,
            'chapter': chapter,
            'page': page
        })
    
    if len(embeddings) != len(descriptions):
        print(f"Warning: Embeddings ({len(embeddings)}) and descriptions ({len(descriptions)}) count mismatch!")
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    print(f"Loaded {len(descriptions)} text descriptions")
    
    # Print manga distribution
    manga_counts = defaultdict(int)
    for m in metadata:
        manga_counts[m['manga']] += 1
    
    print(f"\nDataset contains {len(manga_counts)} unique manga:")
    for manga, count in sorted(manga_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {manga}: {count} descriptions")
    
    return embeddings, descriptions, metadata


def extract_gender_from_description(text: str) -> str:
    """
    Extract gender from character description.
    
    Handles multiple description formats:
    - Old format: "gender; age; hair; ..." (semicolon-separated)
    - New format: "gender, age_range, hair_color, ..." (comma-separated)
    - Keyword detection in any format
    
    Args:
        text: Character description text
    
    Returns:
        Gender: 'male', 'female', 'ambiguous', or 'unknown'
    """
    if not text:
        return 'unknown'
    
    text_lower = text.lower().strip()
    
    # Check for ambiguous first (as it may contain male/female keywords)
    if 'ambiguous' in text_lower:
        return 'ambiguous'
    
    # Try to get first field from the description
    # Handle both semicolon (old format) and comma (new format) separators
    first_field = None
    
    # First, try semicolon separator (old format)
    if ';' in text:
        first_field = text.split(';')[0].strip().lower()
    # Then try comma separator (new format)
    elif ',' in text:
        first_field = text.split(',')[0].strip().lower()
    else:
        # No separator found, use the whole text
        first_field = text_lower
    
    # Check the first field for gender keywords
    if first_field:
        # Check for 'female' first (since 'male' is a substring of 'female')
        if 'female' in first_field:
            return 'female'
        elif 'male' in first_field:
            return 'male'
    
    # Fallback: look for gender keywords anywhere in the text
    # This helps with malformed descriptions or different formats
    
    # Count gender-specific keywords
    female_keywords = ['female', 'woman', 'girl', 'lady', 'she', 'her']
    male_keywords = ['male', 'man', 'boy', 'guy', 'he', 'his']
    
    # Split into words for more accurate matching
    words = text_lower.split()
    
    female_count = sum(1 for kw in female_keywords if kw in words)
    male_count = sum(1 for kw in male_keywords if kw in words)
    
    if female_count > male_count:
        return 'female'
    elif male_count > female_count:
        return 'male'
    
    return 'unknown'


def create_gender_labels(descriptions: list[str]) -> tuple[np.ndarray, dict, dict]:
    """
    Create gender labels from text descriptions.
    
    Args:
        descriptions: List of character description texts
    
    Returns:
        tuple: (labels array, id_to_gender mapping, gender_counts dict)
    """
    print("\nExtracting gender from descriptions...")
    
    genders = []
    for desc in descriptions:
        gender = extract_gender_from_description(desc)
        genders.append(gender)
    
    # Create label mappings
    unique_genders = sorted(set(genders))
    gender_to_id = {gender: i for i, gender in enumerate(unique_genders)}
    id_to_gender = {i: gender for gender, i in gender_to_id.items()}
    
    labels = np.array([gender_to_id[gender] for gender in genders])
    
    # Count distribution
    gender_counts = defaultdict(int)
    for gender in genders:
        gender_counts[gender] += 1
    
    print(f"\nGender distribution:")
    for gender in sorted(gender_counts.keys()):
        count = gender_counts[gender]
        percentage = 100 * count / len(genders)
        print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    return labels, id_to_gender, dict(gender_counts)


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray, label_names: dict):
    """
    Compute clustering metrics.
    
    Args:
        embeddings: Embedding vectors (N x D)
        labels: Cluster labels
        label_names: Mapping from label ID to name
    
    Returns:
        dict: Dictionary containing all metrics
    """
    n_clusters = len(set(labels))
    
    results = {
        'n_clusters': n_clusters,
        'label_names': label_names,
        'total_samples': len(embeddings)
    }
    
    if n_clusters > 1:
        print(f"\nComputing clustering metrics for {n_clusters} gender groups...")
        results['silhouette'] = silhouette_score(embeddings, labels)
        results['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
    else:
        print(f"\nOnly {n_clusters} group in dataset - cannot compute separation metrics")
        results['silhouette'] = None
        results['davies_bouldin'] = None
        results['calinski_harabasz'] = None
    
    return results


def evaluate_clustering_quality(embeddings: np.ndarray, labels: np.ndarray, label_names: dict, gender_counts: dict):
    """
    Evaluate and print clustering quality metrics.
    
    Args:
        embeddings: Embedding vectors
        labels: Cluster labels
        label_names: Mapping from label ID to name
        gender_counts: Count of each gender
    
    Returns:
        dict: Metrics dictionary
    """
    print("\n" + "="*80)
    print("CLUSTERING QUALITY EVALUATION - GENDER")
    print("="*80)
    
    metrics = compute_clustering_metrics(embeddings, labels, label_names)
    metrics['gender_counts'] = gender_counts
    
    print(f"\nNumber of gender groups: {metrics['n_clusters']}")
    print(f"Total samples: {metrics['total_samples']}")
    
    print(f"\nDistribution:")
    for gender, count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / metrics['total_samples']
        print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    if metrics['n_clusters'] > 1:
        print(f"\nClustering Metrics:")
        print(f"  Silhouette Score:        {metrics['silhouette']:.4f} (higher is better, range: -1 to 1)")
        print(f"  Davies-Bouldin Index:    {metrics['davies_bouldin']:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f} (higher is better)")
        
        sil = metrics['silhouette']
        if sil > 0.5:
            print("\n✓ Excellent: Embeddings strongly separate by gender")
        elif sil > 0.3:
            print("\n✓ Good: Embeddings moderately separate by gender")
        elif sil > 0.2:
            print("\n~ Fair: Embeddings weakly separate by gender")
        else:
            print("\n~ Weak separation by gender")
            print("  Note: This is normal for character descriptions encoded by CLIP.")
            print("  Gender may overlap with other attributes (age, build, clothing).")
    else:
        print("Only one gender group in dataset - cannot compute separation metrics")
    
    print("\n" + "="*80)
    
    return metrics


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


def plot_tsne_by_gender(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: dict,
    metrics: dict,
    title: str = "Character Embeddings Clustered by Gender",
    figsize: tuple = (14, 10),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization colored by gender.
    
    Args:
        coords: t-SNE coordinates (N x 2)
        labels: Gender labels
        label_names: Mapping from label ID to gender name
        metrics: Metrics dictionary
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    
    # Define colors for genders
    color_map = {
        'male': '#3498db',      # Blue
        'female': '#e74c3c',    # Red
        'ambiguous': '#95a5a6', # Gray
        'unknown': '#34495e'    # Dark gray
    }
    
    # Plot each gender
    for label in unique_labels:
        mask = labels == label
        label_coords = coords[mask]
        gender_name = label_names.get(label, f"Unknown_{label}")
        color = color_map.get(gender_name, '#2ecc71')  # Default green
        
        ax.scatter(
            label_coords[:, 0],
            label_coords[:, 1],
            c=color,
            label=gender_name.title(),
            s=100,
            alpha=0.6,
            edgecolors='white',
            linewidths=0.5,
        )
    
    # Build metrics text box
    metrics_text = "Grouped by: GENDER\n"
    metrics_text += "="*40 + "\n"
    
    if metrics['n_clusters'] > 1:
        metrics_text += f"GENDER METRICS:\n"
        metrics_text += f"  Silhouette:        {metrics['silhouette']:.4f}\n"
        metrics_text += f"  Davies-Bouldin:    {metrics['davies_bouldin']:.4f}\n"
        metrics_text += f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}\n"
        metrics_text += f"\nGender distribution:\n"
        for gender, count in sorted(metrics['gender_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / metrics['total_samples']
            metrics_text += f"  {gender}: {count} ({percentage:.0f}%)\n"
    
    # Add metrics box to plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(
        loc='upper right',
        fontsize=11,
        title="Gender",
        title_fontsize=12,
        framealpha=0.9
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
    def convert_value(value):
        """Recursively convert numpy types to Python types."""
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, dict):
            return {str(k): convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [convert_value(item) for item in value]
        elif value is None:
            return None
        else:
            return value
    
    metrics_serializable = convert_value(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Saved metrics to: {output_path}")


def save_gender_samples(
    descriptions: list[str],
    metadata: list[dict],
    labels: np.ndarray,
    label_names: dict,
    output_path: Path,
    samples_per_gender: int = 10
):
    """
    Save sample descriptions for each gender to a text file.
    
    Args:
        descriptions: List of character descriptions
        metadata: List of metadata dictionaries
        labels: Gender labels
        label_names: Mapping from label ID to gender name
        output_path: Path to save samples
        samples_per_gender: Number of samples per gender
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SAMPLE CHARACTER DESCRIPTIONS BY GENDER\n")
        f.write("="*80 + "\n\n")
        
        # Group samples by gender
        for label_id in sorted(set(labels)):
            gender = label_names[label_id]
            mask = labels == label_id
            indices = np.where(mask)[0]
            
            # Sample up to N from this gender
            sample_indices = indices[:samples_per_gender]
            
            f.write(f"\n{gender.upper()}\n")
            f.write("-" * 80 + "\n\n")
            
            for idx in sample_indices:
                desc = descriptions[idx]
                meta = metadata[idx]
                f.write(f"Source: {meta['manga']} / {meta['chapter']} / {meta['page']}\n")
                f.write(f"Description: {desc}\n")
                f.write("\n")
    
    print(f"Saved sample descriptions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster text embeddings by character gender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic gender clustering
  python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender
  
  # With custom t-SNE parameters
  python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --perplexity 50 --n-iter 2000
  
  # Skip visualization (metrics only)
  python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --no-viz
  
  # Save sample descriptions
  python metrics/cluster_text.py --input final_dataset_text_embeddings --output clusters/text_by_gender --save-samples
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to text embeddings dataset directory (e.g., final_dataset_text_embeddings)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory (e.g., clusters/text_by_gender)",
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
        "--save-samples",
        action="store_true",
        help="Save sample descriptions for each gender to a text file",
    )
    parser.add_argument(
        "--samples-per-gender",
        type=int,
        default=10,
        help="Number of sample descriptions to save per gender (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Check sklearn availability
    if not SKLEARN_AVAILABLE:
        print("Error: sklearn is required for clustering. Install with: pip install scikit-learn")
        return
    
    # Load embeddings and descriptions
    dataset_dir = Path(args.input)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Loading text embeddings from: {dataset_dir}")
    loaded = load_text_embeddings_and_descriptions(dataset_dir)
    
    if loaded is None:
        print("Error: Failed to load embeddings and descriptions")
        return
    
    embeddings, descriptions, metadata = loaded
    
    if len(embeddings) == 0:
        print("Error: No embeddings found")
        return
    
    # Extract gender labels
    labels, label_names, gender_counts = create_gender_labels(descriptions)
    
    # Compute metrics
    metrics = evaluate_clustering_quality(embeddings, labels, label_names, gender_counts)
    
    # Save metrics
    output_dir = Path(args.output)
    metrics_file = output_dir / "metrics.json"
    save_metrics(metrics, metrics_file)
    
    # Save sample descriptions if requested
    if args.save_samples:
        samples_file = output_dir / "gender_samples.txt"
        save_gender_samples(
            descriptions, metadata, labels, label_names,
            samples_file, args.samples_per_gender
        )
    
    # Compute t-SNE and create visualization if requested
    if not args.no_viz:
        print("\nGenerating t-SNE visualization...")
        coords = compute_tsne(
            embeddings,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
        )
        
        viz_file = output_dir / "tsne_by_gender.png"
        plot_tsne_by_gender(
            coords, labels, label_names, metrics,
            title="Character Embeddings Clustered by Gender",
            output_path=viz_file,
        )
    
    print(f"\n{'='*80}")
    print("CLUSTERING COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics saved to: {metrics_file}")
    if args.save_samples:
        print(f"Sample descriptions saved to: {output_dir / 'gender_samples.txt'}")
    if not args.no_viz:
        print(f"Visualization saved to: {output_dir / 'tsne_by_gender.png'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

