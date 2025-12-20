#!/usr/bin/env python3
"""
Evaluate embedding quality by measuring how well similar content clusters together.

This computes silhouette scores using author/manga as labels and visualizes with t-SNE.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score


def load_embeddings_from_index(index_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """
    Load all embeddings and metadata from a FAISS index.
    
    Returns:
        tuple: (embeddings array, list of metadata dicts)
    """
    from faiss_index import MangaFaissIndex
    
    index = MangaFaissIndex.load(index_dir)
    
    n = len(index)
    dim = index.dimension
    
    # Reconstruct all embeddings
    embeddings = np.zeros((n, dim), dtype=np.float32)
    metadata = []
    
    for i, (int_id, meta) in enumerate(sorted(index.id_to_meta.items())):
        embeddings[i] = index.get_embedding(int_id)
        metadata.append(meta)
    
    return embeddings, metadata


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute t-SNE projection of embeddings.
    
    Args:
        embeddings: (N, D) embedding matrix
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
    
    Returns:
        (N, 2) array of 2D coordinates
    """
    perplexity = min(perplexity, len(embeddings) - 1)
    
    print(f"Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    
    coords = tsne.fit_transform(embeddings)
    return coords


def create_labels(metadata: list[dict], label_type: str) -> tuple[np.ndarray, dict]:
    """
    Create numeric labels from metadata.
    
    Args:
        metadata: List of metadata dicts
        label_type: 'author' or 'manga'
    
    Returns:
        tuple: (label array, id_to_name dict)
    """
    if label_type == "author":
        categories = [m["author"] for m in metadata]
    elif label_type == "manga":
        categories = [m["manga"] for m in metadata]
    else:
        raise ValueError(f"Invalid label_type: {label_type}")
    
    unique_categories = sorted(set(categories))
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    
    labels = np.array([category_to_id[cat] for cat in categories])
    id_to_category = {i: cat for cat, i in category_to_id.items()}
    
    return labels, id_to_category


def evaluate_embedding_quality(embeddings: np.ndarray, metadata: list[dict]):
    """
    Evaluate how well embeddings separate content by author and manga.
    
    Args:
        embeddings: (N, D) embedding array
        metadata: List of metadata dicts
    """
    print("\n" + "="*80)
    print("EMBEDDING QUALITY EVALUATION")
    print("="*80)
    
    results = {}
    
    # Evaluate by author
    print("\n--- Grouping by AUTHOR (Art Style) ---")
    author_labels, author_names = create_labels(metadata, "author")
    
    n_authors = len(set(author_labels))
    print(f"Number of unique authors: {n_authors}")
    
    if n_authors > 1:
        author_silhouette = silhouette_score(embeddings, author_labels)
        author_db = davies_bouldin_score(embeddings, author_labels)
        
        print(f"\nSilhouette Score (author): {author_silhouette:.4f}")
        print(f"  → Higher is better (range: -1 to 1)")
        print(f"  → Measures: Do panels from same author cluster together?")
        
        print(f"\nDavies-Bouldin Index (author): {author_db:.4f}")
        print(f"  → Lower is better")
        print(f"  → Measures: How well-separated are different authors?")
        
        if author_silhouette > 0.5:
            print("\n✓ Excellent: Embeddings strongly capture art style differences")
        elif author_silhouette > 0.3:
            print("\n✓ Good: Embeddings moderately capture art style")
        elif author_silhouette > 0.2:
            print("\n~ Fair: Embeddings weakly capture art style")
        else:
            print("\n✗ Poor: Embeddings don't separate by art style")
        
        results['author_silhouette'] = author_silhouette
        results['author_db'] = author_db
    else:
        print("Only one author in dataset - cannot compute separation metrics")
        results['author_silhouette'] = None
        results['author_db'] = None
    
    # Evaluate by manga
    print("\n--- Grouping by MANGA (Story/Characters) ---")
    manga_labels, manga_names = create_labels(metadata, "manga")
    
    n_mangas = len(set(manga_labels))
    print(f"Number of unique manga: {n_mangas}")
    
    if n_mangas > 1:
        manga_silhouette = silhouette_score(embeddings, manga_labels)
        manga_db = davies_bouldin_score(embeddings, manga_labels)
        
        print(f"\nSilhouette Score (manga): {manga_silhouette:.4f}")
        print(f"  → Higher is better (range: -1 to 1)")
        print(f"  → Measures: Do panels from same manga cluster together?")
        
        print(f"\nDavies-Bouldin Index (manga): {manga_db:.4f}")
        print(f"  → Lower is better")
        print(f"  → Measures: How well-separated are different manga?")
        
        if manga_silhouette > 0.5:
            print("\n✓ Excellent: Embeddings strongly capture manga-specific features")
        elif manga_silhouette > 0.3:
            print("\n✓ Good: Embeddings moderately capture manga-specific features")
        elif manga_silhouette > 0.2:
            print("\n~ Fair: Embeddings weakly capture manga-specific features")
        else:
            print("\n✗ Poor: Embeddings don't separate by manga")
        
        results['manga_silhouette'] = manga_silhouette
        results['manga_db'] = manga_db
    else:
        print("Only one manga in dataset - cannot compute separation metrics")
        results['manga_silhouette'] = None
        results['manga_db'] = None
    
    # Comparison
    if n_authors > 1 and n_mangas > 1:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        if author_silhouette > manga_silhouette + 0.1:
            print(f"\nEmbeddings primarily capture ART STYLE (author)")
            print(f"  Author silhouette ({author_silhouette:.3f}) >> Manga silhouette ({manga_silhouette:.3f})")
            print(f"  → Good for: Finding panels with similar artistic characteristics")
        elif manga_silhouette > author_silhouette + 0.1:
            print(f"\nEmbeddings primarily capture STORY/CHARACTERS (manga)")
            print(f"  Manga silhouette ({manga_silhouette:.3f}) >> Author silhouette ({author_silhouette:.3f})")
            print(f"  → Good for: Finding panels from same series")
        else:
            print(f"\nEmbeddings capture BOTH art style and story")
            print(f"  Author silhouette: {author_silhouette:.3f}")
            print(f"  Manga silhouette: {manga_silhouette:.3f}")
            print(f"  → Balanced representation")
    
    print("\n" + "="*80)
    
    results['n_authors'] = n_authors
    results['n_mangas'] = n_mangas
    
    return results


def plot_tsne_with_metrics(
    coords: np.ndarray,
    metadata: list[dict],
    embeddings: np.ndarray,
    color_by: str = "author",
    title: str = "t-SNE Visualization with Silhouette Scores",
    figsize: tuple = (14, 10),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization with color coding and silhouette metrics.
    Same format as original but with metrics added to title.
    
    Args:
        coords: (N, 2) t-SNE coordinates
        metadata: List of metadata dicts
        embeddings: Original high-dimensional embeddings
        color_by: 'author' or 'manga'
        title: Plot title
        figsize: Figure size
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories
    if color_by == "author":
        categories = [m["author"] for m in metadata]
    else:
        categories = [m["manga"] for m in metadata]
    
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)
    
    # Compute silhouette score for this grouping
    labels, _ = create_labels(metadata, color_by)
    if n_cats > 1:
        silhouette = silhouette_score(embeddings, labels)
    else:
        silhouette = None
    
    # Color map
    cmap = plt.colormaps.get_cmap("tab20" if n_cats <= 20 else "hsv")
    colors = {cat: cmap(i / n_cats) for i, cat in enumerate(unique_cats)}
    
    # Plot each category
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        
        ax.scatter(
            cat_coords[:, 0],
            cat_coords[:, 1],
            c=[colors[cat]],
            label=cat[:30],
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
        )
    
    # Add silhouette score to title
    if silhouette is not None:
        title_with_score = f"{title}\nSilhouette Score ({color_by}): {silhouette:.4f}"
    else:
        title_with_score = title
    
    ax.set_title(title_with_score, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title=color_by.capitalize(),
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tsne_with_thumbnails(
    coords: np.ndarray,
    metadata: list[dict],
    embeddings: np.ndarray,
    thumbnail_size: int = 40,
    figsize: tuple = (20, 16),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization with image thumbnails and metrics.
    Same format as original but with metrics added to title.
    
    Args:
        coords: (N, 2) t-SNE coordinates
        metadata: List of metadata dicts
        embeddings: Original high-dimensional embeddings
        thumbnail_size: Size of thumbnails in pixels
        figsize: Figure size
        output_path: Optional path to save figure
    """
    from PIL import Image
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get author colors for background dots
    authors = [m["author"] for m in metadata]
    unique_authors = sorted(set(authors))
    cmap = plt.colormaps.get_cmap("tab10")
    author_colors = {a: cmap(i % 10) for i, a in enumerate(unique_authors)}
    
    # Compute silhouette scores
    author_labels, _ = create_labels(metadata, "author")
    manga_labels, _ = create_labels(metadata, "manga")
    
    author_sil = silhouette_score(embeddings, author_labels) if len(set(author_labels)) > 1 else None
    manga_sil = silhouette_score(embeddings, manga_labels) if len(set(manga_labels)) > 1 else None
    
    # Plot background points
    for author in unique_authors:
        mask = [a == author for a in authors]
        author_coords = coords[mask]
        ax.scatter(
            author_coords[:, 0],
            author_coords[:, 1],
            c=[author_colors[author]],
            s=20,
            alpha=0.3,
            label=author[:25],
        )
    
    # Add thumbnails
    for i, meta in enumerate(metadata):
        try:
            img = Image.open(meta["path"]).convert("RGB")
            img.thumbnail((thumbnail_size, thumbnail_size))
            
            imagebox = OffsetImage(np.array(img), zoom=0.8)
            ab = AnnotationBbox(
                imagebox,
                (coords[i, 0], coords[i, 1]),
                frameon=True,
                pad=0.1,
                bboxprops=dict(
                    edgecolor=author_colors[meta["author"]],
                    linewidth=2,
                ),
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading {meta['path']}: {e}")
    
    # Title with metrics
    title_parts = ["t-SNE with Thumbnails"]
    if author_sil is not None:
        title_parts.append(f"Silhouette (author): {author_sil:.4f}")
    if manga_sil is not None:
        title_parts.append(f"Silhouette (manga): {manga_sil:.4f}")
    
    ax.set_title("\n".join(title_parts), fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, title="Author")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding quality and visualize with t-SNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_embeddings.py --index-dir datasets/small/faiss_index
  python evaluate_embeddings.py --index-dir datasets/small/faiss_index --color-by author
  python evaluate_embeddings.py --index-dir datasets/small/faiss_index --color-by manga
  python evaluate_embeddings.py --index-dir datasets/small/faiss_index --with-thumbnails
        """
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="datasets/small/faiss_index",
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["author", "manga"],
        default="author",
        help="Color points by author or manga (default: author)",
    )
    parser.add_argument(
        "--with-thumbnails",
        action="store_true",
        help="Show image thumbnails instead of dots",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save plot to file (e.g., tsne.png)",
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
    
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir)
    
    # Load embeddings
    print(f"Loading embeddings from: {index_dir}")
    embeddings, metadata = load_embeddings_from_index(index_dir)
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Evaluate embedding quality
    metrics = evaluate_embedding_quality(embeddings, metadata)
    
    # Compute t-SNE
    coords = compute_tsne(
        embeddings,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )
    
    # Generate visualizations
    if args.with_thumbnails:
        plot_tsne_with_thumbnails(
            coords, metadata, embeddings,
            output_path=Path(args.output) if args.output else None,
        )
    else:
        plot_tsne_with_metrics(
            coords, metadata, embeddings,
            color_by=args.color_by,
            output_path=Path(args.output) if args.output else None,
        )


if __name__ == "__main__":
    main()