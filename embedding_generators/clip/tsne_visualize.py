#!/usr/bin/env python3
"""
t-SNE Visualization for FAISS Index

Visualize the embedding space of indexed manga panels using t-SNE.

Usage:
    python embedding_generators/clip/tsne_visualize.py --index-dir datasets/small/faiss_index
    python embedding_generators/clip/tsne_visualize.py --index-dir datasets/small/faiss_index --with-thumbnails
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_embeddings_from_index(index_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """
    Load all embeddings and metadata from a FAISS index.
    
    Returns:
        tuple: (embeddings array, list of metadata dicts)
    """
    from faiss_image_index import ImageFaissIndex
    
    index = ImageFaissIndex.load(index_dir)
    
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
    # Adjust perplexity if needed
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


def plot_tsne(
    coords: np.ndarray,
    metadata: list[dict],
    title: str = "t-SNE Visualization of Manga Embeddings",
    figsize: tuple = (14, 10),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization with color coding by manga.
    
    Args:
        coords: (N, 2) t-SNE coordinates
        metadata: List of metadata dicts
        title: Plot title
        figsize: Figure size
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique manga
    categories = [m.get("manga", "Unknown") for m in metadata]
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)
    
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
            label=cat[:30],  # Truncate long names
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    # Legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title="Manga",
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
    thumbnail_size: int = 40,
    figsize: tuple = (20, 16),
    output_path: Path | None = None,
) -> None:
    """
    Plot t-SNE visualization with image thumbnails.
    
    Args:
        coords: (N, 2) t-SNE coordinates
        metadata: List of metadata dicts
        thumbnail_size: Size of thumbnails in pixels
        figsize: Figure size
        output_path: Optional path to save figure
    """
    from PIL import Image
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get manga colors for background dots
    mangas = [m.get("manga", "Unknown") for m in metadata]
    unique_mangas = sorted(set(mangas))
    cmap = plt.colormaps.get_cmap("tab10")
    manga_colors = {m: cmap(i % 10) for i, m in enumerate(unique_mangas)}
    
    # Plot background points
    for manga in unique_mangas:
        mask = [m == manga for m in mangas]
        manga_coords = coords[mask]
        ax.scatter(
            manga_coords[:, 0],
            manga_coords[:, 1],
            c=[manga_colors[manga]],
            s=20,
            alpha=0.3,
            label=manga[:25],
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
                    edgecolor=manga_colors[meta.get("manga", "Unknown")],
                    linewidth=2,
                ),
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading {meta['path']}: {e}")
    
    ax.set_title("t-SNE with Thumbnails", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, title="Manga")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_interactive_html(
    coords: np.ndarray,
    metadata: list[dict],
    output_path: Path,
) -> None:
    """
    Create an interactive HTML visualization.
    
    Args:
        coords: (N, 2) t-SNE coordinates
        metadata: List of metadata dicts
        output_path: Path to save HTML file
    """
    # Normalize coords to 0-100 range for CSS positioning
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_norm = (coords[:, 0] - x_min) / (x_max - x_min) * 90 + 5
    y_norm = (coords[:, 1] - y_min) / (y_max - y_min) * 90 + 5
    
    # Generate colors by manga
    mangas = sorted(set(m.get("manga", "Unknown") for m in metadata))
    colors = [
        f"hsl({int(i * 360 / len(mangas))}, 70%, 50%)"
        for i in range(len(mangas))
    ]
    manga_colors = dict(zip(mangas, colors))
    
    # Build HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>t-SNE Visualization</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #1a1a1a; font-family: Arial; }}
        h1 {{ color: white; text-align: center; }}
        .container {{ position: relative; width: 100%; height: 90vh; background: #2a2a2a; border-radius: 10px; }}
        .point {{
            position: absolute;
            width: 60px;
            height: 80px;
            transform: translate(-50%, -50%);
            cursor: pointer;
            transition: transform 0.2s, z-index 0s;
        }}
        .point:hover {{
            transform: translate(-50%, -50%) scale(2);
            z-index: 1000;
        }}
        .point img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 4px;
            border: 2px solid;
        }}
        .tooltip {{
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
        }}
        .point:hover .tooltip {{ display: block; }}
        .legend {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
    </style>
</head>
<body>
    <h1>t-SNE Embedding Visualization</h1>
    <div class="container">
        {points}
    </div>
    <div class="legend">
        <strong>Manga</strong>
        {legend}
    </div>
</body>
</html>"""
    
    # Generate points
    points_html = []
    for i, meta in enumerate(metadata):
        manga = meta.get("manga", "Unknown")
        color = manga_colors.get(manga, "#888888")
        point = f'''
        <div class="point" style="left: {x_norm[i]:.1f}%; top: {y_norm[i]:.1f}%;">
            <img src="file://{meta.get('path', '')}" style="border-color: {color};">
            <div class="tooltip">{manga}<br>{meta.get('chapter', 'Unknown')} / {meta.get('page', 'Unknown')}</div>
        </div>'''
        points_html.append(point)
    
    # Generate legend
    legend_html = []
    for manga, color in manga_colors.items():
        legend_html.append(
            f'<div class="legend-item"><div class="legend-dot" style="background: {color};"></div>{manga}</div>'
        )
    
    html = html.format(
        points="\n".join(points_html),
        legend="\n".join(legend_html),
    )
    
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Saved interactive visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip/tsne_visualize.py --index-dir datasets/small/faiss_index
  python clip/tsne_visualize.py --index-dir datasets/small/faiss_index --with-thumbnails
  python clip/tsne_visualize.py --index-dir datasets/small/faiss_index --html tsne.html
        """
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="datasets/small/faiss_index",
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--with-thumbnails",
        action="store_true",
        help="Show image thumbnails instead of dots",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Generate interactive HTML visualization",
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
    
    # Compute t-SNE
    coords = compute_tsne(
        embeddings,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )
    
    # Generate visualizations
    if args.html:
        plot_interactive_html(coords, metadata, Path(args.html))
    elif args.with_thumbnails:
        plot_tsne_with_thumbnails(
            coords, metadata,
            output_path=Path(args.output) if args.output else None,
        )
    else:
        plot_tsne(
            coords, metadata,
            output_path=Path(args.output) if args.output else None,
        )


if __name__ == "__main__":
    main()
