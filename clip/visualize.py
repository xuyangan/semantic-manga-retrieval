import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import argparse


def load_data(output_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Load clusters and embeddings from output directory."""
    with open(output_dir / "clusters.json") as f:
        clusters = json.load(f)
    embeddings = np.load(output_dir / "embeddings.npy")
    return clusters, embeddings


def plot_cluster_grid(clusters: list[dict], output_path: Path, max_per_cluster: int = 6):
    """Create a grid visualization showing images grouped by cluster."""
    # Group images by cluster
    cluster_groups = {}
    for item in clusters:
        cluster_id = item["cluster"]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(item)

    n_clusters = len(cluster_groups)
    fig, axes = plt.subplots(
        n_clusters, max_per_cluster, figsize=(max_per_cluster * 3, n_clusters * 3)
    )

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for row, cluster_id in enumerate(sorted(cluster_groups.keys())):
        items = cluster_groups[cluster_id][:max_per_cluster]

        for col in range(max_per_cluster):
            ax = axes[row, col]
            ax.axis("off")

            if col < len(items):
                item = items[col]
                try:
                    img = Image.open(item["path"]).convert("RGB")
                    ax.imshow(img)
                    # Show manga name as title for first column
                    if col == 0:
                        ax.set_title(
                            f"Cluster {cluster_id}\n({len(cluster_groups[cluster_id])} images)",
                            fontsize=10,
                            fontweight="bold",
                        )
                    else:
                        ax.set_title(f"{item['manga'][:20]}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster grid to: {output_path}")


def plot_tsne(
    clusters: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    color_by: str = "cluster",
):
    """Create a t-SNE visualization of the embeddings."""
    print("Computing t-SNE projection (this may take a moment)...")

    # Compute t-SNE
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Get colors based on cluster or author
    if color_by == "cluster":
        labels = [item["cluster"] for item in clusters]
        unique_labels = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
        colors = [cmap(unique_labels.index(l)) for l in labels]
        legend_labels = [f"Cluster {l}" for l in unique_labels]
    else:  # color by author
        authors = [item["author"] for item in clusters]
        unique_authors = sorted(set(authors))
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_authors))
        colors = [cmap(unique_authors.index(a)) for a in authors]
        legend_labels = unique_authors
        labels = [unique_authors.index(a) for a in authors]

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=100, alpha=0.7)

    # Add legend
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=10
        )
        for i in range(len(legend_labels))
    ]
    ax.legend(handles, legend_labels, loc="best", fontsize=8)

    ax.set_title(f"t-SNE Visualization (colored by {color_by})", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to: {output_path}")


def plot_tsne_with_thumbnails(
    clusters: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    thumbnail_size: int = 64,
):
    """Create a t-SNE visualization with image thumbnails."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    print("Computing t-SNE projection with thumbnails...")

    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(20, 16))

    # Plot points first (for reference)
    labels = [item["cluster"] for item in clusters]
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=10, alpha=0.3)

    # Add thumbnails
    for i, item in enumerate(clusters):
        try:
            img = Image.open(item["path"]).convert("RGB")
            img.thumbnail((thumbnail_size, thumbnail_size))
            img_array = np.array(img)

            imagebox = OffsetImage(img_array, zoom=0.5)
            ab = AnnotationBbox(imagebox, (coords[i, 0], coords[i, 1]), frameon=True, pad=0.1)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading thumbnail for {item['path']}: {e}")

    ax.set_title("t-SNE Visualization with Thumbnails", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE with thumbnails to: {output_path}")


def generate_html_report(clusters: list[dict], output_path: Path):
    """Generate an interactive HTML report for browsing clusters."""
    # Group images by cluster
    cluster_groups = {}
    for item in clusters:
        cluster_id = item["cluster"]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(item)

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Manga Cluster Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
        h1 {{ text-align: center; }}
        .cluster {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 10px; }}
        .cluster h2 {{ margin-top: 0; color: #4a9eff; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .image-card {{ 
            background: #3a3a3a; 
            border-radius: 8px; 
            padding: 10px; 
            width: 200px;
            transition: transform 0.2s;
        }}
        .image-card:hover {{ transform: scale(1.05); }}
        .image-card img {{ 
            width: 100%; 
            height: 250px; 
            object-fit: contain; 
            background: #000;
            border-radius: 4px;
        }}
        .image-card .meta {{ 
            font-size: 11px; 
            color: #aaa; 
            margin-top: 8px;
            word-wrap: break-word;
        }}
        .image-card .author {{ color: #4a9eff; font-weight: bold; }}
        .image-card .manga {{ color: #9aff4a; }}
        .stats {{ text-align: center; margin: 20px 0; color: #888; }}
        .nav {{ position: fixed; top: 10px; right: 10px; background: #2a2a2a; padding: 10px; border-radius: 8px; }}
        .nav a {{ color: #4a9eff; text-decoration: none; display: block; margin: 5px 0; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>🎨 Manga Cluster Visualization</h1>
    <div class="stats">Total images: {total} | Clusters: {n_clusters}</div>
    
    <div class="nav">
        <strong>Jump to:</strong>
        {nav_links}
    </div>
    
    {cluster_html}
</body>
</html>"""

    cluster_html_parts = []
    nav_links = []

    for cluster_id in sorted(cluster_groups.keys()):
        items = cluster_groups[cluster_id]
        nav_links.append(f'<a href="#cluster-{cluster_id}">Cluster {cluster_id} ({len(items)})</a>')

        images_html = ""
        for item in items:
            # Use relative path for portability
            rel_path = Path(item["path"]).as_posix()
            images_html += f"""
            <div class="image-card">
                <img src="file://{rel_path}" alt="{item['manga']}" loading="lazy">
                <div class="meta">
                    <div class="author">{item['author']}</div>
                    <div class="manga">{item['manga']}</div>
                    <div>{item['chapter']}</div>
                </div>
            </div>"""

        cluster_html_parts.append(f"""
        <div class="cluster" id="cluster-{cluster_id}">
            <h2>Cluster {cluster_id} ({len(items)} images)</h2>
            <div class="images">{images_html}</div>
        </div>""")

    final_html = html.format(
        total=len(clusters),
        n_clusters=len(cluster_groups),
        nav_links="\n".join(nav_links),
        cluster_html="\n".join(cluster_html_parts),
    )

    with open(output_path, "w") as f:
        f.write(final_html)

    print(f"Saved HTML report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize manga clustering results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory containing clusters.json and embeddings.npy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "grid", "tsne", "tsne-thumbs", "html"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["cluster", "author"],
        default="cluster",
        help="How to color points in t-SNE plot",
    )
    args = parser.parse_args()

    # Default output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "output"

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        print("Please run clip.py first to generate clusters.")
        return

    # Load data
    clusters, embeddings = load_data(output_dir)
    print(f"Loaded {len(clusters)} images with {embeddings.shape[1]}-dim embeddings")

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Generate visualizations based on mode
    if args.mode in ["all", "grid"]:
        plot_cluster_grid(clusters, vis_dir / "cluster_grid.png")

    if args.mode in ["all", "tsne"]:
        plot_tsne(clusters, embeddings, vis_dir / "tsne_clusters.png", color_by="cluster")
        plot_tsne(clusters, embeddings, vis_dir / "tsne_authors.png", color_by="author")

    if args.mode in ["all", "tsne-thumbs"]:
        plot_tsne_with_thumbnails(clusters, embeddings, vis_dir / "tsne_thumbnails.png")

    if args.mode in ["all", "html"]:
        generate_html_report(clusters, vis_dir / "clusters.html")

    print(f"\n✨ All visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
