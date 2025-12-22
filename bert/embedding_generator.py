"""
BERT Embedding Generator for Manga Text Descriptions

This module provides functions to:
- Load BERT model (BAAI/bge-large-en-v1.5)
- Extract embeddings from text descriptions
- Parse metadata from text file paths
- Perform k-means clustering
- Visualize embeddings with t-SNE
- Check for token truncation
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
import json
import re
import matplotlib.pyplot as plt
import argparse

# Default max sequence length for BGE models
DEFAULT_MAX_SEQ_LENGTH = 512


def load_bert_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    Load BERT model for generating embeddings.
    
    Args:
        model_name: Name of the pretrained model
    
    Returns:
        SentenceTransformer model
    """
    print(f"Loading BERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model max sequence length: {model.max_seq_length} tokens")
    return model


def count_tokens(model: SentenceTransformer, text: str) -> int:
    """
    Count the number of tokens in a text using the model's tokenizer.
    
    Args:
        model: SentenceTransformer model
        text: Text to tokenize
    
    Returns:
        Number of tokens
    """
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


def check_truncation(
    model: SentenceTransformer,
    texts: list[str],
    paths: list[Path],
    verbose: bool = True
) -> dict:
    """
    Check which texts will be truncated based on model's max sequence length.
    
    Args:
        model: SentenceTransformer model
        texts: List of texts to check
        paths: List of corresponding file paths
        verbose: Whether to print warnings
    
    Returns:
        dict with truncation statistics and details
    """
    max_length = model.max_seq_length
    tokenizer = model.tokenizer
    
    truncation_info = {
        "max_seq_length": max_length,
        "total_texts": len(texts),
        "truncated_count": 0,
        "truncated_files": [],
        "token_counts": [],
        "max_tokens": 0,
        "min_tokens": float('inf'),
        "avg_tokens": 0,
    }
    
    total_tokens = 0
    
    for text, path in zip(texts, paths):
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        truncation_info["token_counts"].append(token_count)
        total_tokens += token_count
        
        if token_count > max_length:
            truncation_info["truncated_count"] += 1
            truncation_info["truncated_files"].append({
                "path": str(path),
                "token_count": token_count,
                "overflow": token_count - max_length
            })
        
        truncation_info["max_tokens"] = max(truncation_info["max_tokens"], token_count)
        truncation_info["min_tokens"] = min(truncation_info["min_tokens"], token_count)
    
    if texts:
        truncation_info["avg_tokens"] = total_tokens / len(texts)
    
    if verbose:
        print(f"\n=== Token Count Analysis ===")
        print(f"Model max sequence length: {max_length} tokens")
        print(f"Total texts: {len(texts)}")
        print(f"Token counts: min={truncation_info['min_tokens']}, "
              f"max={truncation_info['max_tokens']}, "
              f"avg={truncation_info['avg_tokens']:.1f}")
        
        if truncation_info["truncated_count"] > 0:
            print(f"\n⚠️  WARNING: {truncation_info['truncated_count']} texts "
                  f"({100*truncation_info['truncated_count']/len(texts):.1f}%) "
                  f"will be TRUNCATED!")
            print("Truncated files:")
            for item in truncation_info["truncated_files"][:10]:  # Show first 10
                print(f"  - {item['path']}: {item['token_count']} tokens "
                      f"(+{item['overflow']} over limit)")
            if len(truncation_info["truncated_files"]) > 10:
                print(f"  ... and {len(truncation_info['truncated_files']) - 10} more")
            print("\nConsider using a model with longer context (e.g., jina-embeddings-v2-base-en)")
        else:
            print(f"\n✅ All texts fit within the {max_length} token limit. No truncation.")
    
    return truncation_info


def get_all_text_files(text_dir: Path) -> list[Path]:
    """Recursively find all .txt files in a directory."""
    text_files = list(text_dir.rglob("*.txt"))
    return sorted(text_files)


def parse_text_metadata(path: Path) -> dict:
    """
    Parse metadata from text file path.
    Expected structure: .../author/manga/chapter_X/page_XXX.txt
    
    Returns:
        dict with author, manga, chapter, page, string_id
    """
    parts = path.parts
    
    # Extract components from path
    page_name = path.stem  # e.g., "page_006"
    chapter_dir = parts[-2] if len(parts) >= 2 else "unknown"  # e.g., "chapter_20"
    manga = parts[-3] if len(parts) >= 3 else "unknown"
    author = parts[-4] if len(parts) >= 4 else "unknown"
    
    # Extract page number
    page_match = re.search(r'(\d+)', page_name)
    page_num = page_match.group(1) if page_match else "000"
    
    # Extract chapter number
    chapter_match = re.search(r'(\d+)', chapter_dir)
    chapter_num = chapter_match.group(1) if chapter_match else "0"
    
    # Create a clean string ID
    clean_author = re.sub(r'[^a-zA-Z0-9]', '_', author).strip('_')
    clean_manga = re.sub(r'[^a-zA-Z0-9]', '_', manga).strip('_')
    string_id = f"{clean_author}__{clean_manga}__ch{chapter_num}__pg{page_num}"
    
    return {
        "author": author,
        "manga": manga,
        "chapter": chapter_dir,
        "chapter_num": int(chapter_num) if chapter_num.isdigit() else 0,
        "page": page_name,
        "page_num": int(page_num) if page_num.isdigit() else 0,
        "path": str(path),
        "string_id": string_id,
    }


def read_text_file(path: Path) -> str:
    """Read text content from a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""


def extract_embeddings(
    model: SentenceTransformer,
    text_paths: list[Path],
    batch_size: int = 32,
    normalize: bool = True,
    check_tokens: bool = True
) -> tuple[np.ndarray, list[Path], list[dict], dict | None]:
    """
    Extract BERT embeddings for all text files.
    
    Args:
        model: SentenceTransformer model
        text_paths: List of text file paths
        batch_size: Batch size for processing
        normalize: If True, L2 normalize embeddings (for cosine similarity)
        check_tokens: If True, check for token truncation before encoding
    
    Returns:
        tuple: (embeddings array, valid paths list, metadata list, truncation_info)
    """
    embeddings = []
    valid_paths = []
    metadata_list = []
    texts = []

    # Read all texts and collect metadata
    for path in tqdm(text_paths, desc="Reading text files"):
        text = read_text_file(path)
        if text:  # Only process non-empty files
            texts.append(text)
            valid_paths.append(path)
            metadata_list.append(parse_text_metadata(path))

    if not texts:
        return np.array([]).astype(np.float32), [], [], None

    # Check for truncation before encoding
    truncation_info = None
    if check_tokens:
        truncation_info = check_truncation(model, texts, valid_paths, verbose=True)

    # Generate embeddings in batches
    print(f"\nGenerating embeddings for {len(texts)} texts...")
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )

    return all_embeddings.astype(np.float32), valid_paths, metadata_list, truncation_info


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int = 10, random_state: int = 42
) -> np.ndarray:
    """Perform k-means clustering on embeddings."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def save_results(
    metadata_list: list[dict],
    labels: np.ndarray,
    embeddings: np.ndarray,
    output_dir: Path,
):
    """Save clustering results and embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # Add cluster labels to metadata
    results = []
    for meta, label in zip(metadata_list, labels):
        result = meta.copy()
        result["cluster"] = int(label)
        results.append(result)

    # Save clustering results as JSON
    with open(output_dir / "clusters.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print cluster summary
    print("\n=== Cluster Summary ===")
    for cluster_id in range(max(labels) + 1):
        cluster_items = [r for r in results if r["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_items)} texts")
        # Show sample from each cluster
        for item in cluster_items[:3]:
            print(f"  - {item['author']}/{item['manga']}/{item['chapter']}")


def plot_tsne(
    metadata_list: list[dict],
    embeddings: np.ndarray,
    output_path: Path,
    color_by: str = "cluster",
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
):
    """Create a t-SNE visualization of the embeddings."""
    print(f"Computing t-SNE projection (this may take a moment)...")
    
    # Adjust perplexity if needed
    perplexity = min(perplexity, len(embeddings) - 1)
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        init='pca',
        learning_rate='auto',
    )
    coords = tsne.fit_transform(embeddings)

    # Get colors based on cluster or author
    if color_by == "cluster":
        labels = [item["cluster"] for item in metadata_list]
        unique_labels = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
        colors = [cmap(unique_labels.index(l)) for l in labels]
        legend_labels = [f"Cluster {l}" for l in unique_labels]
    else:  # color by author
        authors = [item["author"] for item in metadata_list]
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


def plot_cluster_summary(
    metadata_list: list[dict],
    output_path: Path,
    max_per_cluster: int = 10,
):
    """Create a text summary visualization showing cluster contents."""
    # Group texts by cluster
    cluster_groups = {}
    for item in metadata_list:
        cluster_id = item["cluster"]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(item)

    n_clusters = len(cluster_groups)
    
    # Create a text-based summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CLUSTER SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for cluster_id in sorted(cluster_groups.keys()):
        items = cluster_groups[cluster_id]
        summary_lines.append(f"Cluster {cluster_id}: {len(items)} texts")
        summary_lines.append("-" * 80)
        
        # Group by author/manga for better readability
        by_author_manga = {}
        for item in items:
            key = f"{item['author']}/{item['manga']}"
            if key not in by_author_manga:
                by_author_manga[key] = []
            by_author_manga[key].append(item)
        
        for key, group_items in sorted(by_author_manga.items()):
            summary_lines.append(f"  {key}: {len(group_items)} texts")
            for item in group_items[:max_per_cluster]:
                summary_lines.append(f"    - {item['chapter']}/{item['page']}")
        
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    # Save as text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Saved cluster summary to: {output_path}")


def generate_html_report(metadata_list: list[dict], output_path: Path):
    """Generate an interactive HTML report for browsing clusters."""
    # Group texts by cluster
    cluster_groups = {}
    for item in metadata_list:
        cluster_id = item["cluster"]
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(item)

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Text Cluster Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
        h1 {{ text-align: center; }}
        .cluster {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 10px; }}
        .cluster h2 {{ margin-top: 0; color: #4a9eff; }}
        .texts {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .text-card {{ 
            background: #3a3a3a; 
            border-radius: 8px; 
            padding: 15px; 
            width: 300px;
            transition: transform 0.2s;
        }}
        .text-card:hover {{ transform: scale(1.02); }}
        .text-card .meta {{ 
            font-size: 12px; 
            color: #aaa; 
            margin-bottom: 8px;
        }}
        .text-card .author {{ color: #4a9eff; font-weight: bold; }}
        .text-card .manga {{ color: #9aff4a; }}
        .text-card .content {{
            font-size: 11px;
            color: #ccc;
            max-height: 200px;
            overflow-y: auto;
            line-height: 1.4;
        }}
        .stats {{ text-align: center; margin: 20px 0; color: #888; }}
        .nav {{ position: fixed; top: 10px; right: 10px; background: #2a2a2a; padding: 10px; border-radius: 8px; }}
        .nav a {{ color: #4a9eff; text-decoration: none; display: block; margin: 5px 0; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>📚 Text Cluster Visualization</h1>
    <div class="stats">Total texts: {total} | Clusters: {n_clusters}</div>
    
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

        texts_html = ""
        for item in items:
            # Read text content
            text_content = read_text_file(Path(item["path"]))
            # Truncate if too long
            if len(text_content) > 500:
                text_content = text_content[:500] + "..."
            
            texts_html += f"""
            <div class="text-card">
                <div class="meta">
                    <div class="author">{item['author']}</div>
                    <div class="manga">{item['manga']}</div>
                    <div>{item['chapter']} / {item['page']}</div>
                </div>
                <div class="content">{text_content}</div>
            </div>"""

        cluster_html_parts.append(f"""
        <div class="cluster" id="cluster-{cluster_id}">
            <h2>Cluster {cluster_id} ({len(items)} texts)</h2>
            <div class="texts">{texts_html}</div>
        </div>""")

    final_html = html.format(
        total=len(metadata_list),
        n_clusters=len(cluster_groups),
        nav_links="\n".join(nav_links),
        cluster_html="\n".join(cluster_html_parts),
    )

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(final_html)

    print(f"Saved HTML report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings from manga text descriptions and perform clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Bert/embedding_generator.py --text-dir downloads_text
  python Bert/embedding_generator.py --text-dir downloads_text --n-clusters 15
  python Bert/embedding_generator.py --text-dir downloads_text --output-dir Bert/output
        """
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default="downloads_text",
        help="Path to directory containing text files (default: downloads_text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (default: Bert/output)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Name of the pretrained BERT model (default: BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters for k-means (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations (t-SNE plots, HTML report)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize embeddings (default: normalize)",
    )
    
    args = parser.parse_args()

    # Setup paths
    text_dir = Path(args.text_dir)
    if not text_dir.exists():
        print(f"Text directory not found: {text_dir}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "output"

    # Load model
    model = load_bert_model(args.model_name)

    # Get all text files
    print(f"Scanning for text files in: {text_dir}")
    text_paths = get_all_text_files(text_dir)
    print(f"Found {len(text_paths)} text files")

    if not text_paths:
        print("No text files found!")
        return

    # Extract embeddings (with truncation check)
    embeddings, valid_paths, metadata_list, truncation_info = extract_embeddings(
        model,
        text_paths,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        check_tokens=True
    )
    print(f"\nExtracted embeddings for {len(valid_paths)} texts")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    if len(valid_paths) == 0:
        print("No valid texts to cluster!")
        return

    # Adjust number of clusters if needed
    n_clusters = min(args.n_clusters, len(valid_paths))
    if n_clusters < args.n_clusters:
        print(f"Warning: Reducing clusters from {args.n_clusters} to {n_clusters} (number of texts)")

    # Cluster embeddings
    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

    # Add cluster labels to metadata
    for meta, label in zip(metadata_list, labels):
        meta["cluster"] = int(label)

    # Save results
    save_results(metadata_list, labels, embeddings, output_dir)

    # Save truncation info if available
    if truncation_info:
        truncation_report = {
            "model": args.model_name,
            "max_seq_length": truncation_info["max_seq_length"],
            "total_texts": truncation_info["total_texts"],
            "truncated_count": truncation_info["truncated_count"],
            "truncation_rate": truncation_info["truncated_count"] / truncation_info["total_texts"] if truncation_info["total_texts"] > 0 else 0,
            "token_stats": {
                "min": truncation_info["min_tokens"],
                "max": truncation_info["max_tokens"],
                "avg": round(truncation_info["avg_tokens"], 2)
            },
            "truncated_files": truncation_info["truncated_files"]
        }
        with open(output_dir / "truncation_report.json", "w") as f:
            json.dump(truncation_report, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - embeddings.npy: {embeddings.shape} embedding vectors")
    print(f"  - clusters.json: clustering results with metadata")
    if truncation_info:
        print(f"  - truncation_report.json: token analysis report")

    # Generate visualizations if requested
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # t-SNE plots
        plot_tsne(
            metadata_list,
            embeddings,
            vis_dir / "tsne_clusters.png",
            color_by="cluster"
        )
        plot_tsne(
            metadata_list,
            embeddings,
            vis_dir / "tsne_authors.png",
            color_by="author"
        )
        
        # Cluster summary
        plot_cluster_summary(metadata_list, vis_dir / "cluster_summary.txt")
        
        # HTML report
        generate_html_report(metadata_list, vis_dir / "clusters.html")
        
        print(f"\n✨ All visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
