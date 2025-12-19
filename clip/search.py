#!/usr/bin/env python3
"""
Manga Panel Similarity Search

Search for similar manga panels using CLIP embeddings and FAISS.

Usage:
    python clip/search.py path/to/query/image.png
    python clip/search.py path/to/image.png --k 10
    python clip/search.py path/to/image.png --visualize
    python clip/search.py path/to/image.png --index-dir datasets/medium/faiss_index
"""

import os
# Fix OpenMP duplicate library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

import numpy as np


def visualize_results(
    query_image: Path,
    results: list[dict],
    output_path: Path | None = None,
    max_results: int = 5,
) -> None:
    """
    Visualize search results with query and matching panels.
    
    Args:
        query_image: Path to query image
        results: Search results from search()
        output_path: Optional path to save visualization
        max_results: Maximum number of results to show
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    results = results[:max_results]
    n_results = len(results)
    
    # Create figure: query on left, results on right
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 5))
    
    if n_results == 0:
        axes = [axes]
    
    # Plot query image
    query_img = Image.open(query_image).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontsize=12, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # Add border to query
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color('blue')
        spine.set_linewidth(3)
    
    # Plot results
    for i, r in enumerate(results):
        ax = axes[i + 1]
        
        try:
            result_img = Image.open(r['path']).convert("RGB")
            ax.imshow(result_img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{e}", ha='center', va='center')
        
        # Title with similarity and metadata
        sim_pct = r['similarity'] * 100
        author = r['author'][:15] + '...' if len(r['author']) > 15 else r['author']
        manga = r['manga'][:20] + '...' if len(r['manga']) > 20 else r['manga']
        title = f"#{r['rank']} ({sim_pct:.1f}%)\n{author}\n{manga}\nCh.{r['chapter_num']} Pg.{r['page_num']}"
        
        # Color based on similarity
        if sim_pct >= 80:
            color = 'green'
        elif sim_pct >= 60:
            color = 'orange'
        else:
            color = 'red'
        
        ax.set_title(title, fontsize=10, color=color)
        ax.axis('off')
    
    plt.suptitle(f"Similar Panels to: {query_image.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def search_by_id(
    query_id: int,
    index_dir: Path,
    k: int = 5,
    visualize: bool = False,
    save_viz: Path | None = None,
) -> list[dict]:
    """
    Search using an already-indexed image (no CLIP model needed).
    
    Args:
        query_id: Integer ID of indexed image
        index_dir: Path to FAISS index directory
        k: Number of results to return
        visualize: Whether to display visualization
        save_viz: Optional path to save visualization
    
    Returns:
        List of search results
    """
    from faiss_index import MangaFaissIndex
    
    index_dir = Path(index_dir)
    index = MangaFaissIndex.load(index_dir)
    
    # Get query metadata and embedding
    query_meta = index.get_by_id(query_id)
    if not query_meta:
        raise ValueError(f"ID {query_id} not found in index")
    
    query_emb = index.get_embedding(query_id)
    
    print(f"Query: {query_meta['author']}/{query_meta['manga']}/{query_meta['page']}")
    
    # Search (k+1 to exclude self)
    results = index.search(query_emb, k=k+1)
    results = [r for r in results if r.get('string_id') != query_meta['string_id']][:k]
    
    # Re-rank
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    # Display
    print("\n" + "=" * 60)
    print(f"  Similar Panels")
    print("=" * 60)
    
    for r in results:
        sim_pct = r['similarity'] * 100
        print(f"\n  #{r['rank']} - Similarity: {sim_pct:.1f}%")
        print(f"      Author:  {r['author']}")
        print(f"      Manga:   {r['manga']}")
        print(f"      Chapter: {r['chapter']} (pg {r['page_num']})")
    
    print("\n" + "=" * 60)
    
    if visualize or save_viz:
        visualize_results(Path(query_meta['path']), results, output_path=save_viz)
    
    return results


def search(
    query_image: Path,
    index_dir: Path,
    k: int = 5,
    show_paths: bool = False,
    visualize: bool = False,
    save_viz: Path | None = None,
) -> list[dict]:
    """
    Search for similar manga panels.
    
    Args:
        query_image: Path to query image
        index_dir: Path to FAISS index directory
        k: Number of results to return
        show_paths: Whether to show full file paths
        visualize: Whether to display visualization
        save_viz: Optional path to save visualization
    
    Returns:
        List of search results with metadata
    """
    from faiss_index import MangaFaissIndex, _get_clip_funcs
    from PIL import Image
    import torch
    
    # Validate inputs
    query_image = Path(query_image)
    if not query_image.exists():
        raise FileNotFoundError(f"Query image not found: {query_image}")
    
    index_dir = Path(index_dir)
    if not (index_dir / "faiss.index").exists():
        raise FileNotFoundError(f"FAISS index not found in: {index_dir}")
    
    # Load CLIP model and encode query
    print(f"Encoding query image: {query_image}")
    load_model, _, _, _ = _get_clip_funcs()
    model, preprocess, device = load_model()
    
    # Preprocess and encode
    img = Image.open(query_image).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype(np.float32)
    
    # Load index
    print(f"Loading index from: {index_dir}")
    index = MangaFaissIndex.load(index_dir)
    
    # Search
    print(f"Searching for top {k} similar panels...\n")
    results = index.search(embedding, k=k)
    
    _print_results(query_image, results, show_paths)
    
    # Visualize if requested
    if visualize or save_viz:
        visualize_results(query_image, results, output_path=save_viz)
    
    return results


def _print_results(query_image: Path, results: list[dict], show_paths: bool = False):
    """Print search results."""
    print("=" * 60)
    print(f"  Search Results (Query: {query_image.name})")
    print("=" * 60)
    
    for r in results:
        sim_pct = r['similarity'] * 100
        print(f"\n  #{r['rank']} - Similarity: {sim_pct:.1f}%")
        print(f"      Author:  {r['author']}")
        print(f"      Manga:   {r['manga']}")
        print(f"      Chapter: {r['chapter']} (pg {r['page_num']})")
        if show_paths:
            print(f"      Path:    {r['path']}")
    
    print("\n" + "=" * 60)


def search_batch(
    query_images: list[Path],
    index_dir: Path,
    k: int = 5,
) -> dict[str, list[dict]]:
    """
    Search for multiple query images at once (more efficient).
    
    Args:
        query_images: List of query image paths
        index_dir: Path to FAISS index directory
        k: Number of results per query
    
    Returns:
        Dict mapping query filename to results
    """
    from faiss_index import MangaFaissIndex, _get_clip_funcs
    
    # Load index
    index = MangaFaissIndex.load(index_dir)
    
    # Load CLIP model
    load_model, _, _, _ = _get_clip_funcs()
    model, preprocess, device = load_model()
    
    from PIL import Image
    import torch
    
    # Encode all queries
    embeddings = []
    valid_paths = []
    
    for path in query_images:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy())
                valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    if not embeddings:
        return {}
    
    # Batch search
    query_matrix = np.vstack(embeddings).astype(np.float32)
    all_results = index.search_batch(query_matrix, k=k)
    
    return {str(path): results for path, results in zip(valid_paths, all_results)}


def main():
    parser = argparse.ArgumentParser(
        description="Search for similar manga panels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with an image (uses cached embedding if in index)
  python clip/search.py query.png -v
  
  # Search by index ID (no CLIP model needed)
  python clip/search.py --by-id 0 -v
  
  # Save visualization
  python clip/search.py query.png --save-viz results.png
  
  # Use different index
  python clip/search.py query.png --index-dir datasets/large/faiss_index
        """
    )
    parser.add_argument(
        "query_image",
        type=str,
        nargs="?",
        default=None,
        help="Path to query image",
    )
    parser.add_argument(
        "--by-id",
        type=int,
        default=None,
        help="Search by index ID (no CLIP model needed)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="datasets/small/faiss_index",
        help="Path to FAISS index directory (default: datasets/small/faiss_index)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Display visualization of results",
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Save visualization to file (e.g., results.png)",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Show full file paths in results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--list-ids",
        action="store_true",
        help="List all indexed images with their IDs",
    )
    
    args = parser.parse_args()
    
    try:
        # List IDs mode
        if args.list_ids:
            from faiss_index import MangaFaissIndex
            index = MangaFaissIndex.load(args.index_dir)
            print(f"\nIndexed images ({len(index)} total):\n")
            for int_id, meta in sorted(index.id_to_meta.items()):
                print(f"  {int_id:4d}: {meta['author']}/{meta['manga']}/{meta['chapter']}/{meta['page']}")
            return
        
        # Search by ID mode
        if args.by_id is not None:
            results = search_by_id(
                query_id=args.by_id,
                index_dir=args.index_dir,
                k=args.top_k,
                visualize=args.visualize,
                save_viz=Path(args.save_viz) if args.save_viz else None,
            )
        # Search by image mode
        elif args.query_image:
            results = search(
                query_image=args.query_image,
                index_dir=args.index_dir,
                k=args.top_k,
                show_paths=args.show_paths,
                visualize=args.visualize,
                save_viz=Path(args.save_viz) if args.save_viz else None,
            )
        else:
            parser.print_help()
            print("\nError: Provide query_image or --by-id", file=sys.stderr)
            sys.exit(1)
            return
        
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
