#!/usr/bin/env python3
"""
FAISS Index for Text Search with Image Visualization

Build and search FAISS index from CLIP text embeddings.
Maps text results to corresponding images and deduplicates by page.

Usage:
    python clip/faiss_text_index.py final_dataset_text_embeddings
    python clip/faiss_text_index.py --search final_dataset_text_embeddings/faiss_index --query "tall warrior" --image-dir final_dataset
    python clip/faiss_text_index.py --search final_dataset_text_embeddings/faiss_index --query "tall warrior" -k 100 --visualize
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Lazy FAISS import
_faiss = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class TextFaissIndex:
    """FAISS index for text similarity search using cosine similarity."""
    
    __slots__ = ('dimension', 'index', 'id_to_meta', '_next_id')
    
    def __init__(self, dimension: int = 768, use_ivf: bool = False, nlist: int = 100):
        faiss = _get_faiss()
        self.dimension = dimension
        
        if use_ivf:
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        self.id_to_meta: dict[int, dict] = {}
        self._next_id = 0
    
    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add normalized embeddings with metadata."""
        if len(embeddings) == 0:
            return
        
        faiss = _get_faiss()
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
        
        for i, meta in enumerate(metadata):
            self.id_to_meta[self._next_id + i] = meta
        self._next_id += len(metadata)
        
        self.index.add(embeddings)
        print(f"Indexed {len(embeddings)} texts (total: {self.index.ntotal})")
    
    def search(self, query: np.ndarray, k: int = 10) -> list[dict]:
        """Search for similar texts."""
        faiss = _get_faiss()
        
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, k)
        
        return [
            {"rank": i + 1, "similarity": float(s), **self.id_to_meta[idx]}
            for i, (s, idx) in enumerate(zip(scores[0], indices[0]))
            if idx != -1 and idx in self.id_to_meta
        ]
    
    def save(self, output_dir: Path) -> None:
        """Save index and metadata."""
        faiss = _get_faiss()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump({
                "dimension": self.dimension,
                "total": self.index.ntotal,
                "metadata": {str(k): v for k, v in self.id_to_meta.items()},
            }, f)
        
        print(f"Saved index ({self.index.ntotal} vectors) to {output_dir}")
    
    @classmethod
    def load(cls, index_dir: Path) -> "TextFaissIndex":
        """Load index from disk."""
        faiss = _get_faiss()
        index_dir = Path(index_dir)
        
        instance = cls.__new__(cls)
        instance.index = faiss.read_index(str(index_dir / "faiss.index"))
        
        with open(index_dir / "metadata.json") as f:
            data = json.load(f)
        
        instance.dimension = data.get("dimension", instance.index.d)
        instance.id_to_meta = {int(k): v for k, v in data["metadata"].items()}
        instance._next_id = max(instance.id_to_meta.keys(), default=-1) + 1
        
        print(f"Loaded index: {instance.index.ntotal} vectors")
        return instance
    
    def get_embedding(self, idx: int) -> np.ndarray | None:
        """Reconstruct embedding by ID."""
        return self.index.reconstruct(idx) if idx in self.id_to_meta else None
    
    def __len__(self) -> int:
        return self.index.ntotal


def load_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Load embeddings and metadata from text_embeddings.py output."""
    embeddings_dir = Path(embeddings_dir)
    
    embeddings = np.load(embeddings_dir / "all_embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    
    with open(embeddings_dir / "metadata.json") as f:
        data = json.load(f)
    
    metadata = data.get("files", [])
    input_dir = data.get("input_dir", "")
    
    for m in metadata:
        m["input_dir"] = input_dir
    
    print(f"Loaded metadata: {len(metadata)} entries")
    return embeddings, metadata


def build_index(embeddings_dir: Path, output_dir: Path = None, use_ivf: bool = False) -> TextFaissIndex:
    """Build FAISS index from embeddings directory."""
    print(f"\n{'='*50}\n  FAISS Text Index Builder\n{'='*50}")
    
    embeddings, metadata = load_embeddings(embeddings_dir)
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings found")
    
    index = TextFaissIndex(
        dimension=embeddings.shape[1],
        use_ivf=use_ivf,
        nlist=min(100, len(embeddings) // 10) if use_ivf else 100
    )
    index.add(embeddings, metadata)
    
    output_dir = output_dir or embeddings_dir / "faiss_index"
    index.save(output_dir)
    
    print(f"\n{'='*50}\n  Done! {len(index)} texts indexed\n{'='*50}")
    return index


def get_page_key(meta: dict) -> str:
    """Extract unique page identifier from metadata."""
    # source_file format: "manga/chapter_X/page_XXX.txt"
    source = meta.get("source_file") or meta.get("path", "")
    # Remove extension and return as key
    return str(Path(source).with_suffix(""))


def get_text(meta: dict) -> str:
    """Get text content from metadata."""
    if "text" in meta:
        return meta["text"]
    
    input_dir = meta.get("input_dir", "")
    rel_path = meta.get("path") or meta.get("source_file", "")
    
    if input_dir and rel_path:
        try:
            return (Path(input_dir) / rel_path).read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def find_image_path(meta: dict, image_dir: Path) -> Path | None:
    """Find corresponding image file for a text result."""
    # Get page key (e.g., "Berserk/chapter_105/page_011")
    page_key = get_page_key(meta)
    
    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        img_path = image_dir / (page_key + ext)
        if img_path.exists():
            return img_path
    
    return None


def deduplicate_by_page(results: list[dict]) -> list[dict]:
    """
    Deduplicate results by page, keeping only highest similarity per page.
    
    Args:
        results: List of search results (already sorted by similarity)
    
    Returns:
        Deduplicated results with updated ranks
    """
    seen_pages = {}
    
    for r in results:
        page_key = get_page_key(r)
        
        # Keep only the first (highest similarity) result per page
        if page_key not in seen_pages:
            seen_pages[page_key] = r
    
    # Update ranks
    deduped = list(seen_pages.values())
    for i, r in enumerate(deduped):
        r["rank"] = i + 1
        r["page_key"] = get_page_key(r)
    
    return deduped


def visualize_results(results: list[dict], image_dir: Path, query: str = None, 
                      output_path: Path = None, show: bool = True):
    """
    Visualize search results with corresponding images.
    
    Args:
        results: Deduplicated search results
        image_dir: Directory containing source images
        query: Query text for title
        output_path: Path to save visualization
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    n = len(results)
    if n == 0:
        print("No results to visualize")
        return
    
    # Determine grid layout
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Title
    title = f'Text Search Results: "{query[:50]}..."' if query and len(query) > 50 else f'Text Search Results: "{query}"'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, r in enumerate(results):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Find and display image
        img_path = find_image_path(r, image_dir)
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', fontsize=8)
                ax.set_facecolor('#f0f0f0')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=10)
            ax.set_facecolor('#f0f0f0')
        
        # Title with info
        page_key = r.get("page_key", get_page_key(r))
        parts = page_key.split("/")
        manga = parts[0] if parts else "?"
        rest = "/".join(parts[1:]) if len(parts) > 1 else ""
        
        # Truncate manga name
        if len(manga) > 20:
            manga = manga[:17] + "..."
        
        text_preview = get_text(r)[:40] + "..." if len(get_text(r)) > 40 else get_text(r)
        
        ax.set_title(f"#{r['rank']} [{r['similarity']:.3f}]\n{manga}\n{rest}", fontsize=9)
        ax.set_xlabel(text_preview, fontsize=7, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results(results: list[dict], query: str, output_path: Path, image_dir: Path = None):
    """Save search results to JSON file."""
    # Add image paths to results
    results_with_images = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "input_dir"}  # Exclude input_dir
        if image_dir:
            img_path = find_image_path(r, image_dir)
            r_copy["image_path"] = str(img_path) if img_path else None
        results_with_images.append(r_copy)
    
    data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "results": results_with_images,
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved results to: {output_path}")


def print_results(results: list[dict], image_dir: Path = None):
    """Print search results."""
    print("\nResults:\n" + "-" * 60)
    
    for r in results:
        page_key = r.get("page_key", get_page_key(r))
        line_idx = r.get("line_index", 0)
        
        print(f"\n  #{r['rank']}. [{r['similarity']:.3f}] {page_key} (line {line_idx})")
        
        # Show image path if available
        if image_dir:
            img_path = find_image_path(r, image_dir)
            if img_path:
                print(f"      Image: {img_path}")
        
        # Show text
        text = get_text(r)
        if text:
            text = text[:150] + "..." if len(text) > 150 else text
            print(f"      Text: {text}")


def search(index_dir: Path, query: str = None, k: int = 50,
           image_dir: Path = None, visualize: bool = False, save_viz: Path = None,
           save_json: Path = None):
    """
    Search the index with deduplication and visualization.
    
    Args:
        index_dir: Path to FAISS index
        query: Search query text
        k: Number of candidates to retrieve
        image_dir: Directory containing source images
        visualize: Whether to display visualization
        save_viz: Path to save visualization image
        save_json: Path to save results JSON
    """
    print("\n=== Text Search ===")
    
    if not query:
        print("Error: Query text required for search")
        return
    
    # Load CLIP before FAISS to avoid macOS segfault
    import torch
    import open_clip
    
    print("Loading CLIP model...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    # Encode query
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
    
    # Load index and search
    index = TextFaissIndex.load(index_dir)
    
    print(f"\nQuery: \"{query}\"")
    print(f"Retrieving {k} candidates...")
    
    # Get candidates
    raw_results = index.search(emb, k=k)
    
    # Deduplicate by page
    results = deduplicate_by_page(raw_results)
    
    print(f"Found {len(raw_results)} candidates → {len(results)} unique pages")
    
    # Print results
    print_results(results, image_dir)
    
    # Save results
    if save_json:
        save_results(results, query, save_json, image_dir)
    
    # Visualize
    if (visualize or save_viz) and image_dir:
        visualize_results(results, image_dir, query, save_viz, visualize)
    elif visualize or save_viz:
        print("\nWarning: --image-dir required for visualization")


def main():
    parser = argparse.ArgumentParser(
        description="FAISS text search with image visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build index
    python clip/faiss_text_index.py final_dataset_text_embeddings
    
    # Search with deduplication
    python clip/faiss_text_index.py --search final_dataset_text_embeddings/faiss_index \\
        --query "tall muscular male warrior" -k 50 --image-dir final_dataset
    
    # Search with visualization
    python clip/faiss_text_index.py --search final_dataset_text_embeddings/faiss_index \\
        --query "young female" -k 100 --image-dir final_dataset --visualize
    
    # Save results
    python clip/faiss_text_index.py --search final_dataset_text_embeddings/faiss_index \\
        --query "dark hair" --image-dir final_dataset --save-viz results.png --save-json results.json
        """
    )
    parser.add_argument("embeddings_dir", nargs="?", help="Embeddings directory to index")
    parser.add_argument("--output-dir", help="Output directory for index")
    parser.add_argument("--use-ivf", action="store_true", help="Use IVF for large datasets")
    parser.add_argument("--search", help="Index directory to search")
    parser.add_argument("--query", help="Search query text")
    parser.add_argument("-k", type=int, default=50, help="Number of candidates to retrieve (default: 50)")
    parser.add_argument("--image-dir", help="Directory containing source images for visualization")
    parser.add_argument("--visualize", action="store_true", help="Display visualization")
    parser.add_argument("--save-viz", help="Save visualization to file")
    parser.add_argument("--save-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.search:
        search(
            Path(args.search),
            args.query,
            k=args.k,
            image_dir=Path(args.image_dir) if args.image_dir else None,
            visualize=args.visualize,
            save_viz=Path(args.save_viz) if args.save_viz else None,
            save_json=Path(args.save_json) if args.save_json else None,
        )
    elif args.embeddings_dir:
        if not Path(args.embeddings_dir).exists():
            print(f"Error: {args.embeddings_dir} not found")
            return
        build_index(
            Path(args.embeddings_dir),
            Path(args.output_dir) if args.output_dir else None,
            args.use_ivf
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
