#!/usr/bin/env python3
"""
CLIP Text Embedding Generator

Generates CLIP text embeddings from text files.
Each line in a text file becomes a separate embedding.

Usage:
    python clip/text_embeddings.py final_dataset_text
    python clip/text_embeddings.py final_dataset_text --output-dir custom_embeddings
    python clip/text_embeddings.py final_dataset_text --batch-size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import open_clip
from tqdm import tqdm

# CLIP token limit (including start/end tokens)
CLIP_MAX_TOKENS = 77


def get_device(prefer_cpu: bool = False) -> str:
    """Get best available device."""
    if prefer_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip_model(device: str = None):
    """Load CLIP model and tokenizer."""
    if device is None:
        device = get_device()
    
    print(f"Loading CLIP model (ViT-L-14) on {device}...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    
    return model, tokenizer, device


class TokenCounter:
    """Token counter using tokenizer's encode method for accurate counts."""
    
    __slots__ = ('_tokenizer', '_has_encode')
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # SimpleTokenizer has .encode(), HFTokenizer has .tokenizer.encode()
        self._has_encode = hasattr(tokenizer, 'encode')
    
    def count(self, text: str) -> int:
        """Count actual tokens (including start/end tokens) without truncation."""
        if self._has_encode:
            # SimpleTokenizer.encode() returns BPE tokens without start/end
            # Add 2 for start token (SOT) and end token (EOT)
            return len(self._tokenizer.encode(text)) + 2
        
        # HFTokenizer fallback
        if hasattr(self._tokenizer, 'tokenizer') and hasattr(self._tokenizer.tokenizer, 'encode'):
            return len(self._tokenizer.tokenizer.encode(text, add_special_tokens=False)) + 2
        
        # Last resort: use truncated output (capped at 77)
        tokens = self._tokenizer([text])
        return (tokens[0] != 0).sum().item()
    
    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count(t) for t in texts]


def read_lines(path: Path) -> list[str]:
    """Read non-empty lines from file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []


def extract_embeddings(
    model, tokenizer, device: str, texts: list[str], batch_size: int = 64
) -> np.ndarray:
    """Extract normalized CLIP text embeddings."""
    if not texts:
        return np.array([], dtype=np.float32)
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    
    return np.vstack(embeddings).astype(np.float32)


def process_files(
    input_dir: Path,
    output_dir: Path,
    model,
    tokenizer,
    device: str,
    batch_size: int = 64
) -> dict:
    """Process text files and generate per-line embeddings."""
    
    # Find all text files
    text_files = sorted(input_dir.rglob("*.txt"))
    if not text_files:
        print(f"No text files found in {input_dir}")
        return {"error": "No files found"}
    
    print(f"Found {len(text_files)} text files")
    
    # Initialize token counter
    counter = TokenCounter(tokenizer)
    
    # Collect all lines with metadata in single pass
    print("\nReading files...")
    lines_data = []
    empty_files = 0
    
    for path in tqdm(text_files, desc="Reading"):
        lines = read_lines(path)
        if not lines:
            empty_files += 1
            continue
        
        rel_path = path.relative_to(input_dir)
        rel_parent = rel_path.parent
        stem = rel_path.stem
        
        for idx, text in enumerate(lines):
            lines_data.append({
                "text": text,
                "rel_path": rel_path,
                "rel_parent": rel_parent,
                "stem": stem,
                "line_idx": idx,
            })
    
    if not lines_data:
        print("No valid text lines found!")
        return {"total_files": len(text_files), "empty_files": empty_files}
    
    # Count tokens (batch for efficiency)
    print("\nAnalyzing tokens...")
    texts = [d["text"] for d in lines_data]
    token_counts = counter.count_batch(texts)
    
    # Add token counts to data
    for d, tc in zip(lines_data, token_counts):
        d["tokens"] = tc
    
    # Analyze truncation
    truncated = [(d, tc) for d, tc in zip(lines_data, token_counts) if tc > CLIP_MAX_TOKENS]
    
    print(f"\n=== Token Analysis ===")
    print(f"Total lines: {len(lines_data)}")
    print(f"Min/Max/Avg tokens: {min(token_counts)}/{max(token_counts)}/{sum(token_counts)/len(token_counts):.1f}")
    
    if truncated:
        print(f"\n⚠️  {len(truncated)} lines ({100*len(truncated)/len(lines_data):.1f}%) will be TRUNCATED!")
        print(f"First 10 (tokens > {CLIP_MAX_TOKENS}):")
        for d, tc in truncated[:10]:
            preview = d["text"][:50] + "..." if len(d["text"]) > 50 else d["text"]
            print(f"  {d['rel_path']} line {d['line_idx']}: {tc} tokens (+{tc - CLIP_MAX_TOKENS})")
            print(f"    \"{preview}\"")
        if len(truncated) > 10:
            print(f"  ... and {len(truncated) - 10} more")
    else:
        print(f"\n✅ All lines fit within {CLIP_MAX_TOKENS} tokens.")
    
    # Extract embeddings
    print(f"\nExtracting embeddings for {len(texts)} lines...")
    embeddings = extract_embeddings(model, tokenizer, device, texts, batch_size)
    
    # Pre-create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_dirs = {output_dir / d["rel_parent"] for d in lines_data}
    for d in unique_dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings and collect metadata
    print(f"\nSaving to {output_dir}...")
    file_metadata = []
    
    for i, d in enumerate(tqdm(lines_data, desc="Saving")):
        # Output path: stem_line_X.npy
        out_name = f"{d['stem']}_line_{d['line_idx']}.npy"
        out_path = output_dir / d["rel_parent"] / out_name
        
        np.save(out_path, embeddings[i])
        
        file_metadata.append({
            "source_file": str(d["rel_path"]),
            "line_index": d["line_idx"],
            "embedding_file": str(out_path.relative_to(output_dir)),
            "tokens": d["tokens"],
            "text": d["text"],
        })
    
    # Save combined embeddings
    np.save(output_dir / "all_embeddings.npy", embeddings)
    
    # Save metadata (compact JSON)
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "embedding_dim": int(embeddings.shape[1]),
        "total_embeddings": len(embeddings),
        "total_source_files": len(text_files),
        "files": file_metadata,
        "truncation_stats": {
            "total_truncated": len(truncated),
            "truncated_items": [
                {"file": str(d["rel_path"]), "line": d["line_idx"], "tokens": d["tokens"]}
                for d, _ in truncated
            ],
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "total_files": len(text_files),
        "empty_files": empty_files,
        "total_lines": len(lines_data),
        "processed_lines": len(embeddings),
        "truncated_lines": len(truncated),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP text embeddings (per-line)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clip/text_embeddings.py final_dataset_text
    python clip/text_embeddings.py final_dataset_text --batch-size 128
        """
    )
    parser.add_argument("input_dir", help="Input directory with text files")
    parser.add_argument("--output-dir", help="Output directory (default: input_dir_embeddings)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_embeddings"
    
    print("=" * 60)
    print("  CLIP Text Embedding Generator")
    print("=" * 60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    
    device = "cpu" if args.cpu else get_device()
    model, tokenizer, device = load_clip_model(device)
    
    stats = process_files(input_dir, output_dir, model, tokenizer, device, args.batch_size)
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Files:      {stats.get('total_files', 0)} ({stats.get('empty_files', 0)} empty)")
    print(f"  Lines:      {stats.get('total_lines', 0)}")
    print(f"  Processed:  {stats.get('processed_lines', 0)}")
    print(f"  Truncated:  {stats.get('truncated_lines', 0)}")
    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
