#!/usr/bin/env python3
"""
Manga Image to Text Description Generator

Converts manga page images to text descriptions using Claude's vision API.
Preserves the directory structure: author/manga/chapter/page.png -> page.txt

Usage:
    python embedding_generators/llm_claude/llm_setup.py datasets/large --prompt "Describe this manga page"
    python embedding_generators/llm_claude/llm_setup.py datasets/small -p prompts/basic_prompt.txt
    python embedding_generators/llm_claude/llm_setup.py datasets/medium --resume
"""

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Model configuration
MODEL = "claude-sonnet-4-5-20250929"

# Pricing per 1M tokens (as of late 2024 for claude-sonnet-4-5-20250929)
INPUT_COST_PER_1M = 3.00   # $3 per 1M input tokens
OUTPUT_COST_PER_1M = 15.00  # $15 per 1M output tokens


@dataclass
class TokenStats:
    """Track token usage and timing statistics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    per_image_stats: list = field(default_factory=list)
    
    def add_result(self, input_tokens: int, output_tokens: int, elapsed: float, path: str):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.successful += 1
        self.per_image_stats.append({
            "path": path,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
        })
    
    def add_failure(self, path: str, error: str):
        self.failed += 1
        self.per_image_stats.append({
            "path": path,
            "error": error,
        })
    
    def add_skip(self):
        self.skipped += 1
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def wall_time(self) -> float:
        return (self.end_time or time.time()) - self.start_time
    
    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.successful if self.successful else 0
    
    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.successful if self.successful else 0
    
    @property
    def avg_time_per_image(self) -> float:
        return self.wall_time / self.successful if self.successful else 0
    
    @property
    def estimated_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
        return input_cost + output_cost
    
    def report(self) -> str:
        """Generate a summary report."""
        lines = [
            "",
            "=" * 70,
            "  IMAGE TO TEXT CONVERSION REPORT",
            "=" * 70,
            "",
            f"  Total images found:    {self.total_images:,}",
            f"  Successfully processed: {self.successful:,}",
            f"  Failed:                 {self.failed:,}",
            f"  Skipped (existing):     {self.skipped:,}",
            "",
            "-" * 70,
            "  TOKEN USAGE",
            "-" * 70,
            f"  Total input tokens:     {self.total_input_tokens:,}",
            f"  Total output tokens:    {self.total_output_tokens:,}",
            f"  Total tokens:           {self.total_tokens:,}",
            "",
            f"  Avg input per image:    {self.avg_input_tokens:,.0f}",
            f"  Avg output per image:   {self.avg_output_tokens:,.0f}",
            "",
            "-" * 70,
            "  TIMING",
            "-" * 70,
            f"  Total wall time:        {self.wall_time:.1f}s ({self.wall_time/60:.1f} min)",
            f"  Avg time per image:     {self.avg_time_per_image:.2f}s",
            f"  Throughput:             {self.successful / self.wall_time * 60:.1f} images/min" if self.wall_time > 0 else "",
            "",
            "-" * 70,
            "  COST ESTIMATE",
            "-" * 70,
            f"  Input cost:             ${(self.total_input_tokens / 1_000_000) * INPUT_COST_PER_1M:.4f}",
            f"  Output cost:            ${(self.total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M:.4f}",
            f"  Total estimated cost:   ${self.estimated_cost:.4f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


def encode_image(path: Path) -> tuple[str, str]:
    """Encode image to base64 and determine media type."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".gif":
        media_type = "image/gif"
    elif suffix == ".webp":
        media_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {suffix}")
    
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    
    return data, media_type


def generate_text(client: Anthropic, image_path: Path, prompt: str, max_tokens: int = 1024) -> tuple[str, int, int]:
    """
    Generate text description for an image.
    
    Returns:
        Tuple of (text, input_tokens, output_tokens)
    """
    image_b64, media_type = encode_image(image_path)
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    
    text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    return text, input_tokens, output_tokens


def find_images(input_dir: Path) -> list[Path]:
    """Find all image files in directory following author/manga/chapter/page structure."""
    images = []
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            # Verify it follows the expected structure (at least 3 levels deep)
            rel_path = path.relative_to(input_dir)
            if len(rel_path.parts) >= 3:  # author/manga/chapter/page
                images.append(path)
    
    return images


def get_output_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Convert input image path to output text path."""
    rel_path = image_path.relative_to(input_dir)
    output_path = output_dir / rel_path.with_suffix(".txt")
    return output_path


def process_images(
    input_dir: Path,
    output_dir: Path,
    prompt: str,
    max_tokens: int = 1024,
    resume: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    verbose: bool = False,
) -> TokenStats:
    """
    Process all images in input directory and save text descriptions.
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for text files
        prompt: Prompt for the LLM
        max_tokens: Maximum tokens for response
        resume: Skip already processed images
        dry_run: Only count images, don't process
        limit: Maximum number of images to process
        verbose: Print detailed progress
    
    Returns:
        TokenStats with usage statistics
    """
    stats = TokenStats()
    
    # Find all images
    print(f"\nScanning {input_dir} for images...")
    images = find_images(input_dir)
    stats.total_images = len(images)
    
    if limit:
        images = images[:limit]
        print(f"Limited to first {limit} images")
    
    print(f"Found {stats.total_images} images to process")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for img in images[:10]:
            out = get_output_path(img, input_dir, output_dir)
            print(f"  {img.relative_to(input_dir)} -> {out.relative_to(output_dir)}")
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more")
        stats.end_time = time.time()
        return stats
    
    # Initialize client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    client = Anthropic(api_key=api_key)
    
    # Process images
    print(f"\nProcessing images with model: {MODEL}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    for i, image_path in enumerate(images):
        output_path = get_output_path(image_path, input_dir, output_dir)
        rel_path = image_path.relative_to(input_dir)
        
        # Skip if already exists and resume mode
        if resume and output_path.exists():
            if verbose:
                print(f"[{i+1}/{len(images)}] SKIP (exists): {rel_path}")
            stats.add_skip()
            continue
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image
        try:
            start = time.time()
            text, input_tokens, output_tokens = generate_text(
                client, image_path, prompt, max_tokens
            )
            elapsed = time.time() - start
            
            # Save text
            output_path.write_text(text, encoding="utf-8")
            
            stats.add_result(input_tokens, output_tokens, elapsed, str(rel_path))
            
            # Progress
            if verbose:
                print(f"[{i+1}/{len(images)}] OK: {rel_path}")
                print(f"    Tokens: {input_tokens} in / {output_tokens} out | {elapsed:.1f}s")
            else:
                # Compact progress
                pct = (i + 1) / len(images) * 100
                print(f"\r[{i+1}/{len(images)}] ({pct:.0f}%) {rel_path.name[:40]:<40} ", end="", flush=True)
                
        except Exception as e:
            stats.add_failure(str(rel_path), str(e))
            print(f"\n[{i+1}/{len(images)}] FAIL: {rel_path}")
            print(f"    Error: {e}")
    
    print()  # New line after progress
    stats.end_time = time.time()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert manga images to text descriptions using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in datasets/large
  python LLM-Claude/llm_setup.py datasets/large --prompt "Describe this manga page in detail"
  
  # Use a prompt from a file
  python LLM-Claude/llm_setup.py datasets/small -p prompts/describe.txt
  
  # Resume interrupted processing
  python LLM-Claude/llm_setup.py datasets/medium --resume
  
  # Dry run to see what would be processed
  python LLM-Claude/llm_setup.py datasets/large --dry-run
  
  # Process only first 10 images (for testing)
  python LLM-Claude/llm_setup.py datasets/small --limit 10 -v
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing images (e.g., datasets/large)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: datasets_text/<size>)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text or path to prompt file",
    )
    parser.add_argument(
        "-p", "--prompt-file",
        type=str,
        default=None,
        help="Path to file containing prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count images, don't process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Save statistics to JSON file",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # datasets/large -> datasets_text/large
        parts = input_dir.parts
        if "datasets" in parts:
            idx = parts.index("datasets")
            new_parts = list(parts[:idx]) + ["datasets_text"] + list(parts[idx+1:])
            output_dir = Path(*new_parts)
        else:
            output_dir = input_dir.parent / f"{input_dir.name}_text"
    
    # Get prompt
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        prompt = prompt_path.read_text().strip()
    elif args.prompt:
        # Check if prompt is a file path (only if it's short enough to be a path)
        if len(args.prompt) < 256 and not args.prompt.startswith(("Describe", "What", "Tell", "Analyze", "List")):
            prompt_path = Path(args.prompt)
            if prompt_path.exists() and prompt_path.is_file():
                prompt = prompt_path.read_text().strip()
            else:
                prompt = args.prompt
        else:
            prompt = args.prompt
    else:
        # Default prompt
        prompt = """Describe this manga page in detail. Include:
1. The visual style and art characteristics
2. Panel layout and composition
3. Characters present and their expressions/poses
4. Scene setting and background elements
5. Any text, dialogue, or sound effects visible
6. The mood and atmosphere conveyed"""
    
    # Validate prompt is not empty
    if not prompt:
        print("Error: Prompt cannot be empty. Provide a prompt with --prompt or -p", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 70)
    print("  MANGA IMAGE TO TEXT CONVERTER")
    print("=" * 70)
    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Model:  {MODEL}")
    print(f"  Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"  Prompt: {prompt}")
    print()
    
    try:
        stats = process_images(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt=prompt,
            max_tokens=args.max_tokens,
            resume=args.resume,
            dry_run=args.dry_run,
            limit=args.limit,
            verbose=args.verbose,
        )
        
        # Print report
        print(stats.report())
        
        # Save stats if requested
        if args.save_stats:
            stats_path = Path(args.save_stats)
            stats_dict = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "model": MODEL,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "total_images": stats.total_images,
                "successful": stats.successful,
                "failed": stats.failed,
                "skipped": stats.skipped,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_tokens": stats.total_tokens,
                "avg_input_tokens": stats.avg_input_tokens,
                "avg_output_tokens": stats.avg_output_tokens,
                "wall_time_seconds": stats.wall_time,
                "avg_time_per_image": stats.avg_time_per_image,
                "estimated_cost_usd": stats.estimated_cost,
                "per_image_stats": stats.per_image_stats,
            }
            stats_path.write_text(json.dumps(stats_dict, indent=2))
            print(f"\nStatistics saved to: {stats_path}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
