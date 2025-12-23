#!/usr/bin/env python3
"""
LLM Encoder Module

Generates character descriptions from manga images using Claude LLM.
"""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Model configuration
MODEL = "claude-sonnet-4-5-20250929"

# Path to the base prompt
PROMPT_PATH = Path(__file__).parent.parent / "LLM-Claude" / "prompts" / "basic_prompt.txt"


def load_base_prompt() -> str:
    """Load the character description prompt."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def encode_image_base64(path: Path) -> tuple[str, str]:
    """Encode image to base64 and determine media type for Claude."""
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    
    if suffix not in media_types:
        raise ValueError(f"Unsupported image format: {suffix}")
    
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    
    return data, media_types[suffix]


def generate_character_description(
    client: Anthropic,
    image_path: Path,
    user_query: str,
    max_tokens: int = 256
) -> tuple[str, int, int]:
    """
    Generate a single character description based on user's query.
    
    Args:
        client: Anthropic client
        image_path: Path to the manga image
        user_query: User's text describing which character to describe
        max_tokens: Maximum tokens for response
    
    Returns:
        Tuple of (description, input_tokens, output_tokens)
    """
    base_prompt = load_base_prompt()
    
    full_prompt = f"""{base_prompt}

USER REQUEST: Find what the user wants based on {user_query}. If the user wants to know about the character, find where the character is located at in the image.

Generate EXACTLY ONE LINE for this specific character. Do not describe any other characters."""

    image_b64, media_type = encode_image_base64(image_path)
    
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
                    {"type": "text", "text": full_prompt},
                ],
            }
        ],
    )
    
    text = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    description = lines[0] if lines else text
    
    return description, input_tokens, output_tokens


def encode_query_llm(image_path: Path, user_query: str, verbose: bool = False) -> str:
    """
    Encode a user query (image + text) into a character description.
    
    Args:
        image_path: Path to the query image
        user_query: Text describing which character to describe
        verbose: Print detailed info
    
    Returns:
        Character description string
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env file.")
    
    client = Anthropic(api_key=api_key)
    
    if verbose:
        print(f"Image: {image_path}")
        print(f"Query: {user_query}")
        print(f"Model: {MODEL}")
        print("-" * 50)
    
    description, input_tokens, output_tokens = generate_character_description(
        client, image_path, user_query
    )
    
    if verbose:
        print(f"Input tokens:  {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print("-" * 50)
    
    return description
