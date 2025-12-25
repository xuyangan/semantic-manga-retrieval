#!/usr/bin/env python3
"""
CLIP Encoder Module

Generates CLIP embeddings for images and text.
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
import open_clip


def get_device() -> str:
    """Get best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(device: str = None):
    """
    Load CLIP model for both image and text encoding.
    
    Args:
        device: Device to use (default: auto-detect)
    
    Returns:
        tuple: (model, preprocess, tokenizer, device)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading CLIP model (ViT-L-14) on {device}...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    
    return model, preprocess, tokenizer, device


def encode_image(model, preprocess, device: str, image_path: Path) -> np.ndarray:
    """
    Generate normalized CLIP embedding for a single image.
    
    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device string
        image_path: Path to image file
    
    Returns:
        Normalized embedding array (1, 768)
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().astype(np.float32)


def encode_text(model, tokenizer, device: str, text: str) -> np.ndarray:
    """
    Generate normalized CLIP embedding for a single text.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: Device string
        text: Text string to encode
    
    Returns:
        Normalized embedding array (1, 768)
    """
    tokens = tokenizer([text]).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().astype(np.float32)


def encode_texts(model, tokenizer, device: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate normalized CLIP embeddings for multiple texts.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        device: Device string
        texts: List of text strings
        batch_size: Batch size for processing
    
    Returns:
        Normalized embeddings array (N, 768)
    """
    import torch
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        
        with torch.no_grad():
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
    
    if all_embeddings:
        return np.vstack(all_embeddings).astype(np.float32)
    return np.array([], dtype=np.float32)
