import torch
import open_clip
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import json


def get_device():
    """Get the best available device: mps, cuda, or cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_all_images(downloads_dir: Path) -> list[Path]:
    """Recursively find all images in the downloads directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = []
    for ext in image_extensions:
        images.extend(downloads_dir.rglob(f"*{ext}"))
    return sorted(images)


def extract_embeddings(
    model, preprocess, device: str, image_paths: list[Path], batch_size: int = 16
) -> tuple[np.ndarray, list[Path]]:
    """Extract CLIP embeddings for all images."""
    embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                batch_images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensor)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    if embeddings:
        return np.vstack(embeddings), valid_paths
    return np.array([]), []


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int = 10, random_state: int = 42
) -> np.ndarray:
    """Perform k-means clustering on embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def save_results(
    image_paths: list[Path],
    labels: np.ndarray,
    embeddings: np.ndarray,
    output_dir: Path,
):
    """Save clustering results and embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # Save clustering results as JSON
    results = []
    for path, label in zip(image_paths, labels):
        results.append(
            {
                "path": str(path),
                "cluster": int(label),
                "author": path.parts[-4] if len(path.parts) >= 4 else "unknown",
                "manga": path.parts[-3] if len(path.parts) >= 3 else "unknown",
                "chapter": path.parts[-2] if len(path.parts) >= 2 else "unknown",
            }
        )

    with open(output_dir / "clusters.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print cluster summary
    print("\n=== Cluster Summary ===")
    for cluster_id in range(max(labels) + 1):
        cluster_items = [r for r in results if r["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_items)} images")
        # Show sample from each cluster
        for item in cluster_items[:3]:
            print(f"  - {item['author']}/{item['manga']}/{item['chapter']}")


def main():
    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    model = model.to(device).eval()

    # Get all images from downloads folder
    downloads_dir = Path(__file__).parent.parent / "downloads"
    print(f"Scanning for images in: {downloads_dir}")

    image_paths = get_all_images(downloads_dir)
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found!")
        return

    # Extract embeddings
    embeddings, valid_paths = extract_embeddings(model, preprocess, device, image_paths)
    print(f"Extracted embeddings for {len(valid_paths)} images")

    if len(valid_paths) == 0:
        print("No valid images to cluster!")
        return

    # Determine number of clusters (min of 10 or number of images)
    n_clusters = 5
    print(f"\nPerforming k-means clustering with {n_clusters} clusters...")

    # Cluster embeddings
    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

    # Save results
    output_dir = Path(__file__).parent / "output"
    save_results(valid_paths, labels, embeddings, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - embeddings.npy: {embeddings.shape} embedding vectors")
    print(f"  - clusters.json: clustering results with metadata")


if __name__ == "__main__":
    main()
