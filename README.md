# Manga Vectorizer

A comprehensive system for manga character retrieval using CLIP embeddings, LLM-generated descriptions, and advanced retrieval algorithms including late fusion and Query-Conditioned Feedback Retrieval (QCFR).

## Overview

Manga Vectorizer enables semantic search for manga characters by combining:
- **CLIP embeddings** for visual similarity (ViT-L-14 model)
- **LLM descriptions** (Claude) for rich character descriptions
- **FAISS indexing** for efficient similarity search
- **Advanced retrieval methods**: Late Fusion and QCFR (two-pass retrieval with Rocchio-style query refinement)

## Paper/Report

For detailed information about the methodology, experiments, and results, see the full report:

📄 **[Character Retrieval in Manga via Semantic Query Refinement](report/Character%20Retrieval%20in%20Manga%20via%20Semantic%20Query%20Refinemen.pdf)**

## Features

- 🖼️ **Multi-modal retrieval**: Image + text hybrid search
- 🤖 **LLM-enhanced queries**: Automatic character description generation
- ⚡ **Fast search**: FAISS-based similarity search
- 📊 **Comprehensive evaluation**: Recall@k, mAP@k metrics with visualization
- 🔍 **Grid search**: Hyperparameter optimization for retrieval methods
- 📈 **Visualization**: Recall curves, result visualizations, and clustering

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xuyangan/semantic-manga-retrieval.git
cd semantic-manga-retrieval
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package (development mode):
```bash
pip install -e .
```

### Environment Variables

Create a `.env` file in the project root with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Project Structure

```
manga-vectorizer/
├── src/manga_vectorizer/          # Main package
│   ├── core/                      # Core modules
│   │   ├── clip_encoder.py       # CLIP model and encoding
│   │   ├── llm_encoder.py         # LLM description generation
│   │   ├── faiss_search.py        # FAISS index utilities
│   │   └── fusion.py              # Late fusion reranking
│   ├── retrieval/                 # Retrieval algorithms
│   │   └── qcfr.py                # QCFR implementation
│   └── evaluation/                # Evaluation scripts
│       ├── recall_image.py        # Image-only retrieval eval
│       ├── recall_text.py         # Text-only retrieval eval
│       ├── recall_fusion.py       # Late fusion eval
│       ├── recall_qcfr.py         # QCFR eval
│       └── utils.py               # Evaluation utilities
├── scripts/                       # Utility scripts
│   ├── build_dataset.py           # Dataset builder
│   ├── download_manga.py           # Manga downloader
│   ├── isolate_characters.py      # Character isolation
│   └── preprocess_images.py        # Image preprocessing
├── embedding_generators/          # Embedding generation
│   ├── clip/                      # CLIP embeddings
│   ├── bert/                      # BERT embeddings
│   └── llm_claude/                # LLM utilities
├── queries/                       # Query dataset
│   └── query_*/                   # Individual queries
│       ├── query.png              # Query image
│       ├── query.txt              # User query text
│       ├── labels.txt             # Ground truth labels
│       └── llm_description.txt    # Generated description
└── results/                       # Evaluation results
```

## Quick Start

### 1. Prepare Your Dataset

Organize your manga images in the following structure:
```
final_dataset/
├── manga_name/
│   └── chapter_XX/
│       ├── page_001.jpg
│       ├── page_002.jpg
│       └── ...
```

### 2. Generate Embeddings

#### Image Embeddings
```bash
python embedding_generators/clip/image_embeddings.py \
    --input final_dataset \
    --output dataset_embeddings
```

#### Text Embeddings (from extracted text)
```bash
python embedding_generators/clip/text_embeddings.py \
    --input final_dataset_text \
    --output dataset_text_embeddings
```

### 3. Build FAISS Indexes

```bash
python embedding_generators/clip/faiss_image_index.py \
    --embeddings dataset_embeddings \
    --output dataset_embeddings/faiss_index

python embedding_generators/clip/faiss_text_index.py \
    --embeddings dataset_text_embeddings \
    --output dataset_text_embeddings/faiss_index
```

### 4. Run Evaluation

#### QCFR (Query-Conditioned Feedback Retrieval)

**Single run:**
```bash
python -m manga_vectorizer.evaluation.recall_qcfr \
    --queries queries \
    --alpha 0.8 --m-img 200 --l-pos 20 --b 0.35 --c 0.3 \
    --image-index dataset_embeddings/faiss_index \
    --text-index dataset_text_embeddings/faiss_index \
    --output results/qcfr
```

**Grid search:**
```bash
python -m manga_vectorizer.evaluation.recall_qcfr \
    --queries queries \
    --alphas 0.3 0.5 0.8 \
    --m-imgs 100 200 300 \
    --l-pos-values 20 30 50 \
    --bs 0.2 0.35 0.5 \
    --cs 0.1 0.2 0.3 \
    --image-index dataset_embeddings/faiss_index \
    --text-index dataset_text_embeddings/faiss_index \
    --output results/qcfr_grid
```

#### Late Fusion

**Single run:**
```bash
python -m manga_vectorizer.evaluation.recall_fusion \
    --queries queries \
    --mode image \
    --alpha 0.5 --m 100 \
    --image-index dataset_embeddings/faiss_index \
    --text-index dataset_text_embeddings/faiss_index \
    --output results/fusion
```

**Grid search:**
```bash
python -m manga_vectorizer.evaluation.recall_fusion \
    --queries queries \
    --mode image \
    --alphas 0.1 0.3 0.5 0.7 0.9 \
    --m-values 50 75 100 \
    --image-index dataset_embeddings/faiss_index \
    --text-index dataset_text_embeddings/faiss_index \
    --output results/fusion_grid
```

#### Image-Only Retrieval

```bash
python -m manga_vectorizer.evaluation.recall_image \
    --queries queries \
    --index dataset_embeddings/faiss_index \
    --output results/image_only
```

#### Text-Only Retrieval

```bash
python -m manga_vectorizer.evaluation.recall_text \
    --queries queries \
    --index dataset_text_embeddings/faiss_index \
    --output results/text_only
```

## Retrieval Methods

### 1. QCFR (Query-Conditioned Feedback Retrieval)

**Two-pass retrieval with Rocchio-style query refinement:**

1. **Pass-1**: Retrieve candidates from both image and text channels
2. **Hybrid scoring**: Combine image and text scores: `s[p] = (1-α) * s_img[p] + α * s_txt[p]`
3. **Pseudo-labeling**: Select top L_pos as pseudo-positives, bottom L_neg as pseudo-negatives
4. **Feedback centroids**: Compute softmax-weighted centroids from pseudo-labels
5. **Query refinement**: Rocchio update: `q' = normalize(a*q + b*pos - c*neg + d*tq)`
6. **Pass-2**: Full re-search with refined query

**Hyperparameters:**
- `alpha`: Hybrid score weight (0=image only, 1=text only)
- `m_img`: Number of image candidates in Pass-1
- `l_pos`: Number of pseudo-positives
- `b`: Rocchio positive feedback weight
- `c`: Rocchio negative feedback weight
- `a`: Original query weight (default: 1.0)
- `d`: Text query weight (default: 0.21)

**Reference values:**
- `m_img=100`, `m_txt=300` (3×m_img)
- `l_pos=20`, `l_neg=20`
- `alpha=0.8`
- `a=1.0`, `b=0.35`, `c=0.30`, `d=0.21`

### 2. Late Fusion

**Single-pass retrieval with reranking:**

1. Retrieve top M candidates from image index
2. For each candidate, find corresponding text embeddings
3. Compute max-pooled text similarity per page
4. Rerank using weighted combination: `score = α * s_img + (1-α) * s_txt`
5. Return top K results

**Hyperparameters:**
- `alpha`: Weight for image score (0=text only, 1=image only)
- `m`: Number of initial candidates to retrieve
- `k`: Number of final results

### 3. Image-Only / Text-Only

Standard FAISS similarity search using single modality.

## Query Format

Each query should be in a folder with the following structure:

```
queries/query_1/
├── query.png (or query.jpg)    # Query image
├── query.txt                    # User query text
├── labels.txt                   # Ground truth (one path per line)
└── llm_description.txt         # Optional: cached LLM description
```

**Example `query.txt`:**
```
I want to find a manga that has a character similar to the one that appears on the top most panel.
```

**Example `labels.txt`:**
```
final_dataset/Berserk/chapter_105/page_005.jpg
final_dataset/Berserk/chapter_105/page_011.jpg
final_dataset/Berserk/chapter_154/page_013.jpg
...
```

## Evaluation Metrics

The evaluation scripts compute:

- **Recall@k**: Fraction of relevant items retrieved in top-k
- **mAP@k**: Mean Average Precision at k
- **Statistics**: Mean, std, min, max, median across queries

Results are saved with:
- `recall_results.json`: Detailed per-query results
- `recall_summary_table.csv`: Summary statistics
- `recall_curves/`: Individual and aggregate recall curves
- `visualizations/`: Result visualizations (if `--image-dir` provided)

## Output Structure

```
results/
└── qcfr/
    ├── recall_results.json           # Full results
    ├── recall_summary_table.csv      # Summary table
    ├── recall_curves/
    │   ├── aggregate_all_queries.png # Aggregate curves
    │   └── query_001_*.png           # Per-query curves
    └── visualizations/                # Result images (optional)
```

For grid search:
```
results/
└── qcfr_grid/
    ├── grid_search_comparison.csv    # All configs compared
    ├── best_config.json              # Best hyperparameters
    ├── a0.8_m200_l20_b0.35_c0.3/    # Individual config results
    │   └── ...
    └── ...
```

## API Usage

### Basic Retrieval

```python
from manga_vectorizer import (
    load_clip_model,
    encode_image,
    encode_text,
    encode_query_llm,
    qcfr_search_with_description
)
from pathlib import Path

# Load CLIP model
model, preprocess, tokenizer, device = load_clip_model()

# Generate LLM description
query_image = Path("queries/query_1/query.png")
user_query = "Find similar characters"
description = encode_query_llm(query_image, user_query)

# Generate embeddings
image_emb = encode_image(model, preprocess, device, query_image)
text_emb = encode_text(model, tokenizer, device, description)

# Run QCFR search
results = qcfr_search_with_description(
    image_path=query_image,
    user_query=user_query,
    image_db=Path("final_dataset"),
    text_db=Path("final_dataset_text"),
    image_index_dir=Path("dataset_embeddings/faiss_index"),
    text_index_dir=Path("dataset_text_embeddings/faiss_index"),
    m_img=200,
    l_pos=20,
    alpha=0.8,
    b=0.35,
    c=0.3,
    k=50
)

# Access results
for result in results['final_results']:
    print(f"Rank {result['rank']}: {result['path']} (score: {result['similarity']:.3f})")
```

## Configuration

### CLIP Model

Default: **ViT-L-14** with **laion2b_s32b_b82k** pretrained weights

To use a different model, modify `src/manga_vectorizer/core/clip_encoder.py`.

### LLM

Default: **Claude 4.5 Sonnet** (via Anthropic API)

Requires `ANTHROPIC_API_KEY` in `.env` file.

## Performance Tips

1. **GPU acceleration**: Use CUDA for faster CLIP encoding
2. **Batch processing**: Embedding generators support batch processing
3. **FAISS indexes**: Use IVF indexes for large datasets (>1M vectors)
4. **Caching**: LLM descriptions are cached in query folders
5. **True max-pooling**: Disable with `--no-true-max-pooling` for faster QCFR (slightly less accurate)

## Troubleshooting

### FAISS Import Error
```bash
pip install faiss-cpu  # or faiss-gpu for CUDA
```

### CUDA Out of Memory
- Reduce batch size in embedding generation
- Use CPU mode: `device="cpu"` in model loading
- Process datasets in smaller chunks

### LLM API Errors
- Check `ANTHROPIC_API_KEY` in `.env`
- Verify API quota/rate limits
- Descriptions are cached, so re-runs won't call API

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **FAISS**: Facebook AI Similarity Search
- **Anthropic Claude**: LLM descriptions
- **OpenCLIP**: Open-source CLIP implementations

