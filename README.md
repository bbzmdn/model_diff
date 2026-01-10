### Do Steering Vectors and RLHF Discover the Same Features?

In this repo I've tried to do an an empirical investigation into whether contrastive activation addition (steering vectors) and RLHF learn to represent behavioral traits using the same underlying features in transformer models.

### The Problem
When we steer a language model using activation differences (CAA/steering vectors) versus training it with RLHF, are we operating on the same feature space? Or do these methods carve up the model's internal representations differently?

This codebase trains a differential sparse autoencoder (diff-SAE) on the activations `chat_model - base_model` and then decomposes steering vectors into this learned feature basis to measure overlap.

### Description of the scripts
- `diff_sae.py` - TopK SAE implementation for learning features from RLHF-induced activation differences
- `train_diff_sae.py` - Extracts diff activations from 100M tokens and trains the SAE
- `steer_vectors.py` - Computes steering vectors via contrastive activation addition across behavioral categories (refusal, roleplay, uncertainty, helpfulness, format)
- `analyze_diff_sae.ipynb` - Decomposes steering vectors in the diff-SAE basis and analyzes feature overlap

### Setup

```bash
cp .env.example .env
pip install torch transformers datasets jaxtyping python-dotenv
```

Models default to Gemma-2-2B (base) and Gemma-2-2B-it (chat). Modify `.env` to experiment with other model pairs.

### Usage

```bash
# 1. Generate steering vectors (requires behavioral prompts in data/)
python steer_vectors.py

# 2. Train diff-SAE (downloads FineWeb, extracts 100M tokens)
python train_diff_sae.py

# 3. Analyze in notebook
jupyter notebook analyze_diff_sae.ipynb
```

**Note**: Training requires ~40GB VRAM for Gemma-2-2B.

### Preliminary Findings

Early results suggest steering vectors reconstruct poorly in the diff-SAE basis (cosine similarity ~0.3-0.5), with high L0 sparsity (~8000/18432 features active). This hints that steering and RLHF may discover *related but distinct* feature representations rather than identical ones.

More rigorous analysis ongoing.