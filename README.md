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

Models default to Llama-3.1-8B (base) and Llama-3.1-8B-Instruct (instruct). Modify `.env` to experiment with other model pairs.

### Usage

```bash
# 1. Generate steering vectors (requires behavioral prompts in data/)
python steer_vectors.py

# 2. Train diff-SAE (downloads FineWeb, extracts 100M tokens)
python train_diff_sae.py

# 3. Analyze in notebook
jupyter notebook analyze_diff_sae.ipynb
```

**Note**: Training requires ~40GB VRAM for Llama-3.1-8B.

### Preliminary Findings

- High cosine similarity reflects linear alignment with the diff SAE subspace.
- Base and chat models share highly aligned steering directions.
- Sparse diff vectors provide a partial view; some nuanced category-specific features may be underrepresented.


### Limitations and Controls
The observed cosine similarity indicates geometric alignment between steering vectors and the diff-SAE subspace, but does not establish causal equivalence. Reconstruction measures whether steering vectors can be expressed in RLHF-derived features, not whether those features implement the same computations. Also, sparsity measurements require care; some reported sparsity statistics should be treated cautiously until analysis code is fully validated.

The natural next step is causal testing. If steering and RLHF truly share features, then ablating the diff-SAE features most responsible for steering reconstructions should affect both steering-induced behaviors and behaviors the RLHF model exhibits naturally.

Future work could include:
- causal ablations of shared latents
- per-token decomposition rather than averaged vectors
- random and orthogonal subspace baselines
- cross-layer analysis of feature overlap


