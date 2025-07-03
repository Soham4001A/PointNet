# Point-Cloud Classification: Latent Meta Attention vs. Multi-Head Attention

This repository provides a PyTorch implementation of a 3D point-cloud classification model, designed to serve as a clear and modular test bed for comparing Latent Meta Attention (LMA) against standard Multi-Head Attention (MHA).

The primary goal of this work is to offer a simple, well-documented, and easily modifiable script for researchers and practitioners to experiment with attention mechanisms in the context of point-cloud processing.

## Table of Contents
*   [Key Architectural Innovations](#key-architectural-innovations)
    *   [Latent Meta Attention (LMA)](#latent-meta-attention-lma)
        *   [Introduction to LMA](#introduction-to-lma)
        *   [LMA Transformation Pipeline](#lma-transformation-pipeline)
        *   [Integration and Complexity](#integration-and-complexity)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Command-Line Options](#command-line-options)
    *   [Examples](#examples)

## Key Architectural Innovations

### Latent Meta Attention (LMA)
*(Lead Author: Soham Sane || Johns Hopkins University || Collins Aerospace)*

LMA is a novel attention mechanism designed for computational efficiency and potentially richer feature interactions by operating within a reduced-dimensionality latent space. (A research paper detailing LMA is currently in preparation.)

#### Introduction to LMA

The Transformer architecture [Vaswani et al., 2017] has revolutionized sequence modeling across diverse domains, from natural language processing [Devlin et al., 2018; Brown et al., 2020] to computer vision [Dosovitskiy et al., 2020]. Its success is largely attributed to the Multi-Head Self-Attention (MHA) mechanism, which excels at capturing long-range dependencies.

**The Attention Landscape and Its Challenges**

Standard MHA faces a significant challenge: its computational complexity scales quadratically ($O(L^2 D)$) with the input sequence length $L$ and embedding dimension $D$. This becomes prohibitive for very long sequences. Traditional input processing involves either direct projection or chunk-wise embedding.

**Motivation: Efficiency, Richer Interactions, and Re-evaluating Attention**

The quadratic complexity of MHA and the trend towards larger models necessitate more efficient attention mechanisms. LMA re-evaluates the MHA paradigm, questioning whether the "Query," "Key," and "Value" semantics are strictly necessary, or if attention's effectiveness emerges from its structured sequence of operations. LMA explores alternative operational sequences, particularly those involving structured transformations and dimensionality reduction *before* the core interaction calculation.

**Background & Related Work on Efficient Attention**

Existing efficient attention methods include Grouped-Query Attention (GQA) [Ainslie et al., 2023] and Multi-Query Attention (MQA) [Shazeer, 2019] (reducing KV cache for inference), sparse attention (e.g., Longformer [Beltagy et al., 2020]), linearized attention (e.g., Performer [Choromanski et al., 2020]), and latent space attention like DeepSeek's Multi-head Latent Attention (MLA) [DeepSeek-AI, 2024].

LMA distinguishes itself through a unique **two-stage sequence embedding pipeline** to generate its latent space. Critically, **all LMA components (Q', K', V') operate entirely within this reduced latent space**, unlike MLA's partial reduction.

#### LMA Transformation Pipeline

The transformation from an input block $X_{in}$ to the latent representation $Z$ involves:

**1. Stage 1: Initial Sequencing & Embedding**
If the input is a flat vector $X_{raw} \in \mathbb{R}^{B \times N}$, it's reshaped into $L$ chunks of size $C_{in} = N / L$. A shared embedding layer, $\text{EmbedLayer}_1$, maps each chunk $C_{in} \rightarrow d_0$:
```math
Y = \text{EmbedLayer}_1(\text{Reshape}(X_{raw})) \in \mathbb{R}^{B \times L \times d_0}
```

**2. Stage 2: Head-View Stacking & Latent Re-Embedding**
   **a. Head-View Stacking:** $Y$ is split along its embedding dimension $d_0$ into $N_h$ segments (heads), each $Y_i \in \mathbb{R}^{B \times L \times d_h}$ where $d_h = d_0 / N_h$. These segments are then concatenated *sequentially* along the sequence dimension:
   ```math
   X_{stacked} = \text{Concat}_{\text{axis=1}} (Y_1, Y_2, ..., Y_{N_h}) \in \mathbb{R}^{B \times (L \cdot N_h) \times d_h}
   ```
   This creates a longer, thinner intermediate sequence.

   **b. Re-Chunking & Latent Embedding:** The stacked tensor $X_{stacked}$ (total features per batch item $L \cdot d_0$) is reshaped into a new sequence of length $L'$, where each new "chunk" has size $C' = (L \cdot d_0) / L'$. A second shared embedding layer, $\text{EmbedLayer}_2$, maps each chunk of size $C'$ to the target latent dimension $d'$:
   ```math
   Z = \text{EmbedLayer}_2(\text{Reshape}(\text{Flatten}(X_{stacked}))) \in \mathbb{R}^{B \times L' \times d'}
   ```
   This second embedding stage efficiently compresses the combined head-view information into the final latent space $Z$.

#### LMA Latent Attention Calculation
Attention operates entirely on the latent representation $Z$. Latent Query ($Q'$), Key ($K'$), and Value ($V'$) are computed via linear projections ($W_{Q'}, W_{K'}, W_{V'}$) from $Z$, mapping $d' \rightarrow d'$ or to a latent head dimension $d'_{head}$:
```math
Q' = Z W_{Q'}; \quad K' = Z W_{K'}; \quad V' = Z W_{V'}
```
Scaled dot-product attention is then applied:
```math
\text{AttnOut} = \text{softmax}\left( \frac{Q' {K'}^T}{\sqrt{d'_{head}}} \right) V' \in \mathbb{R}^{B \times L' \times d'}
```
The $O((L')^2)$ complexity of this step provides the main computational speedup. Due to the sequential head-view stacking, this latent attention can be conceptualized as a form of self-comparison on a meta-representation of the original sequence's features.

#### Integration and Residual Connections in LMA
LMA utilizes standard Transformer-style residual connections and Layer Normalization. The input $Z$ to the latent attention module serves directly as the input for the first residual sum:
```math
\text{Out}_1 = \text{LayerNorm}(Z + \text{Dropout}(\text{AttnOut}))
```
This is dimensionally consistent without requiring an extra projection for the residual path. This is followed by a Feed-Forward Network (FFN) operating within the latent dimension $d'$, and a second residual connection:
```math
Z_{out} = \text{LayerNorm}(\text{Out}_1 + \text{Dropout}(\text{FFN}(\text{Out}_1)))
```

#### LMA Complexity Analysis
LMA's computational cost (FLOPs), ignoring biases and activations, is roughly:
```math
O(B (N d_0 + L d_0 d' + 3 L' (d')^2 + 2 (L')^2 d' + 2 L' d' d'_{ffn}))
```
This is compared to MHA's:
```math
O(B (4 L d_0^2 + 2 L^2 d_0 + 2 L d_0 d_{ff}))
```
LMA achieves significant efficiency gains when $L' \ll L$ and $d' \ll d_0$, as the $L^2 d_0$ term in MHA typically dominates for long sequences.
The precise condition for LMA being computationally cheaper is:
```math
B (N d_0 + L d_0 d') + B (3 L' (d')^2 + 2 (L')^2 d') < B (4 L d_0^2 + 2 L^2 d_0)
```
(Comparing embedding and attention costs, excluding FFNs for simplicity here).

## Installation

1.  **Python Environment:** Python 3.9+ is recommended.
2.  **Dependencies:** Install the required Python packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `train.py` script is the main entry point for training and evaluating the model.

### Command-Line Options

*   `--dataset`: ModelNet version to use (`10` or `40`). Default: `10`.
*   `--root`: Directory to cache ModelNet data. Default: `data`.
*   `--points`: Number of points to sample from each mesh. Default: `1024`.
*   `--batch`: Batch size. Default: `32`.
*   `--epochs`: Number of training epochs. Default: `50`.
*   `--lr`: Learning rate. Default: `1e-3`.
*   `--embed`: Initial embedding dimension ($d_0$). Default: `64`.
*   `--heads`: Number of heads ($H$) for attention and LMA. Default: `2`.
*   `--ff_dim`: Feed-forward dimension within Transformer blocks. Default: `128`.
*   `--blocks`: Number of Transformer layers. Default: `2`.
*   `--use_lma`: **Enable this flag to use the LMA transformation.**
*   `--summary`: Print a model summary.
*   `--flops`: Print a FLOPs and parameter count analysis.

### Examples

**Train a standard MHA-based model on ModelNet10:**
```bash
python train.py --dataset 10 --epochs 20
```

**Train an LMA-based model on ModelNet10:**
```bash
python train.py --dataset 10 --epochs 20 --use_lma
```

**Run a quick test with a model summary and FLOPs count:**
```bash
python train.py --dataset 10 --epochs 1 --summary --flops --use_lma
```
