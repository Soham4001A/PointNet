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

LMA is a novel attention mechanism designed for computational efficiency and the potential for richer feature interactions by operating on a transformed latent representation of its input sequence.

#### Introduction to LMA

The Transformer architecture [Vaswani et al., 2017] has become a cornerstone of modern deep learning, with its Multi-Head Self-Attention (MHA) mechanism proving highly effective at modeling long-range dependencies. However, MHA's computational complexity scales quadratically ($O(L^2 D)$) with sequence length $L$, posing a challenge for high-resolution inputs like dense point clouds.

**Motivation for LMA:**
The core motivation behind LMA is to create a more efficient attention pathway by transforming the input sequence *before* the main quadratic attention calculation. By restructuring the feature space, LMA aims to create a "meta-representation" that can be processed more efficiently by subsequent standard Transformer blocks. This is achieved by a unique **head-view stacking and re-projection** pipeline that reshapes the sequence's feature dimensions.

#### LMA Transformation Pipeline

The transformation from an input tensor of point embeddings $\mathbf{X} \in \mathbb{R}^{B \times L \times d_0}$ to the latent representation $\mathbf{Z} \in \mathbb{R}^{B \times L \times d_{\text{new}}}$ involves the following steps. Here, $B$ is the batch size, $L$ is the number of points, $d_0$ is the initial embedding dimension from the PointNet backbone, and $d_{\text{new}}$ is the target latent dimension.

**1. Head-View Splitting**
The input tensor $\mathbf{X}$ is first split along its embedding dimension $d_0$ into $H$ separate heads.
$$
\mathbf{X} \rightarrow \{\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_H\}, \quad \text{where each } \mathbf{X}_h \in \mathbb{R}^{B \times L \times (d_0/H)}
$$

**2. Sequential Stacking**
These heads are then concatenated *sequentially* along the sequence dimension (axis 1). This operation creates a longer, thinner intermediate tensor, effectively placing the feature views of each head one after another.
$$
\mathbf{X}_{\text{stacked}} = \text{Concat}_{\text{axis}=1}(\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_H) \in \mathbb{R}^{B \times (L \cdot H) \times (d_0/H)}
$$

**3. Latent Re-shaping**
The stacked tensor $\mathbf{X}_{\text{stacked}}$ is then reshaped back to the original sequence length $L$. Since the total number of features ($L \cdot H \cdot d_0/H = L \cdot d_0$) is preserved, the resulting tensor has a feature dimension of $d_0$.
$$
\mathbf{X}_{\text{reshaped}} = \text{Reshape}(\mathbf{X}_{\text{stacked}}) \in \mathbb{R}^{B \times L \times d_0}
$$

**4. Latent Projection and Residual Connection**
Finally, the reshaped tensor and the original input tensor $\mathbf{X}$ are projected to the target latent dimension $d_{\text{new}}$ using a shared linear projection layer with weights $\mathbf{W}_p \in \mathbb{R}^{d_0 \times d_{\text{new}}}$. The results are combined with a residual connection.
$$
\mathbf{Y} = \text{ReLU}(\mathbf{X}_{\text{reshaped}} \mathbf{W}_p)
$$
$$
\mathbf{X}_{\text{proj}} = \text{ReLU}(\mathbf{X} \mathbf{W}_p)
$$
The final output of the LMA transform is the sum of these two components:
$$
\mathbf{Z} = \mathbf{Y} + \mathbf{X}_{\text{proj}} \in \mathbb{R}^{B \times L \times d_{\text{new}}}
$$
This final tensor $\mathbf{Z}$ is then passed to a series of standard Transformer blocks that operate in the more efficient $d_{\text{new}}$ dimension.

#### Integration and Complexity

The LMA transformation acts as a pre-processing step before the main Transformer encoder. The subsequent Transformer blocks are standard, but they benefit from operating on a sequence with a smaller embedding dimension, $d_{\text{new}}$.

**Complexity Analysis:**
*   **MHA:** The computational cost is dominated by the self-attention calculation, which is $O(L^2 \cdot d_0)$.
*   **LMA:** The cost of the LMA transformation is primarily from the linear projections, which is $O(L \cdot d_0 \cdot d_{\text{new}})$. The subsequent Transformer blocks have an attention cost of $O(L^2 \cdot d_{\text{new}})$.

The primary efficiency gain of LMA comes from the condition where $d_{\text{new}} < d_0$. This reduces the cost of the expensive quadratic attention operation, making it more suitable for scenarios where the embedding dimension can be compressed without significant information loss.

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