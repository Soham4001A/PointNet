# Point-Cloud Classification: Latent Meta Attention vs. Multi-Head Attention

This repository contains a PyTorch implementation of a 3D point-cloud classification model, designed to compare the performance of Latent Meta Attention (LMA) against standard Multi-Head Attention (MHA). The script is modular and well-documented, allowing for easy experimentation with different hyperparameters and model components.

## High-Level Pipeline

The model follows this general pipeline:

1.  **PointNet Backbone**: A per-point MLP lifts the raw (x, y, z) coordinates of each point in the cloud to a higher-dimensional embedding. The output is a sequence of point embeddings.
2.  **Attention Mechanism**: The sequence of point embeddings is then processed by one of two attention mechanisms:
    *   **Latent Meta Attention (LMA)**: A novel attention mechanism that reshapes the input tensor to create a new sequence of "meta-vectors," which are then processed by a standard Transformer.
    *   **Multi-Head Attention (MHA)**: The standard attention mechanism used in Transformers.
3.  **Classification Head**: A global max-pooling layer aggregates the features from the attention mechanism, and a final linear layer produces the classification scores.

## Latent Meta Attention (LMA)

Latent Meta Attention is a technique that reshapes the input tensor to create a new sequence of "meta-vectors." This is done by splitting the embedding dimension into multiple chunks, stacking them along the sequence dimension, and then re-chunking the sequence to a new length. This process is designed to capture more complex relationships between points in the cloud.

### Mathematical Formulation

Let $\mathbf{X} \in \mathbb{R}^{B \times L \times d_0}$ be the input tensor, where $B$ is the batch size, $L$ is the sequence length, and $d_0$ is the initial embedding dimension. The LMA transformation is defined as follows:

1.  **Split**: The embedding dimension $d_0$ is split into $H$ heads, resulting in a set of tensors ${\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_H}$, where each $\mathbf{X}_h \in \mathbb{R}^{B \times L \times (d_0/H)}$.

2.  **Stack**: The heads are concatenated along the sequence dimension to form a new tensor $\mathbf{X}_{\text{stacked}} \in \mathbb{R}^{B \times (H \cdot L) \times (d_0/H)}$:
    $$
    \mathbf{X}_{\text{stacked}} = \text{Concat}(\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_H)_{\text{dim}=1}
    $$

3.  **Re-chunk and Project**: The stacked tensor is reshaped and then projected to a new embedding dimension $d_{\text{new}}$. First, we reshape $\mathbf{X}_{\text{stacked}}$ to $\mathbf{X}_{\text{reshaped}} \in \mathbb{R}^{B \times L \times d_0}$, and then apply a linear projection $\mathbf{W}_p \in \mathbb{R}^{d_0 \times d_{\text{new}}}$:
    $$
    \mathbf{Y} = \text{ReLU}(\mathbf{X}_{\text{reshaped}} \mathbf{W}_p)
    $$

4.  **Residual Connection**: A residual connection is added from the original input tensor $\mathbf{X}$ after a similar projection:
    $$
    \mathbf{X}_{\text{proj}} = \text{ReLU}(\mathbf{X} \mathbf{W}_p)
    $$
    $$
    \mathbf{Z} = \mathbf{Y} + \mathbf{X}_{\text{proj}}
    $$

The final output of the LMA transformation is the tensor $\mathbf{Z} \in \mathbb{R}^{B \times L \times d_{\text{new}}}$. This transformed tensor is then passed to a standard Transformer network.

## Usage

To use this script, you'll first need to install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, you can run the training script with the following command:

```bash
python train.py [OPTIONS]
```

### Command-Line Options

*   `--dataset`: The ModelNet version to use (10 or 40). Default: `10`.
*   `--root`: The directory where the ModelNet data is stored. Default: `data`.
*   `--points`: The number of points to sample from each CAD mesh. Default: `1024`.
*   `--batch`: The batch size. Default: `32`.
*   `--epochs`: The number of epochs to train for. Default: `50`.
*   `--lr`: The learning rate. Default: `1e-3`.
*   `--val_split`: The fraction of the training set to use for validation. Default: `0.1`.
*   `--embed`: The embedding dimension after the PointNet stem. Default: `64`.
*   `--heads`: The number of attention heads. Default: `2`.
*   `--ff_dim`: The feed-forward dimension in the Transformer blocks. Default: `128`.
*   `--blocks`: The number of Transformer layers. Default: `2`.
*   `--dropout`: The dropout rate. Default: `0.1`.
*   `--use_lma`: Activate the Latent Meta Attention path.
*   `--summary`: Print a torchinfo layer summary.
*   `--flops`: Print ptflops MAC / param counts.

## Example

To train a model on ModelNet10 with LMA enabled, you can run the following command:

```bash
python train.py --dataset 10 --use_lma
```