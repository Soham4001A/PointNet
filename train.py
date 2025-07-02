"""
pointcloud_lma_vs_mha.py
Author  : Soham + ChatGPT
Purpose : Compare Latent Meta Attention (LMA) against ordinary Multi-Head
          Attention (MHA) on a 3-D point-cloud classification task.
          The script is **modular & verbose** so you can tweak hyper-params
          or swap components with minimal friction.
---------------------------------------------------------------------------
High-level pipeline
-------------------
          PointCloud (N×3)
                │
      PointNetBackbone (per-point MLP)
                │        (#L points ⇒ sequence length L)
                │
 ┌──────────────┴──────────────┐  (toggle with --use_lma)
 │ (1) LMAInitialTransform     │  – split/stack/re-chunk/embed once,
 │     OR                      │    residual back-projection included
 │ (2) Identity pass-through   │  – Standard path keeps original tensor
 └──────────────┬──────────────┘
         sequence (B, L*, d*)          *L*=L_new, d*=d_new  if LMA enabled
                │
       N × TransformerBlock
        (ordinary MHA in d*)
                │
 Global Max-Pool (mask-aware)
                │
        Classification head
---------------------------------------------------------------------------
Run:
    python train.py --help           # list CLI flags
    python train.py --use_lma        # LMA variant
    python train.py --summary --flops
"""

# ------------------------- 1. Imports ------------------------------------
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from optim.sgd import DAG as DynAG                    # <- provided in your env

#  ---------------- torch_geometric for ModelNet40 ------------------------
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader   # PyG‑aware loader
from torch_geometric.utils import to_dense_batch
from torch.utils.data import random_split       # still use for splitting

#  ---------------- Optional diagnostic helpers ---------------------------
from ptflops import get_model_complexity_info     # FLOPs / MACs



# =========================================================================
# 2.  Dataset utilities ---------------------------------------------------
# =========================================================================
def get_dataloaders(root: str,
                    name: str,
                    num_points: int,
                    batch_size: int,
                    val_split: float = 0.1,
                    workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download (if needed) ModelNet, sample fixed-size point clouds, and
    return train / val / test PyG DataLoaders.

    * root       : directory where ModelNet files are cached
    * name       : ModelNet version ('10' or '40')
    * num_points : #points per cloud after FPS sampling / random pick
    * val_split  : fraction of training set carved out for validation
    """
    transform = Compose([
        NormalizeScale(),            # put each model inside unit sphere
        SamplePoints(num_points)     # FPS + random noise  (§torch_geometric)
    ])

    # ModelNet provides predefined train/test splits.
    dataset_train = ModelNet(root, name=name, train=True,  transform=transform)
    dataset_test  = ModelNet(root, name=name, train=False, transform=transform)

    # carve validation subset from training split
    n_val   = int(len(dataset_train) * val_split)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(
        dataset_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(0))   # reproducible

    def make_loader(ds, shuffle=False):
        return DataLoader(ds,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=workers,
                          pin_memory=True)

    return (make_loader(train_set, True),
            make_loader(val_set),
            make_loader(dataset_test))


# =========================================================================
# 3.  Model building blocks ----------------------------------------------
# =========================================================================
class PointNetBackbone(nn.Module):
    """
    Minimal per-point MLP that lifts raw (x,y,z) coordinates to a d-dim
    embedding.  Output is padded to equal length within the mini-batch so
    we can treat it as a dense (B,L,d) tensor.
    """
    def __init__(self, d_embed: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, d_embed)              # -> (N_points, d_embed)
        )

    def forward(self, data: Data):
        """
        Returns:
            x_pad  : (B, L_max, d_embed)  zero-padded
            mask   : (B, L_max)  True at valid positions
        """
        x = self.mlp(data.pos)
        x_pad, mask = to_dense_batch(x, data.batch) # (B, L_max, d), (B, L_max)
        return x_pad, mask


# -------------------------------------------------------------------------
# Latent Meta Attention initial transform (split-stack-re-chunk-embed once)
# -------------------------------------------------------------------------
class LMAInitialTransform(nn.Module):
    """
    Implements exactly the four-step pre-processing:

        1. Split embedding dim into H chunks       -> (B,L,d0/H) * H
        2. Stack along sequence dim                -> (B, H·L, d0/H)
        3. Re-chunk to new sequence length L_new   -> (B,L_new, ?)
        4. Dense proj -> d_new                     -> (B,L_new,d_new)

    A linear projection of the **original** flattened tensor is added back
    to preserve a *residual* connection.
    """
    def __init__(self, d0: int, n_heads: int, d_new: int = None):
        """
        d0      : original embed dimension
        n_heads : number of attention heads (for the split)
        d_new   : target reduced embed dim (default d0//2)
        """
        super().__init__()
        assert d0 % n_heads == 0, "d0 must be divisible by #heads"

        self.n_heads   = n_heads
        self.d0        = d0
        self.d_new     = d_new or (d0 // 2)       # sensible default
        self.proj      = nn.Linear(d0, self.d_new, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args
        ----
            x : (B, L, d0)

        Returns
        -------
            y     : (B, L_new, d_new)  compressed tensor
            mask* : we don't recompute mask here (caller trims manually)
            L_new : sequence length after reshaping
            d_new : embedding dim after projection (constant)
        """
        B, L, d0 = x.shape
        H        = self.n_heads

        # ── 1. split along embed dim  ───────────────────────────
        chunks = torch.chunk(x, H, dim=-1)   # H tensors, each (B,L,d0/H)

        # ── 2. stack along sequence   ───────────────────────────
        stacked = torch.cat(chunks, dim=1)   # (B, H*L, d0/H)

        # ── 3. determine L_new so total features preserved ─────
        total_feat = stacked.shape[1] * stacked.shape[2]          # (H*L)*(d0/H) == L*d0
        d_chunk    = total_feat // L
        assert total_feat % L == 0, \
               "L must divide total feature count"
        reshaped   = stacked.reshape(B, L, d_chunk)                # (B,L,d_chunk)

        # ── 4. embed → d_new  ───────────────────────────────────
        y = F.relu(self.proj(reshaped))                           # (B,L,d_new)

        # ── residual: project original flat tensor, reshape same ─
        x_proj     = F.relu(self.proj(x))                    # (B, L, d_new)
        y = y + x_proj                                            # residual add

        return y, L


# -------------------------------------------------------------------------
# Standard transformer block operating in reduced space
# -------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Plain Pre-LN transformer block (MHA + Feed-Forward)"""
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads,
                                            batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1    = nn.LayerNorm(d_model)
        self.ln2    = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        att_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(att_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


# -------------------------------------------------------------------------
# Complete model wrapper
# -------------------------------------------------------------------------
class PointNetClassifier(nn.Module):
    """
    * If use_lma == False:
          Backbone   ->  N×Transformer(d0) -> Head
      If use_lma == True:
          Backbone   ->  LMAInitialTransform -> N×Transformer(d_new) -> Head
    """
    def __init__(self,
                 num_classes: int,
                 n_blocks: int = 4,
                 d0: int      = 256,
                 heads: int   = 4,
                 d_ff: int    = 512,
                 dropout: float = 0.1,
                 use_lma: bool = False):
        super().__init__()

        # --- shared PointNet stem ---------------------------------
        self.backbone = PointNetBackbone(d0)
        self.use_lma  = use_lma

        # --- LMA pre-processing -----------------------------------
        if use_lma:
            self.lma = LMAInitialTransform(d0, heads)

        # -----------------------------------------------------------
        #   Determine embedding dimension after optional LMA stage
        # -----------------------------------------------------------
        if use_lma:
            d_star = self.lma.d_new          # output embed dim from LMA
        else:
            d_star = d0                      # unchanged embed dim

        # --- N standard transformer blocks in the (L*, d*) space ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d_star, heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        # --- Global pooling & head ---------------------------------
        self.pool = nn.AdaptiveMaxPool1d(1)       # max over sequence
        self.head = nn.Linear(d_star, num_classes)

    def forward(self, data: Data):
        x, mask = self.backbone(data)             # (B,L,d0), (B,L)

        if self.use_lma:                          # SSR once
            x, L_new = self.lma(x)
            mask = mask[:, :L_new]                # trim mask to new length

        # N transformer layers (identical d_model across blocks)
        for blk in self.blocks:
            x = blk(x)

        # Mask out padded positions *before* pooling
        x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        # transpose to (B,d,L) for AdaptiveMaxPool1d
        x = self.pool(x.transpose(1, 2)).squeeze(-1)   # (B,d)
        return self.head(x)


# =========================================================================
# 4.  Train / validation / test loops ------------------------------------
# =========================================================================
def run_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              optimiser=None,
              device="cuda") -> Tuple[float, float]:
    """
    One full pass over `loader`.
    Returns tuple (avg_loss, accuracy) for the split.
    If optimiser is None  -> evaluation mode.
    """
    train = optimiser is not None
    model.train(train)

    total, correct, sum_loss = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)               # logits  (B, C)
        loss  = criterion(out, batch.y)

        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        preds = out.argmax(dim=1)
        total  += batch.y.size(0)
        correct += (preds == batch.y).sum().item()
        sum_loss += loss.item() * batch.y.size(0)

    return sum_loss / total, correct / total


# =========================================================================
# 5.  Main driver with CLI ------------------------------------------------
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Point-Cloud Classification: LMA vs MHA")
    # ---- data & training hyper-params -----------------------------------
    parser.add_argument("--dataset",    type=str,   default="10", choices=["10", "40"],
                        help="ModelNet version to use (10 or 40)")
    parser.add_argument("--root",       type=str,   default="data",
                        help="where to download / load ModelNet")
    parser.add_argument("--points",     type=int,   default=1024,
                        help="#points sampled from each CAD mesh")
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.1)
    # ---- model size knobs ----------------------------------------------
    parser.add_argument("--embed",      type=int,   default=64,
                        help="d0: embed dim after PointNet stem")
    parser.add_argument("--heads",      type=int,   default=2)
    parser.add_argument("--ff_dim",     type=int,   default=128)
    parser.add_argument("--blocks",     type=int,   default=2,
                        help="#transformer layers")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--use_lma",    action="store_true",
                        help="activate Latent Meta Attention path")
    # ---- diagnostics ----------------------------------------------------
    parser.add_argument("--summary", action="store_true",
                        help="print torchinfo layer summary")
    parser.add_argument("--flops",   action="store_true",
                        help="print ptflops MAC / param counts")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device = {device}")

    # ---------------------------------------------------------------------
    # Create data loaders
    # ---------------------------------------------------------------------
    print(f"Preparing ModelNet{args.dataset} loaders...")
    train_dl, val_dl, test_dl = get_dataloaders(args.root,
                                                args.dataset,
                                                args.points,
                                                args.batch,
                                                args.val_split)

    # ---------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------
    model = PointNetClassifier(
        num_classes=int(args.dataset),
        n_blocks=args.blocks,
        d0=args.embed,
        heads=args.heads,
        d_ff=args.ff_dim,
        dropout=args.dropout,
        use_lma=args.use_lma
    ).to(device)

    # ---------------------------------------------------------------------
    # Optional FLOP / layer-summary inspection
    # ---------------------------------------------------------------------
    if args.summary:
        # torchinfo doesn't understand torch_geometric.data.Data objects
        # Use a plain textual module print instead.
        print("\n--- Model architecture ---")
        print(model)
        print("--- End of model architecture ---\n")

    if args.flops:
        # ptflops doesn't work directly with torch_geometric.data.Data objects.
        # We create a wrapper that converts a dense tensor to a Data object.
        class PtflopsWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, pos):
                # ptflops adds a batch dimension, so shape is (1, B, N, 3)
                _, B, N, _ = pos.shape
                batch = torch.arange(B, device=pos.device).view(-1, 1).repeat(1, N).view(-1)
                pos_flat = pos.view(-1, 3)
                data = Data(pos=pos_flat, batch=batch)
                return self.model(data)

        wrapped_model = PtflopsWrapper(model)

        with torch.no_grad():
            macs, params = get_model_complexity_info(
                wrapped_model,
                (args.batch, args.points, 3),
                as_strings=True,
                print_per_layer_stat=False)
            print(f"[ptflops]  MACs (≅FLOPs): {macs}    |   Params: {params}")

    # ---------------------------------------------------------------------
    # Loss, Optimiser (DynAG), Scheduler (optional)
    # ---------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimiser = DynAG(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_dl, criterion,
                                          optimiser, device)
        val_loss, val_acc = run_epoch(model, val_dl, criterion,
                                      optimiser=None, device=device)

        if val_acc > best_val:
            best_val = val_acc

        dt = time.time() - t0
        print(f"Epoch {ep:03d}/{args.epochs}  "
              f"train  loss {train_loss:.4f}  acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.3f}   "
              f"[{dt:.1f}s]")

    # ---------------------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------------------
    test_loss, test_acc = run_epoch(model, test_dl, criterion,
                                    optimiser=None, device=device)
    print(f"\nTEST  accuracy: {test_acc:.3%}  (loss {test_loss:.4f})")
    print("Best validation accuracy seen:", best_val)


# =========================================================================
# Kick-off
# =========================================================================
if __name__ == "__main__":
    main()