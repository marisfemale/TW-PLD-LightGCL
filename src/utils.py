"""Shared utilities for TW-PLD-LightGCL."""
from __future__ import annotations

import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scipy_sparse_to_torch_sparse(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse COO float32 tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_dropout(mat: torch.Tensor, dropout: float) -> torch.Tensor:
    """Edge dropout on a sparse COO tensor — preserves structure, zeros some values."""
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    return torch.sparse_coo_tensor(indices, values, mat.size())


def recall_at_k(ranked_items: np.ndarray, ground_truth: list[int], k: int) -> float:
    if len(ground_truth) == 0:
        return 0.0
    top_k = set(ranked_items[:k].tolist())
    hits = sum(1 for item in ground_truth if item in top_k)
    return hits / len(ground_truth)


def ndcg_at_k(ranked_items: np.ndarray, ground_truth: list[int], k: int) -> float:
    """NDCG@k matching the official LightGCL formulation (binary relevance)."""
    if len(ground_truth) == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(loc + 2) for loc in range(min(k, len(ground_truth))))
    top_k = ranked_items[:k].tolist()
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(loc + 2) for loc, item in enumerate(top_k) if item in gt_set)
    return dcg / idcg


def evaluate_batch(
    predictions: np.ndarray,
    test_labels: list[list[int]],
    user_ids: np.ndarray,
    k: int,
) -> tuple[float, float, int]:
    """Sum Recall@k and NDCG@k across users in this batch. Returns (sum_recall, sum_ndcg, n_users_scored)."""
    sum_recall, sum_ndcg, n = 0.0, 0.0, 0
    for i, uid in enumerate(user_ids):
        labels = test_labels[uid]
        if len(labels) == 0:
            continue
        sum_recall += recall_at_k(predictions[i], labels, k)
        sum_ndcg += ndcg_at_k(predictions[i], labels, k)
        n += 1
    return sum_recall, sum_ndcg, n
