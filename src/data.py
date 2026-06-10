"""Dataset loading for TW-PLD-LightGCL.

Reads the processed CSVs produced by preprocess_gowalla.py and builds the
objects the LightGCL backbone needs:

- train_csr: scipy CSR (n_users, n_items), binary
- adj_norm:  torch sparse LightGCN-normalized adjacency 1/sqrt(|O_u|*|O_i|) (Eq. 1)
- svd_*:     low-rank SVD factors of the binary train matrix (Eq. 2),
             kept as factors so global propagation never materializes R̃ (Eq. 3)
- test/val_labels: list-of-lists ground truth for evaluation
- TrainData: torch Dataset emitting (user, pos_item, neg_item, timestamp).
             Timestamp is included for the Phase 3 Temporal Weighting
             component; Phase 1 / baseline ignores it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as torch_data

from utils import scipy_sparse_to_torch_sparse

log = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    dataset: str
    n_users: int
    n_items: int
    n_interactions: int
    n_train: int
    n_val: int
    n_test: int


class TrainData(torch_data.Dataset):
    """Per-pair training data with per-epoch negative resampling.

    Each item is (user_id, pos_item, neg_item, timestamp). resample_negatives()
    must be called once per epoch before iteration begins (matches official
    LightGCL training loop). Negatives are rejection-sampled to avoid items
    the user already interacted with.
    """

    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        timestamps: np.ndarray,
        train_csr: sp.csr_matrix,
        seed: int = 42,
    ):
        self.users = users.astype(np.int64)
        self.pos_items = pos_items.astype(np.int64)
        self.timestamps = timestamps.astype(np.int64)
        self.dok = train_csr.todok()
        self.n_items = train_csr.shape[1]
        self.negs = np.zeros(len(self.users), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def resample_negatives(self) -> None:
        n = len(self.users)
        # Vectorized first guess, then patch collisions individually.
        proposals = self.rng.integers(0, self.n_items, size=n)
        for idx in range(n):
            u = int(self.users[idx])
            j = int(proposals[idx])
            while (u, j) in self.dok:
                j = int(self.rng.integers(0, self.n_items))
            self.negs[idx] = j

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.pos_items[idx], self.negs[idx], self.timestamps[idx]


class RecDataset:
    """Container for everything Phase 1 needs from disk."""

    def __init__(
        self,
        data_dir: Path | str,
        svd_q: int = 5,
        device: str = "cpu",
        seed: int = 42,
    ):
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.device = torch.device(device)
        self.svd_q = svd_q

        with (data_dir / "meta.json").open() as f:
            m = json.load(f)
        self.meta = DatasetMeta(
            dataset=m["dataset"],
            n_users=m["n_users"],
            n_items=m["n_items"],
            n_interactions=m["n_interactions"],
            n_train=m["n_train"],
            n_val=m["n_val"],
            n_test=m["n_test"],
        )

        log.info(
            "Loading %s: %d users, %d items, %d train interactions",
            self.meta.dataset, self.meta.n_users, self.meta.n_items, self.meta.n_train,
        )
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df = pd.read_csv(data_dir / "val.csv")
        test_df = pd.read_csv(data_dir / "test.csv")

        self.train_csr = self._build_csr(train_df)
        self.adj_norm = self._build_normalized_adj(self.train_csr).to(self.device)

        self.train_data = TrainData(
            users=train_df.user_id.values,
            pos_items=train_df.item_id.values,
            timestamps=train_df.timestamp.values,
            train_csr=self.train_csr,
            seed=seed,
        )

        self.val_labels = self._build_labels(val_df, self.meta.n_users)
        self.test_labels = self._build_labels(test_df, self.meta.n_users)

        log.info("Computing low-rank SVD (q=%d) on the normalized adjacency", svd_q)
        # SVD is on the LightGCN-normalized adjacency (matches the HKUDS reference).
        # Using the unnormalized binary R here gives singular values dominated by
        # high-degree nodes; the global view then over-amplifies popular items
        # (~40% drop in Recall@20) and the unnormalized scale pushes InfoNCE
        # logits high enough to overflow exp() after ~40 epochs.
        u, s, v = torch.svd_lowrank(self.adj_norm, q=svd_q)
        diag_s = torch.diag(s)
        self.svd_u_mul_s = (u @ diag_s).contiguous()
        self.svd_v_mul_s = (v @ diag_s).contiguous()
        self.svd_u_t = u.T.contiguous()
        self.svd_v_t = v.T.contiguous()
        del u, s, v, diag_s
        log.info("Dataset ready.")

    def _build_csr(self, df: pd.DataFrame) -> sp.csr_matrix:
        return sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df.user_id.values, df.item_id.values)),
            shape=(self.meta.n_users, self.meta.n_items),
        )

    @staticmethod
    def _build_normalized_adj(csr: sp.csr_matrix) -> torch.Tensor:
        """Symmetric normalization 1/sqrt(|O_u|*|O_i|) — Eq. 1."""
        row_deg = np.array(csr.sum(axis=1)).flatten()
        col_deg = np.array(csr.sum(axis=0)).flatten()
        coo = csr.tocoo()
        norm = coo.data / np.sqrt(row_deg[coo.row] * col_deg[coo.col] + 1e-12)
        norm_coo = sp.coo_matrix(
            (norm.astype(np.float32), (coo.row, coo.col)), shape=csr.shape
        )
        return scipy_sparse_to_torch_sparse(norm_coo).coalesce()

    @staticmethod
    def _build_labels(df: pd.DataFrame, n_users: int) -> list[list[int]]:
        labels: list[list[int]] = [[] for _ in range(n_users)]
        for u, i in zip(df.user_id.values, df.item_id.values):
            labels[int(u)].append(int(i))
        return labels

    def train_loader(self, batch_size: int, num_workers: int = 0) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
