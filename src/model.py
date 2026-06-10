"""LightGCL backbone for TW-PLD-LightGCL (Phase 1 — baseline reproduction).

Implements Equations 1, 3, 4, 5, 6, 11, 12 from docs/equations.md. The numerics
match the official HKUDS/LightGCL reference (clamped pos-score, log-sum-exp
neg-score for CL; sum-aggregation across layers) so baseline Recall@K /
NDCG@K should reproduce the published numbers.

Phase 1 deliberately excludes PLD (Eqs. 8-10) and temporal weighting (Eq. 7);
those land in Phases 2 and 3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from utils import sparse_dropout


class LightGCL(nn.Module):
    """Pure LightGCL encoder + BPR + InfoNCE contrastive loss.

    Args:
        n_users, n_items, embedding_dim, n_layers: model size (|U|, |I|, d, K)
        adj_norm: sparse (n_users, n_items) tensor with 1/sqrt(|O_u||O_i|) values (Eq. 1)
        svd_u_mul_s: U_q @ diag(Σ_q), shape (n_users, q)
        svd_v_mul_s: V_q @ diag(Σ_q), shape (n_items, q)
        svd_u_t: U_q^T, shape (q, n_users)
        svd_v_t: V_q^T, shape (q, n_items)
        contrastive_temp: τ in Eq. 11
        cl_weight: α in Eq. 12
        reg_weight: γ in Eq. 12
        edge_dropout: edge dropout rate on adj_norm during training (0.0 in defaults)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        adj_norm: Tensor,
        svd_u_mul_s: Tensor,
        svd_v_mul_s: Tensor,
        svd_u_t: Tensor,
        svd_v_t: Tensor,
        contrastive_temp: float = 0.2,
        cl_weight: float = 0.2,
        reg_weight: float = 1e-7,
        edge_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.contrastive_temp = contrastive_temp
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.edge_dropout = edge_dropout

        # Initial embeddings — Θ in Eq. 12
        self.e_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, embedding_dim)))
        self.e_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_items, embedding_dim)))

        # Non-trainable graph tensors held by reference. Sparse tensors as
        # nn.Buffer are awkward; we move them to device ourselves and assume
        # the caller has already done so.
        self.adj_norm = adj_norm
        self.svd_u_mul_s = svd_u_mul_s
        self.svd_v_mul_s = svd_v_mul_s
        self.svd_u_t = svd_u_t
        self.svd_v_t = svd_v_t

        # Cached after forward() so predict() can re-use without re-encoding.
        self.e_u_final: Tensor | None = None
        self.e_i_final: Tensor | None = None

    def encode(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate K layers on local and global views (Eqs. 1, 3, 4).

        Aggregation across layers is sum, matching the official reference;
        Eq. 4 specifies mean — the difference is a constant factor and does
        not affect ranking.
        """
        e_u_list: list[Tensor] = [self.e_u_0]
        e_i_list: list[Tensor] = [self.e_i_0]
        g_u_list: list[Tensor] = [self.e_u_0]
        g_i_list: list[Tensor] = [self.e_i_0]

        adj = sparse_dropout(self.adj_norm, self.edge_dropout) if self.training else self.adj_norm
        adj_t = adj.transpose(0, 1)

        for _ in range(self.n_layers):
            prev_e_u = e_u_list[-1]
            prev_e_i = e_i_list[-1]

            # Local view propagation (Eq. 1)
            z_u = torch.sparse.mm(adj, prev_e_i)
            z_i = torch.sparse.mm(adj_t, prev_e_u)
            e_u_list.append(z_u)
            e_i_list.append(z_i)

            # Global view via SVD factors (Eq. 3) — never materialize R̃.
            vt_ei = self.svd_v_t @ prev_e_i               # (q, d)
            g_u_list.append(self.svd_u_mul_s @ vt_ei)     # (n_users, d)
            ut_eu = self.svd_u_t @ prev_e_u               # (q, d)
            g_i_list.append(self.svd_v_mul_s @ ut_eu)     # (n_items, d)

        e_u_final = torch.stack(e_u_list, dim=0).sum(dim=0)
        e_i_final = torch.stack(e_i_list, dim=0).sum(dim=0)
        g_u_final = torch.stack(g_u_list, dim=0).sum(dim=0)
        g_i_final = torch.stack(g_i_list, dim=0).sum(dim=0)
        return e_u_final, e_i_final, g_u_final, g_i_final

    def forward(
        self, uids: Tensor, pos_iids: Tensor, neg_iids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training forward pass. Returns (total_loss, bpr_loss, weighted_cl_loss)."""
        e_u, e_i, g_u, g_i = self.encode()
        self.e_u_final = e_u
        self.e_i_final = e_i

        # BPR loss (Eq. 6)
        u_emb = e_u[uids]
        pos_emb = e_i[pos_iids]
        neg_emb = e_i[neg_iids]
        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)
        bpr_loss = -(pos_score - neg_score).sigmoid().log().mean()

        # InfoNCE contrastive loss (Eq. 11) — symmetric user+item, with the
        # official's clamp-then-mean on the positive term for numerical
        # stability and log-sum-exp on the negative term.
        pos_u = torch.clamp(
            (g_u[uids] * e_u[uids]).sum(-1) / self.contrastive_temp, -5.0, 5.0
        ).mean()
        neg_u = torch.log(
            torch.exp(g_u[uids] @ e_u.T / self.contrastive_temp).sum(1) + 1e-8
        ).mean()

        iids = torch.cat([pos_iids, neg_iids], dim=0)
        pos_i = torch.clamp(
            (g_i[iids] * e_i[iids]).sum(-1) / self.contrastive_temp, -5.0, 5.0
        ).mean()
        neg_i = torch.log(
            torch.exp(g_i[iids] @ e_i.T / self.contrastive_temp).sum(1) + 1e-8
        ).mean()

        cl_loss = -(pos_u + pos_i) + (neg_u + neg_i)

        # L2 regularization over all trainable parameters (Eq. 12)
        reg_loss = sum(p.norm(2).square() for p in self.parameters()) * self.reg_weight

        total_loss = bpr_loss + self.cl_weight * cl_loss + reg_loss
        return total_loss, bpr_loss, self.cl_weight * cl_loss

    @torch.no_grad()
    def predict(self, uids: Tensor, train_csr) -> Tensor:
        """Rank all items for the given users, masking training interactions.

        Returns: long tensor (batch, n_items) of item ids sorted by score (desc).
        """
        e_u, e_i, _, _ = self.encode()
        self.e_u_final, self.e_i_final = e_u, e_i
        scores = e_u[uids] @ e_i.T
        mask_np = train_csr[uids.cpu().numpy()].toarray()
        mask = torch.from_numpy(mask_np).to(scores.device)
        scores = scores * (1 - mask) - 1e8 * mask
        return scores.argsort(dim=1, descending=True)
