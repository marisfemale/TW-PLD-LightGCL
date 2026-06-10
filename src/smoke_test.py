"""Smoke test for Phase 1 wiring: load data, run one forward/backward, run prediction.

Does not check correctness of numerics — only that the pipeline runs end-to-end
without shape mismatches or NaNs. Run before kicking off full training.

Usage (from project root):
    python src/smoke_test.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import RecDataset
from model import LightGCL
from utils import evaluate_batch, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("smoke")


def main() -> int:
    set_seed(42)
    data_dir = ROOT.parent / "data" / "gowalla"
    log.info("Loading dataset from %s", data_dir)
    ds = RecDataset(data_dir, svd_q=5, device="cpu", seed=42)
    log.info("Meta: %s", ds.meta)

    log.info("Instantiating LightGCL (d=64, K=2)")
    model = LightGCL(
        n_users=ds.meta.n_users,
        n_items=ds.meta.n_items,
        embedding_dim=64,
        n_layers=2,
        adj_norm=ds.adj_norm,
        svd_u_mul_s=ds.svd_u_mul_s,
        svd_v_mul_s=ds.svd_v_mul_s,
        svd_u_t=ds.svd_u_t,
        svd_v_t=ds.svd_v_t,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %s", f"{n_params:,}")

    log.info("Resampling negatives (this is one epoch's worth, ~%s pairs)", f"{ds.meta.n_train:,}")
    ds.train_data.resample_negatives()
    log.info("Negatives sampled. First 5 (u, pos, neg, ts): %s",
             list(zip(ds.train_data.users[:5], ds.train_data.pos_items[:5],
                      ds.train_data.negs[:5], ds.train_data.timestamps[:5])))

    log.info("Running one training mini-batch (batch=256)")
    loader = ds.train_loader(batch_size=256)
    uids, pos, neg, _ts = next(iter(loader))
    log.info("  batch shapes: u=%s pos=%s neg=%s", tuple(uids.shape), tuple(pos.shape), tuple(neg.shape))

    model.train()
    opt.zero_grad()
    loss, bpr, cl = model(uids.long(), pos.long(), neg.long())
    loss.backward()
    opt.step()
    log.info("  loss=%.4f  bpr=%.4f  cl_weighted=%.4f", loss.item(), bpr.item(), cl.item())
    assert torch.isfinite(loss), "loss is not finite"

    log.info("Running prediction on 64 users")
    model.eval()
    test_uids = torch.arange(64)
    preds = model.predict(test_uids, ds.train_csr).cpu().numpy()
    assert preds.shape == (64, ds.meta.n_items), f"unexpected pred shape {preds.shape}"
    sum_r, sum_n, n = evaluate_batch(preds, ds.test_labels, test_uids.numpy(), k=20)
    if n > 0:
        log.info("  random-init Recall@20=%.6f NDCG@20=%.6f (expected ~0 on random init)",
                 sum_r / n, sum_n / n)
    else:
        log.warning("  no users had ground-truth labels in this slice")

    log.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
