"""Phase 1 training: LightGCL baseline on Gowalla.

Reproduces the official HKUDS/LightGCL training protocol. No PLD, no TW —
those land in Phases 2 and 3. Goal is to match the published Recall@K /
NDCG@K within seed variance.

Each run writes to outputs/<run_id>/:
  config.json      full hyperparameter dump
  metrics.csv      per-eval Recall@K / NDCG@K and final test metrics
  best_model.pt    checkpoint at best validation Recall@20
  train.log        copy of the stdout log

Usage:
  Full training (Colab GPU):
    python src/train.py --device cuda
  Local CPU sanity (small batch, capped batches, 1 epoch):
    python src/train.py --device cpu --train-batch-size 1024 \\
        --max-batches-per-epoch 50 --epochs 1 --eval-every-epochs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import RecDataset
from model import LightGCL
from utils import evaluate_batch, set_seed


@dataclass
class Config:
    data_dir: str
    output_dir: str
    embedding_dim: int = 64
    n_layers: int = 2
    svd_q: int = 5
    contrastive_temp: float = 0.2
    cl_weight: float = 0.2
    reg_weight: float = 1e-7
    edge_dropout: float = 0.0
    lr: float = 1e-3
    train_batch_size: int = 4096
    eval_batch_size: int = 256
    epochs: int = 100
    eval_every_epochs: int = 3
    early_stop_patience: int = 10
    eval_k_values: tuple[int, ...] = (20, 50)
    seed: int = 42
    device: str = "cpu"
    max_batches_per_epoch: int = -1


def setup_logging(log_file: Path) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def evaluate(model: LightGCL, ds: RecDataset, labels: list[list[int]],
             batch_size: int, ks: tuple[int, ...]) -> dict[str, float]:
    """Recall@k and NDCG@k for every k in ks, averaged over all users with labels."""
    model.eval()
    device = next(model.parameters()).device
    n_users = ds.meta.n_users
    n_batches = (n_users + batch_size - 1) // batch_size
    sums = {k: [0.0, 0.0, 0] for k in ks}

    with torch.no_grad():
        e_u, e_i, _, _ = model.encode()

    for b in tqdm(range(n_batches), desc="eval", leave=False):
        start, end = b * batch_size, min((b + 1) * batch_size, n_users)
        uids_np = np.arange(start, end)
        uids = torch.from_numpy(uids_np).to(device)
        with torch.no_grad():
            scores = e_u[uids] @ e_i.T
            mask = torch.from_numpy(ds.train_csr[uids_np].toarray()).to(device)
            scores = scores * (1 - mask) - 1e8 * mask
            preds = scores.argsort(dim=1, descending=True).cpu().numpy()
        for k in ks:
            sr, sn, n = evaluate_batch(preds, labels, uids_np, k)
            sums[k][0] += sr; sums[k][1] += sn; sums[k][2] += n

    out: dict[str, float] = {}
    for k in ks:
        sr, sn, n = sums[k]
        out[f"recall@{k}"] = float(sr / max(n, 1))
        out[f"ndcg@{k}"] = float(sn / max(n, 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=str(ROOT.parent / "data" / "gowalla"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT.parent / "outputs"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--svd-q", type=int, default=5)
    parser.add_argument("--contrastive-temp", type=float, default=0.2)
    parser.add_argument("--cl-weight", type=float, default=0.2, help="alpha in Eq. 12")
    parser.add_argument("--reg-weight", type=float, default=1e-7, help="gamma in Eq. 12")
    parser.add_argument("--edge-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every-epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches-per-epoch", type=int, default=-1,
                        help="Cap batches per epoch for fast sanity runs (-1 = no cap)")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config_kwargs = {k: v for k, v in vars(args).items() if k != "run_name"}
    config = Config(**config_kwargs)

    run_id = args.run_name or (time.strftime("%Y%m%d_%H%M%S") + "_lightgcl_gowalla")
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir / "train.log")
    log.info("Run dir: %s", run_dir)
    log.info("Config:\n%s", json.dumps(asdict(config), indent=2, default=list))
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2, default=list)

    set_seed(config.seed)
    ds = RecDataset(config.data_dir, svd_q=config.svd_q,
                    device=config.device, seed=config.seed)

    model = LightGCL(
        n_users=ds.meta.n_users, n_items=ds.meta.n_items,
        embedding_dim=config.embedding_dim, n_layers=config.n_layers,
        adj_norm=ds.adj_norm,
        svd_u_mul_s=ds.svd_u_mul_s, svd_v_mul_s=ds.svd_v_mul_s,
        svd_u_t=ds.svd_u_t, svd_v_t=ds.svd_v_t,
        contrastive_temp=config.contrastive_temp,
        cl_weight=config.cl_weight, reg_weight=config.reg_weight,
        edge_dropout=config.edge_dropout,
    ).to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
    log.info("Trainable params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    metrics_log: list[dict] = []
    best_recall_20 = -1.0
    best_epoch = -1
    evals_without_improvement = 0
    ckpt_path = run_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        ds.train_data.resample_negatives()
        t_neg = time.time() - t0
        loader = ds.train_loader(batch_size=config.train_batch_size)

        model.train()
        total_loss = total_bpr = total_cl = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"ep{epoch}", leave=False, dynamic_ncols=True)
        for batch_idx, (uids, pos, neg, _ts) in enumerate(pbar):
            if 0 <= config.max_batches_per_epoch <= batch_idx:
                break
            uids = uids.long().to(config.device)
            pos = pos.long().to(config.device)
            neg = neg.long().to(config.device)

            opt.zero_grad()
            loss, bpr, cl = model(uids, pos, neg)
            loss.backward()
            opt.step()

            total_loss += loss.item(); total_bpr += bpr.item(); total_cl += cl.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             bpr=f"{bpr.item():.3f}", cl=f"{cl.item():.3f}")

        avg_loss = total_loss / max(n_batches, 1)
        avg_bpr = total_bpr / max(n_batches, 1)
        avg_cl = total_cl / max(n_batches, 1)
        log.info("ep%d done: loss=%.4f bpr=%.4f cl=%.4f | %d batches | neg=%.1fs total=%.1fs",
                 epoch, avg_loss, avg_bpr, avg_cl, n_batches, t_neg, time.time() - t0)

        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs:
            val_metrics = evaluate(model, ds, ds.val_labels,
                                   config.eval_batch_size, config.eval_k_values)
            log.info("ep%d val: %s", epoch,
                     " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
            metrics_log.append({
                "epoch": epoch, "avg_loss": avg_loss, "avg_bpr": avg_bpr, "avg_cl": avg_cl,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            val_r20 = val_metrics["recall@20"]
            if val_r20 > best_recall_20:
                best_recall_20 = val_r20
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_metrics": val_metrics}, ckpt_path)
                log.info("  new best val Recall@20=%.4f, saved checkpoint", val_r20)
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= config.early_stop_patience:
                    log.info("Early stopping: %d evals without improvement",
                             evals_without_improvement)
                    break

    log.info("Training done. Best val Recall@20=%.4f at epoch %d",
             best_recall_20, best_epoch)

    if ckpt_path.exists():
        log.info("Loading best checkpoint for final test eval")
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, ds, ds.test_labels,
                                config.eval_batch_size, config.eval_k_values)
        log.info("FINAL TEST: %s",
                 " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
        metrics_log.append({"epoch": "final_test", **{f"test_{k}": v for k, v in test_metrics.items()}})

    if metrics_log:
        keys = sorted({k for r in metrics_log for k in r.keys()})
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in metrics_log:
                w.writerow(r)
        log.info("Wrote metrics to %s", run_dir / "metrics.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
