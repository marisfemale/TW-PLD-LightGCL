"""Preprocess raw Gowalla check-in data into the project's interaction format.

Pipeline (LightGCN/LightGCL convention):
1. Load raw check-ins from Gowalla_cleanCheckins.csv (or totalCheckins.txt).
2. Deduplicate (user, item) pairs, keeping the latest timestamp per pair
   (needed for Eq. 7 — temporal weight uses one timestamp per interaction).
3. Iterative k-core filtering (default k=10): drop users/items with fewer than
   k interactions until the graph is stable.
4. Remap raw IDs to contiguous 0..n_users-1 and 0..n_items-1.
5. Per-user random 80/10/10 train/val/test split.
6. Write {train,val,test}.csv with columns (user_id, item_id, timestamp)
   and meta.json with dataset statistics.

Usage (from project root):
    python src/preprocess_gowalla.py \\
        --input  data/gowalla/Gowalla_cleanCheckins.csv \\
        --output-dir data/gowalla
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_K_CORE = 10
DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1


def load_raw(input_path: Path) -> pd.DataFrame:
    """Load the raw check-in file.

    Supports both Gowalla_cleanCheckins.csv (comma-separated, with header) and
    Gowalla_totalCheckins.txt (tab-separated, no header). Returns a DataFrame
    with columns: raw_user (int64), raw_item (int64), timestamp (int64 seconds).
    """
    log.info("Loading raw check-ins from %s", input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            input_path,
            usecols=["user", "check-in time", "location id"],
            dtype={"user": "int64", "location id": "int64"},
        )
        df = df.rename(
            columns={"user": "raw_user", "location id": "raw_item", "check-in time": "ts"}
        )
    else:
        df = pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            names=["raw_user", "ts", "lat", "lon", "raw_item"],
            usecols=["raw_user", "ts", "raw_item"],
            dtype={"raw_user": "int64", "raw_item": "int64"},
        )

    # Convert to Unix seconds via explicit timedelta arithmetic — robust to
    # pandas' default datetime resolution (ns in 2.x, us in 3.x) and to
    # tz-aware/naive cast restrictions in 3.x.
    ts = pd.to_datetime(df["ts"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
    df["timestamp"] = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("int64")
    df = df.drop(columns=["ts"])
    log.info(
        "Loaded %s check-ins (%s users, %s locations)",
        f"{len(df):,}",
        f"{df.raw_user.nunique():,}",
        f"{df.raw_item.nunique():,}",
    )
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated (user, item) check-ins, keeping the latest timestamp."""
    log.info("Deduplicating (user, item) pairs, keeping latest timestamp")
    before = len(df)
    df = df.groupby(["raw_user", "raw_item"], as_index=False, sort=False)["timestamp"].max()
    log.info("Dedup: %s -> %s interactions", f"{before:,}", f"{len(df):,}")
    return df


def k_core_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than k interactions until stable."""
    log.info("Applying %d-core filtering", k)
    iteration = 0
    while True:
        iteration += 1
        user_counts = df.groupby("raw_user").size()
        item_counts = df.groupby("raw_item").size()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        new_df = df[df.raw_user.isin(keep_users) & df.raw_item.isin(keep_items)]
        log.info(
            "  iter %d: %s interactions, %s users, %s items",
            iteration,
            f"{len(new_df):,}",
            f"{new_df.raw_user.nunique():,}",
            f"{new_df.raw_item.nunique():,}",
        )
        if len(new_df) == len(df):
            break
        df = new_df
    log.info(
        "k-core converged: %s interactions, %s users, %s items",
        f"{len(df):,}",
        f"{df.raw_user.nunique():,}",
        f"{df.raw_item.nunique():,}",
    )
    return df.reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Remap raw user/item IDs to contiguous 0..n-1."""
    log.info("Remapping IDs to contiguous range")
    user_ids = np.sort(df.raw_user.unique())
    item_ids = np.sort(df.raw_item.unique())
    user_map = {int(raw): i for i, raw in enumerate(user_ids)}
    item_map = {int(raw): i for i, raw in enumerate(item_ids)}
    out = pd.DataFrame(
        {
            "user_id": df.raw_user.map(user_map).astype(np.int32),
            "item_id": df.raw_item.map(item_map).astype(np.int32),
            "timestamp": df.timestamp.astype(np.int64),
        }
    )
    return out, user_map, item_map


def split_per_user(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random per-user split. Each user gets at least 1 val and 1 test interaction."""
    log.info(
        "Splitting per user (val=%.0f%%, test=%.0f%%)",
        val_frac * 100,
        test_frac * 100,
    )
    rng = np.random.default_rng(seed)
    train_idx_chunks: list[np.ndarray] = []
    val_idx_chunks: list[np.ndarray] = []
    test_idx_chunks: list[np.ndarray] = []
    for _, group in df.groupby("user_id", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n - n_test - n_val < 1:
            # Fallback for tiny users (shouldn't occur with k-core >= 3).
            n_test, n_val = 1, 1
        test_idx_chunks.append(idx[:n_test])
        val_idx_chunks.append(idx[n_test : n_test + n_val])
        train_idx_chunks.append(idx[n_test + n_val :])
    train_idx = np.concatenate(train_idx_chunks)
    val_idx = np.concatenate(val_idx_chunks)
    test_idx = np.concatenate(test_idx_chunks)
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to Gowalla_cleanCheckins.csv or Gowalla_totalCheckins.txt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write train/val/test/meta")
    parser.add_argument("--k", type=int, default=DEFAULT_K_CORE, help="k-core threshold")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    df = load_raw(args.input)
    df = deduplicate(df)
    df = k_core_filter(df, k=args.k)
    df, user_map, item_map = remap_ids(df)
    train_df, val_df, test_df = split_per_user(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    n_users = len(user_map)
    n_items = len(item_map)
    n_interactions = len(df)
    meta = {
        "dataset": "gowalla",
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density": n_interactions / (n_users * n_items),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "k_core": args.k,
        "split": f"per-user random {int((1 - args.val_frac - args.test_frac) * 100)}/{int(args.val_frac * 100)}/{int(args.test_frac * 100)}",
        "seed": args.seed,
        "time_min_unix": int(df.timestamp.min()),
        "time_max_unix": int(df.timestamp.max()),
        "source": str(args.input),
    }
    with (args.output_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    log.info("Wrote train=%s, val=%s, test=%s",
             f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}")
    log.info("meta.json:\n%s", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
