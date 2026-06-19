import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/Office_Products.jsonl")
OUT_DIR = Path("data/processed")
NEG_RATIO = 4
MIN_INTERACTIONS = 5
SEED = 42


CHUNK_SIZE = 50_000
KEEP_COLS = ["user_id", "parent_asin", "rating", "timestamp"]


def load_raw(path: Path, chunksize: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    Raw JSONL rows carry review text, images, titles, etc. — fields we never
    use. Reading the whole file at once means pandas materializes every one
    of those fields in memory before we get a chance to drop them. Streaming
    in chunks and pruning columns immediately keeps peak RAM bounded to
    roughly one chunk's footprint instead of the full file's.
    """
    reader = pd.read_json(path, lines=True, chunksize=chunksize)

    pieces = []
    for chunk in reader:
        chunk = chunk[KEEP_COLS].rename(columns={"parent_asin": "item_id"})
        chunk["rating"] = 1
        pieces.append(chunk)

    return pd.concat(pieces, ignore_index=True)


def kcore_filter(df: pd.DataFrame, k: int = MIN_INTERACTIONS) -> pd.DataFrame:
    while True:
        user_counts = df.groupby("user_id").size()
        item_counts = df.groupby("item_id").size()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index

        filtered = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)].copy()
        if len(filtered) == len(df):
            return filtered
        df = filtered


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()

    user_enc = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_enc = {v: i for i, v in enumerate(df["item_id"].unique())}

    df.loc[:, "user"] = df["user_id"].map(user_enc)
    df.loc[:, "item"] = df["item_id"].map(item_enc)

    encoded = df[["user", "item", "rating"]].copy()
    return encoded, user_enc, item_enc


def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user", "item"]).reset_index(drop=True)
    test_idx = df.groupby("user").tail(1).index
    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(index=test_idx).reset_index(drop=True)
    return train, test


def sample_bpr_triplets(
    df: pd.DataFrame, n_items: int, ratio: int, rng: np.random.Generator
) -> pd.DataFrame:
    """BPR needs (user, pos_item, neg_item) triplets, not labeled rows."""
    pos_set = set(zip(df["user"], df["item"]))

    users = np.repeat(df["user"].values, ratio)
    pos_items = np.repeat(df["item"].values, ratio)
    neg_candidates = rng.integers(0, n_items, size=len(users))

    mask = np.array(
        [(u, i) not in pos_set for u, i in zip(users, neg_candidates)]
    )

    return pd.DataFrame(
        {
            "user": users[mask],
            "pos_item": pos_items[mask],
            "neg_item": neg_candidates[mask],
        }
    )


def run(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    raw = load_raw(raw_path)
    raw = kcore_filter(raw)
    df, user_enc, item_enc = encode_ids(raw)
    assert {"user", "item", "rating"}.issubset(df.columns), (
        f"encode_ids did not produce expected columns, got {list(df.columns)}"
    )

    n_users = len(user_enc)
    n_items = len(item_enc)

    train_pos, test_pos = leave_one_out_split(df)
    train = sample_bpr_triplets(train_pos, n_items, NEG_RATIO, rng)

    test_records = []
    all_items = np.arange(n_items)
    for row in test_pos.itertuples(index=False):
        neg_pool = all_items[all_items != row.item]
        negs = rng.choice(neg_pool, size=99, replace=False)
        test_records.append(
            {"user": row.user, "pos_item": row.item, "neg_items": negs.tolist()}
        )
    test = pd.DataFrame(test_records)

    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(
            {"encoders": {"user": user_enc, "item": item_enc}, "n_users": n_users, "n_items": n_items},
            f,
        )

    print(
        f"Saved → train: {len(train):,} triplets | "
        f"test: {len(test):,} users | "
        f"{n_users:,} users, {n_items:,} items"
    )


if __name__ == "__main__":
    run()