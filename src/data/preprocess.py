import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/ml-100k/u.data")
OUT_DIR = Path("data/processed")
NEG_RATIO = 4
SEED = 42


def load_and_encode(path: Path) -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="c",
    )
    df["rating"] = 1

    user_enc = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_enc = {v: i for i, v in enumerate(df["item_id"].unique())}

    df["user"] = df["user_id"].map(user_enc)
    df["item"] = df["item_id"].map(item_enc)

    return df[["user", "item", "rating"]], user_enc, item_enc


def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sort once; last interaction per user becomes the test item
    df = df.sort_values(["user", "item"]).reset_index(drop=True)
    test_idx = df.groupby("user").tail(1).index
    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(index=test_idx).reset_index(drop=True)
    return train, test


def sample_negatives(
    df: pd.DataFrame, n_items: int, ratio: int, rng: np.random.Generator
) -> pd.DataFrame:
    pos_set = set(zip(df["user"], df["item"]))
    n_users = df["user"].nunique()

    users, items, labels = [], [], []

    # Positive rows
    users.extend(df["user"].tolist())
    items.extend(df["item"].tolist())
    labels.extend([1] * len(df))

    # Negatives — over-sample then filter collisions; faster than per-row rejection loop
    n_neg = len(df) * ratio
    candidates_u = np.repeat(df["user"].values, ratio)
    candidates_i = rng.integers(0, n_items, size=n_neg)

    mask = np.array(
        [
            (u, i) not in pos_set
            for u, i in zip(candidates_u, candidates_i)
        ]
    )
    users.extend(candidates_u[mask].tolist())
    items.extend(candidates_i[mask].tolist())
    labels.extend([0] * int(mask.sum()))

    return pd.DataFrame({"user": users, "item": items, "label": labels})


def run(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df, user_enc, item_enc = load_and_encode(raw_path)
    n_users = len(user_enc)
    n_items = len(item_enc)

    train_pos, test_pos = leave_one_out_split(df)
    train = sample_negatives(train_pos, n_items, NEG_RATIO, rng)

    # Test set: 1 positive + 99 sampled negatives per user (standard LOO eval protocol)
    test_records = []
    all_items = np.arange(n_items)
    for row in test_pos.itertuples(index=False):
        neg_pool = all_items[all_items != row.item]
        negs = rng.choice(neg_pool, size=99, replace=False)
        test_records.append(
            {
                "user": row.user,
                "pos_item": row.item,
                "neg_items": negs.tolist(),
            }
        )
    test = pd.DataFrame(test_records)

    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    meta = {"n_users": n_users, "n_items": n_items}
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"encoders": {"user": user_enc, "item": item_enc}, **meta}, f)

    print(
        f"Saved → train: {len(train):,} rows | "
        f"test: {len(test):,} users | "
        f"{n_users} users, {n_items} items"
    )


if __name__ == "__main__":
    run()