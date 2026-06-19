import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import get_bpr_loader
from data.preprocess import run as build_dataset
from models.ncf import NeuMF
from training.loss import bpr_loss
from training.trainer import NCFTrainer

CFG = {
    "seed": 42,
    "epochs": 20,
    "batch_size": 1024,
    "layers": [64, 32, 16, 8],
    "latent_dim_gmf": 8,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "eval_every": 1,
    "k": 10,
    "num_workers": 0,
    "loss_type": "bpr",
}

PROCESSED = Path("data/processed")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not (PROCESSED / "train.parquet").exists():
        print("Processed data not found — running preprocessing pipeline...")
        build_dataset()

    with open(PROCESSED / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    n_users = meta["n_users"]
    n_items = meta["n_items"]
    print(f"Dataset: {n_users:,} users | {n_items:,} items")

    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")

    train_loader = get_bpr_loader(
        train_df,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
    )

    model = NeuMF(
        n_users=n_users,
        n_items=n_items,
        layers=CFG["layers"],
        latent_dim_gmf=CFG["latent_dim_gmf"],
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = NCFTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=bpr_loss,
        device=device,
        loss_type=CFG["loss_type"],
        scheduler=scheduler,
    )

    print(f"\nTraining for {CFG['epochs']} epochs with {CFG['loss_type'].upper()} loss...\n")
    history = trainer.fit(
        train_loader=train_loader,
        test_df=test_df,
        epochs=CFG["epochs"],
        eval_every=CFG["eval_every"],
        k=CFG["k"],
    )

    best_precision = max(history["Precision"]) if history["Precision"] else 0.0
    best_ndcg = max(history["ND" \
    "CG"]) if history["NDCG"] else 0.0
    print(f"\nBest Precision@{CFG['k']}: {best_precision:.4f} | Best NDCG@{CFG['k']}: {best_ndcg:.4f}")
    print("Best model saved to saved_models/best_model.pt")


if __name__ == "__main__":
    main()