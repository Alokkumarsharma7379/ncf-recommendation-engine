from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import evaluate


class NCFTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        loss_type: str = "bpr",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = "saved_models",
    ):
        if loss_type not in {"bce", "bpr"}:
            raise ValueError(f"loss_type must be 'bce' or 'bpr', got {loss_type!r}")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.loss_type = loss_type
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_precision = -float("inf")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            self.optimizer.zero_grad(set_to_none=True)

            if self.loss_type == "bpr":
                users, pos_items, neg_items = (t.to(self.device) for t in batch)
                pos_scores = self.model(users, pos_items)
                neg_scores = self.model(users, neg_items)
                loss = self.loss_fn(pos_scores, neg_scores)
            else:
                users, items, labels = (t.to(self.device) for t in batch)
                preds = self.model(users, items)
                loss = self.loss_fn(preds, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader)

    def evaluate(self, test_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
        return evaluate(self.model, test_df, self.device, k=k)

    def save_checkpoint(self, metrics: dict[str, float], epoch: int) -> bool:
        precision_key = next(k for k in metrics if k.startswith("Precision"))
        if metrics[precision_key] <= self.best_precision:
            return False

        self.best_precision = metrics[precision_key]
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, self.checkpoint_dir / "best_model.pt")
        return True

    def fit(
        self,
        train_loader: DataLoader,
        test_df: pd.DataFrame,
        epochs: int,
        eval_every: int = 1,
        k: int = 10,
    ) -> dict[str, list]:
        history = {"loss": [], "Precision": [], "NDCG": []}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            history["loss"].append(loss)

            if epoch % eval_every == 0:
                metrics = self.evaluate(test_df, k=k)
                improved = self.save_checkpoint(metrics, epoch)

                precision_key = next(k for k in metrics if k.startswith("Precision"))
                ndcg_key = next(k for k in metrics if k.startswith("NDCG"))

                history["Precision"].append(metrics[precision_key])
                history["NDCG"].append(metrics[ndcg_key])

                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"{precision_key}: {metrics[precision_key]:.4f} | "
                    f"{ndcg_key}: {metrics[ndcg_key]:.4f}"
                    + (" ✓ saved" if improved else "")
                )
            else:
                print(f"Epoch {epoch:>3}/{epochs} | Loss: {loss:.4f}")

        return history