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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = "saved_models",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_ndcg = -float("inf")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for users, items, labels in tqdm(loader, desc="Training", leave=False):
            users = users.to(self.device)
            items = items.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
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
        ndcg_key = next(k for k in metrics if k.startswith("NDCG"))
        if metrics[ndcg_key] <= self.best_ndcg:
            return False

        self.best_ndcg = metrics[ndcg_key]
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
        history = {"loss": [], "HR": [], "NDCG": []}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            history["loss"].append(loss)

            if epoch % eval_every == 0:
                metrics = self.evaluate(test_df, k=k)
                improved = self.save_checkpoint(metrics, epoch)

                hr_key = next(k for k in metrics if k.startswith("HR"))
                ndcg_key = next(k for k in metrics if k.startswith("NDCG"))

                history["HR"].append(metrics[hr_key])
                history["NDCG"].append(metrics[ndcg_key])

                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"{hr_key}: {metrics[hr_key]:.4f} | "
                    f"{ndcg_key}: {metrics[ndcg_key]:.4f}"
                    + (" ✓ saved" if improved else "")
                )
            else:
                print(f"Epoch {epoch:>3}/{epochs} | Loss: {loss:.4f}")

        return history