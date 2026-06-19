import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from models.ncf import NeuMF

CHECKPOINT = Path("saved_models/best_model.pt")
META = Path("data/processed/meta.pkl")

_model: NeuMF | None = None
_device: torch.device | None = None
_n_users: int | None = None
_n_items: int | None = None


def load_model() -> None:
    global _model, _device, _n_users, _n_items

    with open(META, "rb") as f:
        meta = pickle.load(f)

    _n_users = meta["n_users"]
    _n_items = meta["n_items"]

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT, map_location=_device)

    _model = NeuMF(n_users=_n_users, n_items=_n_items)
    _model.load_state_dict(ckpt["model_state"])
    _model.to(_device)
    _model.eval()


def predict(user_id: int, top_k: int = 10) -> list[dict]:
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    if user_id < 0 or user_id >= _n_users:
        raise ValueError(f"user_id {user_id} out of valid range.")

    items = torch.arange(_n_items, device=_device)
    users = torch.full_like(items, user_id)

    with torch.no_grad():
        scores = _model(users, items)

    top_scores, top_indices = scores.topk(top_k)

    return [
        {"item_id": int(idx), "score": round(float(score), 6)}
        for idx, score in zip(top_indices, top_scores)
    ]