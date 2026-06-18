import math

import torch


def hit_rate_at_k(scores: torch.Tensor, pos_idx: int = 0, k: int = 10) -> float:
    _, topk = scores.topk(k)
    return float((topk == pos_idx).any())


def ndcg_at_k(scores: torch.Tensor, pos_idx: int = 0, k: int = 10) -> float:
    _, topk = scores.topk(k)
    hits = (topk == pos_idx).nonzero(as_tuple=True)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / math.log2(hits[0].item() + 2)


def evaluate(
    model: torch.nn.Module,
    test_df,
    device: torch.device,
    k: int = 10,
) -> dict[str, float]:
    model.eval()
    hr_scores, ndcg_scores = [], []

    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            items = torch.tensor(
                [row.pos_item] + list(row.neg_items), dtype=torch.long, device=device
            )
            user = torch.full_like(items, row.user)

            scores = model(user, items)

            hr_scores.append(hit_rate_at_k(scores, pos_idx=0, k=k))
            ndcg_scores.append(ndcg_at_k(scores, pos_idx=0, k=k))

    return {
        f"HR@{k}": round(sum(hr_scores) / len(hr_scores), 4),
        f"NDCG@{k}": round(sum(ndcg_scores) / len(ndcg_scores), 4),
    }