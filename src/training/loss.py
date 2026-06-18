import torch
import torch.nn as nn


def bce_loss() -> nn.BCELoss:
    return nn.BCELoss()


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()