import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        layers: list[int] = [64, 32, 16, 8],
        latent_dim_gmf: int = 8,
    ):
        super().__init__()

        latent_dim_mlp = layers[0] // 2

        self.gmf_user_emb = nn.Embedding(n_users, latent_dim_gmf)
        self.gmf_item_emb = nn.Embedding(n_items, latent_dim_gmf)

        self.mlp_user_emb = nn.Embedding(n_users, latent_dim_mlp)
        self.mlp_item_emb = nn.Embedding(n_items, latent_dim_mlp)

        mlp_blocks = []
        in_dim = layers[0]
        for out_dim in layers[1:]:
            mlp_blocks += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_blocks)

        self.output = nn.Sequential(
            nn.Linear(latent_dim_gmf + layers[-1], 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in [
            self.gmf_user_emb,
            self.gmf_item_emb,
            self.mlp_user_emb,
            self.mlp_item_emb,
        ]:
            nn.init.normal_(emb.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.output[0].weight, nonlinearity="sigmoid")
        nn.init.zeros_(self.output[0].bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user_emb(user) * self.gmf_item_emb(item)

        mlp_input = torch.cat(
            [self.mlp_user_emb(user), self.mlp_item_emb(item)], dim=-1
        )
        mlp_out = self.mlp(mlp_input)

        return self.output(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)