from __future__ import annotations

import torch
from torch import nn


class MaskedVariableForecaster(nn.Module):
    """One variate-token forecaster whose only route input is an allowed-edge mask."""

    def __init__(
        self,
        history: int,
        horizon: int,
        variables: int,
        hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.history = int(history)
        self.horizon = int(horizon)
        self.variables = int(variables)
        self.hidden = int(hidden)
        self.history_projection = nn.Linear(self.history, self.hidden)
        self.node_embedding = nn.Parameter(torch.zeros(1, self.variables, self.hidden))
        nn.init.normal_(self.node_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=int(heads),
            dim_feedforward=4 * self.hidden,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.horizon),
        )

    def local_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, history, variables]")
        if x.shape[1] != self.history or x.shape[2] != self.variables:
            raise ValueError(
                f"Expected [batch,{self.history},{self.variables}], got {tuple(x.shape)}"
            )
        return self.history_projection(x.transpose(1, 2)) + self.node_embedding

    def forward(
        self,
        x: torch.Tensor,
        allowed_mask: torch.Tensor,
        *,
        return_local_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if allowed_mask.shape != (self.variables, self.variables):
            raise ValueError("allowed_mask must be [variables, variables]")
        local = self.local_features(x)
        encoded = self.interaction(local, mask=~allowed_mask.bool())
        prediction = self.head(encoded).transpose(1, 2)
        if return_local_features:
            return prediction, local
        return prediction


class PriorGate(nn.Module):
    """Node-by-horizon gate using history and forecast-available branch evidence.

    ``branch_features`` contains the normalized A/R disagreement and its
    magnitude for every target node and horizon.  It is computed from model
    outputs that are available at inference time; it never uses the target.
    ``branch_feature_dim=0`` keeps the small legacy interface useful for
    shape-only callers, while the v2 runner uses ``2 * horizon`` features.
    """

    def __init__(
        self,
        hidden: int,
        horizon: int,
        dropout: float = 0.1,
        branch_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.branch_feature_dim = int(branch_feature_dim)
        if self.branch_feature_dim < 0:
            raise ValueError("branch_feature_dim must be non-negative")
        self.horizon_embedding = nn.Embedding(self.horizon, self.hidden)
        self.network = nn.Sequential(
            nn.Linear(3 * self.hidden + self.branch_feature_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        branch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_features.ndim != 3 or local_features.shape[-1] != self.hidden:
            raise ValueError("local_features must be [batch, nodes, hidden]")
        batch, nodes, _ = local_features.shape
        if self.branch_feature_dim:
            if branch_features is None:
                raise ValueError("branch_features are required for this gate")
            if branch_features.shape != (batch, nodes, self.branch_feature_dim):
                raise ValueError(
                    "branch_features must be "
                    f"[batch,nodes,{self.branch_feature_dim}], got {tuple(branch_features.shape)}"
                )
        elif branch_features is not None and branch_features.shape[:2] != (batch, nodes):
            raise ValueError("branch_features batch/node dimensions must match local_features")
        global_context = local_features.mean(dim=1, keepdim=True).expand(batch, nodes, self.hidden)
        local = local_features.unsqueeze(1).expand(batch, self.horizon, nodes, self.hidden)
        global_expanded = global_context.unsqueeze(1).expand(batch, self.horizon, nodes, self.hidden)
        horizon_ids = torch.arange(self.horizon, device=local_features.device)
        horizon = self.horizon_embedding(horizon_ids).view(1, self.horizon, 1, self.hidden)
        horizon = horizon.expand(batch, self.horizon, nodes, self.hidden)
        features = torch.cat([local, global_expanded, horizon], dim=-1)
        if self.branch_feature_dim:
            branch = branch_features.unsqueeze(1).expand(batch, self.horizon, nodes, self.branch_feature_dim)
            features = torch.cat([features, branch], dim=-1)
        return self.network(features).squeeze(-1)


class MatchedPriorGate(nn.Module):
    """Two-head v3 gate for benefit and structure-specificity evidence.

    Protocol v2 applied two BCE losses to one logit.  When the loss target was
    near 0.5 and the stricter specificity target was near zero, the unique
    optimum was an uninformative compromise around one third.  V3 estimates
    the two scientific statements with separate heads and combines their
    probabilities only after both are calibrated.
    """

    def __init__(
        self,
        hidden: int,
        horizon: int,
        dropout: float = 0.1,
        branch_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.branch_feature_dim = int(branch_feature_dim)
        if self.branch_feature_dim < 0:
            raise ValueError("branch_feature_dim must be non-negative")
        self.horizon_embedding = nn.Embedding(self.horizon, self.hidden)
        self.trunk = nn.Sequential(
            nn.Linear(3 * self.hidden + self.branch_feature_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.loss_head = nn.Linear(self.hidden, 1)
        self.specificity_head = nn.Linear(self.hidden, 1)

    def forward(
        self,
        local_features: torch.Tensor,
        branch_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_features.ndim != 3 or local_features.shape[-1] != self.hidden:
            raise ValueError("local_features must be [batch, nodes, hidden]")
        batch, nodes, _ = local_features.shape
        if branch_features.shape != (batch, nodes, self.branch_feature_dim):
            raise ValueError(
                "branch_features must be "
                f"[batch,nodes,{self.branch_feature_dim}], got {tuple(branch_features.shape)}"
            )
        global_context = local_features.mean(dim=1, keepdim=True).expand(
            batch, nodes, self.hidden
        )
        local = local_features.unsqueeze(1).expand(
            batch, self.horizon, nodes, self.hidden
        )
        global_expanded = global_context.unsqueeze(1).expand(
            batch, self.horizon, nodes, self.hidden
        )
        horizon_ids = torch.arange(self.horizon, device=local_features.device)
        horizon = self.horizon_embedding(horizon_ids).view(
            1, self.horizon, 1, self.hidden
        )
        horizon = horizon.expand(batch, self.horizon, nodes, self.hidden)
        branch = branch_features.unsqueeze(1).expand(
            batch, self.horizon, nodes, self.branch_feature_dim
        )
        encoded = self.trunk(
            torch.cat([local, global_expanded, horizon, branch], dim=-1)
        )
        return (
            self.loss_head(encoded).squeeze(-1),
            self.specificity_head(encoded).squeeze(-1),
        )


def matched_prior_probability(
    loss_logits: torch.Tensor,
    specificity_logits: torch.Tensor,
) -> torch.Tensor:
    """Normalized soft-AND used as the selective prior mixture coefficient."""
    if loss_logits.shape != specificity_logits.shape:
        raise ValueError("Matched-prior gate logits must have identical shape")
    loss_probability = torch.sigmoid(loss_logits)
    specificity_probability = torch.sigmoid(specificity_logits)
    return torch.sqrt(loss_probability * specificity_probability)
