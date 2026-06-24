"""HNSD model with incidence-aware drift and stochastic forcing."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .operators import IncidenceOperator


class IncidenceWeights(nn.Module):
    """Learn context-dependent drift or diffusion weights for incidences."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor, operator: IncidenceOperator) -> torch.Tensor:
        edge_features = operator.edge_mean(features)
        context = torch.cat((self.node_projection(features[operator.node_ids]), self.edge_projection(edge_features[operator.edge_ids])), dim=-1)
        return torch.sigmoid(self.score(context)) + 1e-4


class HNSD(nn.Module):
    """Euler-Maruyama HNSD encoder and classifier."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, steps: int, dt: float, dropout: float, input_dropout: float, noise_scale: float) -> None:
        super().__init__()
        self.steps, self.dt, self.dropout, self.input_dropout, self.noise_scale = steps, dt, dropout, input_dropout, noise_scale
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.normalization = nn.BatchNorm1d(hidden_dim)
        self.drift_weights = IncidenceWeights(hidden_dim)
        self.diffusion_weights = IncidenceWeights(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, features: torch.Tensor, operator: IncidenceOperator, stochastic: bool = True) -> torch.Tensor:
        state = F.gelu(self.normalization(self.input_encoder(F.dropout(features, p=self.input_dropout, training=self.training))))
        for _ in range(self.steps):
            gradient = operator.gradient(state)
            drift = -operator.divergence(self.drift_weights(state, operator) * gradient)
            diffusion = operator.divergence(self.diffusion_weights(state, operator) * gradient)
            stochastic_term = self.noise_scale * math.sqrt(self.dt) * diffusion * torch.randn_like(state) if stochastic else 0.0
            state = F.dropout(state + self.dt * drift + stochastic_term, p=self.dropout, training=self.training)
        return state

    def forward(self, features: torch.Tensor, operator: IncidenceOperator, stochastic: bool = True) -> torch.Tensor:
        return self.classifier(self.encode(features, operator, stochastic=stochastic))

