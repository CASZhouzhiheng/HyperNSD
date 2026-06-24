"""Incidence-domain differential operators for hypergraphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IncidenceOperator:
    """Sparse-free implementation of G and G^T over hypergraph incidences."""

    node_ids: torch.Tensor
    edge_ids: torch.Tensor
    inverse_sqrt_node_degree: torch.Tensor
    edge_sizes: torch.Tensor
    num_nodes: int
    num_edges: int

    @classmethod
    def from_edge_index(cls, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> "IncidenceOperator":
        node_ids = edge_index[0].long().to(device)
        _, edge_ids = torch.unique(edge_index[1].long().to(device), sorted=True, return_inverse=True)
        num_edges = int(edge_ids.max().item()) + 1
        node_degree = torch.bincount(node_ids, minlength=num_nodes).float().clamp_min(1.0)
        edge_sizes = torch.bincount(edge_ids, minlength=num_edges).float().clamp_min(1.0)
        return cls(node_ids, edge_ids, node_degree.rsqrt(), edge_sizes, num_nodes, num_edges)

    def edge_mean(self, features: torch.Tensor) -> torch.Tensor:
        output = features.new_zeros((self.num_edges, features.size(-1)))
        output.index_add_(0, self.edge_ids, features[self.node_ids])
        return output / self.edge_sizes.unsqueeze(-1)

    def gradient(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the degree-normalized incidence gradient from Eq. (1)."""
        normalized_features = features * self.inverse_sqrt_node_degree.unsqueeze(-1)
        normalized_edge_mean = self.edge_mean(normalized_features)
        return normalized_features[self.node_ids] - normalized_edge_mean[self.edge_ids]

    def divergence(self, incidence_features: torch.Tensor) -> torch.Tensor:
        scale = self.inverse_sqrt_node_degree[self.node_ids].unsqueeze(-1)
        direct = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        direct.index_add_(0, self.node_ids, incidence_features * scale)
        edge_sum = incidence_features.new_zeros((self.num_edges, incidence_features.size(-1)))
        edge_sum.index_add_(0, self.edge_ids, incidence_features)
        correction = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        correction.index_add_(0, self.node_ids, edge_sum[self.edge_ids] / self.edge_sizes[self.edge_ids].unsqueeze(-1) * scale)
        return direct - correction
