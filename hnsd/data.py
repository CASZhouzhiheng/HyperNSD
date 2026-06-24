"""Dataset loading and OOD construction for the six HNSD benchmarks."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


DATASET_PATHS = {
    "CocitationCora": Path("cocitation/cora"),
    "CocitationCiteseer": Path("cocitation/citeseer"),
    "CoauthorshipCora": Path("coauthorship/cora"),
    "CoauthorshipDBLP": Path("coauthorship/dblp"),
}
LABEL_CUTOFFS = {
    "CocitationCora": 3,
    "CocitationCiteseer": 2,
    "CoauthorshipCora": 3,
    "CoauthorshipDBLP": 3,
    "ModelNet40": 20,
    "NTU2012": 33,
}


def _dense_tensor(value: object) -> torch.Tensor:
    if sp.issparse(value):
        value = value.toarray()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _split_masks(labels: torch.Tensor, train_prop: float, valid_prop: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stratified train, validation, and test masks."""
    generator = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    valid_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    for class_id in torch.unique(labels).tolist():
        indices = torch.where(labels == class_id)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_count = max(1, int(indices.numel() * train_prop))
        valid_count = min(max(1, int(indices.numel() * valid_prop)), max(indices.numel() - train_count - 1, 0))
        train_mask[indices[:train_count]] = True
        valid_mask[indices[train_count:train_count + valid_count]] = True
        test_mask[indices[train_count + valid_count:]] = True
    return train_mask, valid_mask, test_mask


def _edge_list_to_index(edge_list: list[list[int]]) -> torch.Tensor:
    nodes, edges = [], []
    for edge_id, edge in enumerate(edge_list):
        unique_nodes = sorted(set(int(node) for node in edge))
        nodes.extend(unique_nodes)
        edges.extend([edge_id] * len(unique_nodes))
    return torch.tensor([nodes, edges], dtype=torch.long)


def _load_pickle_dataset(dataset_dir: Path) -> dict:
    with (dataset_dir / "features.pickle").open("rb") as handle:
        features = _dense_tensor(pickle.load(handle))
    with (dataset_dir / "labels.pickle").open("rb") as handle:
        labels = torch.as_tensor(np.asarray(pickle.load(handle)), dtype=torch.long).view(-1)
    with (dataset_dir / "hypergraph.pickle").open("rb") as handle:
        hypergraph = pickle.load(handle)
    return {"features": features, "labels": labels, "edge_list": [list(map(int, hypergraph[key])) for key in hypergraph]}


def _load_le_dataset(dataset_dir: Path, dataset: str) -> dict:
    content = np.genfromtxt(dataset_dir / f"{dataset}.content", dtype=str)
    node_ids = content[:, 0].astype(np.int64)
    node_map = {node_id: row for row, node_id in enumerate(node_ids)}
    features = torch.as_tensor(content[:, 1:-1].astype(np.float32))
    labels = torch.as_tensor(content[:, -1].astype(np.int64))
    grouped: dict[int, list[int]] = {}
    for node_id, edge_id in np.genfromtxt(dataset_dir / f"{dataset}.edges", dtype=np.int64):
        if int(node_id) in node_map:
            grouped.setdefault(int(edge_id), []).append(node_map[int(node_id)])
    return {"features": features, "labels": labels, "edge_list": [grouped[key] for key in sorted(grouped)]}


def load_base_dataset(dataset: str, data_root: Path, train_prop: float, valid_prop: float, seed: int) -> dict:
    """Load one supported dataset into a shared HNSD dictionary format."""
    if dataset in DATASET_PATHS:
        result = _load_pickle_dataset(data_root / DATASET_PATHS[dataset])
    elif dataset in {"ModelNet40", "NTU2012"}:
        result = _load_le_dataset(data_root / dataset, dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    train_mask, valid_mask, test_mask = _split_masks(result["labels"], train_prop, valid_prop, seed)
    result.update({
        "edge_index": _edge_list_to_index(result["edge_list"]),
        "num_nodes": result["labels"].numel(),
        "num_classes": int(result["labels"].max().item()) + 1,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "test_mask": test_mask,
        "node_idx": torch.arange(result["labels"].numel()),
    })
    return result


def _interpolate_features(features: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    first = torch.randperm(features.size(0), generator=generator)
    second = torch.randperm(features.size(0), generator=generator)
    weight = torch.rand((features.size(0), 1), generator=generator)
    return weight * features[first] + (1.0 - weight) * features[second]


def _rewire(edge_list: list[list[int]], ratio: float, seed: int) -> list[list[int]]:
    """Use degree-preserving incidence swaps while retaining hyperedge sizes."""
    generator = torch.Generator().manual_seed(seed)
    rewired = [list(sorted(set(edge))) for edge in edge_list]
    incidences = [(edge_id, node) for edge_id, edge in enumerate(rewired) for node in edge]
    for _ in range(int(round(len(incidences) * ratio / 2))):
        first, second = torch.randint(0, len(incidences), (2,), generator=generator).tolist()
        first_edge, first_node = incidences[first]
        second_edge, second_node = incidences[second]
        if first_edge == second_edge or first_node == second_node:
            continue
        if second_node in rewired[first_edge] or first_node in rewired[second_edge]:
            continue
        rewired[first_edge].remove(first_node)
        rewired[first_edge].append(second_node)
        rewired[second_edge].remove(second_node)
        rewired[second_edge].append(first_node)
        incidences[first] = first_edge, second_node
        incidences[second] = second_edge, first_node
    return [sorted(edge) for edge in rewired]


def _feature_ood(base: dict, seed: int) -> dict:
    result = copy.deepcopy(base)
    result["features"] = _interpolate_features(base["features"], seed)
    return result


def _structure_ood(base: dict, ratio: float, seed: int) -> dict:
    result = copy.deepcopy(base)
    result["edge_list"] = _rewire(base["edge_list"], ratio, seed)
    result["edge_index"] = _edge_list_to_index(result["edge_list"])
    return result


def make_task_datasets(args) -> tuple[dict, dict | None, dict | None]:
    """Build ID, OOD-exposure, and OOD-test datasets for the selected task."""
    base = load_base_dataset(args.dataset, args.data_root, args.train_prop, args.valid_prop, args.seed)
    if args.task == "misclassification":
        return base, None, None
    if args.ood_type == "feature":
        return base, _feature_ood(base, args.seed), _feature_ood(base, args.seed + 1)
    if args.ood_type == "structure":
        return base, _structure_ood(base, args.structure_ratio, args.seed), _structure_ood(base, args.structure_ratio, args.seed + 1)
    labels, indices, cutoff = base["labels"], torch.arange(base["num_nodes"]), LABEL_CUTOFFS[args.dataset]
    id_dataset = copy.deepcopy(base)
    id_mask = labels > cutoff
    id_dataset["node_idx"] = indices[id_mask]
    for name in ("train_mask", "valid_mask", "test_mask"):
        id_dataset[name] = id_dataset[name] & id_mask
    exposure = indices[labels == cutoff]
    generator = torch.Generator().manual_seed(args.seed + 17)
    exposure = exposure[torch.randperm(exposure.numel(), generator=generator)]
    split = max(1, int(0.8 * exposure.numel()))
    ood_train = copy.deepcopy(base)
    ood_train["node_idx"], ood_train["valid_node_idx"] = exposure[:split], exposure[split:]
    ood_test = copy.deepcopy(base)
    ood_test["node_idx"] = indices[labels < cutoff]
    return id_dataset, ood_train, ood_test
