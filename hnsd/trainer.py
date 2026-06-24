"""Training and evaluation loops for HNSD."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import make_task_datasets
from .metrics import binary_detection_metrics
from .model import HNSD
from .operators import IncidenceOperator


def set_seed(seed: int) -> None:
    """Configure repeatable CPU and CUDA random streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(data: dict, device: torch.device) -> dict:
    result = dict(data)
    for key in ("features", "labels", "train_mask", "valid_mask", "test_mask", "node_idx"):
        result[key] = data[key].to(device)
    if "valid_node_idx" in data:
        result["valid_node_idx"] = data["valid_node_idx"].to(device)
    result["operator"] = IncidenceOperator.from_edge_index(data["edge_index"], data["num_nodes"], device)
    return result


def _confidence(logits: torch.Tensor, score_type: str) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    if score_type == "msp":
        return probabilities.max(dim=-1).values
    if score_type == "energy":
        return torch.logsumexp(logits, dim=-1)
    return (probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


@torch.no_grad()
def _mean_logits(model: HNSD, data: dict, samples: int) -> torch.Tensor:
    return torch.stack([model(data["features"], data["operator"], stochastic=True) for _ in range(samples)], dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_ood(model: HNSD, id_data: dict, ood_data: dict, mask_name: str, samples: int, score_type: str) -> dict[str, float]:
    """Evaluate ID-versus-OOD detection with matched test or validation indices."""
    model.eval()
    id_logits, ood_logits = _mean_logits(model, id_data, samples), _mean_logits(model, ood_data, samples)
    id_indices = torch.where(id_data[mask_name])[0]
    if mask_name == "valid_mask" and "valid_node_idx" in ood_data:
        ood_indices = ood_data["valid_node_idx"]
    elif ood_data["node_idx"].numel() == id_data["num_nodes"]:
        ood_indices = id_indices
    else:
        ood_indices = ood_data["node_idx"]
    return binary_detection_metrics(
        _confidence(id_logits[id_indices], score_type).cpu().numpy(),
        _confidence(ood_logits[ood_indices], score_type).cpu().numpy(),
    )


@torch.no_grad()
def evaluate_misclassification(model: HNSD, data: dict, mask_name: str, samples: int, score_type: str) -> tuple[dict[str, float], float]:
    """Evaluate correct-versus-incorrect prediction detection."""
    model.eval()
    logits = _mean_logits(model, data, samples)
    indices = torch.where(data[mask_name])[0]
    correct = (logits.argmax(dim=-1)[indices] == data["labels"][indices]).long().cpu().numpy()
    scores = _confidence(logits[indices], score_type).cpu().numpy()
    metrics = binary_detection_metrics(scores[correct == 1], scores[correct == 0])
    return metrics, float(correct.mean())


def _uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probabilities, torch.full_like(log_probabilities, 1.0 / logits.size(-1)), reduction="batchmean")


def _energy_margin_loss(id_logits: torch.Tensor, ood_logits: torch.Tensor, margin_in: float, margin_out: float) -> torch.Tensor:
    """Separate ID and OOD free energies using a margin objective."""
    id_energy, ood_energy = -torch.logsumexp(id_logits, dim=-1), -torch.logsumexp(ood_logits, dim=-1)
    return F.relu(id_energy - margin_in).square().mean() + F.relu(margin_out - ood_energy).square().mean()


def run_experiment(args) -> dict:
    """Train HNSD and select the checkpoint with the best validation AUROC."""
    device = torch.device(args.device if torch.cuda.is_available() and not str(args.device).startswith("cpu") else "cpu")
    set_seed(args.seed)
    id_raw, ood_train_raw, ood_test_raw = make_task_datasets(args)
    id_data = _to_device(id_raw, device)
    ood_train = _to_device(ood_train_raw, device) if ood_train_raw is not None else None
    ood_test = _to_device(ood_test_raw, device) if ood_test_raw is not None else None
    model = HNSD(id_data["features"].size(-1), id_data["num_classes"], args.hidden_dim, args.steps, args.dt, args.dropout, args.input_dropout, args.noise_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        id_logits = model(id_data["features"], id_data["operator"], stochastic=True)
        loss = F.cross_entropy(id_logits[id_data["train_mask"]], id_data["labels"][id_data["train_mask"]])
        if args.task == "ood_detection":
            ood_logits = model(ood_train["features"], ood_train["operator"], stochastic=True)
            id_train_logits, ood_train_logits = id_logits[id_data["train_mask"]], ood_logits[ood_train["node_idx"]]
            regularizer = args.uniform_weight * _uniform_loss(ood_train_logits)
            regularizer += args.energy_weight * _energy_margin_loss(id_train_logits, ood_train_logits, args.energy_margin_in, args.energy_margin_out)
            loss = loss + args.ood_weight * regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        if args.task == "ood_detection":
            valid = evaluate_ood(model, id_data, ood_train, "valid_mask", args.eval_samples, args.ood_score)
            test = evaluate_ood(model, id_data, ood_test, "test_mask", args.eval_samples, args.ood_score)
            test_accuracy = float((id_logits[id_data["test_mask"]].argmax(dim=-1) == id_data["labels"][id_data["test_mask"]]).float().mean().item())
        else:
            valid, _ = evaluate_misclassification(model, id_data, "valid_mask", args.eval_samples, args.mis_score)
            test, test_accuracy = evaluate_misclassification(model, id_data, "test_mask", args.eval_samples, args.mis_score)
        record = {"epoch": epoch, "loss": float(loss.item()), "valid": valid, "test": test, "test_accuracy": test_accuracy}
        if np.isfinite(valid["auroc"]) and (best is None or valid["auroc"] > best["valid"]["auroc"]):
            best = record
            if args.save_checkpoint:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output_dir / f"{args.dataset}_{args.task}_{args.ood_type}_best.pt")
        print(f"epoch={epoch:04d} loss={record['loss']:.4f} val_auroc={100 * valid['auroc']:.2f} test_auroc={100 * test['auroc']:.2f}", flush=True)
    if best is None:
        raise RuntimeError("No finite validation AUROC was produced.")
    result = {"dataset": args.dataset, "task": args.task, "ood_type": args.ood_type if args.task == "ood_detection" else None, "seed": args.seed, "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "best": best}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.dataset}_{args.task}_{args.ood_type}_seed{args.seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

