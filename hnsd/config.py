"""Command-line configuration for HNSD experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = (
    "CocitationCora",
    "CocitationCiteseer",
    "CoauthorshipCora",
    "CoauthorshipDBLP",
    "ModelNet40",
    "NTU2012",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hypergraph Neural Stochastic Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=DATASETS, default="CoauthorshipDBLP")
    parser.add_argument("--task", choices=("ood_detection", "misclassification"), default="ood_detection")
    parser.add_argument("--ood-type", choices=("label", "feature", "structure"), default="label")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--input-dropout", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--ood-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--energy-margin-in", type=float, default=-5.0)
    parser.add_argument("--energy-margin-out", type=float, default=-1.0)
    parser.add_argument("--uniform-weight", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=5)
    parser.add_argument("--structure-ratio", type=float, default=0.50)
    parser.add_argument("--mis-score", choices=("msp", "energy", "entropy"), default="msp")
    parser.add_argument("--ood-score", choices=("msp", "energy", "entropy"), default="entropy")
    parser.add_argument("--train-prop", type=float, default=0.50)
    parser.add_argument("--valid-prop", type=float, default=0.25)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser
