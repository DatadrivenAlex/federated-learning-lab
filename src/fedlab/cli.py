"""Command-line entry point for repeatable synthetic experiments."""

import argparse
import json
from pathlib import Path

from .data import make_synthetic_federation
from .train import train_federated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare federated optimizers on synthetic data.")
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.35)
    parser.add_argument("--proximal-mu", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = make_synthetic_federation(
        seed=args.seed,
        n_clients=args.clients,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    result = train_federated(
        dataset,
        algorithm=args.algorithm,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=args.client_fraction,
        proximal_mu=args.proximal_mu,
        seed=args.seed + 1,
    )
    payload = json.dumps(result.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
