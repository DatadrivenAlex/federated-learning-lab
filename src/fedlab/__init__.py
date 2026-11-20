"""Small, reproducible federated-learning experiments."""

from .data import ClientDataset, FederatedDataset, make_synthetic_federation
from .train import RoundMetrics, TrainingResult, train_federated

__all__ = [
    "ClientDataset",
    "FederatedDataset",
    "RoundMetrics",
    "TrainingResult",
    "make_synthetic_federation",
    "train_federated",
]
