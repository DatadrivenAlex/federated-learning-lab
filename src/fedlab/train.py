"""FedAvg and FedProx training loops with explicit communication accounting."""

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .data import FederatedDataset
from .model import binary_cross_entropy, gradient, predict_probabilities

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    test_loss: float
    test_accuracy: float
    selected_clients: int
    estimated_communication_bytes: int


@dataclass(frozen=True)
class TrainingResult:
    algorithm: str
    weights: tuple[float, ...]
    history: tuple[RoundMetrics, ...]
    total_communication_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "weights": list(self.weights),
            "history": [asdict(point) for point in self.history],
            "total_communication_bytes": self.total_communication_bytes,
        }


def _train_client(
    features: FloatArray,
    labels: FloatArray,
    initial: FloatArray,
    *,
    learning_rate: float,
    local_epochs: int,
    proximal_mu: float,
    l2: float,
) -> FloatArray:
    weights = initial.copy()
    for _ in range(local_epochs):
        weights -= learning_rate * gradient(
            features,
            labels,
            weights,
            l2=l2,
            proximal_to=initial,
            proximal_mu=proximal_mu,
        )
    return weights


def train_federated(
    dataset: FederatedDataset,
    *,
    algorithm: str = "fedavg",
    rounds: int = 30,
    local_epochs: int = 3,
    learning_rate: float = 0.18,
    client_fraction: float = 1.0,
    proximal_mu: float = 0.05,
    l2: float = 1e-4,
    seed: int = 19,
) -> TrainingResult:
    """Train a logistic model by sample-weighted FedAvg or FedProx.

    The simulator keeps client examples local and averages model parameters. It does
    not provide cryptographic secure aggregation or a formal privacy guarantee.
    """
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError("algorithm must be 'fedavg' or 'fedprox'")
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must be positive")
    if learning_rate <= 0 or not 0 < client_fraction <= 1 or proximal_mu < 0 or l2 < 0:
        raise ValueError("invalid learning rate, client fraction, proximal_mu, or l2")
    if len(dataset.clients) < 2:
        raise ValueError("at least two clients are required")

    n_features = dataset.clients[0].features.shape[1]
    if any(client.features.shape[1] != n_features for client in dataset.clients):
        raise ValueError("all client feature matrices must have equal width")
    rng = np.random.default_rng(seed)
    weights = np.zeros(n_features + 1, dtype=np.float64)
    selected_count = max(1, math.ceil(len(dataset.clients) * client_fraction))
    mu = proximal_mu if algorithm == "fedprox" else 0.0
    history: list[RoundMetrics] = []
    total_bytes = 0

    for round_number in range(1, rounds + 1):
        selected = rng.choice(len(dataset.clients), size=selected_count, replace=False)
        local_models: list[FloatArray] = []
        sample_counts: list[int] = []
        for index in selected:
            client = dataset.clients[int(index)]
            local = _train_client(
                client.features,
                client.labels.astype(np.float64),
                weights,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                proximal_mu=mu,
                l2=l2,
            )
            local_models.append(local)
            sample_counts.append(client.labels.size)

        count_array = np.asarray(sample_counts, dtype=np.float64)
        weights = np.average(np.stack(local_models), axis=0, weights=count_array)
        probabilities = predict_probabilities(dataset.test_features, weights)
        accuracy = float(np.mean((probabilities >= 0.5) == dataset.test_labels))
        loss = binary_cross_entropy(
            dataset.test_features, dataset.test_labels.astype(np.float64), weights
        )
        round_bytes = 2 * len(selected) * weights.nbytes
        total_bytes += round_bytes
        history.append(
            RoundMetrics(
                round=round_number,
                test_loss=loss,
                test_accuracy=accuracy,
                selected_clients=len(selected),
                estimated_communication_bytes=round_bytes,
            )
        )

    return TrainingResult(
        algorithm=algorithm,
        weights=tuple(float(value) for value in weights),
        history=tuple(history),
        total_communication_bytes=total_bytes,
    )
