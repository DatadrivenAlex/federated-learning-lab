"""Synthetic client data with controllable label and feature heterogeneity."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ClientDataset:
    """One simulated participant's local binary-classification data."""

    features: FloatArray
    labels: IntArray


@dataclass(frozen=True)
class FederatedDataset:
    """Client-local training partitions and one shared, balanced test set."""

    clients: tuple[ClientDataset, ...]
    test_features: FloatArray
    test_labels: IntArray


def make_synthetic_federation(
    *,
    seed: int = 7,
    n_clients: int = 8,
    samples_per_client: int = 160,
    n_features: int = 8,
    dirichlet_alpha: float = 0.35,
    feature_shift: float = 0.35,
    test_samples: int = 1200,
) -> FederatedDataset:
    """Create synthetic binary data with Dirichlet-skewed client label rates.

    Each client gets a different class prior and a small feature-domain shift. No real
    records, external datasets, model checkpoints, or network calls are used.
    """
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if samples_per_client < 4 or test_samples < 4:
        raise ValueError("sample counts must be at least 4")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if dirichlet_alpha <= 0 or feature_shift < 0:
        raise ValueError("dirichlet_alpha must be positive and feature_shift non-negative")

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    clients: list[ClientDataset] = []

    for _ in range(n_clients):
        positive_rate = float(rng.dirichlet([dirichlet_alpha, dirichlet_alpha])[1])
        positive_count = round(samples_per_client * positive_rate)
        positive_count = min(samples_per_client - 1, max(1, positive_count))
        labels = np.concatenate(
            [np.zeros(samples_per_client - positive_count, dtype=np.int64),
             np.ones(positive_count, dtype=np.int64)]
        )
        rng.shuffle(labels)
        domain_offset = rng.normal(0.0, feature_shift, size=n_features)
        signed_signal = (2.0 * labels.astype(np.float64) - 1.0)[:, None]
        features = rng.normal(size=(samples_per_client, n_features)) + 1.15 * signed_signal * direction
        features += domain_offset
        clients.append(ClientDataset(features.astype(np.float64), labels))

    test_labels = np.tile(np.array([0, 1], dtype=np.int64), (test_samples + 1) // 2)[:test_samples]
    rng.shuffle(test_labels)
    test_features = rng.normal(size=(test_samples, n_features))
    test_features += 1.15 * (2.0 * test_labels.astype(np.float64) - 1.0)[:, None] * direction
    return FederatedDataset(
        clients=tuple(clients),
        test_features=test_features.astype(np.float64),
        test_labels=test_labels,
    )
