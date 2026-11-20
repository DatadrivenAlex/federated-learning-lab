"""Logistic model primitives used by the federated-learning exercise."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def add_bias(features: FloatArray) -> FloatArray:
    return np.column_stack((features, np.ones(features.shape[0], dtype=np.float64)))


def sigmoid(logits: FloatArray) -> FloatArray:
    clipped = np.clip(logits, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_cross_entropy(features: FloatArray, labels: FloatArray, weights: FloatArray) -> float:
    """Mean logistic loss; `weights[-1]` is the intercept."""
    logits = add_bias(features) @ weights
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def gradient(
    features: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    *,
    l2: float = 1e-4,
    proximal_to: FloatArray | None = None,
    proximal_mu: float = 0.0,
) -> FloatArray:
    """Gradient of mean logistic loss, L2 penalty, and optional FedProx term."""
    design = add_bias(features)
    residual = sigmoid(design @ weights) - labels
    grad = (design.T @ residual) / labels.size
    grad[:-1] += l2 * weights[:-1]
    if proximal_to is not None and proximal_mu:
        grad += proximal_mu * (weights - proximal_to)
    return grad


def predict_probabilities(features: FloatArray, weights: FloatArray) -> FloatArray:
    return sigmoid(add_bias(features) @ weights)
