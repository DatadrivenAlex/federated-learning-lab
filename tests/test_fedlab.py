from fedlab.data import make_synthetic_federation
from fedlab.model import binary_cross_entropy
from fedlab.train import train_federated


def test_federation_is_reproducible_and_has_skewed_clients():
    first = make_synthetic_federation(seed=4, n_clients=8, samples_per_client=100)
    second = make_synthetic_federation(seed=4, n_clients=8, samples_per_client=100)
    assert len(first.clients) == 8
    assert (first.clients[0].features == second.clients[0].features).all()
    rates = [client.labels.mean() for client in first.clients]
    assert max(rates) - min(rates) > 0.2
    assert first.test_labels.mean() == 0.5


def test_fedavg_learns_and_records_estimated_communication():
    data = make_synthetic_federation(seed=11)
    result = train_federated(data, rounds=20, local_epochs=4)
    first = result.history[0]
    last = result.history[-1]
    assert last.test_loss < first.test_loss
    assert last.test_accuracy > 0.75
    assert len(result.history) == 20
    assert result.total_communication_bytes == sum(
        metric.estimated_communication_bytes for metric in result.history
    )


def test_fedprox_mu_zero_matches_fedavg():
    data = make_synthetic_federation(seed=12)
    avg = train_federated(data, rounds=5, local_epochs=2, algorithm="fedavg", seed=2)
    prox = train_federated(
        data, rounds=5, local_epochs=2, algorithm="fedprox", proximal_mu=0.0, seed=2
    )
    assert prox.weights == avg.weights


def test_partial_participation_is_reproducible():
    data = make_synthetic_federation(seed=5)
    one = train_federated(data, rounds=4, client_fraction=0.25, seed=9)
    two = train_federated(data, rounds=4, client_fraction=0.25, seed=9)
    assert one.weights == two.weights
    assert all(point.selected_clients == 2 for point in one.history)


def test_training_reduces_loss_from_zero_model():
    data = make_synthetic_federation(seed=7)
    result = train_federated(data, rounds=10)
    assert result.history[-1].test_loss < binary_cross_entropy(
        data.test_features,
        data.test_labels.astype(float),
        __import__("numpy").zeros(data.test_features.shape[1] + 1),
    )
