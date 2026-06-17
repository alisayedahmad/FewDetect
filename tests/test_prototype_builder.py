import torch
from approaches.prototypical.prototype_builder import PrototypeBuilder


def test_prototype_shape():
    builder = PrototypeBuilder(distance="euclidean")

    # 3 classes, 5 shots, dimension 384
    support = [torch.randn(5, 384) for _ in range(3)]
    prototypes = builder.build_prototypes(support)

    assert prototypes.shape == (3, 384), f"Expected (3, 384), got {prototypes.shape}"


def test_prototype_is_mean():
    builder = PrototypeBuilder(distance="euclidean")

    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prototypes = builder.build_prototypes([features])

    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(prototypes, expected), "Prototype should be the mean"


def test_classify_perfect():
    """Queries identical to prototypes should classify correctly."""
    builder = PrototypeBuilder(distance="euclidean")

    prototypes = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    queries = prototypes.clone()
    predictions, probs = builder.classify(queries, prototypes)

    assert predictions.tolist() == [0, 1, 2], f"Got {predictions.tolist()}"


def test_cosine_distance():
    builder = PrototypeBuilder(distance="cosine")

    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    queries = torch.tensor([[2.0, 0.0]])  # same direction as prototype 0

    predictions, _ = builder.classify(queries, prototypes)
    assert predictions.item() == 0, "Should match prototype 0"


def test_probabilities_sum_to_one():
    builder = PrototypeBuilder(distance="euclidean")

    prototypes = torch.randn(5, 384)
    queries = torch.randn(10, 384)

    _, probs = builder.classify(queries, prototypes)

    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(10), atol=1e-5), "Probs should sum to 1"


if __name__ == "__main__":
    test_prototype_shape()
    test_prototype_is_mean()
    test_classify_perfect()
    test_cosine_distance()
    test_probabilities_sum_to_one()
    print("Tous les tests passent!")