from torchvision import datasets
from data.episode_sampler import EpisodeSampler


def test_episode_shape():
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=42)

    support, query, classes = sampler.sample_episode()

    assert len(classes) == 5, "Doit avoir 5 classes"
    assert len(support) == 5, "5 groupes support"
    assert len(query) == 5, "5 groupes query"

    for s in support:
        assert len(s) == 5, "5 shots par classe"
    for q in query:
        assert len(q) == 15, "15 queries par classe"


def test_no_overlap():
    """Support et query ne partagent aucun indice."""
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=42)

    support, query, _ = sampler.sample_episode()

    for s, q in zip(support, query):
        overlap = set(s) & set(q)
        assert len(overlap) == 0, f"Overlap détecté: {overlap}"


def test_reproducible():
    """Même seed = même épisode."""
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    sampler1 = EpisodeSampler(cifar, seed=42)
    sampler2 = EpisodeSampler(cifar, seed=42)

    _, _, classes1 = sampler1.sample_episode()
    _, _, classes2 = sampler2.sample_episode()

    assert classes1 == classes2, "Même seed doit donner même épisode"


if __name__ == "__main__":
    test_episode_shape()
    test_no_overlap()
    test_reproducible()
    print("Tous les tests passent!")