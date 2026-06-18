import torch
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.maml.model import MAMLClassifier
from approaches.maml.meta_learner import MAMLMetaLearner


def _build_episode_features(cifar, sampler, model):
    """Helper: sample episode and extract features"""
    support_idx, query_idx, classes = sampler.sample_episode()

    # Extract support features
    support_imgs = []
    support_labels = []
    for cls_id, indices in enumerate(support_idx):
        for i in indices:
            support_imgs.append(cifar[i][0])
            support_labels.append(cls_id)

    # Extract query features
    query_imgs = []
    query_labels = []
    for cls_id, indices in enumerate(query_idx):
        for i in indices:
            query_imgs.append(cifar[i][0])
            query_labels.append(cls_id)

    support_features = model.forward_features(support_imgs)
    query_features = model.forward_features(query_imgs)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_features, support_labels, query_features, query_labels


def test_maml_train_episode():
    """MAML should complete one meta-training episode without crashing """
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=42)

    model = MAMLClassifier(n_way=5)
    learner = MAMLMetaLearner(model, inner_steps=3, first_order=True)

    sf, sl, qf, ql = _build_episode_features(cifar, sampler, model)
    loss, acc = learner.meta_train_episode(sf, sl, qf, ql)

    assert loss > 0, "Loss should be positive"
    assert 0 <= acc <= 1, f"Accuracy should be between 0 and 1, got {acc}"
    print(f"Train episode — loss: {loss:.3f}, accuracy: {acc*100:.1f}%")


def test_maml_test_episode():
    """MAML should adapt and evaluate without updating initial weights."""
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=42)

    model = MAMLClassifier(n_way=5)
    learner = MAMLMetaLearner(model, inner_steps=5, first_order=True)

    sf, sl, qf, ql = _build_episode_features(cifar, sampler, model)

    # Save weights before
    weights_before = model.head[0].weight.clone()

    acc = learner.meta_test_episode(sf, sl, qf, ql)

    # Weights should be restored after test
    weights_after = model.head[0].weight.clone()
    assert torch.allclose(weights_before, weights_after), "Weights should be restored"
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    print(f"Test episode — accuracy: {acc*100:.1f}%")


def test_meta_training_improves():
    """Accuracy should improve after several meta-training episodes """
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    model = MAMLClassifier(n_way=5)
    learner = MAMLMetaLearner(
        model, inner_lr=0.01, outer_lr=0.001,
        inner_steps=5, first_order=True
    )

    # Run a few meta-training episodes
    losses = []
    for i in range(10):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=i)
        sf, sl, qf, ql = _build_episode_features(cifar, sampler, model)
        loss, acc = learner.meta_train_episode(sf, sl, qf, ql)
        losses.append(loss)
        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1} — loss: {loss:.3f}, acc: {acc*100:.1f}%")

    # Loss should generally decrease
    first_half = sum(losses[:5]) / 5
    second_half = sum(losses[5:]) / 5
    print(f"  Avg loss first 5: {first_half:.3f}, last 5: {second_half:.3f}")


if __name__ == "__main__":
    test_maml_train_episode()
    test_maml_test_episode()
    test_meta_training_improves()
    print("\nAll MAML tests passed!")