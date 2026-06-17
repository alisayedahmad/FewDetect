from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.prototypical.model import PrototypicalDetector
import torch


def test_full_episode():
    """Run a complete 5-way 5-shot episode end to end."""
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=5, seed=42)
    model = PrototypicalDetector(distance="euclidean")

    support_idx, query_idx, classes = sampler.sample_episode()

    # Build PIL image lists
    support_images = []
    for cls_indices in support_idx:
        support_images.append([cifar[i][0] for i in cls_indices])

    query_images = []
    query_labels = []
    for cls_id, cls_indices in enumerate(query_idx):
        for i in cls_indices:
            query_images.append(cifar[i][0])
            query_labels.append(cls_id)

    query_labels = torch.tensor(query_labels)

    accuracy, predictions = model.evaluate_episode(
        support_images, query_images, query_labels
    )

    print(f"Accuracy: {accuracy * 100:.1f}%")
    assert accuracy > 0.2, f"Should beat random (20%), got {accuracy * 100:.1f}%"
    assert predictions.shape[0] == len(query_labels)


if __name__ == "__main__":
    test_full_episode()
    print("Test passed!")