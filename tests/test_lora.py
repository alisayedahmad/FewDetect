import torch
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.finetuning.model import LoRAFinetuner


def test_lora_episode():
    """LoRA fine-tuning should complete one episode and beat random."""
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=5, seed=42)

    model = LoRAFinetuner(lora_rank=4, n_steps=10, lr=1e-3)

    support_idx, query_idx, classes = sampler.sample_episode()

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

    accuracy = model.evaluate_episode(support_images, query_images, query_labels)

    print(f"LoRA episode accuracy: {accuracy*100:.1f}%")
    assert accuracy > 0.2, f"Should beat random, got {accuracy*100:.1f}%"


if __name__ == "__main__":
    test_lora_episode()
    print("Test passed!")