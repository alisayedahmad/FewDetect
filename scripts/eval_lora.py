import torch
import numpy as np
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.finetuning.model import LoRAFinetuner


def main():
    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    n_episodes = 50
    model = LoRAFinetuner(lora_rank=4, n_steps=20, lr=1e-3, distance="cosine")

    # --- 5-shot ---
    print(f"\n=== 5-way 5-shot ===")
    accs_5 = []
    for ep in range(n_episodes):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=ep + 7000)
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
        acc = model.evaluate_episode(support_images, query_images, query_labels)
        accs_5.append(acc)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} — running mean: {np.mean(accs_5)*100:.1f}%")

    mean_5 = np.mean(accs_5) * 100
    ci_5 = 1.96 * np.std(accs_5) * 100 / np.sqrt(n_episodes)

    # --- 1-shot ---
    print(f"\n=== 5-way 1-shot ===")
    accs_1 = []
    for ep in range(n_episodes):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=1, n_query=15, seed=ep + 8000)
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
        acc = model.evaluate_episode(support_images, query_images, query_labels)
        accs_1.append(acc)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} — running mean: {np.mean(accs_1)*100:.1f}%")

    mean_1 = np.mean(accs_1) * 100
    ci_1 = 1.96 * np.std(accs_1) * 100 / np.sqrt(n_episodes)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  LoRA FINE-TUNING RESULTS ON CIFAR-100")
    print(f"{'='*60}")
    print(f"  5-way 5-shot: {mean_5:.1f}% ± {ci_5:.1f}%")
    print(f"  5-way 1-shot: {mean_1:.1f}% ± {ci_1:.1f}%")
    print(f"{'='*60}")
    print(f"\n  Comparison:")
    print(f"  ProtoNets (cosine):  5-shot 95.5%   1-shot 83.3%")
    print(f"  FOMAML:             5-shot 89.4%   1-shot 71.8%")
    print(f"  LoRA:               5-shot {mean_5:.1f}%   1-shot {mean_1:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()