import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.prototypical.model import PrototypicalDetector
from approaches.finetuning.model import LoRAFinetuner


def evaluate_kshot(dataset, model_fn, k_shots, n_episodes=30, seed_offset=0):
    """Evaluate at different K values."""
    results = {}
    for k in k_shots:
        print(f"    K={k}...")
        accs = []
        model = model_fn()
        for ep in range(n_episodes):
            sampler = EpisodeSampler(dataset, n_way=5, k_shot=k,
                                     n_query=15, seed=ep + seed_offset)
            support_idx, query_idx, _ = sampler.sample_episode()

            support_images = [[dataset[i][0] for i in cls] for cls in support_idx]
            query_images = []
            query_labels = []
            for cls_id, cls_indices in enumerate(query_idx):
                for i in cls_indices:
                    query_images.append(dataset[i][0])
                    query_labels.append(cls_id)
            query_labels = torch.tensor(query_labels)

            if isinstance(model, PrototypicalDetector):
                acc, _ = model.evaluate_episode(support_images, query_images, query_labels)
            else:
                acc = model.evaluate_episode(support_images, query_images, query_labels)
            accs.append(acc)

        mean = np.mean(accs) * 100
        ci = 1.96 * np.std(accs) * 100 / np.sqrt(n_episodes)
        results[k] = {'mean': mean, 'ci': ci}
        print(f"      {mean:.1f}% ± {ci:.1f}%")
    return results


def main():
    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    k_shots = [1, 2, 5, 10, 20]
    n_episodes = 30

    # ProtoNets cosine
    print("\nProtoNets (cosine):")
    proto_results = evaluate_kshot(
        cifar,
        lambda: PrototypicalDetector(distance="cosine"),
        k_shots, n_episodes=n_episodes, seed_offset=2000
    )

    # ProtoNets euclidean
    print("\nProtoNets (euclidean):")
    proto_euc_results = evaluate_kshot(
        cifar,
        lambda: PrototypicalDetector(distance="euclidean"),
        k_shots, n_episodes=n_episodes, seed_offset=2000
    )

    # LoRA
    print("\nLoRA:")
    lora_results = evaluate_kshot(
        cifar,
        lambda: LoRAFinetuner(lora_rank=4, n_steps=20, lr=1e-3, distance="cosine"),
        k_shots, n_episodes=n_episodes, seed_offset=2000
    )

    # Plot
    import os
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, results, color, marker in [
        ('ProtoNets (cosine)', proto_results, '#2196F3', 'o'),
        ('ProtoNets (euclidean)', proto_euc_results, '#4CAF50', 's'),
        ('LoRA fine-tuning', lora_results, '#FF9800', '^'),
    ]:
        ks = sorted(results.keys())
        means = [results[k]['mean'] for k in ks]
        cis = [results[k]['ci'] for k in ks]

        ax.errorbar(ks, means, yerr=cis, label=label, color=color,
                    marker=marker, markersize=8, linewidth=2, capsize=5)

    ax.set_xlabel('K (shots per class)', fontsize=13)
    ax.set_ylabel('5-way Accuracy (%)', fontsize=13)
    ax.set_title('K-Shot Curves — CIFAR-100 Few-Shot Classification', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_shots)
    ax.set_ylim(60, 100)

    plt.tight_layout()
    plt.savefig('results/kshot_curves.png', dpi=150)
    print("\nSaved to results/kshot_curves.png")

    # Print table
    print(f"\n{'='*65}")
    print(f"  {'K':<5} {'Proto (cos)':<18} {'Proto (euc)':<18} {'LoRA':<18}")
    print(f"  {'-'*60}")
    for k in k_shots:
        pc = proto_results[k]
        pe = proto_euc_results[k]
        lo = lora_results[k]
        print(f"  {k:<5} {pc['mean']:>5.1f}% ± {pc['ci']:.1f}%"
              f"    {pe['mean']:>5.1f}% ± {pe['ci']:.1f}%"
              f"    {lo['mean']:>5.1f}% ± {lo['ci']:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()