import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from backbone.dinov2 import DINOv2Backbone
from approaches.finetuning.model import LoRAFinetuner


def extract_episode_data(cifar, sampler):
    """Sample episode and return images + labels."""
    support_idx, query_idx, classes = sampler.sample_episode()

    support_images = []
    support_labels = []
    for cls_id, indices in enumerate(support_idx):
        for i in indices:
            support_images.append(cifar[i][0])
            support_labels.append(cls_id)

    query_images = []
    query_labels = []
    for cls_id, indices in enumerate(query_idx):
        for i in indices:
            query_images.append(cifar[i][0])
            query_labels.append(cls_id)

    return (support_images, support_labels, query_images, query_labels,
            support_idx, classes)


def get_dinov2_features(backbone, images):
    """Extract raw DINOv2 features."""
    return backbone.extract_features(images).numpy()


def get_lora_features(lora_model, support_images_by_class, all_images):
    """Extract features after LoRA adaptation on support set."""
    lora_model._reset_lora()

    with torch.enable_grad():
        lora_model._finetune_on_support(support_images_by_class)

    lora_model.model.eval()
    with torch.no_grad():
        pixels = lora_model._process_images(all_images)
        features = lora_model._extract_cls(pixels)

    return features.numpy()


def plot_umap_comparison(cifar, seed=42):
    """Generate UMAP plots comparing feature spaces."""
    sampler = EpisodeSampler(cifar, n_way=5, k_shot=5, n_query=15, seed=seed)

    (support_images, support_labels, query_images, query_labels,
     support_idx, classes) = extract_episode_data(cifar, sampler)

    all_images = support_images + query_images
    all_labels = np.array(support_labels + query_labels)
    n_support = len(support_images)

    # Class names from CIFAR-100
    class_names = cifar.classes
    episode_class_names = [class_names[classes[i]] for i in range(5)]

    print(f"Episode classes: {episode_class_names}")

    # --- 1. Raw DINOv2 features ---
    print("Extracting DINOv2 features...")
    backbone = DINOv2Backbone(frozen=True)
    raw_features = get_dinov2_features(backbone, all_images)

    # --- 2. LoRA adapted features ---
    print("Extracting LoRA features...")
    lora_model = LoRAFinetuner(lora_rank=4, n_steps=20, lr=1e-3)

    support_images_by_class = []
    for cls_indices in support_idx:
        support_images_by_class.append([cifar[i][0] for i in cls_indices])

    lora_features = get_lora_features(lora_model, support_images_by_class, all_images)

    # --- UMAP projection ---
    print("Computing UMAP projections...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    raw_2d = reducer.fit_transform(raw_features)
    lora_2d = reducer.fit_transform(lora_features)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    titles = ['DINOv2 Raw Features (ProtoNets)', 'DINOv2 + LoRA Adapted']
    data_2d = [raw_2d, lora_2d]

    for ax, title, points_2d in zip(axes, titles, data_2d):
        for cls_id in range(5):
            mask = all_labels == cls_id

            # Support points — filled circles
            support_mask = mask.copy()
            support_mask[n_support:] = False
            if support_mask.any():
                ax.scatter(points_2d[support_mask, 0], points_2d[support_mask, 1],
                          c=colors[cls_id], marker='o', s=100, edgecolors='black',
                          linewidths=1.5, label=f'{episode_class_names[cls_id]} (support)',
                          zorder=3)

            # Query points — smaller, semi-transparent
            query_mask = mask.copy()
            query_mask[:n_support] = False
            if query_mask.any():
                ax.scatter(points_2d[query_mask, 0], points_2d[query_mask, 1],
                          c=colors[cls_id], marker='o', s=40, alpha=0.5,
                          zorder=2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(bbox_to_anchor=(0, -0.05), loc='upper left', ncol=3, fontsize=9)

    plt.suptitle('Feature Space Comparison — Same Episode, Same Classes',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/umap_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to results/umap_comparison.png")
    plt.close()


if __name__ == "__main__":
    from torchvision import datasets
    import os

    os.makedirs("results", exist_ok=True)

    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    plot_umap_comparison(cifar, seed=42)
    print("Done!")