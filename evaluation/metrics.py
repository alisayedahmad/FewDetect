import torch
import numpy as np
from data.episode_sampler import EpisodeSampler
from approaches.prototypical.model import PrototypicalDetector


def evaluate_multi_episode(dataset, model, n_episodes=100, n_way=5, k_shot=5,
                           n_query=15, seed=42, verbose=True):
    """Evaluate a model over multiple episodes with confidence intervals.
    
    Args:
        dataset: torchvision dataset with (image, label) pairs
        model: model with evaluate_episode method
        n_episodes: number of episodes to run
        n_way: classes per episode
        k_shot: support examples per class
        n_query: query examples per class
        seed: random seed for reproducibility
        verbose: print progress
        
    Returns:
        dict with mean_accuracy, std, confidence_95, all_accuracies
    """
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot,
                             n_query=n_query, seed=seed)
    accuracies = []

    for ep in range(n_episodes):
        support_idx, query_idx, classes = sampler.sample_episode()

        # Build PIL image lists
        support_images = []
        for cls_indices in support_idx:
            support_images.append([dataset[i][0] for i in cls_indices])

        query_images = []
        query_labels = []
        for cls_id, cls_indices in enumerate(query_idx):
            for i in cls_indices:
                query_images.append(dataset[i][0])
                query_labels.append(cls_id)

        query_labels = torch.tensor(query_labels)

        accuracy, _ = model.evaluate_episode(
            support_images, query_images, query_labels
        )
        accuracies.append(accuracy)

        if verbose and (ep + 1) % 10 == 0:
            running_mean = np.mean(accuracies) * 100
            print(f"  Episode {ep + 1}/{n_episodes} — running mean: {running_mean:.1f}%")

    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    confidence_95 = 1.96 * std_acc / np.sqrt(n_episodes)

    results = {
        "mean_accuracy": mean_acc,
        "std": std_acc,
        "confidence_95": confidence_95,
        "n_episodes": n_episodes,
        "all_accuracies": accuracies,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Results: {mean_acc*100:.1f}% ± {confidence_95*100:.1f}%")
        print(f"  (mean ± 95% CI over {n_episodes} episodes)")
        print(f"  std: {std_acc*100:.1f}%")
        print(f"  min: {accuracies.min()*100:.1f}%  max: {accuracies.max()*100:.1f}%")
        print(f"{'='*50}")

    return results


if __name__ == "__main__":
    from torchvision import datasets

    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    print("Loading model...")
    model = PrototypicalDetector(distance="euclidean")

    print("\n5-way 5-shot evaluation:")
    results_5shot = evaluate_multi_episode(
        cifar, model, n_episodes=50, n_way=5, k_shot=5, n_query=15
    )

    print("\n5-way 1-shot evaluation:")
    results_1shot = evaluate_multi_episode(
        cifar, model, n_episodes=50, n_way=5, k_shot=1, n_query=15
    )