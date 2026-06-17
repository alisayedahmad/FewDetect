from torchvision import datasets
from approaches.prototypical.model import PrototypicalDetector
from evaluation.metrics import evaluate_multi_episode


def main():
    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    n_episodes = 50

    for distance in ["euclidean", "cosine"]:
        print(f"\n{'='*60}")
        print(f"  Distance: {distance.upper()}")
        print(f"{'='*60}")

        model = PrototypicalDetector(distance=distance)

        print(f"\n--- 5-way 1-shot ({distance}) ---")
        r1 = evaluate_multi_episode(
            cifar, model, n_episodes=n_episodes,
            n_way=5, k_shot=1, n_query=15, seed=42
        )

        print(f"\n--- 5-way 5-shot ({distance}) ---")
        r5 = evaluate_multi_episode(
            cifar, model, n_episodes=n_episodes,
            n_way=5, k_shot=5, n_query=15, seed=42
        )

    print(f"\n{'='*60}")
    print("Done. Compare the results above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()