import torch
import random
import numpy as np
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.maml.model import MAMLClassifier
from approaches.maml.meta_learner import MAMLMetaLearner


def build_episode_features(cifar, sampler, model):
    """Sample episode and pre-extract features."""
    support_idx, query_idx, classes = sampler.sample_episode()

    support_imgs = []
    support_labels = []
    for cls_id, indices in enumerate(support_idx):
        for i in indices:
            support_imgs.append(cifar[i][0])
            support_labels.append(cls_id)

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


def main():
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("Loading CIFAR-100...")
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    n_way = 5
    k_shot = 5
    n_query = 15
    n_train_episodes = 500
    n_eval_episodes = 50
    inner_steps = 10
    inner_lr = 0.1
    outer_lr = 0.001

    print(f"\nConfig: {n_way}-way {k_shot}-shot, {inner_steps} inner steps")
    print(f"Training: {n_train_episodes} episodes")
    print(f"Evaluation: {n_eval_episodes} episodes")

    # --- Meta-training ---
    print("\n=== META-TRAINING (FOMAML) ===")
    model = MAMLClassifier(n_way=n_way)
    learner = MAMLMetaLearner(
        model, inner_lr=inner_lr, outer_lr=outer_lr,
        inner_steps=inner_steps, first_order=True
    )

    train_losses = []
    train_accs = []

    for ep in range(n_train_episodes):
        sampler = EpisodeSampler(cifar, n_way=n_way, k_shot=k_shot,
                                 n_query=n_query, seed=ep + 1000)
        sf, sl, qf, ql = build_episode_features(cifar, sampler, model)
        loss, acc = learner.meta_train_episode(sf, sl, qf, ql)
        train_losses.append(loss)
        train_accs.append(acc)

        if (ep + 1) % 20 == 0:
            avg_loss = np.mean(train_losses[-20:])
            avg_acc = np.mean(train_accs[-20:]) * 100
            print(f"  Episode {ep+1}/{n_train_episodes} — "
                  f"loss: {avg_loss:.3f}, acc: {avg_acc:.1f}%")

    # --- Meta-evaluation (5-shot) ---
    print(f"\n=== EVALUATION: {n_way}-way {k_shot}-shot ===")
    eval_accs_5shot = []

    for ep in range(n_eval_episodes):
        sampler = EpisodeSampler(cifar, n_way=n_way, k_shot=k_shot,
                                 n_query=n_query, seed=ep + 5000)
        sf, sl, qf, ql = build_episode_features(cifar, sampler, model)
        acc = learner.meta_test_episode(sf, sl, qf, ql)
        eval_accs_5shot.append(acc)

        if (ep + 1) % 10 == 0:
            running = np.mean(eval_accs_5shot) * 100
            print(f"  Episode {ep+1}/{n_eval_episodes} — running mean: {running:.1f}%")

    mean_5 = np.mean(eval_accs_5shot) * 100
    ci_5 = 1.96 * np.std(eval_accs_5shot) * 100 / np.sqrt(n_eval_episodes)

    # --- Meta-evaluation (1-shot) ---
    print(f"\n=== EVALUATION: {n_way}-way 1-shot ===")
    eval_accs_1shot = []

    for ep in range(n_eval_episodes):
        sampler = EpisodeSampler(cifar, n_way=n_way, k_shot=1,
                                 n_query=n_query, seed=ep + 6000)
        sf, sl, qf, ql = build_episode_features(cifar, sampler, model)
        acc = learner.meta_test_episode(sf, sl, qf, ql)
        eval_accs_1shot.append(acc)

        if (ep + 1) % 10 == 0:
            running = np.mean(eval_accs_1shot) * 100
            print(f"  Episode {ep+1}/{n_eval_episodes} — running mean: {running:.1f}%")

    mean_1 = np.mean(eval_accs_1shot) * 100
    ci_1 = 1.96 * np.std(eval_accs_1shot) * 100 / np.sqrt(n_eval_episodes)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  FOMAML RESULTS ON CIFAR-100")
    print(f"{'='*60}")
    print(f"  5-way 5-shot: {mean_5:.1f}% ± {ci_5:.1f}%")
    print(f"  5-way 1-shot: {mean_1:.1f}% ± {ci_1:.1f}%")
    print(f"{'='*60}")
    print(f"\n  ProtoNets reference (same dataset):")
    print(f"  5-way 5-shot: 94.3% ± 1.4%  (euclidean)")
    print(f"  5-way 1-shot: 81.5% ± 2.8%  (euclidean)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()