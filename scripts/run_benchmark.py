import json
import time
import torch
import numpy as np
from datetime import datetime
from torchvision import datasets
from data.episode_sampler import EpisodeSampler
from approaches.prototypical.model import PrototypicalDetector
from approaches.maml.model import MAMLClassifier
from approaches.maml.meta_learner import MAMLMetaLearner
from approaches.finetuning.model import LoRAFinetuner


def evaluate_prototypical(cifar, n_episodes, k_shot, seed_offset):
    """Evaluate ProtoNets (cosine distance) """
    model = PrototypicalDetector(distance="cosine")
    accs = []

    for ep in range(n_episodes):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=k_shot,
                                 n_query=15, seed=ep + seed_offset)
        support_idx, query_idx, _ = sampler.sample_episode()

        support_images = [[cifar[i][0] for i in cls] for cls in support_idx]
        query_images = []
        query_labels = []
        for cls_id, cls_indices in enumerate(query_idx):
            for i in cls_indices:
                query_images.append(cifar[i][0])
                query_labels.append(cls_id)

        acc, _ = model.evaluate_episode(
            support_images, query_images, torch.tensor(query_labels)
        )
        accs.append(acc)

    return np.array(accs)


def evaluate_maml(cifar, n_episodes, k_shot, seed_offset):
    """Train MAML then evaluate on episodes. Returns array of accuracies"""
    model = MAMLClassifier(n_way=5)
    learner = MAMLMetaLearner(
        model, inner_lr=0.1, outer_lr=0.001,
        inner_steps=10, first_order=True
    )

    # Meta-train
    for ep in range(500):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=5,
                                 n_query=15, seed=ep + 1000)
        support_idx, query_idx, _ = sampler.sample_episode()

        s_imgs, s_labels = [], []
        for cls_id, indices in enumerate(support_idx):
            for i in indices:
                s_imgs.append(cifar[i][0])
                s_labels.append(cls_id)

        q_imgs, q_labels = [], []
        for cls_id, indices in enumerate(query_idx):
            for i in indices:
                q_imgs.append(cifar[i][0])
                q_labels.append(cls_id)

        sf = model.forward_features(s_imgs)
        qf = model.forward_features(q_imgs)
        learner.meta_train_episode(sf, torch.tensor(s_labels),
                                   qf, torch.tensor(q_labels))

    # Evaluate
    accs = []
    for ep in range(n_episodes):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=k_shot,
                                 n_query=15, seed=ep + seed_offset)
        support_idx, query_idx, _ = sampler.sample_episode()

        s_imgs, s_labels = [], []
        for cls_id, indices in enumerate(support_idx):
            for i in indices:
                s_imgs.append(cifar[i][0])
                s_labels.append(cls_id)

        q_imgs, q_labels = [], []
        for cls_id, indices in enumerate(query_idx):
            for i in indices:
                q_imgs.append(cifar[i][0])
                q_labels.append(cls_id)

        sf = model.forward_features(s_imgs)
        qf = model.forward_features(q_imgs)
        acc = learner.meta_test_episode(sf, torch.tensor(s_labels),
                                        qf, torch.tensor(q_labels))
        accs.append(acc)

    return np.array(accs)


def evaluate_lora(cifar, n_episodes, k_shot, seed_offset):
    """Evaluate LoRA fine-tuning on episodes and return array of accuracies"""
    model = LoRAFinetuner(lora_rank=4, n_steps=20, lr=1e-3, distance="cosine")
    accs = []

    for ep in range(n_episodes):
        sampler = EpisodeSampler(cifar, n_way=5, k_shot=k_shot,
                                 n_query=15, seed=ep + seed_offset)
        support_idx, query_idx, _ = sampler.sample_episode()

        support_images = [[cifar[i][0] for i in cls] for cls in support_idx]
        query_images = []
        query_labels = []
        for cls_id, cls_indices in enumerate(query_idx):
            for i in cls_indices:
                query_images.append(cifar[i][0])
                query_labels.append(cls_id)

        acc = model.evaluate_episode(
            support_images, query_images, torch.tensor(query_labels)
        )
        accs.append(acc)

    return np.array(accs)


def format_result(accs):
    """Format accuracy array into dict with stats"""
    return {
        "mean": round(float(np.mean(accs) * 100), 1),
        "std": round(float(np.std(accs) * 100), 1),
        "ci_95": round(float(1.96 * np.std(accs) * 100 / np.sqrt(len(accs))), 1),
        "min": round(float(np.min(accs) * 100), 1),
        "max": round(float(np.max(accs) * 100), 1),
        "n_episodes": len(accs),
    }


def main():
    print("=" * 60)
    print("  FewDetect Benchmark — CIFAR-100")
    print("=" * 60)

    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    n_episodes = 50
    results = {"timestamp": datetime.now().isoformat(), "dataset": "CIFAR-100"}

    # --- ProtoNets ---
    print("\n[1/3] Prototypical Networks (cosine)...")
    t0 = time.time()
    proto_5 = evaluate_prototypical(cifar, n_episodes, k_shot=5, seed_offset=9000)
    proto_1 = evaluate_prototypical(cifar, n_episodes, k_shot=1, seed_offset=9500)
    proto_time = time.time() - t0
    print(f"  5-shot: {np.mean(proto_5)*100:.1f}% ± {1.96*np.std(proto_5)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  1-shot: {np.mean(proto_1)*100:.1f}% ± {1.96*np.std(proto_1)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  Time: {proto_time:.0f}s")

    results["prototypical_networks"] = {
        "distance": "cosine",
        "5_shot": format_result(proto_5),
        "1_shot": format_result(proto_1),
        "time_seconds": round(proto_time),
    }

    # --- MAML ---
    print("\n[2/3] FOMAML (500 meta-train episodes)...")
    t0 = time.time()
    maml_5 = evaluate_maml(cifar, n_episodes, k_shot=5, seed_offset=9000)
    maml_1 = evaluate_maml(cifar, n_episodes, k_shot=1, seed_offset=9500)
    maml_time = time.time() - t0
    print(f"  5-shot: {np.mean(maml_5)*100:.1f}% ± {1.96*np.std(maml_5)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  1-shot: {np.mean(maml_1)*100:.1f}% ± {1.96*np.std(maml_1)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  Time: {maml_time:.0f}s")

    results["fomaml"] = {
        "inner_lr": 0.1,
        "inner_steps": 10,
        "meta_train_episodes": 500,
        "5_shot": format_result(maml_5),
        "1_shot": format_result(maml_1),
        "time_seconds": round(maml_time),
    }

    # --- LoRA ---
    print("\n[3/3] LoRA Fine-tuning...")
    t0 = time.time()
    lora_5 = evaluate_lora(cifar, n_episodes, k_shot=5, seed_offset=9000)
    lora_1 = evaluate_lora(cifar, n_episodes, k_shot=1, seed_offset=9500)
    lora_time = time.time() - t0
    print(f"  5-shot: {np.mean(lora_5)*100:.1f}% ± {1.96*np.std(lora_5)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  1-shot: {np.mean(lora_1)*100:.1f}% ± {1.96*np.std(lora_1)*100/np.sqrt(n_episodes):.1f}%")
    print(f"  Time: {lora_time:.0f}s")

    results["lora_finetuning"] = {
        "lora_rank": 4,
        "n_steps": 20,
        "lr": 0.001,
        "5_shot": format_result(lora_5),
        "1_shot": format_result(lora_1),
        "time_seconds": round(lora_time),
    }

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — 5-way classification on CIFAR-100")
    print(f"{'='*60}")
    print(f"  {'Approach':<25} {'5-shot':>15} {'1-shot':>15}")
    print(f"  {'-'*55}")
    for name, key in [("ProtoNets (cosine)", "prototypical_networks"),
                      ("FOMAML", "fomaml"),
                      ("LoRA", "lora_finetuning")]:
        r5 = results[key]["5_shot"]
        r1 = results[key]["1_shot"]
        print(f"  {name:<25} {r5['mean']:>6.1f}% ± {r5['ci_95']:.1f}%"
              f"  {r1['mean']:>6.1f}% ± {r1['ci_95']:.1f}%")
    print(f"{'='*60}")

    # Save JSON
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to results/benchmark_report.json")


if __name__ == "__main__":
    main()