import torch
import numpy as np
import time
from data.coco_fewshot import COCOFewShot
from data.episode_sampler import EpisodeSampler
from approaches.prototypical.model import PrototypicalDetector
from approaches.finetuning.model import LoRAFinetuner


def evaluate_approach(dataset, model_fn, n_episodes=50, k_shot=5, seed_offset=0):
    """Evaluate an approach over multiple episodes and return array of accuracies"""
    accs = []
    model = model_fn()

    for ep in range(n_episodes):
        sampler = EpisodeSampler(dataset, n_way=5, k_shot=k_shot,
                                 n_query=15, seed=ep + seed_offset)
        support_idx, query_idx, classes = sampler.sample_episode()

        support_images = [[dataset[i][0] for i in cls] for cls in support_idx]

        query_images = []
        query_labels = []
        for cls_id, cls_indices in enumerate(query_idx):
            for i in cls_indices:
                query_images.append(dataset[i][0])
                query_labels.append(cls_id)

        query_labels = torch.tensor(query_labels)

        if hasattr(model, 'evaluate_episode'):
            if isinstance(model, PrototypicalDetector):
                acc, _ = model.evaluate_episode(
                    support_images, query_images, query_labels
                )
            else:
                acc = model.evaluate_episode(
                    support_images, query_images, query_labels
                )
        accs.append(acc)

        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1}/{n_episodes} — running mean: {np.mean(accs)*100:.1f}%")

    return np.array(accs)


def main():
    print("=" * 60)
    print("  FewDetect — COCO Novel Classes Evaluation")
    print("=" * 60)

    dataset = COCOFewShot(split="val2017", class_set="novel")
    print(f"\nNovel classes: {dataset.selected_names}")
    n_episodes = 50

    results = {}

    # --- ProtoNets ---
    print("\n[1/2] Prototypical Networks (cosine)...")
    for k in [5, 1]:
        print(f"\n  --- {k}-shot ---")
        t0 = time.time()
        accs = evaluate_approach(
            dataset,
            lambda: PrototypicalDetector(distance="cosine"),
            n_episodes=n_episodes, k_shot=k, seed_offset=3000
        )
        t = time.time() - t0
        mean = np.mean(accs) * 100
        ci = 1.96 * np.std(accs) * 100 / np.sqrt(n_episodes)
        print(f"  Result: {mean:.1f}% ± {ci:.1f}% ({t:.0f}s)")
        results[f"proto_{k}shot"] = {"mean": mean, "ci": ci}

    # --- LoRA ---
    print("\n[2/2] LoRA Fine-tuning...")
    for k in [5, 1]:
        print(f"\n  --- {k}-shot ---")
        t0 = time.time()
        accs = evaluate_approach(
            dataset,
            lambda: LoRAFinetuner(lora_rank=4, n_steps=20, lr=1e-3, distance="cosine"),
            n_episodes=n_episodes, k_shot=k, seed_offset=3000
        )
        t = time.time() - t0
        mean = np.mean(accs) * 100
        ci = 1.96 * np.std(accs) * 100 / np.sqrt(n_episodes)
        print(f"  Result: {mean:.1f}% ± {ci:.1f}% ({t:.0f}s)")
        results[f"lora_{k}shot"] = {"mean": mean, "ci": ci}

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  COCO NOVEL CLASSES — 5-way classification")
    print(f"{'='*60}")
    print(f"  {'Approach':<25} {'5-shot':>15} {'1-shot':>15}")
    print(f"  {'-'*55}")
    print(f"  {'ProtoNets (cosine)':<25} {results['proto_5shot']['mean']:>6.1f}% ± {results['proto_5shot']['ci']:.1f}%"
          f"  {results['proto_1shot']['mean']:>6.1f}% ± {results['proto_1shot']['ci']:.1f}%")
    print(f"  {'LoRA':<25} {results['lora_5shot']['mean']:>6.1f}% ± {results['lora_5shot']['ci']:.1f}%"
          f"  {results['lora_1shot']['mean']:>6.1f}% ± {results['lora_1shot']['ci']:.1f}%")
    print(f"{'='*60}")
    print(f"\n  CIFAR-100 reference:")
    print(f"  ProtoNets: 5-shot 95.5%  1-shot 83.3%")
    print(f"  LoRA:      5-shot 93.1%  1-shot 82.5%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()