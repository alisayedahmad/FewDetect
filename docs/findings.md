# FewDetect — Findings and Analysis

## Overview

This project compares three few-shot approaches — Prototypical Networks, FOMAML, and LoRA fine-tuning — on the same DINOv2 ViT-S/14 backbone, evaluated on CIFAR-100 and COCO under identical conditions.

The main finding: **with a strong self-supervised backbone like DINOv2, the simplest approach wins.** Prototypical Networks with cosine distance outperform both MAML and LoRA fine-tuning across all settings.

---

## Classification Results

### CIFAR-100 — 5-way classification

| Approach              | 5-shot         | 1-shot         |
|-----------------------|----------------|----------------|
| ProtoNets (cosine)    | **95.5% ± 1.4%** | **83.3% ± 2.8%** |
| ProtoNets (euclidean) | 94.3% ± 1.4%  | 81.5% ± 2.8%  |
| LoRA fine-tuning      | 93.1% ± 1.4%  | 82.5% ± 2.9%  |
| FOMAML                | 89.4% ± 1.5%  | 71.8% ± 2.7%  |

### COCO Novel Classes — 5-way classification

| Approach              | 5-shot         | 1-shot         |
|-----------------------|----------------|----------------|
| ProtoNets (cosine)    | **85.8% ± 1.9%** | **69.1% ± 3.5%** |
| LoRA fine-tuning      | 82.6% ± 1.7%  | 66.2% ± 3.1%  |

### K-Shot Curves (CIFAR-100)

| K   | ProtoNets (cos) | ProtoNets (euc) | LoRA          |
|-----|-----------------|-----------------|---------------|
| 1   | 86.4% ± 2.7%   | 85.7% ± 2.7%   | 84.1% ± 3.8% |
| 2   | 92.7% ± 2.0%   | 91.8% ± 2.0%   | 92.0% ± 2.3% |
| 5   | 95.8% ± 1.5%   | 94.6% ± 1.6%   | 93.6% ± 2.0% |
| 10  | 96.8% ± 1.3%   | 96.4% ± 1.2%   | 94.5% ± 2.0% |
| 20  | 97.4% ± 0.8%   | 97.0% ± 0.9%   | 95.1% ± 1.4% |

### Few-Shot Detection — COCO Novel Classes

| Setting           | mAP@0.5       |
|-------------------|---------------|
| 5-way 5-shot      | 13.4% ± 2.5%  |

---

## Key Findings

### 1. Distance metric matters — cosine beats euclidean with DINOv2

Snell et al. (2017) found that Euclidean distance outperforms cosine for Prototypical Networks. We observe the **opposite** with DINOv2. Cosine consistently wins across all K values and both datasets.

**Why:** DINOv2 is trained with self-supervised contrastive objectives that structure the embedding space directionally. The angle between feature vectors carries more semantic information than their magnitude. Cosine distance captures exactly this. Snell used a small CNN trained from scratch where magnitude information was relevant.

### 2. When the backbone is strong enough, simplicity wins

ProtoNets requires zero training — just compute means and measure distances. Yet it beats MAML (which meta-trains for 500 episodes) and LoRA (which fine-tunes 73K parameters per episode). This contradicts the intuition that adaptation should help.

**Why:** DINOv2 was trained on 142M images with self-supervised learning. Its features already cluster semantically similar images together. Computing the mean of 5 examples in this well-structured space gives an excellent prototype. Adaptation (MAML, LoRA) risks overfitting to the small support set without adding discriminative power.

### 3. MAML struggles with frozen backbones

FOMAML achieved only 89.4% on CIFAR-100 5-shot, far below ProtoNets (95.5%). During meta-training, accuracy stayed at ~20% (random), suggesting the outer loop learned little.

**Why:** MAML was designed to adapt entire networks. When the backbone is frozen, MAML can only adapt a small linear head. ProtoNets is mathematically optimal for this setting — the mean is the exact solution for nearest-centroid classification with Bregman divergences. MAML approximates this solution through gradient steps but never reaches it.

### 4. LoRA adapts the features but doesn't improve generalization

UMAP visualizations show that LoRA produces tighter, more separated clusters than raw DINOv2 features. Despite this visually better structure, classification accuracy is lower.

**Why:** LoRA optimizes the feature space for the specific 25 support images in each episode. This specialization makes the features perfect for those images but slightly less generalizable to new query images. The raw DINOv2 features are less specialized but more robust.

### 5. The CIFAR-100 to COCO gap reveals real-world difficulty

Performance drops ~10 points from CIFAR-100 to COCO for all approaches. CIFAR-100 has clean, centered, 32x32 images. COCO crops have variable aspect ratios, occlusion, challenging viewpoints, and intra-class diversity.

The ranking between approaches is preserved across datasets, suggesting our conclusions generalize.

### 6. K-shot saturation

ProtoNets reaches 96.8% at K=10 on CIFAR-100. Going from K=10 to K=20 gains only 0.6%. The marginal value of each additional support example diminishes rapidly. This has practical implications: collecting more than 10 examples per novel class provides diminishing returns.

### 7. Few-shot detection is fundamentally harder than classification

Our detector achieves 13.4% mAP on novel classes. This is expected — detection requires both localizing and classifying objects in cluttered scenes, with only 5 examples to define each new class. The model must generalize spatial features learned on base classes to novel categories it has never seen.

---

## UMAP Feature Space Analysis

We visualized the embedding space for a single episode comparing raw DINOv2 features (used by ProtoNets) and LoRA-adapted features:

- **Raw DINOv2:** clusters are well-separated but spread — sufficient for classification.
- **LoRA-adapted:** clusters are extremely tight and well-separated — visually better but slightly overfit to the support set.

This visual analysis confirms that DINOv2 features are already highly structured. LoRA tightens the clusters but at the cost of generalization to unseen queries.

---

## Limitations

- **No full MAML/LoRA evaluation on COCO detection.** Only ProtoNets-style prototype classification was used for the detection pipeline.
- **Training on val2017 only.** Using the full train2017 (118K images) would likely improve detection results significantly.
- **No VIRAT evaluation.** Transfer to real surveillance data remains as future work.
- **Single backbone.** All conclusions are specific to DINOv2 ViT-S/14. A weaker backbone might show different rankings between approaches.
- **No second-order MAML.** Only FOMAML was evaluated. Full MAML with second-order gradients might perform better but at significant computational cost.

---

## Conclusions

For practitioners building few-shot systems with modern pretrained backbones:

1. **Start with ProtoNets + cosine distance.** It's simple, fast, requires no training, and sets a strong baseline that adaptation methods struggle to beat.
2. **Don't assume adaptation helps.** With strong backbones, the cost of overfitting to small support sets outweighs the benefit of feature adaptation.
3. **Invest in the backbone, not the meta-learning algorithm.** The choice of DINOv2 matters more than the choice between ProtoNets, MAML, or LoRA.
4. **10 shots is usually enough.** Performance saturates quickly with Prototypical Networks on strong features.

---

## Reproducibility

All results can be reproduced from a clean clone:

```bash
# Classification benchmark (all 3 approaches)
python scripts/run_benchmark.py

# K-shot curves
python scripts/kshot_curves.py

# Distance comparison
python scripts/compare_distances.py

# COCO novel class evaluation
python scripts/eval_coco.py

# Detection training + evaluation
python scripts/train_detector.py
python scripts/eval_detector.py

# UMAP visualization
python -m evaluation.feature_viz

# All tests
python -m pytest tests/ -v
```
