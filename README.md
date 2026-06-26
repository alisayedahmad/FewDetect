# FewDetect

Few-shot object detection benchmark comparing **Prototypical Networks**, **MAML**, and **DINOv2 fine-tuning** on the same backbone under identical conditions.

---

## Key Result

With a strong self-supervised backbone (DINOv2 ViT-S/14), **the simplest approach wins**. Prototypical Networks with cosine distance outperform both MAML and LoRA across all settings — without any training.

| Approach              | CIFAR-100 5-shot | COCO Novel 5-shot |
|-----------------------|------------------|-------------------|
| ProtoNets (cosine)    | **95.5%**        | **85.8%**         |
| LoRA fine-tuning      | 93.1%            | 82.6%             |
| FOMAML                | 89.4%            | —                 |

Few-shot detection on COCO novel classes: **13.4% mAP@0.5** (5-way 5-shot).

---

## What this project does

Classical detectors need thousands of labeled examples per class. In surveillance, that's not realistic — suspicious individuals, abandoned objects, stolen vehicles are rare by definition.

FewDetect answers: **which few-shot approach generalizes best for detecting objects from only 1-20 examples?**

---

## Approaches

**Prototypical Networks** — each class is represented by the mean of its support embeddings. Classification by nearest prototype. No training required.

**FOMAML** — meta-learns a weight initialization such that a few gradient steps on the support set produce a good classifier. First-order approximation for stability.

**LoRA Fine-tuning** — injects low-rank adaptation matrices (0.33% of parameters) into DINOv2 and fine-tunes on each episode's support set.

All three share the same DINOv2 ViT-S/14 backbone for fair comparison.

---

## K-Shot Curves (CIFAR-100)

| K   | ProtoNets (cos) | ProtoNets (euc) | LoRA          |
|-----|-----------------|-----------------|---------------|
| 1   | 86.4% ± 2.7%   | 85.7% ± 2.7%   | 84.1% ± 3.8% |
| 2   | 92.7% ± 2.0%   | 91.8% ± 2.0%   | 92.0% ± 2.3% |
| 5   | 95.8% ± 1.5%   | 94.6% ± 1.6%   | 93.6% ± 2.0% |
| 10  | 96.8% ± 1.3%   | 96.4% ± 1.2%   | 94.5% ± 2.0% |
| 20  | 97.4% ± 0.8%   | 97.0% ± 0.9%   | 95.1% ± 1.4% |

---

## Project Structure

```
FewDetect/
├── backbone/
│   ├── dinov2.py              # DINOv2 wrapper with feature extraction
│   └── fpn.py                 # Feature Pyramid Network (multi-scale)
├── detection_head/
│   ├── fcos_head.py           # FCOS detection head (few-shot + standard)
│   └── losses.py              # Focal loss, IoU loss, centerness loss
├── approaches/
│   ├── prototypical/
│   │   ├── model.py           # ProtoNets classifier
│   │   ├── prototype_builder.py  # Prototype computation + distance metrics
│   │   └── detector.py        # Full few-shot detector (DINOv2+FPN+FCOS)
│   ├── maml/
│   │   ├── model.py           # MAML classifier head
│   │   └── meta_learner.py    # FOMAML inner/outer loop with `higher`
│   └── finetuning/
│       └── model.py           # LoRA fine-tuning with `peft`
├── data/
│   ├── episode_sampler.py     # N-way K-shot episode generation
│   ├── coco_fewshot.py        # COCO 60/20 base/novel split
│   └── coco_detection.py      # COCO detection loader + target assigner
├── evaluation/
│   ├── metrics.py             # Multi-episode evaluation with confidence intervals
│   └── feature_viz.py         # UMAP visualization of feature spaces
├── scripts/
│   ├── run_benchmark.py       # Full 3-approach benchmark
│   ├── kshot_curves.py        # Performance vs K shots
│   ├── compare_distances.py   # Euclidean vs cosine comparison
│   ├── train_maml.py          # FOMAML meta-training
│   ├── eval_lora.py           # LoRA evaluation
│   ├── eval_coco.py           # COCO novel class evaluation
│   ├── train_detector.py      # Train FCOS on base classes
│   └── eval_detector.py       # Few-shot detection evaluation
├── tests/                     # pytest tests for all components
├── docs/
│   └── findings.md            # Detailed results and analysis
└── results/                   # Generated outputs (gitignored)
```

---

## Quick Start

```bash
git clone https://github.com/alisayedahmad/FewDetect.git
cd FewDetect
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft higher pycocotools umap-learn matplotlib pytest
```

Run classification benchmark:
```bash
python scripts/kshot_curves.py
```

Run few-shot detection:
```bash
python scripts/train_detector.py
python scripts/eval_detector.py
```

Run tests:
```bash
python -m pytest tests/ -v
```

---

## Findings

See [docs/findings.md](docs/findings.md) for detailed analysis. Key insights:

1. **Cosine beats Euclidean with DINOv2** — opposite of Snell et al. (2017), because DINOv2's contrastive training structures features directionally.
2. **Simplicity wins with strong backbones** — ProtoNets (no training) beats MAML (500 episodes meta-training) and LoRA (per-episode fine-tuning).
3. **Invest in the backbone** — the choice of DINOv2 matters more than the choice of meta-learning algorithm.
4. **K-shot saturation** — ProtoNets reaches 96.8% at K=10, gaining only 0.6% more at K=20. The marginal value of each additional example diminishes rapidly.

---

## Datasets

**CIFAR-100** — 100 classes, used for rapid prototyping and comparison of all three approaches.

**MS-COCO** — 80 classes split into 60 base (training) and 20 novel (evaluation). Standard few-shot detection benchmark.

---

## References

- Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
- Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation", ICML 2017
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019

## License

MIT.
