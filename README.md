# FewDetect

Benchmark comparing three few-shot object detection approaches on surveillance data:
**Prototypical Networks**, **MAML**, and **DINOv2 fine-tuning**.

Same backbone. Same dataset. Same evaluation protocol. Rigorous comparison.

---

## What this is

Classical detectors like YOLO need thousands of labeled examples per class.
In surveillance, that's not realistic — a suspicious individual, an abandoned bag,
a stolen vehicle. These categories are rare by definition.

Few-shot detection addresses this: how do you train a detector that recognizes
a new class from only 1, 5, or 10 examples?

This project answers: which approach generalizes best, and under what conditions.

---

## Approaches

- **Prototypical Networks** — class prototypes as mean embeddings, distance-based detection
- **MAML** — meta-learned initialization for fast adaptation
- **DINOv2 fine-tuning** — adapting a strong self-supervised backbone with LoRA

---

## Status

Setting up. Starting with the papers and the data pipeline.