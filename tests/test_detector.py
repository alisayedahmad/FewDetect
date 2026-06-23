import torch
from torchvision import datasets
from approaches.prototypical.detector import FewShotDetector
from data.episode_sampler import EpisodeSampler


def test_build_prototypes():
    """Should build prototypes from support images & return correct shape """
    detector = FewShotDetector(fpn_dim=256, num_convs=2, frozen_backbone=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    sampler = EpisodeSampler(cifar, n_way=5, k_shot=3, n_query=5, seed=42)
    support_idx, _, classes = sampler.sample_episode()

    support_images = [[cifar[i][0] for i in cls] for cls in support_idx]
    prototypes = detector.build_prototypes(support_images)

    assert prototypes.shape == (5, 256), f"Expected (5, 256), got {prototypes.shape}"
    print(f"Prototypes shape: {prototypes.shape}")


def test_detect_returns_valid_format():
    """Detection should return boxes, scores, labels per image """
    detector = FewShotDetector(fpn_dim=256, num_convs=2, frozen_backbone=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    sampler = EpisodeSampler(cifar, n_way=5, k_shot=3, n_query=5, seed=42)
    support_idx, query_idx, classes = sampler.sample_episode()

    support_images = [[cifar[i][0] for i in cls] for cls in support_idx]
    query_images = [cifar[query_idx[0][0]][0]]  # just 1 query image

    prototypes = detector.build_prototypes(support_images)
    detections = detector.detect(query_images, prototypes, score_threshold=0.1)

    assert len(detections) == 1, "Should have 1 detection result"
    det = detections[0]
    assert 'boxes' in det
    assert 'scores' in det
    assert 'labels' in det
    assert det['boxes'].ndim == 2
    assert det['boxes'].shape[1] == 4 or det['boxes'].shape[0] == 0

    n_det = len(det['scores'])
    print(f"Detected {n_det} objects")
    if n_det > 0:
        print(f"  Scores range: [{det['scores'].min():.3f}, {det['scores'].max():.3f}]")
        print(f"  Labels: {det['labels'].tolist()[:10]}")


def test_detect_batch():
    """Detection should work with multiple query images."""
    detector = FewShotDetector(fpn_dim=256, num_convs=2, frozen_backbone=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)

    sampler = EpisodeSampler(cifar, n_way=3, k_shot=2, n_query=4, seed=42)
    support_idx, query_idx, classes = sampler.sample_episode()

    support_images = [[cifar[i][0] for i in cls] for cls in support_idx]
    query_images = [cifar[query_idx[0][i]][0] for i in range(2)]

    prototypes = detector.build_prototypes(support_images)
    detections = detector.detect(query_images, prototypes, score_threshold=0.1)

    assert len(detections) == 2, "Should have 2 detection results"
    print(f"Image 0: {len(detections[0]['scores'])} detections")
    print(f"Image 1: {len(detections[1]['scores'])} detections")


if __name__ == "__main__":
    test_build_prototypes()
    test_detect_returns_valid_format()
    test_detect_batch()
    print("\nAll detector tests passed!")