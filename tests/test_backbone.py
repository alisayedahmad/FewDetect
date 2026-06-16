import torch
from torchvision import datasets
from backbone.dinov2 import DINOv2Backbone


def test_cls_token_shape():
    backbone = DINOv2Backbone(frozen=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    
    img, _ = cifar[0]
    features = backbone.extract_features(img)
    
    assert features.shape == (1, 384), f"Expected (1, 384), got {features.shape}"


def test_batch_extraction():
    backbone = DINOv2Backbone(frozen=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    
    images = [cifar[i][0] for i in range(4)]
    features = backbone.extract_features(images)
    
    assert features.shape == (4, 384), f"Expected (4, 384), got {features.shape}"


def test_patch_tokens_shape():
    backbone = DINOv2Backbone(frozen=True)
    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    
    img, _ = cifar[0]
    patches = backbone.extract_features(img, return_patch_tokens=True)
    
    # DINOv2 ViT-S/14: 16x16 patches on 224x224 = 256 patches
    assert patches.shape[0] == 1
    assert patches.shape[2] == 384
    assert patches.shape[1] > 0, "Should have patch tokens"


def test_frozen_by_default():
    backbone = DINOv2Backbone(frozen=True)
    
    for param in backbone.model.parameters():
        assert not param.requires_grad, "Should be frozen"


def test_unfreeze():
    backbone = DINOv2Backbone(frozen=True)
    backbone.unfreeze()
    
    for param in backbone.model.parameters():
        assert param.requires_grad, "Should be unfrozen"


if __name__ == "__main__":
    test_cls_token_shape()
    test_batch_extraction()
    test_patch_tokens_shape()
    test_frozen_by_default()
    test_unfreeze()
    print("Tous les tests passent!")