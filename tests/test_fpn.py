import torch
from backbone.fpn import LightFPN


def test_fpn_output_shapes():
    """FPN should produce 3 levels with correct shapes."""
    fpn = LightFPN(embed_dim=384, fpn_dim=256, grid_size=16)

    # Simulated DINOv2 patch tokens
    patch_tokens = torch.randn(2, 256, 384)  # batch=2, 16x16=256 patches, dim=384

    features = fpn(patch_tokens)

    assert features['p3'].shape == (2, 256, 16, 16), f"P3: {features['p3'].shape}"
    assert features['p4'].shape == (2, 256, 8, 8), f"P4: {features['p4'].shape}"
    assert features['p5'].shape == (2, 256, 4, 4), f"P5: {features['p5'].shape}"
    print("P3: (2, 256, 16, 16)")
    print("P4: (2, 256, 8, 8)")
    print("P5: (2, 256, 4, 4)")


def test_fpn_gradient_flow():
    """Gradients should flow through all levels."""
    fpn = LightFPN(embed_dim=384, fpn_dim=256, grid_size=16)
    patch_tokens = torch.randn(1, 256, 384, requires_grad=True)

    features = fpn(patch_tokens)

    # Sum all levels and backprop
    loss = sum(f.sum() for f in features.values())
    loss.backward()

    assert patch_tokens.grad is not None, "Gradients should flow back"
    assert patch_tokens.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_fpn_with_dinov2():
    """FPN should work with real DINOv2 output."""
    from backbone.dinov2 import DINOv2Backbone
    from torchvision import datasets

    backbone = DINOv2Backbone(frozen=True)
    fpn = LightFPN(embed_dim=384, fpn_dim=256, grid_size=16)

    cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
    img, _ = cifar[0]

    # Get patch tokens from DINOv2
    patch_tokens = backbone.extract_features(img, return_patch_tokens=True)
    print(f"Patch tokens: {patch_tokens.shape}")

    features = fpn(patch_tokens)
    print(f"P3: {features['p3'].shape}")
    print(f"P4: {features['p4'].shape}")
    print(f"P5: {features['p5'].shape}")


if __name__ == "__main__":
    test_fpn_output_shapes()
    test_fpn_gradient_flow()
    test_fpn_with_dinov2()
    print("\nAll FPN tests passed!")