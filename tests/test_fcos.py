import torch
from detection_head.fcos_head import FCOSHead


def test_fcos_standard_mode():
    """FCOS should work with a fixed number of classes  """
    head = FCOSHead(fpn_dim=256, n_classes=20, num_convs=2)

    fpn_features = {
        'p3': torch.randn(2, 256, 16, 16),
        'p4': torch.randn(2, 256, 8, 8),
        'p5': torch.randn(2, 256, 4, 4),
    }

    output = head(fpn_features)

    assert len(output['cls_scores']) == 3
    assert len(output['bbox_preds']) == 3
    assert len(output['centerness']) == 3

    # Check shapes
    assert output['cls_scores'][0].shape == (2, 20, 16, 16)
    assert output['bbox_preds'][0].shape == (2, 4, 16, 16)
    assert output['centerness'][0].shape == (2, 1, 16, 16)
    print("Standard mode shapes OK")


def test_fcos_fewshot_mode():
    """FCOS should classify by prototype distance in few-shot mode."""
    head = FCOSHead(fpn_dim=256, n_classes=None, num_convs=2)

    fpn_features = {
        'p3': torch.randn(1, 256, 16, 16),
        'p4': torch.randn(1, 256, 8, 8),
        'p5': torch.randn(1, 256, 4, 4),
    }

    prototypes = torch.randn(5, 256)  # 5-way

    output = head(fpn_features, prototypes=prototypes)

    assert output['cls_scores'][0].shape == (1, 5, 16, 16)
    assert output['cls_scores'][1].shape == (1, 5, 8, 8)
    assert output['cls_scores'][2].shape == (1, 5, 4, 4)
    print("Few-shot mode shapes OK")


def test_fcos_bbox_positive():
    """Bounding box predictions should be positive (exp activation)."""
    head = FCOSHead(fpn_dim=256, n_classes=5, num_convs=2)

    fpn_features = {
        'p3': torch.randn(1, 256, 16, 16),
        'p4': torch.randn(1, 256, 8, 8),
        'p5': torch.randn(1, 256, 4, 4),
    }

    output = head(fpn_features)

    for bbox in output['bbox_preds']:
        assert (bbox > 0).all(), "All bbox predictions should be positive"
    print("Bbox all positive OK")


def test_fcos_centerness_range():
    """Centerness should be in [0, 1] (sigmoid activation)."""
    head = FCOSHead(fpn_dim=256, n_classes=5, num_convs=2)

    fpn_features = {
        'p3': torch.randn(1, 256, 16, 16),
        'p4': torch.randn(1, 256, 8, 8),
        'p5': torch.randn(1, 256, 4, 4),
    }

    output = head(fpn_features)

    for c in output['centerness']:
        assert (c >= 0).all() and (c <= 1).all(), "Centerness should be in [0, 1]"
    print("Centerness in range OK")


def test_fcos_gradient_flow():
    """Gradients should flow through the entire head."""
    head = FCOSHead(fpn_dim=256, n_classes=5, num_convs=2)

    fpn_features = {
        'p3': torch.randn(1, 256, 16, 16, requires_grad=True),
        'p4': torch.randn(1, 256, 8, 8, requires_grad=True),
        'p5': torch.randn(1, 256, 4, 4, requires_grad=True),
    }

    output = head(fpn_features)

    loss = sum(c.sum() + b.sum() + ct.sum()
               for c, b, ct in zip(output['cls_scores'],
                                   output['bbox_preds'],
                                   output['centerness']))
    loss.backward()

    for level in ['p3', 'p4', 'p5']:
        assert fpn_features[level].grad is not None
    print("Gradient flow OK")


if __name__ == "__main__":
    test_fcos_standard_mode()
    test_fcos_fewshot_mode()
    test_fcos_bbox_positive()
    test_fcos_centerness_range()
    test_fcos_gradient_flow()
    print("\nAll FCOS tests passed!")