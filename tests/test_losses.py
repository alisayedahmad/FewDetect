import torch
from detection_head.losses import FocalLoss, IoULoss, CenternessLoss, compute_centerness_targets


def test_focal_loss_basic():
    """Focal loss should return a positive scalar  """
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    preds = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    loss = loss_fn(preds, targets)
    assert loss.ndim == 0, "Should be scalar"
    assert loss.item() > 0, "Should be positive"
    print(f"Focal loss: {loss.item():.4f}")


def test_focal_loss_easy_vs_hard():
    """Focal loss should be lower for confident correct predictions   """
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Easy: high confidence correct
    easy_preds = torch.tensor([[10.0, -10.0, -10.0]])
    easy_targets = torch.tensor([0])

    # Hard: low confidence
    hard_preds = torch.tensor([[0.1, 0.0, -0.1]])
    hard_targets = torch.tensor([0])

    easy_loss = loss_fn(easy_preds, easy_targets)
    hard_loss = loss_fn(hard_preds, hard_targets)

    assert easy_loss < hard_loss, "Easy examples should have lower loss"
    print(f"Easy: {easy_loss.item():.6f}, Hard: {hard_loss.item():.6f}")


def test_iou_loss_perfect():
    """Perfect overlap should give near-zero loss."""
    loss_fn = IoULoss()

    boxes = torch.tensor([[2.0, 3.0, 2.0, 3.0]])  # l, t, r, b
    loss = loss_fn(boxes, boxes)

    assert loss.item() < 0.01, f"Perfect overlap loss should be ~0, got {loss.item()}"
    print(f"Perfect IoU loss: {loss.item():.6f}")


def test_iou_loss_no_overlap():
    """No overlap should give high loss          """
    loss_fn = IoULoss()

    pred = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    loss = loss_fn(pred, target)
    assert loss.item() > 1.0, "No overlap should have high loss"
    print(f"No overlap IoU loss: {loss.item():.4f}")


def test_centerness_targets():
    """Centerness should be 1.0 for perfectly centered points."""
    # Point exactly at center: l=r, t=b
    boxes = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    centerness = compute_centerness_targets(boxes)
    assert torch.allclose(centerness, torch.tensor([1.0]), atol=1e-5)
    print(f"Center point centerness: {centerness.item():.4f}")

    # Point at edge: l >> r
    boxes_edge = torch.tensor([[9.0, 5.0, 1.0, 5.0]])
    centerness_edge = compute_centerness_targets(boxes_edge)
    assert centerness_edge.item() < 0.5, "Edge point should have low centerness"
    print(f"Edge point centerness: {centerness_edge.item():.4f}")


def test_centerness_loss():
    """Centerness loss should work with sigmoid predictions."""
    loss_fn = CenternessLoss()

    preds = torch.tensor([0.9, 0.1, 0.5])
    targets = torch.tensor([1.0, 0.0, 0.5])

    loss = loss_fn(preds, targets)
    assert loss.item() > 0, "Should be positive"
    print(f"Centerness loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_focal_loss_basic()
    test_focal_loss_easy_vs_hard()
    test_iou_loss_perfect()
    test_iou_loss_no_overlap()
    test_centerness_targets()
    test_centerness_loss()
    print("\nAll loss tests passed!")