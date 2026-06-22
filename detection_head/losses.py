import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection (Lin et al., 2017)
    Reduces the loss contribution from easy negatives (background) so the model focuses on hard examples (actual objects)
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)


    
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        Args:
            predictions: tensor [N, C] raw logits
            targets: tensor [N] class indices (0 = background)
            
        Returns:
            scalar focal loss
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        return loss.mean()


class IoULoss(nn.Module):
    """IoU Loss for bounding box regression.
    
    Measures overlap between predicted and target boxes.
    Works with FCOS format: (left, top, right, bottom) distances
    from each point to the box edges.
    """

    def __init__(self, loss_type="iou"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, predictions, targets):
        """
        Args:
            predictions: tensor [N, 4] predicted (l, t, r, b) distances
            targets: tensor [N, 4] target (l, t, r, b) distances
            
        Returns:
            scalar IoU loss
        """
        pred_l, pred_t, pred_r, pred_b = predictions.unbind(dim=1)
        tgt_l, tgt_t, tgt_r, tgt_b = targets.unbind(dim=1)

        pred_area = (pred_l + pred_r) * (pred_t + pred_b)
        tgt_area = (tgt_l + tgt_r) * (tgt_t + tgt_b)

        inter_w = torch.min(pred_l, tgt_l) + torch.min(pred_r, tgt_r)
        inter_h = torch.min(pred_t, tgt_t) + torch.min(pred_b, tgt_b)
        inter_area = inter_w * inter_h

        union_area = pred_area + tgt_area - inter_area
        iou = inter_area / (union_area + 1e-6)

        loss = -torch.log(iou + 1e-6)
        return loss.mean()


class CenternessLoss(nn.Module):
    """Binary cross-entropy loss for centerness prediction.
    
    Centerness measures how close a point is to the center of its
    assigned object. Points at the center have centerness = 1,
    points at the edges have centerness close to 0.
    """

    def forward(self, predictions, targets):
        """
        Args:
            predictions: tensor [N] predicted centerness (after sigmoid)
            targets: tensor [N] target centerness values in [0, 1]
            
        Returns:
            scalar centerness loss
        """
        return F.binary_cross_entropy(predictions, targets)


def compute_centerness_targets(bbox_targets):
    """Compute centerness from FCOS bbox targets.
    
    centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
    
    Args:
        bbox_targets: tensor [N, 4] with (left, top, right, bottom)
        
    Returns:
        centerness: tensor [N] values in [0, 1]
    """
    l, t, r, b = bbox_targets.unbind(dim=1)

    lr_min = torch.min(l, r)
    lr_max = torch.max(l, r)
    tb_min = torch.min(t, b)
    tb_max = torch.max(t, b)

    centerness = torch.sqrt(
        (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
    )
    return centerness