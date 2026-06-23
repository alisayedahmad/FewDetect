import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.dinov2 import DINOv2Backbone
from backbone.fpn import LightFPN
from detection_head.fcos_head import FCOSHead


class FewShotDetector(nn.Module):
    """Few-shot object detector: DINOv2 + FPN + FCOS with prototype classification 
    
    Pipeline:
    1. Extract patch tokens from DINOv2
    2. Build multi-scale features with FPN
    3. For support set: extract prototypes through cls_tower
    4. For query images: detect objects using prototype-based classification
    """

    def __init__(self, fpn_dim=256, num_convs=4, frozen_backbone=True):
        super().__init__()
        self.backbone = DINOv2Backbone(frozen=frozen_backbone)
        self.fpn = LightFPN(embed_dim=self.backbone.embed_dim, fpn_dim=fpn_dim)
        self.head = FCOSHead(fpn_dim=fpn_dim, n_classes=None, num_convs=num_convs)
        self.fpn_dim = fpn_dim

    def extract_fpn_features(self, images):
        """PIL images -> FPN multi-scale feature maps 
        Args:
            images: list of PIL images
            
        Returns:
            dict with 'p3', 'p4', 'p5' feature maps
        """
        patch_tokens = self.backbone.extract_features(images, return_patch_tokens=True)
        fpn_features = self.fpn(patch_tokens)
        return fpn_features

    def build_prototypes(self, support_images, level='p3'):
        """Build class prototypes from support set images.
        
        For each class, extract features through backbone + FPN + cls_tower,
        then average the spatial features to get one prototype per class
        
        Args:
            support_images: list of lists of PIL images
                support_images[i] = K images for class i
            level: which FPN level to use for prototypes
            
        Returns:
            prototypes: tensor [n_way, fpn_dim]
        """
        prototypes = []

        for class_images in support_images:
            fpn_features = self.extract_fpn_features(class_images)
            feat = fpn_features[level]  # [K, fpn_dim, H, W]

            # Pass through cls_tower for richer features
            cls_feat = self.head.cls_tower(feat)  # [K, fpn_dim, H, W]

            # Global average pool: [K, fpn_dim, H, W] -> [K, fpn_dim]
            pooled = cls_feat.mean(dim=[2, 3])  # [K, fpn_dim]

            # Average over K shots
            prototype = pooled.mean(dim=0)  # [fpn_dim]
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # [n_way, fpn_dim]
        return prototypes

    @torch.no_grad()
    def detect(self, query_images, prototypes, score_threshold=0.3):
        """Run detection on query images using prototypes
        
        Args:
            query_images: list of PIL images
            prototypes: tensor [n_way, fpn_dim]
            score_threshold: minimum score to keep a detection
            
        Returns:
            list of dicts, one per image, each with:
                'boxes': tensor [N, 4] in (x1, y1, x2, y2) format
                'scores': tensor [N]
                'labels': tensor [N]
        """
        fpn_features = self.extract_fpn_features(query_images)
        output = self.head(fpn_features, prototypes=prototypes)

        batch_size = len(query_images)
        all_detections = []

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level_idx, level in enumerate(['p3', 'p4', 'p5']):
                cls_scores = output['cls_scores'][level_idx][b]  # [n_way, H, W]
                bbox_preds = output['bbox_preds'][level_idx][b]  # [4, H, W]
                centerness = output['centerness'][level_idx][b]  # [1, H, W]

                n_way, H, W = cls_scores.shape

                # Compute stride for this level
                stride = 224 // H  # image_size / feature_size

                # Generate grid points
                shifts_x = torch.arange(0, W, device=cls_scores.device) * stride + stride // 2
                shifts_y = torch.arange(0, H, device=cls_scores.device) * stride + stride // 2
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

                # Get scores: cls_score * centerness
                cls_probs = cls_scores.sigmoid()
                center_score = centerness[0]  # [H, W]
                scores = cls_probs * center_score  # [n_way, H, W]

                # For each class, get positions above threshold
                for cls_id in range(n_way):
                    cls_score_map = scores[cls_id]  # [H, W]
                    mask = cls_score_map > score_threshold

                    if mask.sum() == 0:
                        continue

                    matched_scores = cls_score_map[mask]
                    matched_x = shift_x[mask].float()
                    matched_y = shift_y[mask].float()

                    # Get bbox predictions at matched locations
                    l = bbox_preds[0][mask]
                    t = bbox_preds[1][mask]
                    r = bbox_preds[2][mask]
                    bbot = bbox_preds[3][mask]

                    # Convert (l, t, r, b) distances to (x1, y1, x2, y2)
                    x1 = matched_x - l
                    y1 = matched_y - t
                    x2 = matched_x + r
                    y2 = matched_y + bbot

                    boxes = torch.stack([x1, y1, x2, y2], dim=1)

                    all_boxes.append(boxes)
                    all_scores.append(matched_scores)
                    all_labels.append(torch.full((len(matched_scores),), cls_id))

            if len(all_boxes) > 0:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Simple NMS per class
                keep = self._nms(all_boxes, all_scores, all_labels, iou_threshold=0.5)
                all_boxes = all_boxes[keep]
                all_scores = all_scores[keep]
                all_labels = all_labels[keep]
            else:
                all_boxes = torch.zeros(0, 4)
                all_scores = torch.zeros(0)
                all_labels = torch.zeros(0, dtype=torch.long)

            all_detections.append({
                'boxes': all_boxes,
                'scores': all_scores,
                'labels': all_labels,
            })

        return all_detections

    def _nms(self, boxes, scores, labels, iou_threshold=0.5):
        """Per-class Non-Maximum Suppression."""
        keep = []
        unique_labels = labels.unique()

        for cls_id in unique_labels:
            cls_mask = labels == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            # Sort by score descending
            order = cls_scores.argsort(descending=True)
            cls_boxes = cls_boxes[order]
            cls_indices = cls_indices[order]

            suppressed = torch.zeros(len(cls_boxes), dtype=torch.bool)

            for i in range(len(cls_boxes)):
                if suppressed[i]:
                    continue
                keep.append(cls_indices[i].item())

                if i == len(cls_boxes) - 1:
                    break

                # Compute IoU with remaining boxes
                remaining = cls_boxes[i + 1:]
                remaining_mask = ~suppressed[i + 1:]

                if remaining_mask.sum() == 0:
                    break

                ious = self._compute_iou(cls_boxes[i].unsqueeze(0), remaining)
                suppressed[i + 1:] |= (ious > iou_threshold)

        return keep

    def _compute_iou(self, box, boxes):
        """Compute IoU between one box and multiple boxes."""
        x1 = torch.max(box[:, 0], boxes[:, 0])
        y1 = torch.max(box[:, 1], boxes[:, 1])
        x2 = torch.min(box[:, 2], boxes[:, 2])
        y2 = torch.min(box[:, 3], boxes[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - inter
        return inter / (union + 1e-6)