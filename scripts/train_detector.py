import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import DataLoader
from backbone.dinov2 import DINOv2Backbone
from backbone.fpn import LightFPN
from detection_head.fcos_head import FCOSHead
from detection_head.losses import FocalLoss, IoULoss
from data.coco_detection import COCODetection, FCOSTargetAssigner


class DetectorTrainer:
    """Trains FPN + FCOS on base classes."""

    def __init__(self, n_classes=60, fpn_dim=256, num_convs=2, lr=1e-3,
                 image_size=224, device='cuda'):
        self.device = device
        self.image_size = image_size
        self.n_classes = n_classes

        # Backbone — frozen, just for feature extraction
        self.backbone = DINOv2Backbone(frozen=True)
        self.backbone.model.to(device)

        # Trainable parts
        self.fpn = LightFPN(embed_dim=384, fpn_dim=fpn_dim).to(device)
        self.head = FCOSHead(fpn_dim=fpn_dim, n_classes=n_classes + 1,
                             num_convs=num_convs).to(device)

        # Losses
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.bbox_loss_fn = IoULoss()

        # Optimizer — only FPN and head
        params = list(self.fpn.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        # Target assigner
        self.assigner = FCOSTargetAssigner(image_size=image_size)

        trainable = sum(p.numel() for p in params if p.requires_grad)
        print(f"Trainable params: {trainable:,}")

    @torch.no_grad()
    def _extract_patch_tokens(self, images):
        """Extract DINOv2 patch tokens from image tensors.
        
        Args:
            images: tensor [B, 3, H, W] on device
            
        Returns:
            patch_tokens: tensor [B, 256, 384]
        """
        # DINOv2 processor expects specific normalization
        # Apply DINOv2 normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        normalized = (images - mean) / std

        output = self.backbone.model(pixel_values=normalized)
        return output.last_hidden_state[:, 1:, :]  # skip CLS

    def train_one_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.fpn.train()
        self.head.train()

        total_cls_loss = 0
        total_bbox_loss = 0
        total_center_loss = 0
        n_batches = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)

            # Extract features
            patch_tokens = self._extract_patch_tokens(images)
            fpn_features = self.fpn(patch_tokens)
            output = self.head(fpn_features)

            # Compute loss for each image in batch
            batch_cls_loss = 0
            batch_bbox_loss = 0
            batch_center_loss = 0
            valid_images = 0

            for b in range(len(targets)):
                level_targets = self.assigner.assign(targets[b], self.n_classes)

                for level_idx, level_name in enumerate(['p3', 'p4', 'p5']):
                    cls_pred = output['cls_scores'][level_idx][b]  # [C+1, H, W]
                    bbox_pred = output['bbox_preds'][level_idx][b]  # [4, H, W]
                    center_pred = output['centerness'][level_idx][b]  # [1, H, W]

                    cls_tgt = level_targets[level_name]['cls_targets'].to(self.device)
                    bbox_tgt = level_targets[level_name]['bbox_targets'].to(self.device)
                    center_tgt = level_targets[level_name]['centerness_targets'].to(self.device)

                    H, W = cls_tgt.shape

                    # Classification loss — all points
                    cls_pred_flat = cls_pred.permute(1, 2, 0).reshape(-1, self.n_classes + 1)
                    cls_tgt_flat = cls_tgt.reshape(-1)
                    batch_cls_loss += self.cls_loss_fn(cls_pred_flat, cls_tgt_flat)

                    # Bbox and centerness loss — positive points only
                    pos_mask = cls_tgt > 0
                    n_pos = pos_mask.sum().item()

                    if n_pos > 0:
                        bbox_pred_pos = bbox_pred[:, pos_mask].permute(1, 0)
                        bbox_tgt_pos = bbox_tgt[pos_mask]
                        batch_bbox_loss += self.bbox_loss_fn(bbox_pred_pos, bbox_tgt_pos)

                        center_pred_pos = center_pred[0, pos_mask]
                        center_tgt_pos = center_tgt[pos_mask]
                        batch_center_loss += F.binary_cross_entropy(
                            center_pred_pos.clamp(1e-6, 1 - 1e-6),
                            center_tgt_pos
                        )

                valid_images += 1

            if valid_images == 0:
                continue

            loss = (batch_cls_loss + batch_bbox_loss + batch_center_loss) / valid_images

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.fpn.parameters()) + list(self.head.parameters()),
                max_norm=10.0
            )
            self.optimizer.step()

            total_cls_loss += batch_cls_loss.item()
            total_bbox_loss += batch_bbox_loss.item() if isinstance(batch_bbox_loss, torch.Tensor) else 0
            total_center_loss += batch_center_loss.item() if isinstance(batch_center_loss, torch.Tensor) else 0
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_cls = total_cls_loss / n_batches
                avg_bbox = total_bbox_loss / n_batches
                avg_center = total_center_loss / n_batches
                print(f"  [{batch_idx+1}/{len(dataloader)}] "
                      f"cls: {avg_cls:.4f}  bbox: {avg_bbox:.4f}  center: {avg_center:.4f}")

        return {
            'cls_loss': total_cls_loss / max(n_batches, 1),
            'bbox_loss': total_bbox_loss / max(n_batches, 1),
            'center_loss': total_center_loss / max(n_batches, 1),
        }

    def save(self, path):
        """Save trained FPN + head weights."""
        torch.save({
            'fpn': self.fpn.state_dict(),
            'head': self.head.state_dict(),
        }, path)
        print(f"Saved to {path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load dataset
    dataset = COCODetection(split="val2017", class_set="base", image_size=224)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=COCODetection.collate_fn
    )

    n_classes = len(dataset.selected_names)
    print(f"Classes: {n_classes}")

    # Train
    trainer = DetectorTrainer(
        n_classes=n_classes, fpn_dim=256, num_convs=2,
        lr=1e-3, device=device
    )

    n_epochs = 5
    for epoch in range(n_epochs):
        t0 = time.time()
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        losses = trainer.train_one_epoch(dataloader, epoch)
        t = time.time() - t0
        print(f"Epoch {epoch+1} done in {t:.0f}s — "
              f"cls: {losses['cls_loss']:.4f}  "
              f"bbox: {losses['bbox_loss']:.4f}  "
              f"center: {losses['center_loss']:.4f}")

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    trainer.save("results/detector_base_classes.pth")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()