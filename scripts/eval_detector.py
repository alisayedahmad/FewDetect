import os
import torch
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from backbone.dinov2 import DINOv2Backbone
from backbone.fpn import LightFPN
from detection_head.fcos_head import FCOSHead
from data.coco_fewshot import NOVEL_CLASSES


class FewShotDetectorEvaluator:
    """Evaluate few-shot detection on novel classes using trained FPN + FCOS    """

    def __init__(self, checkpoint_path, fpn_dim=256, num_convs=2,
                 image_size=224, device='cuda'):
        self.device = device
        self.image_size = image_size
        self.fpn_dim = fpn_dim

        #Load backbone
        self.backbone = DINOv2Backbone(frozen=True)
        self.backbone.model.to(device)

        #Load trained FPN + head 
        self.fpn = LightFPN(embed_dim=384, fpn_dim=fpn_dim).to(device)
        self.head = FCOSHead(fpn_dim=fpn_dim, n_classes=None,
                             num_convs=num_convs).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.fpn.load_state_dict(checkpoint['fpn'])

        # Load head weights selectively — skip cls_logits since we use prototypes
        head_state = {}
        for k, v in checkpoint['head'].items():
            if 'cls_logits' not in k:
                head_state[k] = v
        self.head.load_state_dict(head_state, strict=False)

        self.fpn.eval()
        self.head.eval()

        print("Detector loaded.")

        # Normalization for DINOv2
        self.normalize = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, pil_images):
        """PIL images -> normalized tensor on device."""
        tensors = [self.normalize(img) for img in pil_images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def _extract_fpn(self, images_tensor):
        """Image tensor -> FPN features."""
        output = self.backbone.model(pixel_values=images_tensor)
        patch_tokens = output.last_hidden_state[:, 1:, :]
        return self.fpn(patch_tokens)

    @torch.no_grad()
    def build_prototypes(self, support_images):
        """Build prototypes from support images for each class 
        
        Args:
            support_images: list of lists of PIL images (one list per class)
            
        Returns:
            prototypes: tensor [n_way, fpn_dim]
        """
        prototypes = []
        for class_images in support_images:
            tensors = self._preprocess(class_images)
            fpn_features = self._extract_fpn(tensors)
            cls_feat = self.head.cls_tower(fpn_features['p3'])
            pooled = cls_feat.mean(dim=[2, 3])  # [K, fpn_dim]
            prototype = pooled.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    @torch.no_grad()
    def detect(self, query_images, prototypes, score_threshold=0.15):
        """Run detection on query images using the given prototypes for classification 
        
        Returns:
            list of dicts with 'boxes', 'scores', 'labels'
        """
        tensors = self._preprocess(query_images)
        fpn_features = self._extract_fpn(tensors)
        output = self.head(fpn_features, prototypes=prototypes)

        all_detections = []
        for b in range(len(query_images)):
            boxes, scores, labels = [], [], []

            for level_idx, level in enumerate(['p3', 'p4', 'p5']):
                cls_scores = output['cls_scores'][level_idx][b]
                bbox_preds = output['bbox_preds'][level_idx][b]
                centerness = output['centerness'][level_idx][b]

                n_way, H, W = cls_scores.shape
                stride = self.image_size // H

                shifts_x = torch.arange(0, W, device=self.device) * stride + stride // 2
                shifts_y = torch.arange(0, H, device=self.device) * stride + stride // 2
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

                cls_probs = cls_scores.sigmoid()
                center_score = centerness[0]
                final_scores = cls_probs * center_score

                for cls_id in range(n_way):
                    mask = final_scores[cls_id] > score_threshold
                    if mask.sum() == 0:
                        continue

                    s = final_scores[cls_id][mask]
                    cx = shift_x[mask].float()
                    cy = shift_y[mask].float()

                    l = bbox_preds[0][mask]
                    t = bbox_preds[1][mask]
                    r = bbox_preds[2][mask]
                    bot = bbox_preds[3][mask]

                    x1 = (cx - l).clamp(min=0)
                    y1 = (cy - t).clamp(min=0)
                    x2 = (cx + r).clamp(max=self.image_size)
                    y2 = (cy + bot).clamp(max=self.image_size)

                    boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
                    scores.append(s)
                    labels.append(torch.full_like(s, cls_id, dtype=torch.long))

            if boxes:
                boxes = torch.cat(boxes)
                scores = torch.cat(scores)
                labels = torch.cat(labels)
                keep = self._nms(boxes, scores, labels)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = torch.zeros(0, 4)
                scores = torch.zeros(0)
                labels = torch.zeros(0, dtype=torch.long)

            all_detections.append({
                'boxes': boxes, 'scores': scores, 'labels': labels
            })

        return all_detections

    def _nms(self, boxes, scores, labels, iou_threshold=0.5):
        keep = []
        for cls_id in labels.unique():
            cls_mask = labels == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            order = cls_scores.argsort(descending=True)
            cls_boxes = cls_boxes[order]
            cls_indices = cls_indices[order]
            suppressed = torch.zeros(len(cls_boxes), dtype=torch.bool, device=boxes.device)

            for i in range(len(cls_boxes)):
                if suppressed[i]:
                    continue
                keep.append(cls_indices[i].item())
                if i < len(cls_boxes) - 1:
                    remaining = cls_boxes[i + 1:]
                    ious = self._iou(cls_boxes[i].unsqueeze(0), remaining)
                    suppressed[i + 1:] |= (ious > iou_threshold)
        return keep

    def _iou(self, box, boxes):
        x1 = torch.max(box[:, 0], boxes[:, 0])
        y1 = torch.max(box[:, 1], boxes[:, 1])
        x2 = torch.min(box[:, 2], boxes[:, 2])
        y2 = torch.min(box[:, 3], boxes[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        a1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (a1 + a2 - inter + 1e-6)


def compute_ap(detections, ground_truths, iou_threshold=0.5):
    """Compute Average Precision for one class.
    
    Args:
        detections: list of (score, box) tuples
        ground_truths: list of boxes [N, 4]
        
    Returns:
        AP value
    """
    if len(ground_truths) == 0:
        return 0.0 if len(detections) > 0 else 1.0

    if len(detections) == 0:
        return 0.0

    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    matched = [False] * len(ground_truths)

    tp = []
    fp = []

    for score, det_box in detections:
        best_iou = 0
        best_gt = -1

        for gt_idx, gt_box in enumerate(ground_truths):
            x1 = max(det_box[0], gt_box[0])
            y1 = max(det_box[1], gt_box[1])
            x2 = min(det_box[2], gt_box[2])
            y2 = min(det_box[3], gt_box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
            a2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            iou = inter / (a1 + a2 - inter + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_iou >= iou_threshold and not matched[best_gt]:
            tp.append(1)
            fp.append(0)
            matched[best_gt] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(ground_truths)
    precision = tp / (tp + fp)

    # AP as area under PR curve (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p_at_r = precision[recall >= t]
        ap += max(p_at_r) / 11 if len(p_at_r) > 0 else 0

    return ap


def evaluate_fewshot_detection(checkpoint_path, coco_root="data/coco",
                                n_episodes=30, n_way=5, k_shot=5):
    """Full few-shot detection evaluation on novel classes."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    evaluator = FewShotDetectorEvaluator(checkpoint_path, device=device)

    # Load COCO val for novel classes
    coco = COCO(os.path.join(coco_root, "annotations", "instances_val2017.json"))
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c['name']: c['id'] for c in cats}

    novel_cat_ids = [name_to_id[n] for n in NOVEL_CLASSES if n in name_to_id]
    novel_names = [n for n in NOVEL_CLASSES if n in name_to_id]

    print(f"\nNovel classes: {novel_names}")
    print(f"Evaluating {n_episodes} episodes, {n_way}-way {k_shot}-shot")

    import random
    random.seed(42)

    all_aps = []

    for ep in range(n_episodes):
        # Sample n_way novel classes
        chosen_cats = random.sample(list(zip(novel_names, novel_cat_ids)), n_way)

        support_images = []
        query_data = []  # list of (images, gt_boxes) per class

        for cls_name, cat_id in chosen_cats:
            img_ids = coco.getImgIds(catIds=[cat_id])
            sampled_ids = random.sample(img_ids, min(k_shot + 5, len(img_ids)))

            support_ids = sampled_ids[:k_shot]
            query_ids = sampled_ids[k_shot:k_shot + 5]

            # Load support: crop objects
            support_crops = []
            for img_id in support_ids:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(coco_root, "val2017", img_info['file_name'])
                img = Image.open(img_path).convert('RGB')

                anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id]))
                if anns:
                    x, y, w, h = anns[0]['bbox']
                    crop = img.crop((x, y, x + w, y + h))
                    support_crops.append(crop)

            if len(support_crops) < k_shot:
                support_crops = support_crops * k_shot
                support_crops = support_crops[:k_shot]

            support_images.append(support_crops)

            # Load query: full images with GT boxes
            cls_query = []
            for img_id in query_ids:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(coco_root, "val2017", img_info['file_name'])
                img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img.size

                anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id]))
                gt_boxes = []
                for ann in anns:
                    if ann.get('iscrowd', 0):
                        continue
                    x, y, w, h = ann['bbox']
                    # Normalize to image_size
                    scale_x = 224.0 / orig_w
                    scale_y = 224.0 / orig_h
                    gt_boxes.append([
                        x * scale_x, y * scale_y,
                        (x + w) * scale_x, (y + h) * scale_y
                    ])
                cls_query.append((img, gt_boxes))

            query_data.append(cls_query)

        # Build prototypes
        prototypes = evaluator.build_prototypes(support_images)

        # Detect and evaluate per class
        episode_aps = []
        for cls_id in range(n_way):
            cls_detections = []
            cls_gt_boxes = []

            for query_img, gt_boxes in query_data[cls_id]:
                dets = evaluator.detect([query_img], prototypes, score_threshold=0.1)
                det = dets[0]

                # Get detections for this class
                cls_mask = det['labels'] == cls_id
                for s, b in zip(det['scores'][cls_mask], det['boxes'][cls_mask]):
                    cls_detections.append((s.item(), b.cpu().tolist()))

                cls_gt_boxes.extend(gt_boxes)

            ap = compute_ap(cls_detections, cls_gt_boxes, iou_threshold=0.5)
            episode_aps.append(ap)

        mean_ap = np.mean(episode_aps)
        all_aps.append(mean_ap)

        if (ep + 1) % 5 == 0:
            running = np.mean(all_aps) * 100
            print(f"  Episode {ep+1}/{n_episodes} — running mAP: {running:.1f}%")

    final_map = np.mean(all_aps) * 100
    ci = 1.96 * np.std(all_aps) * 100 / np.sqrt(n_episodes)

    print(f"\n{'='*60}")
    print(f"  FEW-SHOT DETECTION — COCO Novel Classes")
    print(f"{'='*60}")
    print(f"  {n_way}-way {k_shot}-shot mAP@0.5: {final_map:.1f}% ± {ci:.1f}%")
    print(f"  ({n_episodes} episodes)")
    print(f"{'='*60}")

    return final_map, ci


if __name__ == "__main__":
    evaluate_fewshot_detection(
        checkpoint_path="results/detector_base_classes.pth",
        n_episodes=30, n_way=5, k_shot=5
    )