import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from data.coco_fewshot import NOVEL_CLASSES


class COCODetection(Dataset):
    """COCO dataset for object detection with full images and bbox annotations 
    Returns images with all their bounding box annotations for training
    a detector on base classes.
    """

    def __init__(self, root="data/coco", split="val2017", class_set="base",
                 image_size=224):
        self.root = root
        self.img_dir = os.path.join(root, split)
        self.image_size = image_size
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")

        print(f"Loading COCO {split} for detection...")
        self.coco = COCO(ann_file)

        # Build class mappings
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {c['id']: c['name'] for c in cats}
        self.name_to_cat_id = {c['name']: c['id'] for c in cats}

        novel_names = [n for n in NOVEL_CLASSES if n in self.name_to_cat_id]
        all_names = sorted([c['name'] for c in cats])

        if class_set == "base":
            selected_names = [n for n in all_names if n not in novel_names]
        elif class_set == "novel":
            selected_names = novel_names
        else:
            selected_names = all_names

        self.selected_names = sorted(selected_names)
        self.name_to_label = {n: i for i, n in enumerate(self.selected_names)}
        self.selected_cat_ids = set(
            self.name_to_cat_id[n] for n in self.selected_names
        )

        # Get image IDs that contain at least one selected class
        all_img_ids = set()
        for cat_id in self.selected_cat_ids:
            all_img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = sorted(list(all_img_ids))

        print(f"Class set: {class_set} ({len(self.selected_names)} classes)")
        print(f"Images: {len(self.img_ids)}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Returns (image_tensor, target_dict).
        
        image_tensor: [3, image_size, image_size]
        target_dict: {
            'boxes': [N, 4] in (x1, y1, x2, y2) normalized to [0, 1],
            'labels': [N] class indices
        }
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        orig_w, orig_h = img.size

        # Get annotations for selected classes
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id not in self.selected_cat_ids:
                continue

            name = self.cat_id_to_name[cat_id]
            label = self.name_to_label[name]

            x, y, w, h = ann['bbox']
            if w < 1 or h < 1:
                continue

            # Normalize to [0, 1]
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        # Resize image
        img = img.resize((self.image_size, self.image_size))

        # To tensor [3, H, W], normalized to [0, 1]
        img_tensor = torch.tensor(
            list(img.getdata()), dtype=torch.float32
        ).reshape(self.image_size, self.image_size, 3).permute(2, 0, 1) / 255.0

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels, 'image_id': img_id}
        return img_tensor, target

    def get_class_name(self, label):
        return self.selected_names[label]

    @staticmethod
    def collate_fn(batch):
        """Custom collate since targets have variable number of boxes."""
        images = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return images, targets


class FCOSTargetAssigner:
    """Assigns ground truth boxes to FCOS feature map points    
    For each point on each FPN level, determines:
    - Is this point inside a ground truth box?
    - If yes, which class and what are the (l, t, r, b) distances?
    - What is the centerness target?
    """

    def __init__(self, image_size=224, fpn_levels=None):
        self.image_size = image_size
        # Size ranges for each FPN level
        if fpn_levels is None:
            self.fpn_levels = {
                'p3': {'size': 16, 'stride': 14, 'range': (0, 64)},
                'p4': {'size': 8, 'stride': 28, 'range': (64, 128)},
                'p5': {'size': 4, 'stride': 56, 'range': (128, 512)},
            }

    def assign(self, targets, n_classes):
        """Assign targets for one image 
        Args:
            targets: dict with 'boxes' [N, 4] normalized and 'labels' [N]
            n_classes: number of classes
            
        Returns:
            dict with per-level targets:
                'cls_targets': [H, W] class labels (-1 = ignore, 0 = bg, 1+ = class)
                'bbox_targets': [H, W, 4] (l, t, r, b) distances in pixels
                'centerness_targets': [H, W] centerness values
        """
        boxes = targets['boxes'] * self.image_size  # denormalize
        labels = targets['labels']

        level_targets = {}

        for level_name, level_info in self.fpn_levels.items():
            H = W = level_info['size']
            stride = level_info['stride']
            size_min, size_max = level_info['range']

            cls_target = torch.zeros(H, W, dtype=torch.long)
            bbox_target = torch.zeros(H, W, 4)
            centerness_target = torch.zeros(H, W)

            # Generate grid points
            shifts_x = torch.arange(0, W) * stride + stride // 2
            shifts_y = torch.arange(0, H) * stride + stride // 2

            for i in range(H):
                for j in range(W):
                    cx = shifts_x[j].float()
                    cy = shifts_y[i].float()

                    best_area = float('inf')
                    best_box_idx = -1

                    for box_idx in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[box_idx]

                        # Check if point is inside box
                        if cx < x1 or cx > x2 or cy < y1 or cy > y2:
                            continue

                        # Compute (l, t, r, b) distances
                        l = cx - x1
                        t = cy - y1
                        r = x2 - cx
                        b = y2 - cy

                        max_dist = max(l, t, r, b).item()

                        # Check size range for this level
                        if max_dist < size_min or max_dist > size_max:
                            continue

                        # Keep smallest box if multiple match
                        area = (x2 - x1) * (y2 - y1)
                        if area < best_area:
                            best_area = area
                            best_box_idx = box_idx

                    if best_box_idx >= 0:
                        x1, y1, x2, y2 = boxes[best_box_idx]
                        l = cx - x1
                        t = cy - y1
                        r = x2 - cx
                        b = y2 - cy

                        cls_target[i, j] = labels[best_box_idx] + 1  # 0 = bg
                        bbox_target[i, j] = torch.tensor([l, t, r, b])

                        lr_min = min(l, r)
                        lr_max = max(l, r)
                        tb_min = min(t, b)
                        tb_max = max(t, b)
                        centerness_target[i, j] = (
                            (lr_min / (lr_max + 1e-6)) *
                            (tb_min / (tb_max + 1e-6))
                        ).sqrt()

            level_targets[level_name] = {
                'cls_targets': cls_target,
                'bbox_targets': bbox_target,
                'centerness_targets': centerness_target,
            }

        return level_targets