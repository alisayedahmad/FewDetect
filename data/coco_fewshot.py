import os
import torch
from PIL import Image
from pycocotools.coco import COCO


# 20 novel classes — standard split from Kang et al., ICCV 2019 paper 
NOVEL_CLASSES = [
    'person', 'airplane', 'boat', 'parking meter', 'dog',
    'elephant', 'backpack', 'suitcase', 'sports ball', 'skateboard',
    'wine glass', 'spoon', 'sandwich', 'hot dog', 'chair',
    'dining table', 'mouse', 'microwave', 'scissors', 'teddy bear',
]


class COCOFewShot:
    """COCO dataset split into base(60) and novel (20) classes
    
    Each sample is a cropped object from COCO, turned into a classification image with a class label
    
    
    Args:
        root: path to data/coco/
        split: 'val2017' or 'train2017'
        class_set: 'base', 'novel', or 'all'
        min_area: minimum bounding box area to include
    """

    def __init__(self, root="data/coco", split="val2017", class_set="novel",
                 min_area=32*32):
        self.root = root
        self.img_dir = os.path.join(root, split)
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")

        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(ann_file)

        # Get all category names and IDs
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.all_class_names = [c['name'] for c in cats]
        self.cat_id_to_name = {c['id']: c['name'] for c in cats}
        self.name_to_cat_id = {c['name']: c['id'] for c in cats}

        # Split into base and novel
        self.novel_names = [n for n in NOVEL_CLASSES if n in self.name_to_cat_id]
        self.base_names = [n for n in self.all_class_names if n not in self.novel_names]

        if class_set == "novel":
            selected_names = self.novel_names
        elif class_set == "base":
            selected_names = self.base_names
        else:
            selected_names = self.all_class_names

        # Build label mapping: class_name -> local label (0, 1, 2, ...)
        self.selected_names = sorted(selected_names)
        self.name_to_label = {name: i for i, name in enumerate(self.selected_names)}

        # Collect all valid crops
        self.samples = []  # list of (image_id, bbox, label)
        selected_cat_ids = [self.name_to_cat_id[n] for n in self.selected_names]

        for cat_id in selected_cat_ids:
            ann_ids = self.coco.getAnnIds(catIds=[cat_id])
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue
                bbox = ann['bbox']  # [x, y, w, h]
                area = bbox[2] * bbox[3]
                if area < min_area:
                    continue

                name = self.cat_id_to_name[cat_id]
                label = self.name_to_label[name]
                self.samples.append((ann['image_id'], bbox, label))

        # Count per class
        class_counts = {}
        for _, _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        print(f"Class set: {class_set} ({len(self.selected_names)} classes)")
        print(f"Total crops: {len(self.samples)}")
        print(f"Crops per class: min={min(class_counts.values())}, "
              f"max={max(class_counts.values())}, "
              f"mean={sum(class_counts.values())//len(class_counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (PIL image crop, label)"""
        image_id, bbox, label = self.samples[idx]

        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Crop bounding box [x, y, w, h] -> [left, top, right, bottom]
        x, y, w, h = bbox
        crop = img.crop((x, y, x + w, y + h))

        return crop, label

    def get_class_name(self, label):
        """Get class name from label index"""
        return self.selected_names[label]

    def __repr__(self):
        return (f"COCOFewShot({len(self.selected_names)} classes, "
                f"{len(self.samples)} crops)")