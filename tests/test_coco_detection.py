import os
import torch
import pytest
from data.coco_detection import COCODetection, FCOSTargetAssigner


@pytest.mark.skipif(
    not os.path.exists("data/coco/val2017"),
    reason="COCO val2017 not downloaded"
)
class TestCOCODetection:

    def test_load_image(self):
        dataset = COCODetection(split="val2017", class_set="base", image_size=224)
        img, target = dataset[0]

        assert img.shape == (3, 224, 224), f"Image shape: {img.shape}"
        assert img.min() >= 0 and img.max() <= 1, "Should be normalized"
        assert 'boxes' in target and 'labels' in target
        print(f"Image shape: {img.shape}")
        print(f"Boxes: {target['boxes'].shape}, Labels: {target['labels'].shape}")

    def test_boxes_normalized(self):
        dataset = COCODetection(split="val2017", class_set="base", image_size=224)
        img, target = dataset[0]

        if len(target['boxes']) > 0:
            assert target['boxes'].min() >= 0, "Boxes should be >= 0"
            assert target['boxes'].max() <= 1, "Boxes should be <= 1"
            print(f"Box range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")

    def test_collate_fn(self):
        dataset = COCODetection(split="val2017", class_set="base", image_size=224)
        batch = [dataset[i] for i in range(4)]
        images, targets = COCODetection.collate_fn(batch)

        assert images.shape == (4, 3, 224, 224)
        assert len(targets) == 4
        print(f"Batch images: {images.shape}")

    def test_target_assigner(self):
        dataset = COCODetection(split="val2017", class_set="base", image_size=224)
        assigner = FCOSTargetAssigner(image_size=224)

        _, target = dataset[0]
        n_classes = len(dataset.selected_names)

        level_targets = assigner.assign(target, n_classes)

        assert 'p3' in level_targets
        assert 'p4' in level_targets
        assert 'p5' in level_targets

        p3 = level_targets['p3']
        assert p3['cls_targets'].shape == (16, 16)
        assert p3['bbox_targets'].shape == (16, 16, 4)
        assert p3['centerness_targets'].shape == (16, 16)

        n_pos = (p3['cls_targets'] > 0).sum().item()
        print(f"P3 positive points: {n_pos}")
        print(f"P3 class targets unique: {p3['cls_targets'].unique().tolist()}")


if __name__ == "__main__":
    TestCOCODetection().test_load_image()
    TestCOCODetection().test_boxes_normalized()
    TestCOCODetection().test_collate_fn()
    TestCOCODetection().test_target_assigner()
    print("\nAll detection data tests passed!")