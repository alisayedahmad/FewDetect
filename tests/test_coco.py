import os
import pytest
from data.coco_fewshot import COCOFewShot


@pytest.mark.skipif(
    not os.path.exists("data/coco/val2017"),
    reason="COCO val2017 not downloaded"
)
class TestCOCO:

    def test_novel_classes(self):
        dataset = COCOFewShot(split="val2017", class_set="novel")
        assert len(dataset.selected_names) == 20
        assert len(dataset) > 0

    def test_base_classes(self):
        dataset = COCOFewShot(split="val2017", class_set="base")
        assert len(dataset.selected_names) == 60
        assert len(dataset) > 0

    def test_no_overlap(self):
        novel = COCOFewShot(split="val2017", class_set="novel")
        base = COCOFewShot(split="val2017", class_set="base")
        overlap = set(novel.selected_names) & set(base.selected_names)
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_load_crop(self):
        dataset = COCOFewShot(split="val2017", class_set="novel")
        img, label = dataset[0]
        assert img.size[0] > 0 and img.size[1] > 0
        assert 0 <= label < 20
        print(f"Crop size: {img.size}, class: {dataset.get_class_name(label)}")

    def test_compatible_with_sampler(self):
        from data.episode_sampler import EpisodeSampler
        dataset = COCOFewShot(split="val2017", class_set="novel")
        sampler = EpisodeSampler(dataset, n_way=5, k_shot=5, n_query=15, seed=42)
        support, query, classes = sampler.sample_episode()
        assert len(classes) == 5
        print(f"Episode classes: {[dataset.get_class_name(c) for c in classes]}")


if __name__ == "__main__":
    TestCOCO().test_novel_classes()
    TestCOCO().test_base_classes()
    TestCOCO().test_no_overlap()
    TestCOCO().test_load_crop()
    TestCOCO().test_compatible_with_sampler()
    print("\nAll COCO tests passed!")