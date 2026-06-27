import torch
import os
from PIL import Image, ImageDraw, ImageFont
import random
from pycocotools.coco import COCO
from data.coco_fewshot import NOVEL_CLASSES


def visualize_detection_examples():
    """Generate simple visualization of query images from novel classes."""
    
    # Load COCO
    coco = COCO("data/coco/annotations/instances_val2017.json")
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c['name']: c['id'] for c in cats}

    novel_names = [n for n in NOVEL_CLASSES if n in name_to_id]

    os.makedirs("docs/images", exist_ok=True)

    random.seed(42)

    # Generate 2 example visualizations
    for episode in range(2):
        # Sample 2 classes
        chosen_names = random.sample(novel_names, 2)

        # Create grid: 2 classes × 3 images = 6 images total
        grid_width = 700
        grid_height = 400
        grid = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid)

        try:
            font_title = ImageFont.truetype("arial.ttf", 14)
            font_label = ImageFont.truetype("arial.ttf", 12)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()

        # Title
        title = f"Few-Shot Detection Example {episode + 1}: {', '.join(chosen_names)}"
        draw.text((10, 5), title, fill='black', font=font_title)

        x_offset = 10
        img_width = 100
        img_height = 100

        for cls_idx, cls_name in enumerate(chosen_names):
            cat_id = name_to_id[cls_name]
            img_ids = coco.getImgIds(catIds=[cat_id])
            sampled_ids = random.sample(img_ids, min(3, len(img_ids)))

            y_offset = 40 + cls_idx * 160

            # Draw class label
            draw.text((x_offset, y_offset - 25), f"{cls_name}:", fill='#1976d2', font=font_label)

            # Draw 3 example images
            for i, img_id in enumerate(sampled_ids):
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join("data/coco/val2017", img_info['file_name'])
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # Get bounding box for this class
                    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id]))
                    if anns:
                        x, y, w, h = anns[0]['bbox']
                        # Crop around the object with some padding
                        pad = 10
                        crop_box = (
                            max(0, int(x - pad)), max(0, int(y - pad)),
                            min(img.width, int(x + w + pad)), min(img.height, int(y + h + pad))
                        )
                        crop = img.crop(crop_box)
                    else:
                        crop = img
                    
                    # Resize to thumbnail
                    crop.thumbnail((img_width, img_height))
                    
                    # Paste onto grid
                    grid.paste(crop, (x_offset + i * 120, y_offset))
                except Exception as e:
                    print(f"Skipping image {img_id}: {e}")

        # Save
        grid.save(f"docs/images/detection_example_{episode + 1}.png")
        print(f"✓ Saved docs/images/detection_example_{episode + 1}.png")


if __name__ == "__main__":
    visualize_detection_examples()
    print("\nVisualization complete!")