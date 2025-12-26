import json
import os
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = "./crack-1"  


def explore_dataset():
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(DATASET_PATH, split)
        if os.path.exists(split_path):
            images = [f for f in os.listdir(split_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\n{split.upper()} set: {len(images)} images")
            
            anno_file = os.path.join(split_path, '_annotations.coco.json')
            if os.path.exists(anno_file):
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                print(f"  - Number of annotations: {len(data['annotations'])}")
                print(f"  - Number of categories: {len(data['categories'])}")
                
                print(f"  - Categories:")
                for cat in data['categories']:
                    print(f"    * {cat['name']} (id: {cat['id']})")
    
    print("\n" + "=" * 60)
    print("Sample visualization will appear in a new window...")
    print("=" * 60)
    
    visualize_sample()

def visualize_sample():
    train_path = os.path.join(DATASET_PATH, 'train')
    images = [f for f in os.listdir(train_path) 
             if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        img_path = os.path.join(train_path, images[0])
        img = Image.open(img_path)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Sample Image: {images[0]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/sample_image.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Sample image saved to: results/sample_image.png")
        plt.show()

if __name__ == "__main__":
    explore_dataset()