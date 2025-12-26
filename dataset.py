"""
dataset.py - Custom Dataset for Crack Segmentation
Handles COCO format annotations from Roboflow
"""

import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as coco_mask


class CrackSegmentationDataset(Dataset):
    """
    Dataset for crack segmentation from COCO format annotations
    """
    def __init__(self, root_dir, split='train', transform=None, img_size=512):
        """
        Args:
            root_dir: Path to dataset (e.g., 'crack-bphdr-1')
            split: 'train', 'valid', or 'test'
            transform: Albumentations transform
            img_size: Image size for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Paths
        self.img_dir = os.path.join(root_dir, split)
        self.anno_file = os.path.join(self.img_dir, '_annotations.coco.json')
        
        # Load annotations
        with open(self.anno_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.img_to_annos = {}
        for anno in self.coco_data['annotations']:
            img_id = anno['image_id']
            if img_id not in self.img_to_annos:
                self.img_to_annos[img_id] = []
            self.img_to_annos[img_id].append(anno)
        
        # Get all images
        self.images = self.coco_data['images']
        
        print(f"{split} set: {len(self.images)} images, "
              f"{len(self.coco_data['annotations'])} annotations")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # Load image
        img_path = os.path.join(self.img_dir, img_filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        if img_id in self.img_to_annos:
            for anno in self.img_to_annos[img_id]:
                # Handle different annotation formats
                if 'segmentation' in anno:
                    seg = anno['segmentation']
                    
                    # Polygon format
                    if isinstance(seg, list):
                        for polygon in seg:
                            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 1)
                    
                    # RLE format
                    elif isinstance(seg, dict):
                        rle = coco_mask.frPyObjects(seg, height, width)
                        m = coco_mask.decode(rle)
                        mask = np.maximum(mask, m)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


def get_training_augmentation(img_size=512):
    """
    Training augmentations for crack segmentation
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Color transforms (important for concrete/road variations)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_validation_augmentation(img_size=512):
    """
    Validation augmentations (only resize and normalize)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# Test the dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test dataset loading
    dataset = CrackSegmentationDataset(
        root_dir='crack-1',
        split='train',
        transform=get_training_augmentation(img_size=512)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a sample
    image, mask = dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {mask.unique()}")
    
    # Denormalize for visualization
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    image_np = np.clip(image_np, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask.numpy(), cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/dataset_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Test visualization saved to results/dataset_test.png")
    plt.show()