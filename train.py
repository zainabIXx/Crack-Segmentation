
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np

from dataset import CrackSegmentationDataset, get_training_augmentation, get_validation_augmentation
from models import SegmentationModel, get_model_config


def calculate_metrics(outputs, masks, threshold=0.5):
    """Calculate IoU and Dice"""
    with torch.no_grad():
        preds = (torch.sigmoid(outputs) > threshold).float()
        masks = masks.float()
        
        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum() - intersection
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        dice = (2 * intersection + 1e-7) / (preds.sum() + masks.sum() + 1e-7)
        
    return iou.item(), dice.item()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    losses, ious, dices = [], [], []
    
    for images, masks in tqdm(loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        iou, dice = calculate_metrics(outputs, masks)
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)
    
    return np.mean(losses), np.mean(ious), np.mean(dices)


def validate(model, loader, criterion, device):
    """Validate"""
    model.eval()
    losses, ious, dices = [], [], []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            iou, dice = calculate_metrics(outputs, masks)
            losses.append(loss.item())
            ious.append(iou)
            dices.append(dice)
    
    return np.mean(losses), np.mean(ious), np.mean(dices)


def plot_metrics(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_iou'], 'r-', label='Val')
    axes[1].set_title('IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_dice'], 'r-', label='Val')
    axes[2].set_title('Dice Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    config = {
        'dataset_path': 'crack-1',
        'img_size': 256,
        'batch_size': 4,
        'num_workers': 2,
        'epochs': 10,
        'lr': 1e-4,
        'model_config': 'unet_resnet34',
        'train_split': 0.8,
        'mixed_precision': torch.cuda.is_available(),
        'save_every': 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{config['model_config']}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Loading dataset...")
    full_dataset = CrackSegmentationDataset(
        root_dir=config['dataset_path'],
        split='train',
        transform=get_training_augmentation(config['img_size'])
    )
    
    total = len(full_dataset)
    train_size = int(config['train_split'] * total)
    val_size = total - train_size
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_dataset = Subset(full_dataset, indices[:train_size])
    
    val_dataset_full = CrackSegmentationDataset(
        root_dir=config['dataset_path'],
        split='train',
        transform=get_validation_augmentation(config['img_size'])
    )
    val_dataset = Subset(val_dataset_full, indices[train_size:])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("Creating model...")
    model_cfg = get_model_config(config['model_config'])
    model = SegmentationModel.create_model(
        architecture=model_cfg['architecture'],
        encoder=model_cfg['encoder'],
        encoder_weights='imagenet',
        classes=1,
        activation=None
    ).to(device)
    
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    scaler = GradScaler() if config['mixed_precision'] else None
    
    print(f"\nStarting training for {config['epochs']} epochs...\n")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': []
    }
    best_iou = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_iou)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, f"{exp_dir}/best_model.pth")
            print(f"✓ Best model saved (IoU: {val_iou:.4f})")
        
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, f"{exp_dir}/checkpoint_epoch_{epoch+1}.pth")
    
    plot_metrics(history, f"{exp_dir}/training_curves.png")
    
    with open(f"{exp_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()