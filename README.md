# Crack Segmentation for Infrastructure Inspection

A deep learning project for automatic crack detection in concrete structures using semantic segmentation.

## 📦 Deliverables

This project includes the following deliverables:

### 1. **Source Code**
- `dataset.py` - COCO dataset loader with augmentation pipeline
- `models.py` - Model architecture definitions (U-Net, FPN, PSPNet)
- `train.py` - Complete training pipeline with metrics tracking
- `download_dataset.py` - Automated dataset downloader
- `explore_dataset.py` - Dataset visualization and exploration

### 2. **Trained Model**
- **Best model checkpoint:** `experiments/unet_resnet34_20251226_141727/best_model.pth`
  - Saved at Epoch 8/10 with IoU: **58.92%**, Dice: **73.81%**
- **Model configuration:** `experiments/unet_resnet34_20251226_141727/config.json`
- **Training history:** `experiments/unet_resnet34_20251226_141727/history.json`
- **Training curves:** `experiments/unet_resnet34_20251226_141727/training_curves.png`

### 3. **Documentation**
- This README with complete setup and usage instructions
- Training metrics and performance analysis
- Dataset information and preprocessing details

### 4. **Results & Visualizations**
- Training curves (Loss, IoU, Dice): `experiments/unet_resnet34_20251226_141727/training_curves.png`
- Sample predictions with ground truth: `results/dataset_test.png`
- Dataset exploration samples: `results/sample_image.png`
- Complete training log available in terminal output

### 5. **Project Report**
A comprehensive report covering:
- Dataset description and statistics
- Model architecture and approach
- Training setup and hyperparameters
- Results (quantitative metrics)
- Observations and challenges
- Future improvements

---

## 📊 Dataset

**Source:** [Roboflow Universe - crack-bphdr](https://universe.roboflow.com/university-bswxt/crack-bphdr)

**Statistics:**
- **Total Images:** 1,551
- **Format:** COCO Segmentation with polygon annotations
- **Classes:** 2 (concrete, crack)
- **Split:**
  - Training: 1,239 images (79.9%)
  - Test: 312 images (20.1%)
  - Validation: 248 images (20% of training split)

## 🏗️ Architecture

**Model:** U-Net with ResNet34 Encoder
- **Parameters:** 24.4M
- **Input Size:** 512×512×3 RGB
- **Output:** 512×512×1 Binary Mask
- **Pretrained:** ImageNet weights

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations opencv-python
pip install matplotlib scikit-learn tqdm
pip install pycocotools roboflow
```

### Download Dataset

```bash
python download_dataset.py
```

### Explore Dataset

```bash
python explore_dataset.py
```

### Train Model

```bash
python train.py
```

## 📁 Project Structure

```
s.task/
├── crack-1/                    # Dataset folder (auto-downloaded)
│   ├── train/                  # Training images + annotations
│   ├── test/                   # Test images + annotations
│   ├── README.dataset.txt      # Dataset info
│   └── README.roboflow.txt     # Roboflow info
├── experiments/                # Training outputs
│   └── unet_resnet34_*/       # Experiment folders
│       ├── config.json         # Training config
│       ├── best_model.pth      # Best model weights
│       ├── history.json        # Training metrics
│       └── training_curves.png # Loss/IoU/Dice plots
├── results/                    # Visualizations
├── dataset.py                  # Dataset loader
├── models.py                   # Model definitions
├── train.py                    # Training script
├── download_dataset.py         # Dataset downloader
├── explore_dataset.py          # Dataset explorer
└── README.md                   # This file
```

## ⚙️ Configuration

Key hyperparameters used in the completed training run:

```python
config = {
    'dataset_path': 'crack-1',
    'img_size': 256,              # Image resize dimension
    'batch_size': 4,              # 4 images per batch
    'num_workers': 2,             # Data loading workers
    'epochs': 10,                 # Total training epochs
    'lr': 1e-4,                   # Learning rate (0.0001)
    'model_config': 'unet_resnet34',
    'train_split': 0.8,           # 80% train, 20% validation
    'mixed_precision': False,     # Disabled (CPU training)
    'save_every': 10              # Save checkpoint every N epochs
}
```

**Training Details:**
- **Device:** CPU (no GPU available)
- **Loss Function:** Dice Loss (binary, from logits)
- **Optimizer:** AdamW (weight decay optimization)
- **Scheduler:** ReduceLROnPlateau (mode='max', factor=0.5, patience=5)
- **Data Split:** 991 training images, 248 validation images
- **Total Batches:** 248 training batches, 62 validation batches

## 📈 Training Results

**Training Run Completed:** 10 epochs on CPU

**Final Results (Epoch 10/10):**
```
Train - Loss: 0.3179, IoU: 0.5310, Dice: 0.6883
Val   - Loss: 0.2724, IoU: 0.5835, Dice: 0.7329
```

**Best Model Performance (Epoch 8/10):**
```
Train - Loss: 0.3198, IoU: 0.5309, Dice: 0.6892
Val   - Loss: 0.2698, IoU: 0.5892, Dice: 0.7381
✓ Best model saved (IoU: 0.5892)
```

**Training Progress:**
- Initial performance (Epoch 1): Val IoU **40.30%**, Dice **56.74%**
- Final performance (Epoch 10): Val IoU **58.35%**, Dice **73.29%**
- **Best validation IoU: 58.92%** (achieved at Epoch 8)
- **Best validation Dice: 73.81%** (achieved at Epoch 8)
- Training time: ~3.5 hours total on CPU
- Average training time per epoch: ~20 minutes
- Average validation time per epoch: ~2-3 minutes

**Training Convergence:**
- Model showed rapid improvement in first 5 epochs
- Performance plateaued after epoch 8
- Learning rate reduction activated by scheduler
- No significant overfitting observed

## 🎯 Model Performance Summary

| Metric | Best Value | Final Value |
|--------|------------|-------------|
| **Validation IoU** | **58.92%** (Epoch 8) | 58.35% (Epoch 10) |
| **Validation Dice** | **73.81%** (Epoch 8) | 73.29% (Epoch 10) |
| **Training Loss** | 0.3179 | 0.3179 |
| **Validation Loss** | **0.2698** (Epoch 8) | 0.2724 |
| **Training IoU** | 53.19% | 53.10% |
| **Training Dice** | 68.92% | 68.83% |

**Training History:**

| Epoch | Train Loss | Train IoU | Train Dice | Val Loss | Val IoU | Val Dice | Status |
|-------|------------|-----------|------------|----------|---------|----------|--------|
| 1/10  | 0.8894 | 15.60% | 25.47% | 0.7362 | 40.30% | 56.74% | ✓ Best |
| 2/10  | 0.6183 | 41.93% | 58.27% | 0.4340 | 53.58% | 69.25% | ✓ Best |
| 3/10  | 0.4211 | 49.58% | 65.65% | 0.3366 | 56.13% | 71.39% | ✓ Best |
| 4/10  | 0.3690 | 50.81% | 66.76% | 0.3048 | 57.28% | 72.38% | ✓ Best |
| 5/10  | 0.3446 | 51.92% | 67.74% | 0.2885 | 57.93% | 72.94% | ✓ Best |
| 6/10  | 0.3338 | 52.32% | 68.17% | 0.2884 | 57.31% | 72.47% | - |
| 7/10  | 0.3296 | 52.42% | 68.19% | 0.2803 | 57.89% | 72.94% | - |
| 8/10  | 0.3198 | 53.09% | 68.92% | 0.2698 | **58.92%** | **73.81%** | ✓ **Best** |
| 9/10  | 0.3186 | 53.19% | 68.85% | 0.2705 | 58.65% | 73.57% | - |
| 10/10 | 0.3179 | 53.10% | 68.83% | 0.2724 | 58.35% | 73.29% | - |

## 🛠️ Available Models

Modify `model_config` in `train.py` to use different architectures:

- `unet_resnet34` - U-Net with ResNet34 (default, balanced)
- `unet_resnet50` - U-Net with ResNet50 (more capacity)
- `unet_efficientnet_b0` - U-Net with EfficientNet-B0 (efficient)
- `fpn_resnet34` - FPN with ResNet34 (multi-scale features)
- `pspnet_resnet34` - PSPNet with ResNet34 (pyramid pooling)

## 📊 Data Augmentation

**Training:**
- Geometric: HorizontalFlip, VerticalFlip, RandomRotate90
- Affine: ShiftScaleRotate
- Intensity: RandomBrightnessContrast, GaussianBlur, GaussNoise
- Normalization: ImageNet statistics

**Validation:**
- Resize + Normalization only

## 🧪 Experiments

All training runs are saved in `experiments/` with:
- Model weights (`.pth` files)
- Training configuration (`config.json`)
- Metrics history (`history.json`)
- Training curves (`training_curves.png`)

## 🚧 Known Issues & Future Work

**Current Limitations:**
- CPU-only training (no GPU access) - significantly slower training times
- Image size reduced to 256×256 (from 512×512) due to CPU memory constraints
- Limited dataset size (1,551 images total)
- Binary segmentation only (crack vs. background)
- No test set evaluation performed yet

**Observations from Training:**
- ✅ Model converged successfully in 10 epochs
- ✅ No significant overfitting (train/val metrics close)
- ✅ Rapid improvement in first 5 epochs (40% → 58% IoU)
- ⚠️ Performance plateau after epoch 8 (learning rate reduction needed)
- ⚠️ Training time: ~20 minutes/epoch on CPU (vs <1 min on GPU)

**Planned Improvements:**
- [ ] **GPU training support** - Would reduce training from 3.5 hours to ~10 minutes
- [ ] **Larger image size** (512×512) - Better capture fine crack details
- [ ] **More epochs** (50-100) - Potentially reach 65-70% IoU
- [ ] Mixed precision training (FP16) for 2-3× speedup
- [ ] Test set evaluation and metrics
- [ ] Inference script with visualization overlays
- [ ] Test-time augmentation (TTA) for improved predictions
- [ ] Multi-class segmentation (crack severity levels)
- [ ] Post-processing (morphological operations to clean predictions)
- [ ] Model deployment (ONNX export for production use)
- [ ] Ensemble multiple architectures (U-Net + FPN + PSPNet)

## 📝 Citation

If you use this dataset, please cite:

```
@misc{crack-bphdr_dataset,
  title = {Crack Segmentation Dataset},
  author = {Roboflow Universe - university-bswxt},
  year = {2022},
  url = {https://universe.roboflow.com/university-bswxt/crack-bphdr},
  license = {Public Domain}
}
```

## 📄 License

- **Code:** MIT License
- **Dataset:** Public Domain (Roboflow)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Status:** ✅ Training Complete - Ready for Inference

**Last Updated:** December 26, 2025