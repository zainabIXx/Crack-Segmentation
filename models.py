"""
models.py - Model Architectures for Crack Segmentation
Supports: UNet, FPN, PSPNet with various encoders
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationModel:
    """
    Factory class for creating segmentation models
    """
    
    AVAILABLE_ARCHITECTURES = ['unet', 'fpn', 'pspnet', 'unetplusplus']
    AVAILABLE_ENCODERS = [
        'resnet34', 'resnet50', 'resnet101',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4',
        'resnext50_32x4d', 'densenet121', 'vgg16'
    ]
    
    @staticmethod
    def create_model(
        architecture='unet',
        encoder='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    ):
        """
        Create a segmentation model
        
        Args:
            architecture: Model architecture (unet, fpn, pspnet)
            encoder: Encoder backbone
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation (None, 'sigmoid', 'softmax')
        
        Returns:
            PyTorch model
        """
        architecture = architecture.lower()
        
        if architecture not in SegmentationModel.AVAILABLE_ARCHITECTURES:
            raise ValueError(
                f"Architecture '{architecture}' not supported. "
                f"Choose from: {SegmentationModel.AVAILABLE_ARCHITECTURES}"
            )
        
        # Create model based on architecture
        if architecture == 'unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        
        elif architecture == 'fpn':
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        
        elif architecture == 'pspnet':
            model = smp.PSPNet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        
        elif architecture == 'unetplusplus':
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        
        return model
    
    @staticmethod
    def print_model_info(model, input_size=(3, 512, 512)):
        """
        Print model information
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Test forward pass
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("="*60 + "\n")


# Model configurations for different experiments
MODEL_CONFIGS = {
    # Lightweight models (faster training)
    'unet_resnet34': {
        'architecture': 'unet',
        'encoder': 'resnet34',
        'description': 'UNet with ResNet34 - Good balance of speed and accuracy'
    },
    'unet_efficientnet_b0': {
        'architecture': 'unet',
        'encoder': 'efficientnet-b0',
        'description': 'UNet with EfficientNet-B0 - Most efficient'
    },
    
    # Medium models (better accuracy)
    'unet_resnet50': {
        'architecture': 'unet',
        'encoder': 'resnet50',
        'description': 'UNet with ResNet50 - Better feature extraction'
    },
    'fpn_resnet34': {
        'architecture': 'fpn',
        'encoder': 'resnet34',
        'description': 'FPN with ResNet34 - Multi-scale features'
    },
    'pspnet_resnet34': {
        'architecture': 'pspnet',
        'encoder': 'resnet34',
        'description': 'PSPNet with ResNet34 - Pyramid pooling'
    },
    
    # Heavy models (best accuracy)
    'unet_efficientnet_b3': {
        'architecture': 'unet',
        'encoder': 'efficientnet-b3',
        'description': 'UNet with EfficientNet-B3 - Stronger encoder'
    },
    'fpn_resnet50': {
        'architecture': 'fpn',
        'encoder': 'resnet50',
        'description': 'FPN with ResNet50 - Best multi-scale'
    },
    'pspnet_resnet50': {
        'architecture': 'pspnet',
        'encoder': 'resnet50',
        'description': 'PSPNet with ResNet50 - Best context aggregation'
    }
}


def get_model_config(config_name):
    """
    Get model configuration by name
    """
    if config_name not in MODEL_CONFIGS:
        print(f"\nAvailable model configurations:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  - {name}: {config['description']}")
        raise ValueError(f"Config '{config_name}' not found")
    
    return MODEL_CONFIGS[config_name]


# Test models
if __name__ == "__main__":
    print("Testing model creation...\n")
    
    # Test different architectures
    configs_to_test = ['unet_resnet34', 'fpn_resnet34', 'pspnet_resnet34']
    
    for config_name in configs_to_test:
        config = get_model_config(config_name)
        print(f"\n{'='*60}")
        print(f"Creating: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        model = SegmentationModel.create_model(
            architecture=config['architecture'],
            encoder=config['encoder'],
            encoder_weights='imagenet',
            classes=1,
            activation=None  # We'll use BCEWithLogitsLoss
        )
        
        SegmentationModel.print_model_info(model)
    
    print("\n✓ All models created successfully!")
    
    # Print available configurations
    print("\n" + "="*60)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("="*60)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Architecture: {config['architecture'].upper()}")
        print(f"  Encoder: {config['encoder']}")
        print(f"  Description: {config['description']}")
    print("="*60)