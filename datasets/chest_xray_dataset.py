import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXrayDataset(Dataset):
    """Chest X-ray lung segmentation dataset."""
    
    def __init__(self, metadata_path, split_path, split='train', transform=None, image_size=(512, 512)):
        """
        Args:
            metadata_path: path to metadata.csv
            split_path: path to split.csv
            split: one of 'train', 'val', or 'test'
            transform: albumentations transform pipeline
            image_size: target image size (H, W)
        """
        self.split = split
        self.image_size = image_size
        
        # Load metadata and split info
        self.metadata = pd.read_csv(metadata_path)
        self.split_info = pd.read_csv(split_path)
        
        # Filter rows for the current split
        split_names = self.split_info[self.split_info['split'] == split]['image_name'].tolist()
        self.data = self.metadata[self.metadata['image_name'].isin(split_names)].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} samples for {split} split")
        
        # Set default transforms if not provided
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform
    
    def _get_default_transform(self, split):
        """Get default augmentation pipeline."""
        if split == 'train':
            # Stronger augmentation for training
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            # Validation & test: resize + normalize only
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Fetch a single sample."""
        row = self.data.iloc[idx]
        
        # Read image & mask
        image = self._load_image(row['image_path'])
        mask = self._load_mask(row['mask_path'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has correct shape/type
        if isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0)  # add channel dim
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'image_name': row['image_name'],
            'original_size': (row['height'], row['width'])
        }
    
    def _load_image(self, image_path):
        """Load image from disk."""
        try:
            image = Image.open(image_path).convert('RGB')  # force 3 channels
            image = np.array(image)
            
            # Convert to grayscale (common for chest X-rays)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback: return a blank image with target size
            return np.zeros(self.image_size, dtype=np.uint8)
    
    def _load_mask(self, mask_path):
        """Load segmentation mask from disk."""
        try:
            mask = Image.open(mask_path).convert('L')  # grayscale
            mask = np.array(mask)
            
            # Binarize mask
            mask = (mask > 127).astype(np.float32)
            
            return mask
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            # Fallback: return a blank mask
            return np.zeros(self.image_size, dtype=np.float32)
    
    def get_positive_weight(self):
        """Compute positive class weight for imbalance handling."""
        total_positive = sum(self.data['positive_pixels'])
        total_pixels = sum(self.data['total_pixels'])
        positive_ratio = total_positive / total_pixels
        
        # Balanced weight
        pos_weight = (1 - positive_ratio) / positive_ratio
        return pos_weight
    
    def get_sample_info(self, idx):
        """Return detailed information of a sample."""
        row = self.data.iloc[idx]
        return {
            'image_name': row['image_name'],
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'size': (row['height'], row['width']),
            'positive_ratio': row['positive_ratio'],
            'positive_pixels': row['positive_pixels']
        }

def create_data_loaders(config, batch_size=None, num_workers=None):
    """Create PyTorch data loaders for train/val/test splits."""
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if num_workers is None:
        num_workers = config.get('num_workers', 4)
    
    # Datasets
    train_dataset = ChestXrayDataset(
        metadata_path=config['data']['metadata_path'],
        split_path=config['data']['split_path'],
        split='train',
        image_size=config['dataset']['image_size']
    )
    
    val_dataset = ChestXrayDataset(
        metadata_path=config['data']['metadata_path'],
        split_path=config['data']['split_path'],
        split='val',
        image_size=config['dataset']['image_size']
    )
    
    test_dataset = ChestXrayDataset(
        metadata_path=config['data']['metadata_path'],
        split_path=config['data']['split_path'],
        split='test',
        image_size=config['dataset']['image_size']
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_positive_weight()

# Convenience visualization helper
def visualize_sample(dataset, idx):
    """Visualize one sample from the dataset."""
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image']
    mask = sample['mask']
    
    # Convert tensors to numpy for display
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:  # grayscale
            image = image.squeeze(0).numpy()
        else:  # RGB
            image = image.permute(1, 2, 0).numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Overlay mask in red
    if len(image.shape) == 2:
        overlay = np.stack([image, image, image], axis=-1)
    else:
        overlay = image.copy()
    overlay[mask > 0.5] = [1, 0, 0]  # red for mask area
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print sample info
    info = dataset.get_sample_info(idx)
    print("Sample Info:")
    print(f"  Name: {info['image_name']}")
    print(f"  Size: {info['size']}")
    print(f"  Positive Ratio: {info['positive_ratio']:.3f}")
    
    return sample
