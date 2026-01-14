import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tqdm import tqdm
import yaml

def scan_data_folders(image_dir, mask_dir):
    """Scan image and mask folders and build paired file lists."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Collect all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    # Collect all mask files
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(list(mask_dir.glob(f'*{ext}')))
        mask_files.extend(list(mask_dir.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(mask_files)} mask files")
    
    # Build filename maps
    image_dict = {f.stem: str(f) for f in image_files}
    mask_dict = {f.stem: str(f) for f in mask_files}
    
    # Match image–mask pairs by stem
    paired_data = []
    for image_name, image_path in image_dict.items():
        if image_name in mask_dict:
            paired_data.append({
                'image_name': image_name,
                'image_path': image_path,
                'mask_path': mask_dict[image_name]
            })
        else:
            print(f"Warning: No mask found for image {image_name}")
    
    print(f"Found {len(paired_data)} paired image–mask files")
    return paired_data

def validate_image_mask_pair(image_path, mask_path):
    """Validate that image and mask have identical spatial size."""
    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        if image.size != mask.size:
            print(f"Size mismatch: {image_path} {image.size} vs {mask_path} {mask.size}")
            return False
        return True
    except Exception as e:
        print(f"Error validating {image_path}: {e}")
        return False

def calculate_mask_stats(mask_path):
    """Compute mask statistics (size, positives, positive ratio)."""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Binarize the mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        total_pixels = mask.shape[0] * mask.shape[1]
        positive_pixels = np.sum(binary_mask > 0)
        positive_ratio = positive_pixels / total_pixels
        
        return {
            'height': mask.shape[0],
            'width': mask.shape[1],
            'total_pixels': total_pixels,
            'positive_pixels': positive_pixels,
            'positive_ratio': positive_ratio
        }
    except Exception as e:
        print(f"Error calculating stats for {mask_path}: {e}")
        return None

def prepare_dataset(image_dir, mask_dir, output_dir="data", test_size=0.3, val_size=0.5, random_seed=42):
    """
    Prepare the dataset and generate metadata.csv and split.csv.
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scanning data folders...")
    paired_data = scan_data_folders(image_dir, mask_dir)
    
    if not paired_data:
        raise ValueError("No paired image–mask data found!")
    
    print("Validating image–mask pairs...")
    valid_data = []
    for data in tqdm(paired_data, desc="Validating"):
        if validate_image_mask_pair(data['image_path'], data['mask_path']):
            # Compute mask statistics
            stats = calculate_mask_stats(data['mask_path'])
            if stats:
                data.update(stats)
                valid_data.append(data)
    
    print(f"Valid pairs: {len(valid_data)}")
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(valid_data)
    
    # Extra dataset stats
    print("\nDataset Statistics:")
    print(f"Total samples: {len(metadata_df)}")
    print(f"Average positive ratio: {metadata_df['positive_ratio'].mean():.3f}")
    print(f"Image size range: {metadata_df['width'].min()}x{metadata_df['height'].min()} "
          f"to {metadata_df['width'].max()}x{metadata_df['height'].max()}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    # Split the dataset
    print("\nSplitting dataset...")
    
    # First, create a test split
    train_val_idx, test_idx = train_test_split(
        range(len(metadata_df)), 
        test_size=test_size, 
        random_state=random_seed,
        stratify=pd.cut(metadata_df['positive_ratio'], bins=5, labels=False)  # stratify by positive_ratio
    )
    
    # Then, split train/val from the remaining data
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_seed,
        stratify=pd.cut(metadata_df.iloc[train_val_idx]['positive_ratio'], bins=5, labels=False)
    )
    
    # Build split DataFrame
    split_data = []
    for idx in train_idx:
        split_data.append({'image_name': metadata_df.iloc[idx]['image_name'], 'split': 'train'})
    for idx in val_idx:
        split_data.append({'image_name': metadata_df.iloc[idx]['image_name'], 'split': 'val'})
    for idx in test_idx:
        split_data.append({'image_name': metadata_df.iloc[idx]['image_name'], 'split': 'test'})
    
    split_df = pd.DataFrame(split_data)
    
    # Save split info
    split_path = os.path.join(output_dir, "split.csv")
    split_df.to_csv(split_path, index=False)
    
    print("Dataset split:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/len(metadata_df)*100:.1f}%)")
    print(f"  Val: {len(val_idx)} samples ({len(val_idx)/len(metadata_df)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} samples ({len(test_idx)/len(metadata_df)*100:.1f}%)")
    print(f"Split info saved to {split_path}")
    
    return metadata_df, split_df

def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Convenience function for calling directly in a notebook
def prepare_data_for_notebook():
    """Convenience function for notebooks."""
    config = load_config('configs/default.yaml')
    
    metadata_df, split_df = prepare_dataset(
        image_dir=config['data']['image_dir'],
        mask_dir=config['data']['mask_dir'],
        output_dir='data',
        test_size=config['split']['test'] + config['split']['val'],
        val_size=config['split']['val'] / (config['split']['test'] + config['split']['val']),
        random_seed=config['split']['random_seed']
    )
    
    return metadata_df, split_df

if __name__ == "__main__":
    # Use default config when running as a script
    prepare_data_for_notebook()

