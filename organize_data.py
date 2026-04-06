"""
Organize downloaded datasets into training structure
"""
import argparse
import logging
import shutil
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def organize_aerial_segmentation(source_dir, target_dir, train_ratio=0.8, logger=None):
    """
    Organize Kaggle Aerial Segmentation dataset
    Source structure: data/raw/aerial_segmentation/Semantic segmentation dataset/Tile X/images|masks/
    Target structure: data/aerial_segmentation/train|val/images|masks/
    """
    source_path = Path(source_dir) / "Semantic segmentation dataset"
    target_path = Path(target_dir)
    
    if not source_path.exists():
        if logger:
            logger.error(f"Source path not found: {source_path}")
        return False
    
    # Create target directories
    (target_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (target_path / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (target_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (target_path / "val" / "masks").mkdir(parents=True, exist_ok=True)
    
    # Collect all images from all tiles with unique naming
    all_images = []
    for tile_dir in sorted(source_path.glob("Tile *")):
        tile_num = tile_dir.name.split()[-1]  # Get tile number
        images_dir = tile_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                mask_file = tile_dir / "masks" / img_file.with_suffix('.png').name
                if mask_file.exists():
                    # Create unique names with tile prefix
                    new_name = f"tile{tile_num}_{img_file.stem}"
                    all_images.append((img_file, mask_file, new_name))
    
    if not all_images:
        if logger:
            logger.error("No image-mask pairs found!")
        return False
    
    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    
    train_pairs = all_images[:split_idx]
    val_pairs = all_images[split_idx:]
    
    if logger:
        logger.info(f"Found {len(all_images)} image-mask pairs across {len(list(source_path.glob('Tile *')))} tiles")
        logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
        logger.info("Copying files...")
    
    # Copy training files with unique names
    for img, mask, new_name in tqdm(train_pairs, desc="Copying train data"):
        new_img_name = f"{new_name}.jpg"
        new_mask_name = f"{new_name}.png"
        shutil.copy2(img, target_path / "train" / "images" / new_img_name)
        shutil.copy2(mask, target_path / "train" / "masks" / new_mask_name)
    
    # Copy validation files with unique names
    for img, mask, new_name in tqdm(val_pairs, desc="Copying val data"):
        new_img_name = f"{new_name}.jpg"
        new_mask_name = f"{new_name}.png"
        shutil.copy2(img, target_path / "val" / "images" / new_img_name)
        shutil.copy2(mask, target_path / "val" / "masks" / new_mask_name)
    
    if logger:
        logger.info(f"✓ Aerial segmentation organized at {target_path}")
        logger.info(f"  Train: {len(train_pairs)} pairs")
        logger.info(f"  Val: {len(val_pairs)} pairs")
    
    return True

def organize_solar_panels(source_dir, target_dir, train_ratio=0.8, logger=None):
    """
    Organize Solar Antwerp dataset from GitHub
    Note: This is a placeholder - actual structure depends on the repo
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        if logger:
            logger.warning(f"Source path not found: {source_path}")
            logger.info("Skipping solar panels organization")
        return False
    
    # Create target directories
    (target_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (target_path / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    (target_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (target_path / "val" / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Check various possible structures
    possible_image_dirs = [
        source_path / "images",
        source_path / "data" / "images",
        source_path / "orthophotos",
    ]
    
    image_dir = None
    for dir_path in possible_image_dirs:
        if dir_path.exists():
            image_dir = dir_path
            break
    
    if not image_dir:
        if logger:
            logger.warning("Could not find images directory in solar_antwerp")
            logger.info("Please manually organize this dataset")
        return False
    
    # Look for images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.tif"))
    
    if not image_files:
        if logger:
            logger.warning("No images found in solar_antwerp")
        return False
    
    # Simple split (assumes masks/annotations are in similar structure)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    if logger:
        logger.info(f"Found {len(image_files)} solar panel images")
        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
        logger.info("Note: You may need to manually organize annotations")
    
    # Copy images (annotations need manual organization)
    for img in tqdm(train_images, desc="Copying train images"):
        shutil.copy2(img, target_path / "train" / "images" / img.name)
    
    for img in tqdm(val_images, desc="Copying val images"):
        shutil.copy2(img, target_path / "val" / "images" / img.name)
    
    if logger:
        logger.info(f"✓ Solar panels images organized at {target_path}")
        logger.info("⚠ Please manually organize annotations if needed")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Organize downloaded datasets for training"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root data directory"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs='+',
        choices=['aerial', 'solar', 'all'],
        default=['all'],
        help="Which datasets to organize"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Set random seed
    random.seed(args.seed)
    
    logger.info("="*80)
    logger.info("Dataset Organization Tool")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Train ratio: {args.train_ratio}")
    logger.info(f"Random seed: {args.seed}")
    
    data_path = Path(args.data_dir)
    results = {}
    
    datasets_to_organize = args.datasets
    if 'all' in datasets_to_organize:
        datasets_to_organize = ['aerial', 'solar']
    
    # Organize aerial segmentation
    if 'aerial' in datasets_to_organize:
        logger.info("\n" + "="*80)
        logger.info("Organizing Aerial Segmentation Dataset")
        logger.info("="*80)
        
        source = data_path / "raw" / "aerial_segmentation"
        target = data_path / "aerial_segmentation"
        
        results['aerial'] = organize_aerial_segmentation(
            source, target, args.train_ratio, logger
        )
    
    # Organize solar panels
    if 'solar' in datasets_to_organize:
        logger.info("\n" + "="*80)
        logger.info("Organizing Solar Panels Dataset")
        logger.info("="*80)
        
        source = data_path / "raw" / "solar_antwerp"
        target = data_path / "solar_panels"
        
        results['solar'] = organize_solar_panels(
            source, target, args.train_ratio, logger
        )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Organization Summary")
    logger.info("="*80)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset:20} {status}")
    
    logger.info("\n" + "="*80)
    logger.info("Next Steps:")
    logger.info("="*80)
    logger.info("1. Verify organized data structure:")
    logger.info("   - data/aerial_segmentation/train/{images,masks}")
    logger.info("   - data/aerial_segmentation/val/{images,masks}")
    logger.info("2. Update train_all.sh/bat if needed")
    logger.info("3. Run training: ./train_all.sh or train_all.bat")

if __name__ == "__main__":
    main()
