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
    Organize Solar Plants Brazil dataset from Hugging Face.
    Expected source layout: train|val|test/{input,labels}
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        if logger:
            logger.warning(f"Source path not found: {source_path}")
            logger.info("Skipping solar panels organization")
        return False

    found_any = False
    for split_name in ["train", "val", "test"]:
        input_dir = source_path / split_name / "input"
        label_dir = source_path / split_name / "labels"

        if not input_dir.exists() or not label_dir.exists():
            continue

        found_any = True
        image_out = target_path / split_name / "images"
        mask_out = target_path / split_name / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            list(input_dir.glob("*.tif")) +
            list(input_dir.glob("*.tiff")) +
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.jpeg"))
        )

        copied = 0
        for image_file in tqdm(image_files, desc=f"Copying {split_name} images"):
            mask_candidates = [
                label_dir / image_file.name.replace("img", "target", 1),
                label_dir / image_file.with_suffix(".tif").name.replace("img", "target", 1),
                label_dir / image_file.with_suffix(".png").name.replace("img", "target", 1),
                label_dir / image_file.name,
            ]
            mask_file = next((candidate for candidate in mask_candidates if candidate.exists()), None)
            if mask_file is None:
                continue

            shutil.copy2(image_file, image_out / image_file.name)
            shutil.copy2(mask_file, mask_out / mask_file.name)
            copied += 1

        if logger:
            logger.info(f"{split_name.title()} pairs copied: {copied}")

    if not found_any:
        if logger:
            logger.warning("Could not find train/val/test input-label folders in the solar dataset")
        return False

    if logger:
        logger.info(f"✓ Solar panel dataset organized at {target_path}")
        logger.info("  Expected layout: train|val|test/{images,masks}")

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

        source = data_path / "raw" / "solar_plants_brazil"
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
