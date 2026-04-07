"""
Verification script - Check if everything is ready for training
"""
import sys
from pathlib import Path

def check_item(name, condition, fix_hint=""):
    """Check and print status of an item"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"  # Green or Red
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {name}")
    if not condition and fix_hint:
        print(f"  → {fix_hint}")
    return condition

def main():
    print("="*60)
    print("🔍 Aerial AI - Pre-Training Verification")
    print("="*60)
    print()
    
    all_good = True
    
    # Check Python packages
    print("📦 Required Packages:")
    packages = ['torch', 'transformers', 'PIL', 'cv2', 'numpy', 'dotenv', 'streamlit', 'tqdm']
    for pkg in packages:
        try:
            if pkg == 'PIL':
                from PIL import Image
            elif pkg == 'cv2':
                import cv2
            elif pkg == 'dotenv':
                from dotenv import load_dotenv
            else:
                __import__(pkg)
            check_item(pkg, True)
        except ImportError:
            all_good &= check_item(pkg, False, f"Run: pip install {pkg}")
    
    print()
    
    # Check .env file
    print("🔐 Configuration:")
    env_exists = Path(".env").exists()
    all_good &= check_item(".env file exists", env_exists, "Create .env from .env.example")
    
    if env_exists:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        token = os.getenv("KAGGLE_API_TOKEN")
        has_token = token and token.startswith("KGAT_")
        all_good &= check_item("Kaggle API token set", has_token, "Add KAGGLE_API_TOKEN to .env")
    
    print()
    
    # Check data directories
    print("📂 Data Structure:")
    data_checks = [
        ("data/aerial_segmentation/train/images", "Training images"),
        ("data/aerial_segmentation/train/masks", "Training masks"),
        ("data/aerial_segmentation/val/images", "Validation images"),
        ("data/aerial_segmentation/val/masks", "Validation masks"),
    ]
    
    for path_str, name in data_checks:
        path = Path(path_str)
        exists = path.exists()
        all_good &= check_item(name, exists, f"Run: python organize_data.py --datasets aerial")
    
    print()
    
    # Check image counts
    print("📊 Dataset Statistics:")
    train_img_path = Path("data/aerial_segmentation/train/images")
    val_img_path = Path("data/aerial_segmentation/val/images")
    
    if train_img_path.exists():
        train_count = len(list(train_img_path.glob("*.jpg")))
        has_train = train_count > 0
        all_good &= check_item(f"Training images: {train_count}", has_train, "Run: python organize_data.py --datasets aerial")
    else:
        all_good &= check_item("Training images: 0", False, "Run: python organize_data.py --datasets aerial")
    
    if val_img_path.exists():
        val_count = len(list(val_img_path.glob("*.jpg")))
        has_val = val_count > 0
        all_good &= check_item(f"Validation images: {val_count}", has_val, "Run: python organize_data.py --datasets aerial")
    else:
        all_good &= check_item("Validation images: 0", False, "Run: python organize_data.py --datasets aerial")
    
    print()
    
    # Check output directories
    print("📁 Output Directories:")
    output_dirs = [
        "output",
        "logs",
        "logs/semantic",
        "logs/instance"
    ]
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        check_item(dir_path, path.exists())
    
    print()
    print("="*60)
    
    if all_good:
        print("✅ All checks passed! Ready to train.")
        print()
        print("Next step:")
        print("  python train_semantic.py \\")
        print("    --train_image_dir ./data/aerial_segmentation/train/images \\")
        print("    --train_mask_dir ./data/aerial_segmentation/train/masks \\")
        print("    --val_image_dir ./data/aerial_segmentation/val/images \\")
        print("    --val_mask_dir ./data/aerial_segmentation/val/masks \\")
        print("    --epochs 20 --batch_size 4")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Setup .env: cp .env.example .env (then edit)")
        print("  3. Download data: python download_datasets.py --datasets all")
        print("  4. Organize data: python organize_data.py --datasets aerial")
        sys.exit(1)
    
    print("="*60)

if __name__ == "__main__":
    main()
