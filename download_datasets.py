"""
Download and prepare datasets for aerial image segmentation training
"""
import argparse
import logging
import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import json
from dotenv import load_dotenv

# Setup logging
def setup_logging(log_dir):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_datasets_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up"""
    # First, try to load from .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loading credentials from {env_path}")

    # Check for new API token (preferred method)
    api_token = os.getenv("KAGGLE_API_TOKEN")
    if api_token:
        logging.info("✓ Kaggle API token found in environment")
        # Set the token in environment for kaggle library
        os.environ["KAGGLE_API_TOKEN"] = api_token
        return True

    # Fall back to checking kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        logging.error("Kaggle credentials not found!")
        logging.error("Please use one of these methods:")
        logging.error("Method 1 (Recommended): Add KAGGLE_API_TOKEN to .env file")
        logging.error("Method 2: Save kaggle.json to ~/.kaggle/kaggle.json")
        logging.error("")
        logging.error("To get API token:")
        logging.error("1. Go to https://www.kaggle.com/settings/account")
        logging.error("2. Click 'Create New API Token'")
        logging.error("3. Add token to .env: KAGGLE_API_TOKEN=your_token_here")
        return False

    logging.info("✓ Kaggle credentials found in kaggle.json")
    return True

def install_dependencies(logger):
    """Install required packages for downloading"""
    logger.info("Checking required packages...")

    required_packages = ['kaggle', 'gdown', 'requests', 'python-dotenv', 'huggingface_hub', 'tifffile']

    for package in required_packages:
        try:
            if package == 'python-dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✓ {package} installed successfully")

def download_kaggle_dataset(dataset_name, output_dir, logger):
    """Download dataset from Kaggle"""
    try:
        import kaggle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        logger.info(f"Output directory: {output_path}")

        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_path),
            unzip=True
        )

        logger.info(f"✓ Successfully downloaded {dataset_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {e}")
        return False

def download_github_repo(repo_url, output_dir, logger):
    """Download repository from GitHub"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cloning GitHub repository: {repo_url}")
        logger.info(f"Output directory: {output_path}")

        # Clone repository
        subprocess.check_call([
            "git", "clone", repo_url, str(output_path)
        ])

        logger.info(f"✓ Successfully cloned repository")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to clone repository: {e}")
        return False

def download_from_url(url, output_file, logger):
    """Download file from URL"""
    try:
        import requests

        logger.info(f"Downloading from: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                print()  # New line after download

        logger.info(f"✓ Successfully downloaded to {output_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download: {e}")
        return False

def extract_archive(archive_path, output_dir, logger):
    """Extract zip or tar archive"""
    try:
        archive_path = Path(archive_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {archive_path.name}...")

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(output_path)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False

        logger.info(f"✓ Extracted to {output_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to extract: {e}")
        return False

def organize_dataset(source_dir, target_dir, train_ratio=0.8, logger=None):
    """Organize dataset into train/val splits"""
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        if logger:
            logger.info(f"Organizing dataset from {source_path} to {target_path}")

        # Create target directories
        (target_path / "train" / "images").mkdir(parents=True, exist_ok=True)
        (target_path / "train" / "masks").mkdir(parents=True, exist_ok=True)
        (target_path / "val" / "images").mkdir(parents=True, exist_ok=True)
        (target_path / "val" / "masks").mkdir(parents=True, exist_ok=True)

        # This is a placeholder - actual organization depends on dataset structure
        # You'll need to customize this based on how each dataset is organized

        if logger:
            logger.info(f"✓ Created directory structure at {target_path}")

        return True

    except Exception as e:
        if logger:
            logger.error(f"✗ Failed to organize dataset: {e}")
        return False

def download_aerial_segmentation(data_dir, logger):
    """Download Kaggle Aerial Segmentation dataset"""
    logger.info("\n" + "="*80)
    logger.info("Dataset 1/3: Kaggle Aerial Segmentation")
    logger.info("="*80)

    dataset_name = "humansintheloop/semantic-segmentation-of-aerial-imagery"
    output_dir = Path(data_dir) / "raw" / "aerial_segmentation"

    success = download_kaggle_dataset(dataset_name, output_dir, logger)

    if success:
        # Organize into train/val
        target_dir = Path(data_dir) / "aerial_segmentation"
        organize_dataset(output_dir, target_dir, logger=logger)

        logger.info(f"Dataset location: {target_dir}")
        logger.info("Note: You may need to manually organize images/masks into train/val folders")

    return success

def download_solar_panels(data_dir, logger):
    """Download Solar Plants Brazil dataset from Hugging Face"""
    logger.info("\n" + "="*80)
    logger.info("Dataset 2/3: Solar Panels (Solar Plants Brazil)")
    logger.info("="*80)

    repo_id = "FederCO23/solar-plants-brazil"
    output_dir = Path(data_dir) / "raw" / "solar_plants_brazil"

    try:
        from huggingface_hub import snapshot_download

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading dataset from: https://huggingface.co/datasets/{repo_id}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"✓ Successfully downloaded {repo_id}")
        success = True
    except Exception as e:
        logger.error(f"✗ Failed to download {repo_id}: {e}")
        success = False

    if success:
        target_dir = Path(data_dir) / "solar_panels"
        logger.info(f"Dataset location: {target_dir}")
        logger.info("Note: Run organize_data.py --datasets solar to copy train/val TIFFs into the expected layout")

    return success

def download_indian_demo(data_dir, logger):
    """Download the Indian demo dataset"""
    logger.info("\n" + "="*80)
    logger.info("Dataset 3/3: Indian Demo")
    logger.info("="*80)

    dataset_url = "https://data.mendeley.com/public-files/datasets/xj2v49zt26/files/caf935d8-ef3d-42c0-a7da-0ccc85f10669/file_downloaded"
    output_dir = Path(data_dir) / "raw" / "indian_demo"

    success = download_from_url(dataset_url, output_dir / "indian_demo_dataset", logger)

    if success:
        target_dir = Path(data_dir) / "indian_demo"
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset location: {target_dir}")
        logger.info("Note: This dataset is for demo purposes")

    return success

def download_indiasat(data_dir, logger):
    """Backward-compatible alias for the Indian demo dataset"""
    return download_indian_demo(data_dir, logger)

def create_dataset_info(data_dir, logger):
    """Create dataset info file"""
    info = {
        "download_date": datetime.now().isoformat(),
        "datasets": {
            "aerial_segmentation": {
                "source": "Kaggle - humansintheloop/semantic-segmentation-of-aerial-imagery",
                "purpose": "Semantic segmentation training (buildings, roads, water)",
                "location": str(Path(data_dir) / "aerial_segmentation")
            },
            "solar_panels": {
                "source": "Hugging Face - FederCO23/solar-plants-brazil",
                "purpose": "Instance segmentation training (solar panels)",
                "location": str(Path(data_dir) / "solar_panels")
            },
            "indian_demo": {
                "source": "Mendeley - xj2v49zt26",
                "purpose": "Demo dataset for Indian aerial images",
                "location": str(Path(data_dir) / "indian_demo")
            }
        },
        "structure": {
            "semantic": "data/aerial_segmentation/{train,val}/{images,masks}",
            "instance": "data/solar_panels/{train,val}/{images,masks}"
        }
    }

    info_file = Path(data_dir) / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"\n✓ Dataset info saved to {info_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare aerial segmentation datasets"
    )

    # Arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for datasets (default: ./data)"
    )

    parser.add_argument(
        "--datasets",
        nargs='+',
        choices=['aerial', 'solar', 'solar_plants_brazil', 'india', 'indian_demo', 'all'],
        default=['all'],
        help="Which datasets to download (default: all)"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/downloads",
        help="Directory for log files"
    )

    parser.add_argument(
        "--skip_kaggle_check",
        action="store_true",
        help="Skip Kaggle credentials check"
    )

    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Install required dependencies"
    )

    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file with API tokens (default: .env)"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)

    # Setup logging
    logger = setup_logging(args.log_dir)

    logger.info("="*80)
    logger.info("Aerial AI - Dataset Download Tool")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Datasets to download: {args.datasets}")

    # Install dependencies if requested
    if args.install_deps:
        install_dependencies(logger)

    # Check Kaggle credentials
    if not args.skip_kaggle_check and ('all' in args.datasets or 'aerial' in args.datasets):
        if not check_kaggle_credentials():
            logger.error("\nPlease set up Kaggle credentials and try again")
            logger.error("Use --skip_kaggle_check to skip this check")
            sys.exit(1)

    # Create data directory
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Download datasets
    results = {}

    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['aerial', 'solar_plants_brazil', 'indian_demo']

    if 'aerial' in datasets_to_download:
        results['aerial'] = download_aerial_segmentation(args.data_dir, logger)

    if 'solar' in datasets_to_download or 'solar_plants_brazil' in datasets_to_download:
        results['solar'] = download_solar_panels(args.data_dir, logger)

    if 'india' in datasets_to_download or 'indian_demo' in datasets_to_download:
        results['indian_demo'] = download_indian_demo(args.data_dir, logger)

    # Create dataset info
    create_dataset_info(args.data_dir, logger)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Download Summary")
    logger.info("="*80)

    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset:20} {status}")

    logger.info("\n" + "="*80)
    logger.info("Next Steps:")
    logger.info("="*80)
    logger.info("1. Check downloaded datasets in ./data/")
    logger.info("2. Manually organize images/masks into train/val if needed")
    logger.info("3. Update paths in train_all.sh or train_all.bat")
    logger.info("4. Run training: ./train_all.sh or train_all.bat")
    logger.info("\nFor detailed organization instructions, see TRAINING_GUIDE.md")

if __name__ == "__main__":
    main()
