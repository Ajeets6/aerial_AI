#!/bin/bash
# Download all datasets for aerial segmentation training

echo "=========================================="
echo "Aerial AI - Dataset Download Script"
echo "=========================================="

# Configuration
DATA_DIR="./data"
LOG_DIR="./logs/downloads"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo ""
echo "This script will download the following datasets:"
echo "  1. Kaggle Aerial Segmentation (~2GB)"
echo "  2. Solar Antwerp (GitHub) (~500MB)"
echo "  3. IndiaSat Dataset (~1GB)"
echo ""
echo "Total download size: ~3.5GB"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

# Install dependencies
echo ""
echo "Installing required packages..."
python3 download_datasets.py --install_deps --skip_kaggle_check --datasets aerial 2>/dev/null || true

# Check for Kaggle credentials
echo ""
echo "Checking Kaggle credentials..."

# Check for .env file first
if [ -f ".env" ]; then
    if grep -q "KAGGLE_API_TOKEN" .env; then
        echo "✓ Found API token in .env file"
    else
        echo "⚠️  .env file exists but no KAGGLE_API_TOKEN found"
        echo ""
        echo "Please add your Kaggle API token to .env:"
        echo "  KAGGLE_API_TOKEN=KGAT_your_token_here"
        echo ""
        echo "Get your token from: https://www.kaggle.com/settings/account"
        exit 1
    fi
elif [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "⚠️  Kaggle credentials not found!"
    echo ""
    echo "Please use one of these methods:"
    echo ""
    echo "Method 1 (Recommended): Create .env file with API token"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Copy the token and add to .env file:"
    echo "     KAGGLE_API_TOKEN=KGAT_your_token_here"
    echo ""
    echo "Method 2: Use kaggle.json"
    echo "  1. Download kaggle.json from Kaggle settings"
    echo "  2. Move to ~/.kaggle/kaggle.json"
    echo "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Download datasets
echo ""
echo "Starting download..."
echo ""

python3 download_datasets.py \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --datasets all

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Download Complete!"
    echo "=========================================="
    echo ""
    echo "Datasets saved to: $DATA_DIR"
    echo "Logs saved to: $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check the downloaded data in $DATA_DIR"
    echo "  2. Review TRAINING_GUIDE.md for dataset organization"
    echo "  3. Run ./train_all.sh to start training"
else
    echo ""
    echo "=========================================="
    echo "✗ Download failed"
    echo "=========================================="
    echo "Check logs in $LOG_DIR for details"
    exit 1
fi
