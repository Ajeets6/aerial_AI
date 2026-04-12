# 🛰️ Aerial Image Multi-Segmenter

A Streamlit dashboard for segmenting aerial images with both semantic and instance segmentation capabilities.

## Features

- **Semantic Segmentation**: Buildings, Roads, Water
- **Instance Segmentation**: Solar Panels
- Interactive dropdown to highlight specific features
- Real-time statistics (area, count)

## Models

| Task | Model | Source |
|------|-------|--------|
| Semantic | `nvidia/segformer-b1-finetuned-cityscapes-1024-1024` | Hugging Face |
| Instance | `facebook/mask2former-swin-base-coco-instance` | Hugging Face |

> **Note**: Current models are pretrained on Cityscapes/COCO. For optimal results, fine-tune on aerial datasets.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd aerial_AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kaggle API (for datasets)

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Kaggle API token
KAGGLE_API_TOKEN=KGAT_your_token_here
```

**Get your token**: https://www.kaggle.com/settings/account → "Create New API Token"

### 3. Download & Organize Datasets (for training)

```bash
# Download datasets
./download_datasets.sh      # Linux/Mac
download_datasets.bat        # Windows

# Organize data for training
python organize_data.py --datasets aerial
```

This will:
- Download ~3.5GB of aerial imagery datasets
- Organize into train/val splits (80/20)
- Create properly structured folders

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. **Upload Image**: Click "Upload Aerial Image" and select your aerial/satellite image
2. **Select Feature**: Choose from dropdown (Buildings, Roads, Water, or Solar Panels)
3. **View Results**: See original image alongside highlighted features
4. **Check Statistics**: View area calculations and detection counts

## Datasets (for Fine-tuning)

| Purpose | Dataset | Classes |
|---------|---------|---------|
| Semantic training | [Kaggle Aerial Segmentation](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) | building, road, water |
| Instance training | [Solar Plants Brazil](https://huggingface.co/datasets/FederCO23/solar-plants-brazil) | solar panels (4-band GeoTIFF, binary masks) |
| Indian demo | [Mendeley Indian demo dataset](https://data.mendeley.com/public-files/datasets/xj2v49zt26/files/caf935d8-ef3d-42c0-a7da-0ccc85f10669/file_downloaded) | Indian aerial imagery demo |

**Quick Download**: Run `./download_datasets.sh` (or `.bat` on Windows) to automatically download all datasets. See [DATASET_GUIDE.md](DATASET_GUIDE.md) for details.

## Training Models

To fine-tune models on aerial datasets:

```bash
# Download datasets first
./download_datasets.sh

# Organize data for training
python organize_data.py --datasets aerial

# Train both models
./train_all.sh      # Linux/Mac
train_all.bat       # Windows
```

**📖 Documentation:**
- **[QUICKSTART.md](QUICKSTART.md)** - Copy-paste commands to get started fast
- **[WORKFLOW.md](WORKFLOW.md)** - Complete end-to-end workflow
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training instructions
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** - Dataset download and setup

## Project Structure

```
aerial_AI/
├── app.py                    # Streamlit web application
├── train_semantic.py         # SegFormer training script
├── train_instance.py         # Mask2Former training script
├── download_datasets.py      # Dataset download tool
├── organize_data.py          # Dataset organization script
├── train_all.sh/bat         # Master training scripts
├── download_datasets.sh/bat # Dataset download scripts
├── requirements.txt         # Python dependencies
├── .env                     # API tokens (create from .env.example)
├── README.md               # This file
├── QUICKSTART.md           # Quick reference commands
├── WORKFLOW.md             # Complete workflow guide
├── TRAINING_GUIDE.md       # Training documentation
└── DATASET_GUIDE.md        # Dataset setup guide
```

## Future Enhancements

- Add more feature classes (vegetation, parking lots, etc.)
- Export segmentation masks
- Batch processing
- Measurement tools (distance, area)
- Real-time video processing

## License

MIT

## Acknowledgments

- SegFormer by NVIDIA
- Mask2Former by Facebook Research
- Hugging Face Transformers
