#!/bin/bash
# Master training script to fine-tune both models

echo "=========================================="
echo "Aerial AI - Complete Training Pipeline"
echo "=========================================="

# Configuration
EPOCHS_SEMANTIC=20
EPOCHS_INSTANCE=10
BATCH_SIZE_SEMANTIC=2
BATCH_SIZE_INSTANCE=1
LEARNING_RATE_SEMANTIC=5e-5
LEARNING_RATE_INSTANCE=1e-5
GRADIENT_ACCUMULATION=2
NUM_WORKERS=0

# Paths - Update these to your dataset locations
SEMANTIC_TRAIN_IMG="./data/aerial_segmentation/train/images"
SEMANTIC_TRAIN_MASK="./data/aerial_segmentation/train/masks"
SEMANTIC_VAL_IMG="./data/aerial_segmentation/val/images"
SEMANTIC_VAL_MASK="./data/aerial_segmentation/val/masks"

INSTANCE_TRAIN_IMG="./data/solar_panels/train/images"
INSTANCE_TRAIN_ANN="./data/solar_panels/train/masks"
INSTANCE_VAL_IMG="./data/solar_panels/val/images"
INSTANCE_VAL_ANN="./data/solar_panels/val/masks"

echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')" 2>/dev/null || echo "WARNING: PyTorch not installed or GPU not available"

echo ""
echo "Step 1/2: Training Semantic Segmentation Model (SegFormer)"
echo "=========================================="
python train_semantic.py \
    --train_image_dir "$SEMANTIC_TRAIN_IMG" \
    --train_mask_dir "$SEMANTIC_TRAIN_MASK" \
    --val_image_dir "$SEMANTIC_VAL_IMG" \
    --val_mask_dir "$SEMANTIC_VAL_MASK" \
    --epochs $EPOCHS_SEMANTIC \
    --batch_size $BATCH_SIZE_SEMANTIC \
    --learning_rate $LEARNING_RATE_SEMANTIC \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --num_workers $NUM_WORKERS \
    --num_classes 4 \
    --output_dir "./output/semantic" \
    --log_dir "./logs/semantic" \
    --save_every 5 \
    --device cuda

echo ""
echo "Step 2/2: Training Instance Segmentation Model (Mask2Former)"
echo "=========================================="
python train_instance.py \
    --train_image_dir "$INSTANCE_TRAIN_IMG" \
    --train_mask_dir "$INSTANCE_TRAIN_ANN" \
    --val_image_dir "$INSTANCE_VAL_IMG" \
    --val_mask_dir "$INSTANCE_VAL_ANN" \
    --epochs $EPOCHS_INSTANCE \
    --batch_size $BATCH_SIZE_INSTANCE \
    --learning_rate $LEARNING_RATE_INSTANCE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --num_workers $NUM_WORKERS \
    --num_classes 2 \
    --output_dir "./output/instance" \
    --log_dir "./logs/instance" \
    --save_every 2 \
    --device cuda

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Semantic model: ./output/semantic/best_model"
echo "Instance model: ./output/instance/best_model"
echo ""
echo "View logs:"
echo "  - Semantic: ./logs/semantic/"
echo "  - Instance: ./logs/instance/"
echo ""
echo "Training histories:"
echo "  - ./output/semantic/training_history.json"
echo "  - ./output/instance/training_history.json"
