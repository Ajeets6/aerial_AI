@echo off
REM Master training script to fine-tune both models (Windows)

echo ==========================================
echo Aerial AI - Complete Training Pipeline
echo ==========================================

REM Configuration
set EPOCHS_SEMANTIC=20
set EPOCHS_INSTANCE=10
set BATCH_SIZE_SEMANTIC=4
set BATCH_SIZE_INSTANCE=2
set LEARNING_RATE_SEMANTIC=5e-5
set LEARNING_RATE_INSTANCE=1e-5

REM Paths - Update these to your dataset locations
set SEMANTIC_TRAIN_IMG=.\data\aerial_segmentation\train\images
set SEMANTIC_TRAIN_MASK=.\data\aerial_segmentation\train\masks
set SEMANTIC_VAL_IMG=.\data\aerial_segmentation\val\images
set SEMANTIC_VAL_MASK=.\data\aerial_segmentation\val\masks

set INSTANCE_TRAIN_IMG=.\data\solar_panels\train\images
set INSTANCE_TRAIN_ANN=.\data\solar_panels\train\annotations
set INSTANCE_VAL_IMG=.\data\solar_panels\val\images
set INSTANCE_VAL_ANN=.\data\solar_panels\val\annotations

echo.
echo Checking GPU availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}') if torch else print('PyTorch not installed')" 2>nul || echo WARNING: PyTorch not installed or GPU not available

echo.
echo Step 1/2: Training Semantic Segmentation Model (SegFormer)
echo ==========================================
python train_semantic.py ^
    --train_image_dir "%SEMANTIC_TRAIN_IMG%" ^
    --train_mask_dir "%SEMANTIC_TRAIN_MASK%" ^
    --val_image_dir "%SEMANTIC_VAL_IMG%" ^
    --val_mask_dir "%SEMANTIC_VAL_MASK%" ^
    --epochs %EPOCHS_SEMANTIC% ^
    --batch_size %BATCH_SIZE_SEMANTIC% ^
    --learning_rate %LEARNING_RATE_SEMANTIC% ^
    --num_classes 4 ^
    --output_dir ".\output\semantic" ^
    --log_dir ".\logs\semantic" ^
    --save_every 5 ^
    --device cuda

echo.
echo Step 2/2: Training Instance Segmentation Model (Mask2Former)
echo ==========================================
python train_instance.py ^
    --train_image_dir "%INSTANCE_TRAIN_IMG%" ^
    --train_annotation_dir "%INSTANCE_TRAIN_ANN%" ^
    --val_image_dir "%INSTANCE_VAL_IMG%" ^
    --val_annotation_dir "%INSTANCE_VAL_ANN%" ^
    --epochs %EPOCHS_INSTANCE% ^
    --batch_size %BATCH_SIZE_INSTANCE% ^
    --learning_rate %LEARNING_RATE_INSTANCE% ^
    --num_classes 2 ^
    --output_dir ".\output\instance" ^
    --log_dir ".\logs\instance" ^
    --save_every 2 ^
    --device cuda

echo.
echo ==========================================
echo Training Complete!
echo ==========================================
echo Semantic model: .\output\semantic\best_model
echo Instance model: .\output\instance\best_model
echo.
echo View logs:
echo   - Semantic: .\logs\semantic\
echo   - Instance: .\logs\instance\
echo.
echo Training histories:
echo   - .\output\semantic\training_history.json
echo   - .\output\instance\training_history.json

pause
