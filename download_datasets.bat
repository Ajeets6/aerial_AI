@echo off
REM Download all datasets for aerial segmentation training

echo ==========================================
echo Aerial AI - Dataset Download Script
echo ==========================================

REM Configuration
set DATA_DIR=.\data
set LOG_DIR=.\logs\downloads

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

echo.
echo This script will download the following datasets:
echo   1. Kaggle Aerial Segmentation (~2GB^)
echo   2. Solar Plants Brazil (Hugging Face^) (~358MB^)
echo   3. Indian Demo Dataset (Mendeley^)
echo.
echo Total download size: ~3.5GB
echo.

REM Ask for confirmation
set /p confirm="Continue? (y/n) "
if /i not "%confirm%"=="y" (
    echo Download cancelled
    exit /b 0
)

REM Install dependencies
echo.
echo Installing required packages...
python download_datasets.py --install_deps --skip_kaggle_check --datasets aerial 2>nul

REM Check for Kaggle credentials
echo.
echo Checking Kaggle credentials...

REM Check for .env file first
if exist ".env" (
    findstr /C:"KAGGLE_API_TOKEN" .env >nul
    if errorlevel 1 (
        echo.
        echo WARNING: .env file exists but no KAGGLE_API_TOKEN found
        echo.
        echo Please add your Kaggle API token to .env:
        echo   KAGGLE_API_TOKEN=KGAT_your_token_here
        echo.
        echo Get your token from: https://www.kaggle.com/settings/account
        exit /b 1
    ) else (
        echo SUCCESS: Found API token in .env file
    )
) else if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo.
    echo WARNING: Kaggle credentials not found!
    echo.
    echo Please use one of these methods:
    echo.
    echo Method 1 (Recommended^): Create .env file with API token
    echo   1. Go to https://www.kaggle.com/settings/account
    echo   2. Click 'Create New API Token'
    echo   3. Copy the token and add to .env file:
    echo      KAGGLE_API_TOKEN=KGAT_your_token_here
    echo.
    echo Method 2: Use kaggle.json
    echo   1. Download kaggle.json from Kaggle settings
    echo   2. Move to %USERPROFILE%\.kaggle\kaggle.json
    echo.
    exit /b 1
)

REM Download datasets
echo.
echo Starting download...
echo.

python download_datasets.py ^
    --data_dir "%DATA_DIR%" ^
    --log_dir "%LOG_DIR%" ^
    --datasets all

REM Check exit code
if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo SUCCESS: Download Complete!
    echo ==========================================
    echo.
    echo Datasets saved to: %DATA_DIR%
    echo Logs saved to: %LOG_DIR%
    echo.
    echo Next steps:
    echo   1. Check the downloaded data in %DATA_DIR%
    echo   2. Review TRAINING_GUIDE.md for dataset organization
    echo   3. Run train_all.bat to start training
) else (
    echo.
    echo ==========================================
    echo ERROR: Download failed
    echo ==========================================
    echo Check logs in %LOG_DIR% for details
    exit /b 1
)

pause
