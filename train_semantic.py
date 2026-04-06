"""
Fine-tune SegFormer for aerial semantic segmentation (buildings, roads, water)
"""
import argparse
import logging
import os
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForSemanticSegmentation,
    get_scheduler
)
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

# Setup logging
def setup_logging(log_dir):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_semantic_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class AerialSemanticDataset(Dataset):
    """Dataset for aerial semantic segmentation"""
    
    def __init__(self, image_dir, mask_dir, processor, class_map=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor
        
        # Class mapping based on actual RGB colors in the Kaggle Aerial Segmentation dataset
        # Format: (R, G, B): class_id
        self.class_map = class_map or {
            (60, 16, 152): 0,      # Class 0 - Unknown/Other (purple)
            (132, 41, 246): 1,     # Class 1 - Building (purple/magenta)
            (110, 193, 228): 2,    # Class 2 - Land (blue)
            (254, 221, 58): 3,     # Class 3 - Road (yellow)
            (226, 169, 41): 4,     # Class 4 - Vegetation (orange)
            (155, 155, 155): 5,    # Class 5 - Water (gray)
        }
        
        # We'll map to fewer classes for simpler training
        # Building, Road, Water, Background
        self.simplified_map = {
            0: 0,  # Other -> Background
            1: 1,  # Building -> Building
            2: 0,  # Land -> Background  
            3: 2,  # Road -> Road
            4: 0,  # Vegetation -> Background
            5: 3,  # Water -> Water
        }
        
        self.images = sorted(list(self.image_dir.glob("*.jpg")) + 
                           list(self.image_dir.glob("*.png")))
        
        logging.info(f"Loaded {len(self.images)} images from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask - convert image filename to mask filename (.png)
        mask_filename = img_path.stem + ".png"
        mask_path = self.mask_dir / mask_filename
        
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}\n"
                f"Expected mask for image: {img_path.name}\n"
                f"Make sure masks have .png extension with same base name as images"
            )
        
        mask = Image.open(mask_path)
        
        # Handle both RGB and palette mode masks
        if mask.mode == 'P':
            # Palette mode: convert to RGB first
            mask = mask.convert('RGB')
        elif mask.mode not in ['RGB', 'P']:
            # Convert any other mode to RGB
            mask = mask.convert('RGB')
        
        # Convert RGB mask to class indices
        mask_array = np.array(mask)
        
        if len(mask_array.shape) != 3 or mask_array.shape[2] != 3:
            raise ValueError(f"Expected RGB mask after conversion, got shape {mask_array.shape}, mode was {mask.mode}")
        
        # Create segmentation mask from RGB colors
        h, w = mask_array.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.int64)
        
        # Map RGB colors to class IDs
        for rgb_color, class_id in self.class_map.items():
            # Create mask for this color
            color_mask = np.all(mask_array == rgb_color, axis=-1)
            # Get simplified class
            simplified_class = self.simplified_map[class_id]
            seg_mask[color_mask] = simplified_class
        
        # Process inputs with proper resizing
        encoded_inputs = self.processor(
            images=image, 
            segmentation_maps=seg_mask,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoded_inputs = {k: v.squeeze(0) for k, v in encoded_inputs.items()}
        
        return encoded_inputs

def compute_metrics(pred_masks, true_masks, num_classes):
    """Compute IoU and pixel accuracy"""
    ious = []
    pixel_correct = 0
    pixel_total = 0
    
    for pred, true in zip(pred_masks, true_masks):
        for class_id in range(num_classes):
            pred_class = (pred == class_id)
            true_class = (true == class_id)
            
            intersection = (pred_class & true_class).sum().item()
            union = (pred_class | true_class).sum().item()
            
            if union > 0:
                ious.append(intersection / union)
        
        pixel_correct += (pred == true).sum().item()
        pixel_total += true.numel()
    
    mean_iou = np.mean(ious) if ious else 0.0
    pixel_acc = pixel_correct / pixel_total if pixel_total > 0 else 0.0
    
    return mean_iou, pixel_acc

def collate_fn(batch):
    """Custom collate function to handle batching"""
    # Stack pixel_values
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Stack labels (masks)
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    if num_batches == 0:
        logger.error("No batches in dataloader! Check your dataset.")
        raise ValueError("Dataloader is empty - dataset has 0 samples")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} - Batch {batch_idx+1}/{num_batches} - "
                    f"Loss: {loss.item():.4f} - LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                )
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            logger.error(f"Batch keys: {batch.keys()}")
            logger.error(f"Pixel values shape: {batch['pixel_values'].shape}")
            logger.error(f"Labels shape: {batch['labels'].shape}")
            logger.error(f"Labels unique values: {torch.unique(batch['labels'])}")
            raise
    
    avg_loss = total_loss / num_batches
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, device, logger, num_classes):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Validating")
    
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        preds = upsampled_logits.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mean_iou, pixel_acc = compute_metrics(all_preds, all_labels, num_classes)
    
    avg_loss = total_loss / len(dataloader)
    
    logger.info(
        f"Validation - Loss: {avg_loss:.4f} - "
        f"mIoU: {mean_iou:.4f} - Pixel Acc: {pixel_acc:.4f}"
    )
    
    return avg_loss, mean_iou, pixel_acc

def save_checkpoint(model, processor, optimizer, epoch, metrics, output_dir, logger):
    """Save model checkpoint"""
    checkpoint_dir = Path(output_dir) / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_dir / "training_state.pt")
    
    # Save metrics to JSON
    with open(checkpoint_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer for aerial segmentation")
    
    # Data arguments
    parser.add_argument("--train_image_dir", type=str, required=True,
                       help="Directory containing training images")
    parser.add_argument("--train_mask_dir", type=str, required=True,
                       help="Directory containing training masks")
    parser.add_argument("--val_image_dir", type=str, required=True,
                       help="Directory containing validation images")
    parser.add_argument("--val_mask_dir", type=str, required=True,
                       help="Directory containing validation masks")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
                       help="Pretrained model name")
    parser.add_argument("--num_classes", type=int, default=4,
                       help="Number of segmentation classes (including background)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output/semantic",
                       help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/semantic",
                       help="Directory for log files")
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("="*80)
    logger.info("Starting Semantic Segmentation Training")
    logger.info("="*80)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    logger.info(f"Loading model: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    logger.info(f"Model loaded with {args.num_classes} classes")
    
    # Create datasets
    logger.info("Creating datasets...")
    try:
        train_dataset = AerialSemanticDataset(
            args.train_image_dir,
            args.train_mask_dir,
            processor
        )
        val_dataset = AerialSemanticDataset(
            args.val_image_dir,
            args.val_mask_dir,
            processor
        )
        
        if len(train_dataset) == 0:
            logger.error(f"No images found in {args.train_image_dir}")
            logger.error("Make sure the directory exists and contains .jpg or .png files")
            sys.exit(1)
        
        if len(val_dataset) == 0:
            logger.error(f"No images found in {args.val_image_dir}")
            logger.error("Make sure the directory exists and contains .jpg or .png files")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Training loop
    best_miou = 0.0
    training_history = []
    
    logger.info("="*80)
    logger.info("Starting Training Loop")
    logger.info("="*80)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, lr_scheduler, 
            device, logger, epoch
        )
        
        # Validate
        val_loss, val_miou, val_acc = validate(
            model, val_loader, device, logger, args.num_classes
        )
        
        # Log epoch summary
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_pixel_acc': val_acc,
        }
        training_history.append(epoch_metrics)
        
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f}")
        logger.info(f"  Val mIoU:   {val_miou:.4f}")
        logger.info(f"  Val Acc:    {val_acc:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, processor, optimizer, epoch, 
                epoch_metrics, args.output_dir, logger
            )
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            logger.info(f"🎯 New best mIoU: {best_miou:.4f}")
            best_dir = Path(args.output_dir) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            
            with open(best_dir / "metrics.json", 'w') as f:
                json.dump(epoch_metrics, f, indent=2)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best mIoU: {best_miou:.4f}")
    logger.info(f"Final model saved to: {args.output_dir}/best_model")
    
    # Save training history
    history_file = Path(args.output_dir) / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to: {history_file}")

if __name__ == "__main__":
    main()
