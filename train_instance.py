"""
Fine-tune Mask2Former for solar panel instance segmentation
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
    Mask2FormerForUniversalSegmentation,
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
    log_file = log_dir / f"train_instance_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SolarPanelDataset(Dataset):
    """Dataset for solar panel instance segmentation"""
    
    def __init__(self, image_dir, annotation_dir, processor, target_size=512):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.processor = processor
        self.target_size = target_size
        
        self.images = sorted(list(self.image_dir.glob("*.jpg")) + 
                           list(self.image_dir.glob("*.png")))
        
        logging.info(f"Loaded {len(self.images)} images from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def load_instance_masks(self, annotation_path):
        """Load instance masks from annotation file"""
        # Assuming annotations are in COCO format or separate mask files
        # This is a simplified version - adapt based on your dataset format
        
        # For solar-antwerp dataset, typically each panel is a separate mask
        annotation_files = sorted(self.annotation_dir.glob(f"{annotation_path.stem}_*.png"))
        
        if not annotation_files:
            # Try loading single mask file - use .png extension
            mask_filename = annotation_path.stem + ".png"
            mask_path = self.annotation_dir / mask_filename
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                # Convert to instance masks using connected components
                num_labels, labels = cv2.connectedComponents(mask)
                masks = []
                for i in range(1, num_labels):
                    instance_mask = (labels == i).astype(np.uint8)
                    masks.append(instance_mask)
                return masks
        else:
            # Load individual instance masks
            masks = []
            for mask_file in annotation_files:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                masks.append((mask > 128).astype(np.uint8))
            return masks
        
        return []
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load instance masks
        masks = self.load_instance_masks(img_path)
        
        if not masks:
            # Create dummy mask if none found
            masks = [np.zeros((image.height, image.width), dtype=np.uint8)]
        
        # Create annotations in COCO format
        annotations = []
        for i, mask in enumerate(masks):
            annotations.append({
                "id": i,
                "category_id": 1,  # Solar panel class
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": self.get_bbox(mask),
                "iscrowd": 0
            })
        
        # Process with Mask2Former processor
        # The processor expects specific format
        encoded_inputs = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoded_inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                         for k, v in encoded_inputs.items()}
        
        return encoded_inputs
    
    def get_bbox(self, mask):
        """Get bounding box from mask"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return [0, 0, 1, 1]
        ymin = int(pos[0].min())
        ymax = int(pos[0].max())
        xmin = int(pos[1].min())
        xmax = int(pos[1].max())
        return [xmin, ymin, xmax - xmin, ymax - ymin]

def compute_instance_metrics(predictions, targets):
    """Compute instance segmentation metrics (AP, AR)"""
    # Simplified metric computation
    # In practice, you'd use proper AP/AR calculation
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # This is a placeholder - implement proper IoU-based matching
    num_pred = len(predictions.get('masks', []))
    num_gt = len(targets.get('masks', []))
    
    # Simplified: just count as recall metric
    recall = min(num_pred, num_gt) / max(num_gt, 1)
    precision = min(num_pred, num_gt) / max(num_pred, 1)
    
    return precision, recall

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            
            # Mask2Former expects specific input format
            inputs = {
                "pixel_values": pixel_values,
            }
            
            # Add labels if present
            if "class_labels" in batch:
                inputs["class_labels"] = batch["class_labels"].to(device)
            if "mask_labels" in batch:
                inputs["mask_labels"] = batch["mask_labels"].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
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
            logger.warning(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, device, logger):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validating")
    
    for batch in progress_bar:
        try:
            pixel_values = batch["pixel_values"].to(device)
            
            inputs = {"pixel_values": pixel_values}
            if "class_labels" in batch:
                inputs["class_labels"] = batch["class_labels"].to(device)
            if "mask_labels" in batch:
                inputs["mask_labels"] = batch["mask_labels"].to(device)
            
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
        
        except Exception as e:
            logger.warning(f"Error in validation batch: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}")
    
    return avg_loss

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
    parser = argparse.ArgumentParser(description="Fine-tune Mask2Former for solar panel detection")
    
    # Data arguments
    parser.add_argument("--train_image_dir", type=str, required=True,
                       help="Directory containing training images")
    parser.add_argument("--train_annotation_dir", type=str, required=True,
                       help="Directory containing training annotations/masks")
    parser.add_argument("--val_image_dir", type=str, required=True,
                       help="Directory containing validation images")
    parser.add_argument("--val_annotation_dir", type=str, required=True,
                       help="Directory containing validation annotations/masks")
    
    # Model arguments
    parser.add_argument("--model_name", type=str,
                       default="facebook/mask2former-swin-base-coco-instance",
                       help="Pretrained model name")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes (background + solar panel)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output/instance",
                       help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/instance",
                       help="Directory for log files")
    parser.add_argument("--save_every", type=int, default=2,
                       help="Save checkpoint every N epochs")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("="*80)
    logger.info("Starting Instance Segmentation Training (Solar Panels)")
    logger.info("="*80)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    logger.info(f"Loading model: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    logger.info(f"Model loaded with {args.num_classes} classes")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = SolarPanelDataset(
        args.train_image_dir,
        args.train_annotation_dir,
        processor
    )
    val_dataset = SolarPanelDataset(
        args.val_image_dir,
        args.val_annotation_dir,
        processor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x  # Custom collation needed for Mask2Former
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Training loop
    best_loss = float('inf')
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
        val_loss = validate(model, val_loader, device, logger)
        
        # Log epoch summary
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        training_history.append(epoch_metrics)
        
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, processor, optimizer, epoch,
                epoch_metrics, args.output_dir, logger
            )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(f"🎯 New best validation loss: {best_loss:.4f}")
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
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Final model saved to: {args.output_dir}/best_model")
    
    # Save training history
    history_file = Path(args.output_dir) / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to: {history_file}")

if __name__ == "__main__":
    main()
