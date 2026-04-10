"""
Fine-tune Mask2Former for Solar Plants Brazil instance segmentation.
"""
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tifffile import imread as tiff_imread
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, get_scheduler


def setup_logging(log_dir):
    """Setup logging to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_instance_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(stream=sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


class SolarPlantsBrazilDataset(Dataset):
    """Dataset for Solar Plants Brazil TIFF imagery and binary masks."""

    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor

        self.images = sorted(
            list(self.image_dir.glob("*.tif"))
            + list(self.image_dir.glob("*.tiff"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
        )

        logging.info(f"Loaded {len(self.images)} images from {image_dir}")

    def __len__(self):
        return len(self.images)

    def _normalize_rgb(self, rgb_array):
        """Scale an RGB array to uint8 for Hugging Face processors."""
        rgb_array = rgb_array.astype(np.float32)
        if np.nanmax(rgb_array) <= 1.5:
            rgb_array = rgb_array * 255.0
        else:
            normalized = np.zeros_like(rgb_array, dtype=np.float32)
            for channel_idx in range(rgb_array.shape[2]):
                channel = rgb_array[:, :, channel_idx]
                low, high = np.nanpercentile(channel, (2, 98))
                if high <= low:
                    normalized[:, :, channel_idx] = np.clip(channel, 0, 255)
                else:
                    normalized[:, :, channel_idx] = np.clip((channel - low) / (high - low), 0.0, 1.0) * 255.0
            rgb_array = normalized

        return np.clip(rgb_array, 0, 255).astype(np.uint8)

    def _load_image(self, image_path):
        """Load a 4-band TIFF and convert it to RGB for the processor."""
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            image_array = tiff_imread(str(image_path))

            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.ndim == 3 and image_array.shape[0] in {3, 4} and image_array.shape[-1] not in {3, 4}:
                image_array = np.moveaxis(image_array, 0, -1)

            if image_array.ndim != 3:
                raise ValueError(f"Unsupported TIFF shape for {image_path}: {image_array.shape}")

            if image_array.shape[2] < 3:
                image_array = np.repeat(image_array, 3, axis=2)

            rgb_array = image_array[:, :, :3]
            return Image.fromarray(self._normalize_rgb(rgb_array))

        return Image.open(image_path).convert("RGB")

    def load_instance_masks(self, image_path):
        """Load binary masks and convert connected components into instance masks."""
        mask_candidates = [
            self.mask_dir / image_path.name.replace("img", "target", 1),
            self.mask_dir / image_path.with_suffix(".tif").name.replace("img", "target", 1),
            self.mask_dir / image_path.with_suffix(".png").name.replace("img", "target", 1),
            self.mask_dir / image_path.name,
        ]

        mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
        if mask_path is None:
            return []

        if mask_path.suffix.lower() in {".tif", ".tiff"}:
            mask = tiff_imread(str(mask_path))
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return []

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        binary_mask = (mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)

        masks = []
        for label_idx in range(1, num_labels):
            instance_mask = (labels == label_idx).astype(np.uint8)
            if instance_mask.sum() > 0:
                masks.append(instance_mask)

        return masks

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = self._load_image(image_path)

        masks = self.load_instance_masks(image_path)
        if not masks:
            masks = [np.zeros((image.height, image.width), dtype=np.uint8)]

        annotations = []
        for mask_index, mask in enumerate(masks):
            annotations.append({
                "id": mask_index,
                "category_id": 1,
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": self.get_bbox(mask),
                "iscrowd": 0,
            })

        encoded_inputs = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt",
        )

        return {
            key: value.squeeze(0) if isinstance(value, torch.Tensor) else value
            for key, value in encoded_inputs.items()
        }

    def get_bbox(self, mask):
        """Get bounding box from mask."""
        positions = np.where(mask)
        if len(positions[0]) == 0:
            return [0, 0, 1, 1]

        ymin = int(positions[0].min())
        ymax = int(positions[0].max())
        xmin = int(positions[1].min())
        xmax = int(positions[1].max())
        return [xmin, ymin, xmax - xmin, ymax - ymin]


def collate_fn(batch):
    """Collate a batch for Mask2Former training."""
    if not batch:
        return {}

    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch])
    }

    for key in batch[0].keys():
        if key == "pixel_values":
            continue

        values = [item[key] for item in batch if key in item]
        if not values:
            continue

        if all(isinstance(value, torch.Tensor) and value.shape == values[0].shape for value in values):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated


def train_epoch(model, dataloader, optimizer, lr_scheduler, device, logger, epoch, scaler, use_amp, gradient_accumulation_steps, max_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(progress_bar):
        try:
            pixel_values = batch["pixel_values"].to(device, non_blocking=device.type == "cuda")
            inputs = {"pixel_values": pixel_values}

            if "class_labels" in batch:
                class_labels = batch["class_labels"]
                if isinstance(class_labels, torch.Tensor):
                    inputs["class_labels"] = class_labels.to(device, non_blocking=device.type == "cuda")
                else:
                    inputs["class_labels"] = [
                        item.to(device, non_blocking=device.type == "cuda") if isinstance(item, torch.Tensor) else item
                        for item in class_labels
                    ]

            if "mask_labels" in batch:
                mask_labels = batch["mask_labels"]
                if isinstance(mask_labels, torch.Tensor):
                    inputs["mask_labels"] = mask_labels.to(device, non_blocking=device.type == "cuda")
                else:
                    inputs["mask_labels"] = [
                        item.to(device, non_blocking=device.type == "cuda") if isinstance(item, torch.Tensor) else item
                        for item in mask_labels
                    ]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
                loss = outputs.loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item() * gradient_accumulation_steps
            total_loss += current_loss

            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
            })

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} - Batch {batch_idx + 1}/{num_batches} - "
                    f"Loss: {current_loss:.4f} - LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                )

        except Exception as error:
            logger.warning(f"Error in batch {batch_idx}: {error}")
            continue

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.inference_mode()
def validate(model, dataloader, device, logger, use_amp):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Validating")

    for batch in progress_bar:
        try:
            pixel_values = batch["pixel_values"].to(device, non_blocking=device.type == "cuda")
            inputs = {"pixel_values": pixel_values}

            if "class_labels" in batch:
                class_labels = batch["class_labels"]
                if isinstance(class_labels, torch.Tensor):
                    inputs["class_labels"] = class_labels.to(device, non_blocking=device.type == "cuda")
                else:
                    inputs["class_labels"] = [
                        item.to(device, non_blocking=device.type == "cuda") if isinstance(item, torch.Tensor) else item
                        for item in class_labels
                    ]

            if "mask_labels" in batch:
                mask_labels = batch["mask_labels"]
                if isinstance(mask_labels, torch.Tensor):
                    inputs["mask_labels"] = mask_labels.to(device, non_blocking=device.type == "cuda")
                else:
                    inputs["mask_labels"] = [
                        item.to(device, non_blocking=device.type == "cuda") if isinstance(item, torch.Tensor) else item
                        for item in mask_labels
                    ]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)

            total_loss += outputs.loss.item()
            num_batches += 1

        except Exception as error:
            logger.warning(f"Error in validation batch: {error}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation - Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, processor, optimizer, epoch, metrics, output_dir, logger):
    """Save model checkpoint."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, checkpoint_dir / "training_state.pt")

    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mask2Former for Solar Plants Brazil")

    parser.add_argument("--train_image_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--train_annotation_dir", "--train_mask_dir", dest="train_annotation_dir", type=str, required=True, help="Directory containing training binary masks")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--val_annotation_dir", "--val_mask_dir", dest="val_annotation_dir", type=str, required=True, help="Directory containing validation binary masks")

    parser.add_argument("--model_name", type=str, default="facebook/mask2former-swin-base-coco-instance", help="Pretrained model name")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (background + solar panel)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")

    parser.add_argument("--output_dir", type=str, default="./output/instance", help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/instance", help="Directory for log files")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients across N batches before stepping")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision")
    parser.set_defaults(use_amp=torch.cuda.is_available())

    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else min(2, os.cpu_count() or 1), help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("Starting Instance Segmentation Training (Solar Plants Brazil)")
    logger.info("=" * 80)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError as error:
            logger.warning(f"Gradient checkpointing not enabled: {error}")

    model.to(device)
    logger.info(f"Model loaded with {args.num_classes} classes")

    logger.info("Creating datasets...")
    train_dataset = SolarPlantsBrazilDataset(args.train_image_dir, args.train_annotation_dir, processor)
    val_dataset = SolarPlantsBrazilDataset(args.val_image_dir, args.val_annotation_dir, processor)

    if len(train_dataset) == 0:
        logger.error(f"No training images found in {args.train_image_dir}")
        logger.error("Expected organized data under data/solar_panels/train/images")
        logger.error("Run: python organize_data.py --datasets solar")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.error(f"No validation images found in {args.val_image_dir}")
        logger.error("Expected organized data under data/solar_panels/val/images")
        logger.error("Run: python organize_data.py --datasets solar")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=1 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=1 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    logger.info(f"Total training steps: {num_training_steps}")
    logger.info("=" * 80)
    logger.info("Starting Training Loop")
    logger.info("=" * 80)

    best_loss = float("inf")
    training_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'=' * 80}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            device,
            logger,
            epoch,
            scaler,
            args.use_amp and device.type == "cuda",
            args.gradient_accumulation_steps,
            args.max_grad_norm,
        )

        val_loss = validate(model, val_loader, device, logger, args.use_amp and device.type == "cuda")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        training_history.append(epoch_metrics)

        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f}")

        if epoch % args.save_every == 0:
            save_checkpoint(model, processor, optimizer, epoch, epoch_metrics, args.output_dir, logger)

        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(f"New best validation loss: {best_loss:.4f}")
            best_dir = Path(args.output_dir) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            with open(best_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(epoch_metrics, handle, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Final model saved to: {args.output_dir}/best_model")

    history_file = Path(args.output_dir) / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as handle:
        json.dump(training_history, handle, indent=2)
    logger.info(f"Training history saved to: {history_file}")


if __name__ == "__main__":
    main()