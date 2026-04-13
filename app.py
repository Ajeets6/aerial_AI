import os

# Prevent OpenMP duplicate runtime crash on Windows (common with torch + numpy/opencv stacks).
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForInstanceSegmentation
import io
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Aerial Image Multi-Segmenter",
    page_icon="🛰️",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_semantic_model():
    """Load SegFormer model for semantic segmentation"""
    model_path = Path("./output/semantic/best_model")
    source = str(model_path) if model_path.exists() else "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForSemanticSegmentation.from_pretrained(source)
    return processor, model

@st.cache_resource
def load_instance_model():
    """Load Mask2Former model for instance segmentation (solar panels)"""
    model_path = Path("./output/instance/best_model")
    source = str(model_path) if model_path.exists() else "facebook/mask2former-swin-base-coco-instance"
    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForInstanceSegmentation.from_pretrained(source)
    return processor, model

def predict_semantic(image, processor, model):
    """Run semantic segmentation on image"""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Upsample to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return pred_seg

def predict_instance(image, processor, model, target_class="solar panel"):
    """Run instance segmentation for solar panels"""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get instance masks
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    return results

def create_semantic_overlay(image, seg_mask, feature_idx, alpha=0.6):
    """Create overlay highlighting specific semantic feature"""
    img_array = np.array(image)
    overlay = img_array.copy()

    # Create color mask for selected feature
    mask = (seg_mask == feature_idx).astype(np.uint8) * 255

    # Apply color overlay
    color_map = {
        0: [255, 100, 100],  # Buildings - Red
        1: [100, 100, 255],  # Roads - Blue
        2: [100, 255, 255],  # Water - Cyan
    }

    color = color_map.get(feature_idx, [255, 255, 100])
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask == 255] = color

    # Blend
    result = cv2.addWeighted(img_array, 1-alpha, colored_mask, alpha, 0)
    return result, mask

def create_instance_overlay(image, instances, alpha=0.6):
    """Create overlay for instance segmentation (solar panels)"""
    img_array = np.array(image)
    overlay = img_array.copy()

    if 'masks' not in instances or len(instances['masks']) == 0:
        return img_array, None, 0

    # Combine all instance masks
    combined_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    num_instances = len(instances['masks'])

    for i, mask in enumerate(instances['masks']):
        mask_array = mask.cpu().numpy().astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask_array)

        # Draw yellow overlay for each panel
        colored_mask = np.zeros_like(img_array)
        colored_mask[mask_array == 255] = [255, 255, 100]  # Yellow
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

    return overlay, combined_mask, num_instances

def calculate_stats(mask, pixel_to_meter=0.3):
    """Calculate statistics from segmentation mask"""
    if mask is None:
        return {}

    area_pixels = np.sum(mask == 255)
    area_m2 = area_pixels * (pixel_to_meter ** 2)

    return {
        'area_pixels': int(area_pixels),
        'area_m2': round(area_m2, 2)
    }

# Main UI
st.title("🛰️ Aerial Image Multi-Segmenter")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    st.markdown("""
    **Features:**
    - Semantic: Buildings, Roads, Water
    - Instance: Solar Panels

    **Models:**
    - SegFormer (Cityscapes)
    - Mask2Former (COCO)
    """)

    st.markdown("---")
    st.info("💡 Upload an aerial image to start segmentation")

# File uploader (including TIFF for demo/remote-sensing datasets)
uploaded_file = st.file_uploader("Upload Aerial Image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Feature selection dropdown
    feature_options = {
        "Buildings": 0,
        "Roads": 1,
        "Water": 2,
        "Solar Panels": 3
    }

    selected_feature = st.selectbox(
        "Select feature to highlight:",
        options=list(feature_options.keys())
    )

    # Layout: Original | Highlighted
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=True)

    with col2:
        st.subheader(f"Highlighted: {selected_feature}")

        with st.spinner(f"Processing {selected_feature}..."):
            feature_idx = feature_options[selected_feature]

            if feature_idx < 3:  # Semantic segmentation
                semantic_processor, semantic_model = load_semantic_model()
                seg_mask = predict_semantic(image, semantic_processor, semantic_model)

                # Map cityscapes classes to our classes (simplified mapping)
                # This is a placeholder - would need proper fine-tuning for aerial images
                highlighted, mask = create_semantic_overlay(image, seg_mask, feature_idx)

                st.image(highlighted, width=True)

                # Statistics
                stats = calculate_stats(mask)
                st.markdown("### Statistics")
                st.metric(f"{selected_feature} Area (pixels)", f"{stats.get('area_pixels', 0):,}")
                st.metric(f"{selected_feature} Area (m²)", f"{stats.get('area_m2', 0):.2f}")

            else:  # Instance segmentation (solar panels)
                instance_processor, instance_model = load_instance_model()
                instances = predict_instance(image, instance_processor, instance_model)

                highlighted, mask, num_panels = create_instance_overlay(image, instances)

                st.image(highlighted, width=True)

                # Statistics
                st.markdown("### Statistics")
                st.metric("Solar Panels Detected", num_panels)
                if num_panels > 0:
                    stats = calculate_stats(mask)
                    st.metric("Total Panel Area (pixels)", f"{stats.get('area_pixels', 0):,}")
                    st.metric("Total Panel Area (m²)", f"{stats.get('area_m2', 0):.2f}")

    st.markdown("---")
    st.caption("ℹ️ **Note:** Models are pretrained on Cityscapes/COCO. Fine-tune on aerial datasets for better accuracy.")

else:
    st.info("👆 Upload an aerial image to begin segmentation")

    # Show example UI mockup
    st.markdown("### Expected Output Preview")
    st.code("""
    ┌─────────────────────────────────────────────────────────────┐
    │  Original Image              │  Highlighted (Buildings)     │
    │  ┌─────────────────────┐    │  ┌─────────────────────┐    │
    │  │                     │    │  │                     │    │
    │  │   Aerial view       │    │  │  [Red overlay on    │    │
    │  │                     │    │  │   building regions] │    │
    │  └─────────────────────┘    │  └─────────────────────┘    │
    │                                                             │
    │  Statistics:                                                │
    │  - Buildings area: 124,567 pixels                          │
    │  - Buildings area: 11,211 m²                               │
    └─────────────────────────────────────────────────────────────┘
    """, language="text")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Models: SegFormer + Mask2Former")
