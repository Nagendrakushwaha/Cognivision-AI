import os
import io
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Dict, Any, Optional
from ..models.architectures import build_model, GradCAM
from ..utils.image_processing import (
    load_image_from_bytes,
    image_to_base64,
    preprocess_image_tensor,
    generate_gradcam_overlay
)
from .predict_service import predict_service

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]

class ExplainService:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir

    def explain_image(self, image: Image.Image, target_class_idx: Optional[int] = None) -> Dict[str, Any]:
        if not predict_service._ensure_model_loaded():
            return {
                "is_trained": False,
                "error": "Model is not trained yet. Train a model first to generate visual explanations."
            }

        cam_engine = predict_service.cam_engine
        if cam_engine is None:
            return {
                "is_trained": True,
                "error": "Grad-CAM engine not supported on current model architecture."
            }

        tensor = preprocess_image_tensor(image)
        cam_result = cam_engine.generate(tensor, target_class=target_class_idx)
        heatmap = cam_result["heatmap"] # (H, W) in [0, 1]
        predicted_idx = cam_result["target_class_idx"]
        predicted_class = predict_service.classes[predicted_idx]
        confidence = cam_result["confidence"]

        orig_w, orig_h = image.size

        # 1. Overlay image
        overlay_b64 = generate_gradcam_overlay(image, heatmap, alpha=0.55)

        # 2. Standalone heatmap image (JET on black background)
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        heatmap_arr = np.array(heatmap_pil) / 255.0
        colormap = cm.get_cmap("jet")
        colored = (colormap(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
        heatmap_only_b64 = image_to_base64(Image.fromarray(colored))

        # 3. Original preview image
        thumb = image.copy()
        thumb.thumbnail((400, 500))
        orig_b64 = image_to_base64(thumb)

        # 4. Spatial quadrant analysis
        # Divide receipt vertically into 3 zones: Header (0-30%), Body/Items (30-75%), Footer/Totals (75-100%)
        h_len = heatmap.shape[0]
        header_zone = heatmap[:int(h_len * 0.35), :]
        body_zone = heatmap[int(h_len * 0.35):int(h_len * 0.75), :]
        footer_zone = heatmap[int(h_len * 0.75):, :]

        header_score = float(np.mean(header_zone))
        body_score = float(np.mean(body_zone))
        footer_score = float(np.mean(footer_zone))

        total_score = header_score + body_score + footer_score + 1e-6
        region_importance = {
            "Header / Vendor Identification": round((header_score / total_score) * 100, 1),
            "Document Body / Itemized Entries": round((body_score / total_score) * 100, 1),
            "Footer / Totals & Tax Stamps": round((footer_score / total_score) * 100, 1)
        }

        # Qualitative explanation
        top_region = max(region_importance, key=region_importance.get)
        explanation_text = (
            f"The deep neural network classified this receipt as '{predicted_class}' with {confidence*100:.1f}% confidence. "
            f"Grad-CAM visual inspection shows peak gradient activations primarily centered on the {top_region} ({region_importance[top_region]}% weight). "
            f"This indicates the model leveraged typography, distinctive merchant branding, layout structure, and line-item syntax in this zone to resolve the classification."
        )

        return {
            "is_trained": True,
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "original_image_base64": orig_b64,
            "overlay_base64": overlay_b64,
            "heatmap_only_base64": heatmap_only_b64,
            "region_importance": region_importance,
            "explanation_text": explanation_text,
            "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)"
        }

explain_service = ExplainService()
