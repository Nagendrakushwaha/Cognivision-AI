import os
import io
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

from ..models.architectures import build_model, GradCAM, get_device
from ..utils.paths import BEST_MODEL_PATH, LATEST_MODEL_PATH
from ..utils.image_processing import (
    load_image_from_bytes,
    image_to_base64,
    preprocess_image_tensor,
    generate_gradcam_overlay
)

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]

class PredictService:
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.model_name = None
        self.cam_engine = None
        self.classes = CLASS_NAMES
        self._last_loaded_mtime = 0

    def reload_model(self):
        """Forces an in-memory model reload from disk."""
        self.model = None
        self.cam_engine = None
        return self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        target_path = None
        if BEST_MODEL_PATH.exists():
            target_path = BEST_MODEL_PATH
        elif LATEST_MODEL_PATH.exists():
            target_path = LATEST_MODEL_PATH
        else:
            return False

        mtime = os.path.getmtime(str(target_path))
        if self.model is not None and mtime <= self._last_loaded_mtime:
            return True

        print(f"Loading/Updating inference model from {target_path}...")
        checkpoint = torch.load(str(target_path), map_location=self.device)
        self.model_name = checkpoint.get("model_name", "mobilenet_v3")
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        self.classes = checkpoint.get("classes", CLASS_NAMES)

        model = build_model(model_name=self.model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self._last_loaded_mtime = mtime

        try:
            self.cam_engine = GradCAM(self.model, device=self.device)
        except Exception as e:
            print(f"Warning: GradCAM initialization note: {e}")
            self.cam_engine = None

        return True

    def get_active_model_info(self) -> Dict[str, Any]:
        """Returns the currently active trained model info, or untrained if none exists."""
        loaded = self._ensure_model_loaded()
        if not loaded or self.model is None:
            return {
                "is_trained": False,
                "model_name": None,
                "display_name": "Untrained"
            }
        
        display_name = "MobileNetV3-Small" if "mobilenet" in self.model_name else "CogniNet-CNN"
        return {
            "is_trained": True,
            "model_name": self.model_name,
            "display_name": display_name,
            "architecture": display_name
        }

    def predict_image(self, image: Image.Image, include_gradcam: bool = True) -> Dict[str, Any]:
        if not self._ensure_model_loaded():
            return {
                "error": "Model not trained yet. Please train a model first.",
                "is_trained": False
            }

        start_t = time.time()
        
        # Verify valid RGB PIL Image
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a valid PIL Image object.")
            image = image.convert("RGB")
        except Exception as e:
            return {
                "error": f"Invalid image format: {str(e)}",
                "is_trained": True
            }

        tensor = preprocess_image_tensor(image).to(self.device) # (1, 3, 224, 224)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = self.classes[pred_idx]

        # Top K ranking
        top_k_indices = np.argsort(probs)[::-1]
        top_k = []
        for idx in top_k_indices:
            top_k.append({
                "class_name": self.classes[idx],
                "probability": round(float(probs[idx]), 4),
                "percentage": round(float(probs[idx]) * 100, 2)
            })

        # Generate GradCAM if requested
        gradcam_b64 = None
        if include_gradcam:
            try:
                # Re-initialize GradCAM if needed or use existing
                cam = GradCAM(self.model, device=self.device)
                cam_res = cam.generate(tensor.clone(), target_class=pred_idx)
                heatmap = cam_res["heatmap"]
                gradcam_b64 = generate_gradcam_overlay(image, heatmap, alpha=0.5)
            except Exception as e:
                print(f"GradCAM generation note: {e}")

        # Thumbnail for preview
        thumb = image.copy()
        thumb.thumbnail((400, 500))
        img_b64 = image_to_base64(thumb)

        elapsed_ms = round((time.time() - start_t) * 1000, 2)

        return {
            "is_trained": True,
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "top_k": top_k,
            "image_preview_base64": img_b64,
            "gradcam_base64": gradcam_b64,
            "inference_time_ms": elapsed_ms,
            "model_used": self.model_name or "mobilenet_v3"
        }

    def predict_batch(self, images_list: List[tuple]) -> Dict[str, Any]:
        """
        Takes list of (filename, PIL.Image).
        """
        if not self._ensure_model_loaded():
            return {
                "error": "Model not trained yet. Please train a model first.",
                "is_trained": False
            }

        start_t = time.time()
        results = []

        for filename, img in images_list:
            try:
                img_rgb = img.convert("RGB")
                tensor = preprocess_image_tensor(img_rgb).to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                top_indices = np.argsort(probs)[::-1]
                top_idx = int(top_indices[0])
                top_2_idx = int(top_indices[1])

                results.append({
                    "filename": filename,
                    "predicted_class": self.classes[top_idx],
                    "confidence": round(float(probs[top_idx]), 4),
                    "confidence_percentage": round(float(probs[top_idx]) * 100, 2),
                    "top_2_class": self.classes[top_2_idx],
                    "top_2_confidence": round(float(probs[top_2_idx]), 4)
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "predicted_class": "Error Processing Image",
                    "confidence": 0.0,
                    "confidence_percentage": 0.0,
                    "top_2_class": "N/A",
                    "top_2_confidence": 0.0
                })

        elapsed_ms = round((time.time() - start_t) * 1000, 2)
        return {
            "is_trained": True,
            "total_processed": len(results),
            "results": results,
            "model_used": self.model_name or "mobilenet_v3",
            "inference_time_ms": elapsed_ms
        }

predict_service = PredictService()
