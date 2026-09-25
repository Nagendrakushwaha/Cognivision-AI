import os
import io
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from ..models.architectures import build_model, GradCAM
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
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "models", "best_model.pt")
        self.model = None
        self.model_name = None
        self.cam_engine = None
        self.classes = CLASS_NAMES

    def _ensure_model_loaded(self):
        if self.model is not None:
            return True
            
        if not os.path.exists(self.model_path):
            latest_path = os.path.join(self.data_dir, "models", "latest_model.pt")
            if os.path.exists(latest_path):
                self.model_path = latest_path
            else:
                return False

        print(f"Loading inference model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model_name = checkpoint.get("model_name", "mobilenet_v3")
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        self.classes = checkpoint.get("classes", CLASS_NAMES)

        self.model = build_model(model_name=self.model_name, num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        try:
            self.cam_engine = GradCAM(self.model)
        except Exception as e:
            print(f"Could not initialize GradCAM: {e}")
            self.cam_engine = None

        return True

    def predict_image(self, image: Image.Image, include_gradcam: bool = True) -> Dict[str, Any]:
        if not self._ensure_model_loaded():
            return {
                "error": "Model not trained yet. Please train a model first.",
                "is_trained": False
            }

        start_t = time.time()
        tensor = preprocess_image_tensor(image) # (1, 3, 224, 224)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

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
        if include_gradcam and self.cam_engine:
            try:
                cam_res = self.cam_engine.generate(tensor.clone(), target_class=pred_idx)
                heatmap = cam_res["heatmap"]
                gradcam_b64 = generate_gradcam_overlay(image, heatmap, alpha=0.5)
            except Exception as e:
                print(f"GradCAM generation error: {e}")

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
            tensor = preprocess_image_tensor(img)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]

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

        elapsed_ms = round((time.time() - start_t) * 1000, 2)
        return {
            "is_trained": True,
            "total_processed": len(results),
            "results": results,
            "model_used": self.model_name or "mobilenet_v3",
            "inference_time_ms": elapsed_ms
        }

predict_service = PredictService()
