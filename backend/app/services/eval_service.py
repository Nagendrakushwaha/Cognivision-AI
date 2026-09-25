import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from ..models.architectures import build_model, get_device
from ..utils.paths import TEST_CACHE_FILE, METRICS_FILE, BEST_MODEL_PATH, LATEST_MODEL_PATH, ARTIFACTS_DIR

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]

class EvalService:
    def __init__(self):
        self.device = get_device()

    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        path_to_use = None
        if model_path and os.path.exists(model_path):
            path_to_use = model_path
        elif BEST_MODEL_PATH.exists():
            path_to_use = str(BEST_MODEL_PATH)
        elif LATEST_MODEL_PATH.exists():
            path_to_use = str(LATEST_MODEL_PATH)
        else:
            return {
                "is_trained": False,
                "error": f"Model not trained yet. No model checkpoint found at {BEST_MODEL_PATH}"
            }

        if not TEST_CACHE_FILE.exists():
            return {
                "is_trained": False,
                "error": f"Test dataset cache not found at {TEST_CACHE_FILE}. Please run prepare_cache.py."
            }

        print(f"Loading model checkpoint from {path_to_use} for evaluation on {self.device}...")
        checkpoint = torch.load(path_to_use, map_location=self.device)
        model_name = checkpoint.get("model_name", "mobilenet_v3")
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        classes = checkpoint.get("classes", CLASS_NAMES)

        model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loading test cache from {TEST_CACHE_FILE}...")
        test_cache = torch.load(str(TEST_CACHE_FILE), map_location="cpu")
        test_images = test_cache["images"] # (N, 3, 224, 224)
        test_labels = test_cache["labels"].numpy()

        all_preds = []
        all_probs = []

        batch_size = 32
        num_samples = len(test_images)
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_x = test_images[i:i+batch_size].to(self.device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.append(probs)

        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)

        # Calculate evaluation metrics
        acc = float(accuracy_score(test_labels, all_preds))
        macro_prec = float(precision_score(test_labels, all_preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(test_labels, all_preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(test_labels, all_preds, average="macro", zero_division=0))
        
        weighted_prec = float(precision_score(test_labels, all_preds, average="weighted", zero_division=0))
        weighted_rec = float(recall_score(test_labels, all_preds, average="weighted", zero_division=0))
        weighted_f1 = float(f1_score(test_labels, all_preds, average="weighted", zero_division=0))

        # ROC-AUC calculation (One-vs-Rest)
        roc_auc = None
        try:
            y_onehot = np.zeros((num_samples, num_classes))
            for idx, label in enumerate(test_labels):
                y_onehot[idx, label] = 1.0
            roc_auc = float(roc_auc_score(y_onehot, all_probs, multi_class="ovr", average="macro"))
        except Exception as e:
            print(f"ROC-AUC calculation note: {e}")

        # Confusion matrix
        cm = confusion_matrix(test_labels, all_preds, labels=list(range(num_classes)))
        cm_list = cm.tolist()

        # Class-wise metrics
        clf_rep = classification_report(test_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
        class_wise = []
        for c in classes:
            metrics = clf_rep.get(c, {})
            class_wise.append({
                "class_name": c,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0))
            })

        report = {
            "is_trained": True,
            "model_name": model_name,
            "evaluated_samples": num_samples,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_prec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm_list,
            "classes": classes,
            "class_wise_metrics": class_wise,
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save to artifacts
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def get_latest_metrics(self) -> Dict[str, Any]:
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                pass
        
        if BEST_MODEL_PATH.exists() or LATEST_MODEL_PATH.exists():
            return self.evaluate_model()

        return {
            "is_trained": False,
            "message": "Model not trained yet. Train the model from the Model Training page."
        }

eval_service = EvalService()
