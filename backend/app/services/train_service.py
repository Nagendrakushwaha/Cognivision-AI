import os
import time
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from ..models.architectures import build_model, get_device
from ..utils.paths import (
    TRAIN_CACHE_FILE,
    HISTORY_FILE,
    MODELS_DIR,
    BEST_MODEL_PATH,
    LATEST_MODEL_PATH,
    DATASET_TRAIN_PARQUET,
    ensure_directories
)
from .eval_service import eval_service

CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]

class TrainService:
    def __init__(self):
        ensure_directories()
        self.device = get_device()
        self.current_job_id = None
        self.status = {
            "status": "idle", # "idle", "training", "completed", "error"
            "job_id": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_percent": 0.0,
            "current_train_loss": None,
            "current_val_loss": None,
            "current_train_acc": None,
            "current_val_acc": None,
            "history": [],
            "error_message": None,
            "model_name": None,
            "started_at": None,
            "completed_at": None
        }
        self._lock = threading.Lock()
        self._load_saved_history()

    def _load_saved_history(self):
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r") as f:
                    saved = json.load(f)
                    self.status["history"] = saved.get("history", [])
                    self.status["model_name"] = saved.get("model_name", "mobilenet_v3")
                    self.status["status"] = "completed" if self.status["history"] else "idle"
                    if self.status["history"]:
                        last = self.status["history"][-1]
                        self.status["current_epoch"] = last["epoch"]
                        self.status["total_epochs"] = len(self.status["history"])
                        self.status["progress_percent"] = 100.0
                        self.status["current_train_loss"] = last["train_loss"]
                        self.status["current_val_loss"] = last["val_loss"]
                        self.status["current_train_acc"] = last["train_acc"]
                        self.status["current_val_acc"] = last["val_acc"]
            except Exception as e:
                print(f"Error loading training history: {e}")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.status)

    def validate_pretraining(self, config: Dict[str, Any]):
        """
        Pre-training validation check:
        Verifies dataset existence, class mappings, model creation, and forward/backward sanity pass.
        Raises an informative RuntimeError if anything fails.
        """
        model_name = config.get("model_name", "mobilenet_v3")
        
        # 1. Dataset cache check (or build on demand if missing)
        if not TRAIN_CACHE_FILE.exists():
            if not DATASET_TRAIN_PARQUET.exists():
                raise FileNotFoundError(
                    f"Train parquet dataset not found at expected path: {DATASET_TRAIN_PARQUET}. "
                    f"Please verify train-00000-of-00001.parquet is located in the project root."
                )
            from scripts.prepare_cache import build_cache
            print("Train cache not found. Triggering automated build...")
            build_cache()

        # 2. Load dataset tensors
        train_cache = torch.load(str(TRAIN_CACHE_FILE), map_location="cpu")
        images = train_cache.get("images")
        labels = train_cache.get("labels")

        if images is None or len(images) == 0:
            raise ValueError(f"Train dataset at {TRAIN_CACHE_FILE} contains 0 image tensors.")
        if labels is None or len(labels) == 0:
            raise ValueError(f"Train dataset at {TRAIN_CACHE_FILE} contains 0 label tensors.")
        if len(images) != len(labels):
            raise ValueError(f"Dataset mismatch: {len(images)} images vs {len(labels)} labels.")

        num_classes = len(CLASS_NAMES)

        # 3. Model initialization sanity check
        try:
            test_model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
            test_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model architecture '{model_name}': {e}")

        # 4. Single-batch forward-loss-backward check
        try:
            dummy_batch = images[:min(2, len(images))].to(self.device)
            dummy_labels = labels[:min(2, len(labels))].to(self.device)
            logits = test_model(dummy_batch)
            if logits.shape != (len(dummy_batch), num_classes):
                raise ValueError(f"Expected model output shape {(len(dummy_batch), num_classes)}, got {logits.shape}")
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, dummy_labels)
            loss.backward()
        except Exception as e:
            raise RuntimeError(f"Pre-training forward/backward sanity check failed for '{model_name}': {e}")

        return True

    def start_training(self, config: Dict[str, Any]):
        with self._lock:
            if self.status["status"] == "training":
                return {
                    "status": "error",
                    "success": False,
                    "message": "A training session is already actively running."
                }
            
            # Run pre-training validation check synchronously
            try:
                self.validate_pretraining(config)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "success": False,
                    "message": f"Pre-training validation failed: {str(e)}"
                }

            job_id = f"job-{uuid.uuid4().hex[:8]}"
            self.current_job_id = job_id
            self.status["status"] = "training"
            self.status["job_id"] = job_id
            self.status["error_message"] = None
            self.status["history"] = []
            self.status["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status["completed_at"] = None

        thread = threading.Thread(target=self._run_training, args=(config, job_id), daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "success": True,
            "job_id": job_id,
            "message": "Training started successfully on CPU"
        }

    def _run_training(self, config: Dict[str, Any], job_id: str):
        try:
            model_name = config.get("model_name", "mobilenet_v3")
            epochs = int(config.get("epochs", 5))
            batch_size = int(config.get("batch_size", 32))
            lr = float(config.get("learning_rate", 0.001))
            val_split = float(config.get("validation_split", 0.2))
            seed = int(config.get("random_seed", 42))
            optimizer_name = config.get("optimizer", "Adam")
            freeze_backbone = config.get("freeze_backbone", True)

            torch.manual_seed(seed)

            with self._lock:
                self.status["total_epochs"] = epochs
                self.status["model_name"] = model_name

            train_cache = torch.load(str(TRAIN_CACHE_FILE), map_location="cpu")
            images = train_cache["images"] # (N, 3, 224, 224)
            labels = train_cache["labels"]

            full_dataset = TensorDataset(images, labels)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_ds, val_ds = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # Build Model and send to device
            model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True)
            model.to(self.device)
            
            # Fine-tuning: if transfer learning model with .features, optionally freeze earlier layers
            if freeze_backbone and hasattr(model, 'features'):
                for param in list(model.features.parameters())[:-4]:
                    param.requires_grad = False

            criterion = nn.CrossEntropyLoss()
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            if optimizer_name.lower() == "adamw":
                optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
            elif optimizer_name.lower() == "sgd":
                optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

            best_val_acc = 0.0
            history = []

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                
                # Training phase
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for x_b, y_b in train_loader:
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    preds = model(x_b)
                    loss = criterion(preds, y_b)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * x_b.size(0)
                    _, pred_classes = torch.max(preds, 1)
                    total += y_b.size(0)
                    correct += (pred_classes == y_b).sum().item()

                train_loss = running_loss / max(total, 1)
                train_acc = correct / max(total, 1)

                # Validation phase
                model.eval()
                val_loss_running = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        preds = model(x_val)
                        loss = criterion(preds, y_val)
                        val_loss_running += loss.item() * x_val.size(0)
                        _, pred_classes = torch.max(preds, 1)
                        val_total += y_val.size(0)
                        val_correct += (pred_classes == y_val).sum().item()

                val_loss = val_loss_running / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)
                epoch_dur = round(time.time() - epoch_start, 2)

                log_entry = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "duration_seconds": epoch_dur
                }
                history.append(log_entry)

                with self._lock:
                    self.status["current_epoch"] = epoch
                    self.status["progress_percent"] = round((epoch / epochs) * 100, 1)
                    self.status["current_train_loss"] = log_entry["train_loss"]
                    self.status["current_val_loss"] = log_entry["val_loss"]
                    self.status["current_train_acc"] = log_entry["train_acc"]
                    self.status["current_val_acc"] = log_entry["val_acc"]
                    self.status["history"] = list(history)

                # Save best checkpoint
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "num_classes": len(CLASS_NAMES),
                        "classes": CLASS_NAMES,
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "config": config,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, str(BEST_MODEL_PATH))

            # Save latest checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "num_classes": len(CLASS_NAMES),
                "classes": CLASS_NAMES,
                "epoch": epochs,
                "val_acc": val_acc,
                "config": config,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, str(LATEST_MODEL_PATH))

            # Save training history JSON
            with open(HISTORY_FILE, "w") as f:
                json.dump({
                    "model_name": model_name,
                    "epochs": epochs,
                    "history": history,
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)

            with self._lock:
                self.status["status"] = "completed"
                self.status["progress_percent"] = 100.0
                self.status["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Invalidate cached models in prediction service so new model is loaded immediately!
            try:
                from .predict_service import predict_service
                predict_service.reload_model()
            except Exception as e:
                print(f"Warning: could not reload predict_service model: {e}")

            # Trigger automated evaluation on test set!
            print("Training finished! Triggering evaluation on actual test set...")
            eval_service.evaluate_model()

        except Exception as e:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.status["status"] = "error"
                self.status["error_message"] = str(e)

train_service = TrainService()
