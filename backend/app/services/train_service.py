import os
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from ..models.architectures import build_model
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
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.train_cache_file = os.path.join(data_dir, "artifacts", "cache", "train_cache.pt")
        self.history_file = os.path.join(data_dir, "artifacts", "training_history.json")
        self.model_save_dir = os.path.join(data_dir, "models")
        
        self.status = {
            "status": "idle", # "idle", "training", "completed", "error"
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
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
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

    def start_training(self, config: Dict[str, Any]):
        with self._lock:
            if self.status["status"] == "training":
                return {"success": False, "message": "Training is already in progress."}
            self.status["status"] = "training"
            self.status["error_message"] = None
            self.status["history"] = []
            self.status["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status["completed_at"] = None

        thread = threading.Thread(target=self._run_training, args=(config,), daemon=True)
        thread.start()
        return {"success": True, "message": "Training started successfully."}

    def _run_training(self, config: Dict[str, Any]):
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

            if not os.path.exists(self.train_cache_file):
                raise FileNotFoundError(f"Train cache not found at {self.train_cache_file}. Please run cache preparation.")

            train_cache = torch.load(self.train_cache_file, map_location="cpu")
            images = train_cache["images"] # (626, 3, 224, 224)
            labels = train_cache["labels"]

            full_dataset = TensorDataset(images, labels)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # Build Model
            model = build_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True)
            
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

                # Save best model
                os.makedirs(self.model_save_dir, exist_ok=True)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_save_path = os.path.join(self.model_save_dir, "best_model.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "num_classes": len(CLASS_NAMES),
                        "classes": CLASS_NAMES,
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "config": config
                    }, best_save_path)

            # Save latest model
            latest_save_path = os.path.join(self.model_save_dir, "latest_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "num_classes": len(CLASS_NAMES),
                "classes": CLASS_NAMES,
                "epoch": epochs,
                "val_acc": val_acc,
                "config": config
            }, latest_save_path)

            # Save history to JSON
            with open(self.history_file, "w") as f:
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

            # Automatically evaluate on test set after training!
            print("Training finished! Triggering evaluation on actual test set...")
            eval_service.evaluate_model()

        except Exception as e:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.status["status"] = "error"
                self.status["error_message"] = str(e)

train_service = TrainService()
