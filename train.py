import argparse
import sys
import os
import time
import json
import torch

# Ensure current working directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.app.services.train_service import train_service
from backend.app.services.eval_service import eval_service

def main():
    parser = argparse.ArgumentParser(description="Train Cognivision AI Model on SROIE Receipt Dataset")
    parser.add_argument("--model", type=str, default="mobilenet_v3", choices=["mobilenet_v3", "cogninet_cnn"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"], help="Optimizer")
    
    args = parser.parse_args()

    print("=" * 60)
    print("           COGNIVISION AI - MODEL TRAINING PIPELINE          ")
    print("=" * 60)
    print(f"Model Architecture : {args.model}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Optimizer          : {args.optimizer}")
    print(f"Hardware Mode      : CPU (Ryzen 5 5500U optimized)")
    print("=" * 60)

    # Check cache existence; if missing, build it
    if not os.path.exists("artifacts/cache/train_cache.pt") or not os.path.exists("artifacts/cache/test_cache.pt"):
        print("Dataset cache not found. Generating preprocessed cache from parquet files...")
        from scripts.prepare_cache import build_cache
        build_cache()

    config = {
        "model_name": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "validation_split": args.val_split,
        "random_seed": args.seed,
        "optimizer": args.optimizer,
        "freeze_backbone": True
    }

    # Start training synchronously in CLI mode
    print("\nStarting training execution...")
    train_service._run_training(config)
    
    status = train_service.get_status()
    if status["status"] == "error":
        print(f"\n[ERROR] Training failed: {status['error_message']}")
        sys.exit(1)

    print("\nTraining completed successfully!")
    print("Epochs history:")
    for h in status["history"]:
        print(f"  Epoch {h['epoch']}/{args.epochs} - Train Loss: {h['train_loss']:.4f}, Train Acc: {h['train_acc']*100:.1f}% | Val Loss: {h['val_loss']:.4f}, Val Acc: {h['val_acc']*100:.1f}% ({h['duration_seconds']}s)")

    print("\nEvaluating trained model on independent test dataset (361 samples)...")
    eval_res = eval_service.evaluate_model()
    
    print("\n" + "=" * 60)
    print("                   TEST SET EVALUATION METRICS              ")
    print("=" * 60)
    print(f"Overall Accuracy   : {eval_res['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score     : {eval_res['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1 Score  : {eval_res['weighted_f1'] * 100:.2f}%")
    print(f"Macro Precision    : {eval_res['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall       : {eval_res['macro_recall'] * 100:.2f}%")
    if eval_res.get("roc_auc"):
        print(f"ROC-AUC (OVR)      : {eval_res['roc_auc']:.4f}")
        
    print("\nClass-wise Performance:")
    print(f"{'Class Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for c in eval_res["class_wise_metrics"]:
        print(f"{c['class_name']:<28} | {c['precision']*100:>8.1f}% | {c['recall']*100:>8.1f}% | {c['f1_score']*100:>8.1f}% | {c['support']:>8}")

    print("\nModel saved to: models/best_model.pt")
    print("Metrics saved to: artifacts/evaluation_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
