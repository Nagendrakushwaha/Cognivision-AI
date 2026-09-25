import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))
from backend.app.services.eval_service import eval_service

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cognivision AI Model on Test Dataset")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("         COGNIVISION AI - TEST SET EVALUATION PIPELINE       ")
    print("=" * 60)
    print(f"Model Checkpoint : {args.model_path}")

    res = eval_service.evaluate_model(args.model_path)
    if not res.get("is_trained", False):
        print(f"\n[ERROR] Evaluation failed: {res.get('error', 'Unknown error')}")
        sys.exit(1)

    print(f"Evaluated Samples: {res['evaluated_samples']}")
    print(f"Model Evaluated  : {res['model_name']}")
    print("-" * 60)
    print(f"Accuracy         : {res['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score   : {res['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1 Score: {res['weighted_f1'] * 100:.2f}%")
    print(f"Macro Precision  : {res['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall     : {res['macro_recall'] * 100:.2f}%")
    if res.get("roc_auc") is not None:
        print(f"ROC-AUC (OVR)    : {res['roc_auc']:.4f}")

    print("\nConfusion Matrix (6x6):")
    classes = res["classes"]
    print(f"{'Actual \\ Pred':<26} | " + " | ".join([f"{c[:6]:>6}" for c in classes]))
    print("-" * 75)
    for i, row in enumerate(res["confusion_matrix"]):
        row_str = " | ".join([f"{val:>6}" for val in row])
        print(f"{classes[i]:<26} | {row_str}")

    print("\nClass-Wise Breakdown:")
    print(f"{'Class Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for c in res["class_wise_metrics"]:
        print(f"{c['class_name']:<28} | {c['precision']*100:>8.1f}% | {c['recall']*100:>8.1f}% | {c['f1_score']*100:>8.1f}% | {c['support']:>8}")

    print("\nSaved evaluation report to artifacts/evaluation_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
