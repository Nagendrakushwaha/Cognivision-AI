import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))
from fastapi.testclient import TestClient
from backend.app.main import app

def run_tests():
    client = TestClient(app)
    
    print("=== RUNNING FASTAPI END-TO-END INTEGRATION TESTS ===")
    
    # 1. Root & Health
    r = client.get("/")
    assert r.status_code == 200, f"Root failed: {r.text}"
    print("[PASS] GET / :", r.json()["project"])

    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("[PASS] GET /api/health :", r.json())

    # 2. Dataset Summary
    r = client.get("/api/dataset/summary")
    assert r.status_code == 200, f"Dataset summary failed: {r.text}"
    data = r.json()
    assert data["total_records"] == 987, f"Expected 987 records, got {data['total_records']}"
    assert data["train_records"] == 626, f"Expected 626 train, got {data['train_records']}"
    assert data["test_records"] == 361, f"Expected 361 test, got {data['test_records']}"
    print(f"[PASS] GET /api/dataset/summary : Total={data['total_records']}, Classes={data['num_classes']}")

    # 3. Dataset Records
    r = client.get("/api/dataset/records?split=train&page=1&page_size=5")
    assert r.status_code == 200
    records = r.json()["records"]
    assert len(records) == 5
    print(f"[PASS] GET /api/dataset/records : Loaded {len(records)} sample records successfully.")

    # 4. Receipt Details
    r = client.get("/api/dataset/receipt/test/0")
    assert r.status_code == 200
    detail = r.json()
    assert "tokens" in detail and "image_base64" in detail
    print(f"[PASS] GET /api/dataset/receipt/test/0 : Key={detail['key']}, Tokens={len(detail['tokens'])}, ImageB64Len={len(detail['image_base64'])}")

    # 5. Model Status
    r = client.get("/api/model/status")
    assert r.status_code == 200
    status = r.json()
    print(f"[PASS] GET /api/model/status : Status={status['status']}, Epochs={len(status['history'])}")

    # 6. Model Performance
    r = client.get("/api/model/performance")
    assert r.status_code == 200
    perf = r.json()
    assert perf["is_trained"] is True
    print(f"[PASS] GET /api/model/performance : Acc={perf['accuracy']*100:.2f}%, MacroF1={perf['macro_f1']*100:.2f}%, CM={len(perf['confusion_matrix'])}x{len(perf['confusion_matrix'][0])}")

    # 7. Prediction Sample
    r = client.post("/api/predict/sample?split=test&idx=3")
    assert r.status_code == 200
    pred = r.json()
    assert "predicted_class" in pred and "confidence" in pred
    print(f"[PASS] POST /api/predict/sample : Predicted='{pred['predicted_class']}' ({pred['confidence_percentage']}%), Latency={pred['inference_time_ms']}ms")

    # 8. Explainability Sample (Grad-CAM)
    r = client.post("/api/explain/sample?split=test&idx=3")
    assert r.status_code == 200
    explain = r.json()
    assert "overlay_base64" in explain and "region_importance" in explain
    print(f"[PASS] POST /api/explain/sample : Method='{explain['method']}', Regions={explain['region_importance']}")

    # 9. Batch Prediction on Test Set
    r = client.get("/api/predict/batch/sample_test?count=5")
    assert r.status_code == 200
    batch = r.json()
    assert batch["total_processed"] == 5
    print(f"[PASS] GET /api/predict/batch/sample_test : Processed {batch['total_processed']} samples in {batch['inference_time_ms']}ms")

    print("\n>>> ALL API AND ML INTEGRATION TESTS PASSED 100%! <<<")

if __name__ == "__main__":
    run_tests()
