import os
import io
import json
import torch
import pyarrow.parquet as pq
import urllib.request
import urllib.error
from PIL import Image

BASE_URL = "http://127.0.0.1:8000/api"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pt")
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "latest_model.pt")
TEST_PARQUET = os.path.join(PROJECT_ROOT, "test-00000-of-00001.parquet")

def test_downstream():
    print("=== TEST 4: Verifying Checkpoint Creation ===")
    assert os.path.exists(BEST_MODEL_PATH) or os.path.exists(LATEST_MODEL_PATH), "Checkpoint does not exist!"
    target_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else LATEST_MODEL_PATH
    print(f"Checkpoint found at: {target_path} (size: {os.path.getsize(target_path)} bytes)")

    print("\n=== TEST 5: Loading Checkpoint and Inspecting Keys ===")
    ckpt = torch.load(target_path, map_location="cpu")
    print(f"Keys in checkpoint: {list(ckpt.keys())}")
    print(f"Model Architecture: {ckpt.get('model_name')}")
    print(f"Classes: {ckpt.get('classes')}")
    print(f"Metrics: {ckpt.get('metrics')}")
    assert "model_state_dict" in ckpt, "Missing model_state_dict"
    assert ckpt.get("num_classes") == 6, f"Expected 6 classes, got {ckpt.get('num_classes')}"
    print("PASS: Checkpoint loaded and validated!")

    print("\n=== Checking Active Model in /api/health ===")
    with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
        health_data = json.loads(resp.read().decode())
        print(f"Health Response: {json.dumps(health_data, indent=2)}")
        assert health_data.get("model_trained") is True, "model_trained is not True"
        active_model = health_data.get("active_model", {})
        print(f"Active Model Display Name: {active_model.get('display_name')}")
        print(f"Active Model Architecture: {active_model.get('model_name')}")
        assert active_model.get("model_name") == "cogninet_cnn", f"Expected cogninet_cnn, got {active_model.get('model_name')}"

    print("\n=== TEST 6: Extracting a real receipt image from SROIE parquet ===")
    table = pq.read_table(TEST_PARQUET)
    row = table.to_pandas().iloc[0]
    img_bytes = row['image']['bytes']
    receipt_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Extracted receipt '{row['key']}' size: {receipt_img.size}")
    
    # Save a temporary JPG receipt
    sample_jpg_path = os.path.join(PROJECT_ROOT, "artifacts", "sample_receipt.jpg")
    receipt_img.save(sample_jpg_path, format="JPEG")
    print(f"Saved test receipt to: {sample_jpg_path}")

    print("\n=== TEST 7 & 8: Generating Prediction & Real Grad-CAM via /api/predict/image ===")
    import requests
    with open(sample_jpg_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": ("sample_receipt.jpg", f, "image/jpeg")},
            data={"include_gradcam": "true"}
        )
    print(f"Predict Response Code: {response.status_code}")
    res_json = response.json()
    print("Prediction Result:")
    print(f"  Predicted Class: {res_json.get('predicted_class')}")
    print(f"  Confidence: {res_json.get('confidence')}")
    print(f"  Top Classes: {json.dumps(res_json.get('top_k', []), indent=2)}")
    print(f"  Inference Latency: {res_json.get('inference_time_ms')} ms")
    print(f"  Model Architecture Used: {res_json.get('model_used')}")
    assert res_json.get("predicted_class") is not None, "Missing predicted class"
    assert res_json.get("confidence") is not None, "Missing confidence"
    assert res_json.get("gradcam_base64") is not None, "Grad-CAM heatmap base64 is missing!"
    print(f"  Grad-CAM Base64 length: {len(res_json.get('gradcam_base64'))} chars")
    print("PASS: Real prediction and Grad-CAM successfully generated!")

    print("\n=== Testing Explainability Endpoint /api/explain/image directly ===")
    with open(sample_jpg_path, "rb") as f:
        exp_response = requests.post(
            f"{BASE_URL}/explain/image",
            files={"file": ("sample_receipt.jpg", f, "image/jpeg")}
        )
    print(f"Explain Response Code: {exp_response.status_code}")
    exp_json = exp_response.json()
    assert exp_json.get("overlay_base64") is not None, "Explain endpoint missing overlay_base64"
    assert exp_json.get("heatmap_base64") is not None, "Explain endpoint missing heatmap_base64"
    print(f"  Explain Overlay Base64 length: {len(exp_json.get('overlay_base64'))} chars")
    print("PASS: /api/explain/image passed without error!")

    print("\n=== TEST 9: Verifying Invalid / Non-Receipt Image Handling ===")
    # Create a blank 10x10 image or non-receipt image
    dummy_img = Image.new("RGB", (20, 20), color=(128, 128, 128))
    dummy_bytes = io.BytesIO()
    dummy_img.save(dummy_bytes, format="PNG")
    dummy_bytes.seek(0)
    
    inv_response = requests.post(
        f"{BASE_URL}/predict/image",
        files={"file": ("dummy.png", dummy_bytes, "image/png")},
        data={"include_gradcam": "true"}
    )
    print(f"Dummy Image Predict Status: {inv_response.status_code}")
    assert inv_response.status_code == 200, f"Server crashed on dummy image! Code: {inv_response.status_code}"
    inv_json = inv_response.json()
    print(f"  Dummy Image Prediction: {inv_json.get('predicted_class')} (Confidence: {inv_json.get('confidence')})")

    # Send random corrupt text bytes
    corrupt_response = requests.post(
        f"{BASE_URL}/predict/image",
        files={"file": ("corrupt.txt", io.BytesIO(b"not an image file at all"), "text/plain")},
        data={"include_gradcam": "true"}
    )
    print(f"Corrupt File Status Code: {corrupt_response.status_code} (Expected 400)")
    assert corrupt_response.status_code == 400, f"Expected 400 for corrupt file, got {corrupt_response.status_code}"
    print(f"  Corrupt File Detail: {corrupt_response.json().get('detail')}")
    print("PASS: Invalid/non-receipt images properly handled without crashing!")

    return True

if __name__ == "__main__":
    success = test_downstream()
    exit(0 if success else 1)
