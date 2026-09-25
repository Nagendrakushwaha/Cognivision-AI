import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000/api"

def test_training_pipeline():
    print("=== TEST 1 & 2: Starting 1-epoch CPU training for cogninet_cnn ===")
    config = {
        "model_name": "cogninet_cnn",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "validation_split": 0.2,
        "random_seed": 42
    }
    
    req = urllib.request.Request(
        f"{BASE_URL}/model/train",
        data=json.dumps(config).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Start Response Status: {resp.status}")
            print(f"Response Payload: {json.dumps(data, indent=2)}")
            assert data.get("status") == "started", f"Unexpected status: {data.get('status')}"
            assert "job_id" in data, "No job_id in response"
            print("PASS: Training successfully initiated!")
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTPError {e.code}: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    print("\n=== TEST 3: Monitoring Real Loss/Accuracy Metrics ===")
    max_wait_seconds = 180
    start_time = time.time()
    last_epoch = -1
    
    while time.time() - start_time < max_wait_seconds:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{BASE_URL}/model/status") as resp:
                status = json.loads(resp.read().decode("utf-8"))
                curr_status = status.get("status")
                epoch = status.get("current_epoch", 0)
                train_loss = status.get("current_train_loss")
                val_loss = status.get("current_val_loss")
                train_acc = status.get("current_train_acc")
                val_acc = status.get("current_val_acc")
                progress = status.get("progress_percent", 0)
                
                if epoch != last_epoch or curr_status != "training":
                    print(f"Status: {curr_status} | Epoch: {epoch}/1 | Progress: {progress}% | "
                          f"TrainLoss: {train_loss} | ValLoss: {val_loss} | "
                          f"TrainAcc: {train_acc} | ValAcc: {val_acc}")
                    last_epoch = epoch
                
                if curr_status == "completed":
                    print("\nTraining completed successfully!")
                    print(f"Final History: {json.dumps(status.get('history', []), indent=2)}")
                    assert len(status.get("history", [])) > 0, "No epoch history recorded"
                    print("PASS: Real training metrics verified!")
                    return True
                elif curr_status == "failed":
                    print(f"FAIL: Training failed with error: {status.get('error_message')}")
                    return False
        except Exception as e:
            print(f"Warning while polling: {e}")

    print("FAIL: Timed out waiting for training to complete")
    return False

if __name__ == "__main__":
    success = test_training_pipeline()
    exit(0 if success else 1)
