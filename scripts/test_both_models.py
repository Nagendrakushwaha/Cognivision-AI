import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))
from backend.app.models.architectures import build_model, GradCAM

def test_models():
    print("Testing CogniNetCNN and MobileNetV3 with Grad-CAM...")
    for model_name in ["cogninet_cnn", "mobilenet_v3"]:
        print(f"\n--- Testing {model_name} ---")
        model = build_model(model_name=model_name, num_classes=6, pretrained=False)
        model.eval()
        
        # Test forward pass with dummy batch
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"Forward output shape: {out.shape}")
        assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"
        
        # Test backward pass
        loss_fn = torch.nn.CrossEntropyLoss()
        y = torch.tensor([0, 3], dtype=torch.long)
        loss = loss_fn(out, y)
        loss.backward()
        print(f"Loss backward succeeded. Loss = {loss.item():.4f}")
        
        # Test Grad-CAM
        print("Testing Grad-CAM generation...")
        cam = GradCAM(model)
        x_single = torch.randn(1, 3, 224, 224)
        try:
            res = cam.generate(x_single, target_class=1)
            print(f"GradCAM succeeded for {model_name}! Heatmap shape: {res['heatmap'].shape}, min: {res['heatmap'].min():.3f}, max: {res['heatmap'].max():.3f}")
        except Exception as e:
            import traceback
            print(f"GradCAM FAILED for {model_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_models()
