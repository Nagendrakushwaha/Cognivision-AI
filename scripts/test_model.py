import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import time

def test_model():
    print("Testing PyTorch MobileNetV3-Small on CPU...")
    start = time.time()
    
    # Instantiate lightweight MobileNetV3-Small
    # We can use weights=None or lightweight pretrained
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Replace final linear layer for 6 classes
    num_classes = 6
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    model.eval()
    
    # Dummy input (1, 3, 224, 224)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Forward pass output shape:", out.shape)
    probs = torch.softmax(out, dim=1)
    print("Softmax probabilities shape:", probs.shape)
    print(f"Elapsed time: {time.time() - start:.2f}s")
    print("PyTorch model ready and working on CPU!")

if __name__ == "__main__":
    test_model()
