import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Pool gradients across channels
        # gradients: [1, C, H, W], activations: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, output

if __name__ == "__main__":
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap, pred_class, out = cam_engine.generate(x)
    print("Grad-CAM succeeded!")
    print(f"Heatmap shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
    print(f"Predicted class index: {pred_class}")
