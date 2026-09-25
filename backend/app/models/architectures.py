import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class CogniNetCNN(nn.Module):
    """
    Lightweight, high-speed custom CNN tailored for receipt visual document classification
    on CPU hardware without CUDA requirement.
    """
    def __init__(self, num_classes=6, in_channels=3):
        super(CogniNetCNN, self).__init__()
        
        # Stage 1: (B, 3, 224, 224) -> (B, 32, 112, 112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Stage 4: (B, 128, 28, 28) -> (B, 256, 14, 14)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

    def get_cam_target_layer(self):
        # Target layer for Grad-CAM
        return self.conv4[0]


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-Small optimized for transfer learning and fast CPU inference.
    """
    def __init__(self, num_classes=6, pretrained=True):
        super(MobileNetV3Classifier, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        in_features = base_model.classifier[0].in_features # 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_cam_target_layer(self):
        return self.features[-1]


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visual explainability engine for convolutional vision models.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.get_cam_target_layer()
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

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Spatial average of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam_np = cam.squeeze().cpu().numpy()
        min_v, max_v = np.min(cam_np), np.max(cam_np)
        if max_v > min_v:
            cam_np = (cam_np - min_v) / (max_v - min_v + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)
            
        confidence = probs[0, target_class].item()
        
        return {
            "heatmap": cam_np,
            "target_class_idx": target_class,
            "confidence": confidence,
            "all_probabilities": probs[0].detach().cpu().numpy().tolist()
        }


def build_model(model_name="mobilenet_v3", num_classes=6, pretrained=True):
    name = model_name.lower().replace("-", "_")
    if "cogninet" in name or "cnn" in name:
        return CogniNetCNN(num_classes=num_classes)
    else:
        return MobileNetV3Classifier(num_classes=num_classes, pretrained=pretrained)
