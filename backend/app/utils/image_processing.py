import io
import base64
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torchvision.transforms as transforms
import torch

# Standard ImageNet normalization for PyTorch models
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def image_to_base64(image: Image.Image, format="JPEG", quality=85) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def preprocess_image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = INFERENCE_TRANSFORMS(image)
    return tensor.unsqueeze(0) # (1, 3, 224, 224)

def generate_gradcam_overlay(original_image: Image.Image, heatmap: np.ndarray, alpha=0.55) -> str:
    """
    Overlays a Grad-CAM heatmap onto the original image and returns base64 JPEG string.
    """
    orig_w, orig_h = original_image.size
    
    # Resize heatmap to match original image dimensions using PIL bilinear
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0
    
    # Apply JET colormap (RGB)
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_resized)[:, :, :3] # (H, W, 3) in [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored_heatmap_uint8)
    
    # Blend with original image
    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
    return image_to_base64(blended)
