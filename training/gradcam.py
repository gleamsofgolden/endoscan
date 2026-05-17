"""
Grad-CAM Heatmap Generator
===========================
Produces visual explanations of model predictions by highlighting
which image regions influenced the classification decision.

Critical for medical AI — doctors need to see WHY the model decided
what it decided, not just the prediction score.

Usage:
    from gradcam import GradCAM, overlay_heatmap
    
    cam = GradCAM(model)
    heatmap = cam.generate(image_tensor)
    result_img = overlay_heatmap(original_image, heatmap)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Hooks into the last convolutional layer to capture gradients
    and activations, then combines them to produce a spatial heatmap.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Auto-detect last conv layer for EfficientNet
        if target_layer is None:
            target_layer = self._find_last_conv(model)

        self._register_hooks(target_layer)

    def _find_last_conv(self, model):
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        return last_conv

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        """
        Args:
            image_tensor: (1, C, H, W) tensor, normalised
            class_idx: which class to explain (None = predicted class)
        
        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions → importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx


def overlay_heatmap(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Blend Grad-CAM heatmap onto the original image.

    Args:
        original_image: PIL Image or (H, W, 3) numpy array
        heatmap: (h, w) numpy array from GradCAM.generate()
        alpha: heatmap opacity (0 = invisible, 1 = fully opaque)
        colormap: OpenCV colormap

    Returns:
        PIL Image with heatmap overlaid
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image.convert("RGB"))
    else:
        img_np = original_image.copy()

    h, w = img_np.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


def predict_with_explanation(model, image_tensor, original_image, class_names, device):
    """
    Full pipeline: predict + generate explanation heatmap.

    Returns dict with:
        prediction    : class name
        confidence    : float 0–1
        probabilities : {class_name: probability}
        heatmap_image : PIL Image with Grad-CAM overlay
        gradcam_raw   : raw heatmap array
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    cam_generator = GradCAM(model)

    with torch.enable_grad():
        heatmap, class_idx = cam_generator.generate(image_tensor.unsqueeze(0))

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    heatmap_img = overlay_heatmap(original_image, heatmap)

    return {
        "prediction":    class_names[class_idx],
        "confidence":    float(probs[class_idx]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
        "heatmap_image": heatmap_img,
        "gradcam_raw":   heatmap,
    }
