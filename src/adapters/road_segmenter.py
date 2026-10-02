"""
Road Segmenter Adapter
Adapter for idd-road-segmentation U-Net model (drivable road surface vs non-drivable).

IMPORTANT: Requires custom trained weights from IDD dataset.
Do NOT substitute generic models for road segmentation.
"""
import os
from typing import Dict, Any
import numpy as np

from .base import BaseModelAdapter

class RoadSegmenterAdapter(BaseModelAdapter):
    """
    Adapter for idd-road-segmentation binary drivable-area U-Net model.
    """

    def __init__(self, weights_path: str = "weights/road_segmentation_best.pt"):
        if not os.path.exists(weights_path) and os.path.exists("idd-road-segmentation/runs/2026-08-11_1628/best.pt"):
            weights_path = "idd-road-segmentation/runs/2026-08-11_1628/best.pt"
        super().__init__(weights_path=weights_path)
        self.model_name = "IDD U-Net (ResNet50 Encoder)"
        self.load()

    def load(self) -> bool:
        if not self.weights_path or not os.path.exists(self.weights_path):
            self.status = "FAIL"
            self.status_message = f"Missing weights: Pretrained checkpoint not found at '{self.weights_path}'"
            self.is_loaded = False
            return False

        try:
            import torch
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                classes=2
            )
            state_dict = torch.load(self.weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.is_loaded = True
            self.status = "PASS"
            self.status_message = f"Component loaded successfully ({self.weights_path})"
            return True
        except ImportError as e:
            self.status = "FAIL"
            self.status_message = f"Dependency error: Required PyTorch/SMP module missing ({str(e)})"
            self.is_loaded = False
            return False
        except Exception as e:
            self.status = "FAIL"
            self.status_message = f"Failed to load segmentation checkpoint: {str(e)}"
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Cannot run inference: {self.status_message}")

        import torch
        import cv2

        h, w = image.shape[:2]
        resized = cv2.resize(image, (512, 512))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        drivable_pct = float(np.sum(mask_resized == 1) / mask_resized.size * 100.0)

        return {
            "component": "road_segmenter",
            "model": self.model_name,
            "drivable_area_percentage": round(drivable_pct, 2),
            "mask_shape": mask_resized.shape
        }
