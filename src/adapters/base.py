"""
Base Model Adapter Interface
Defines the standard contract for all ML model adapters in the system.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np

class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    Ensures decoupled model loading, status reporting, and inference execution.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self.is_loaded = False
        self.status = "UNINITIALIZED"
        self.status_message = ""
        self.model = None

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model and weights.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single numpy image frame (BGR format).
        Returns standardized dictionary of model predictions.
        """
        pass

    def get_status(self) -> Tuple[str, str]:
        """
        Returns (STATUS_CODE, STATUS_MESSAGE)
        STATUS_CODE can be: 'PASS', 'FAIL', 'WARN'
        """
        return self.status, self.status_message
