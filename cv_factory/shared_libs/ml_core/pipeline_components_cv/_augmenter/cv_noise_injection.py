# cv_factory/shared_libs/ml_core/pipeline_components_cv/_augmenter/cv_noise_injection.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.augmenters.atomic.noise_injection import NoiseInjection 

logger = logging.getLogger(__name__)

class CVNoiseInjection(BaseComponent):
    """
    Adapter component for injecting random noise (e.g., Gaussian, Salt and Pepper).
    
    This is an X-only augmentation (REQUIRES_TARGET_DATA=False).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, noise_type: str = 'gaussian', strength: float = 0.05):
        """
        Initializes the Adapter and the Atomic Augmenter.
        
        Args:
            noise_type (str): Type of noise to inject.
            strength (float): Intensity of the noise.
        """
        self.noise_type = noise_type
        self.strength = strength
        
        self.atomic_augmenter = NoiseInjection(
            noise_type=self.noise_type, 
            strength=self.strength
        )
        
        logger.info(f"Initialized CVNoiseInjection Adapter with type: {self.noise_type}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVNoiseInjection':
        logger.info("CVNoiseInjection is stateless, no fitting required.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies noise injection by delegating execution to the atomic augmenter.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The augmented image array(s).
        """
        # Atomic logic does not need y
        return self.atomic_augmenter.transform(X) 

    def save(self, path: str) -> None:
        """Saves the component's state (parameters) using the utility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {'noise_type': self.noise_type, 'strength': self.strength}
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic augmenter.
        """
        state = load_artifact(path)
            
        self.noise_type = state['noise_type']
        self.strength = state['strength']
        
        self.atomic_augmenter = NoiseInjection(
            noise_type=self.noise_type, 
            strength=self.strength
        )