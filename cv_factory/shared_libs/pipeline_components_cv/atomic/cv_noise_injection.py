# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_noise_injection.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.augmenters.atomic.noise_injection import NoiseInjection 

logger = logging.getLogger(__name__)

class CVNoiseInjection(BaseComponent):
    """
    Adapter component for injecting random noise (e.g., Gaussian, Salt and Pepper).
    
    Adheres to BaseComponent, manages parameters (mean, std), and delegates execution.
    """
    
    def __init__(self, noise_type: str = 'gaussian', strength: float = 0.05):
        """
        Initializes the Adapter and the Atomic Augmenter.
        """
        # 1. Manage State/Parameters
        self.noise_type = noise_type
        self.strength = strength
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_augmenter = NoiseInjection(
            noise_type=self.noise_type, 
            strength=self.strength
        )
        
        logger.info(f"Initialized CVNoiseInjection Adapter with type: {self.noise_type}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVNoiseInjection':
        logger.info("CVNoiseInjection is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies noise injection by delegating execution to the atomic augmenter.
        """
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        return self.atomic_augmenter.transform(X) 

    def save(self, path: str) -> None:
        """
        Saves the component's state (parameters).
        """
        state = {'noise_type': self.noise_type, 'strength': self.strength}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVNoiseInjection state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic augmenter.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.noise_type = state['noise_type']
        self.strength = state['strength']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_augmenter = NoiseInjection(
            noise_type=self.noise_type, 
            strength=self.strength
        )
        logger.info(f"CVNoiseInjection state loaded from {path}.")