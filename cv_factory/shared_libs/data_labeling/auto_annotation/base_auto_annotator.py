# shared_libs/data_labeling/auto_annotation/base_auto_annotator.py (Hardened)

import numpy as np
import abc
import logging
from typing import List, Dict, Any, Union
from torch import nn, Tensor

# Import các Label Schema đã được định nghĩa ở bước I
from ...configs.label_schema import ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel, StandardLabel # <--- Updated Import

logger = logging.getLogger(__name__)

class BaseAutoAnnotator(abc.ABC):
    """
    Abstract Base Class (ABC) for all automatic labeling methods (Proposal Annotators).
    
    Defines the interface for model loading and label generation from raw data/images.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the annotator.
        model (Union[nn.Module, Any]): The loaded machine learning model.
        min_confidence (float): The minimum confidence threshold to accept an auto-generated label.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Annotator and loads the necessary model.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.min_confidence: float = config.get("min_confidence", 0.7)
        self.model: Union[nn.Module, Any] = self._load_model()
        

    def _load_model(self) -> Union[nn.Module, Any]:
        """
        Default method to load the model artifact. Subclasses should override this 
        for specific framework loading logic (PyTorch, TensorFlow, ONNX).
        
        Checks for required 'model_path' and 'model_type' in config.

        Returns:
            The loaded model object or None if path is missing.
        """
        model_path = self.config.get("model_path")
        model_type = self.config.get("model_type", "default")
        
        if not model_path:
            logger.warning(f"No model path provided for {self.__class__.__name__}. Running in heuristic/dummy mode.")
            return None
            
        # Hardening: Add check for model path existence before attempting load
        # NOTE: This requires integration with the Data Ingestion layer (Connector)
        # For now, we simulate the loading process.
        logger.info(f"Loading {model_type} model from {model_path}...")
        
        # In a real system, you would call a ModelLoader utility here:
        # return ModelLoader.load(model_path, model_type)
        
        return object() # Return a dummy object

    @abc.abstractmethod
    def _run_inference(self, image_data: np.ndarray) -> Any:
        """
        Runs inference on the loaded model and returns the raw prediction result.
        
        Args:
            image_data: The input image as a NumPy array (H, W, C).
            
        Returns:
            Any: The raw, unnormalized output from the ML model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _normalize_output(self, raw_prediction: Any, metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes the raw prediction into validated Pydantic Label Schema objects.
        
        Args:
            raw_prediction: The raw output from `_run_inference`.
            metadata: Contains contextual information like 'image_path' and 'image_data'.
            
        Returns:
            List[StandardLabel]: A list of Pydantic label objects that passed confidence/schema validation.
        """
        raise NotImplementedError

    def annotate(self, raw_input: Dict[str, Any]) -> List[StandardLabel]:
        """
        Public interface: Executes the entire automatic annotation process.
        
        Args:
            raw_input: Dictionary containing 'image_data' (np.ndarray) and 'image_path'.
            
        Returns:
            List[StandardLabel]: List of validated auto-generated labels.
            
        Raises:
            ValueError: If required 'image_data' is missing.
        """
        image_data = raw_input.get("image_data")
        
        if image_data is None:
            raise ValueError(f"Image data is required for annotation.")

        # 1. Run Inference
        raw_prediction = self._run_inference(image_data)
        
        # 2. Normalize and Validate (metadata includes raw_input)
        # Hardening: Ensure image_path is present in metadata for BaseLabel validation
        if 'image_path' not in raw_input:
             raw_input['image_path'] = f"in_memory_{hash(image_data.tobytes())}"
             
        return self._normalize_output(raw_prediction, raw_input)