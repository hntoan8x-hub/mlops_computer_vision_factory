# cv_factory/shared_libs/inference/cv_predictor.py (UPDATED WITH ADAPTER LOGIC)

import logging
from typing import Any, Dict, List, Union
import numpy as np
import warnings
import torch 
import base64 # Required for decoding API/Kafka input
import io
from PIL import Image # Required for decoding image bytes

# --- Import Abstractions and Core Services ---
from .base_cv_predictor import BaseCVPredictor, RawInput, ModelInput, PredictionOutput 
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator
from shared_libs.ml_core.mlflow_service.implementations.mlflow_client_wrapper import MLflowClientWrapper 

logger = logging.getLogger(__name__)

# Helper function for decoding (mimicking core_utils.image_utils)
def _decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """Converts a base64 encoded string back into an RGB NumPy array."""
    try:
        # 1. Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        # 2. Open image from bytes stream using PIL
        img_stream = io.BytesIO(img_bytes)
        with Image.open(img_stream) as img:
            # 3. Convert to RGB and then to NumPy array
            return np.array(img.convert("RGB"))
    except Exception as e:
        raise ValueError(f"Base64 decoding or PIL conversion failed: {e}")

class CVPredictor(BaseCVPredictor):
    """
    A concrete implementation of BaseCVPredictor, now including the Adapter/Router 
    logic within preprocess() to handle various raw data inputs (File, Stream, Base64).
    """

    def __init__(self, predictor_id: str, config: Dict[str, Any], postprocessor: Any):
        # 1. Base Init (Handles DI for postprocessor and config validation)
        super().__init__(predictor_id, config, postprocessor) 

        self.model_config = self.config.get('model', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        # 2. Initialize Preprocessing Orchestrator (Context must be 'inference')
        self.preprocessor = CVPreprocessingOrchestrator(
            config=self.preprocessing_config,
            context='inference'
        )
        
        self.mlflow_client = MLflowClientWrapper()
        self.device = torch.device(self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
    # --- BaseCVPredictor Contract Implementation ---

    def load_model(self, model_uri: str) -> None:
        """Loads the trained model artifact using the MLflow client."""
        if self.is_loaded:
            logger.info(f"[{self.predictor_id}] Model already loaded.")
            return

        logger.info(f"[{self.predictor_id}] Loading model from URI: {model_uri}")
        
        try:
            self.model = self.mlflow_client.load_model(model_uri=model_uri)
            
            if isinstance(self.model, torch.nn.Module):
                self.model.to(self.device)
                self.model.eval()

            self.is_loaded = True
            logger.info(f"[{self.predictor_id}] Model loaded successfully on device: {self.device}")

        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load model from {model_uri}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def preprocess(self, raw_input: RawInput) -> ModelInput:
        """
        Applies necessary transformations. This method now acts as the Adapter/Router 
        to standardize input formats (Base64/JSON) into a NumPy array before 
        calling the CVPreprocessingOrchestrator.
        """
        logger.debug(f"[{self.predictor_id}] Running preprocessing pipeline.")
        
        # --- 1. ADAPTER/ROUTER LOGIC ---
        image_data: Union[np.ndarray, List[np.ndarray], None] = None
        
        if isinstance(raw_input, (np.ndarray, list)):
            # Case 1: Stream Connector (Camera) or Batch Connector (Video/Image) already provides NumPy/List[NumPy]
            image_data = raw_input
            logger.debug("Input type is native NumPy/List. Skipping Base64 decode.")
            
        elif isinstance(raw_input, dict):
            # Case 2: API Connector or Kafka Message (JSON/Dict)
            # Look for common keys containing Base64 data
            base64_string = raw_input.get('image_base64') or raw_input.get('data')
            if base64_string and isinstance(base64_string, str):
                try:
                    # Use the local helper function to decode
                    image_data = _decode_base64_to_numpy(base64_string)
                    logger.info("Decoded Base64 input successfully.")
                except Exception as e:
                    raise ValueError(f"Input dictionary contains invalid Base64 image data: {e}") from e
            else:
                raise ValueError("Input dictionary from API/Kafka does not contain valid Base64 image data.")
        
        else:
            raise TypeError(f"Unsupported RawInput type: {type(raw_input)}. Must be NumPy array, List[NumPy array], or Dict[Base64 string].")

        if image_data is None:
             raise ValueError("Raw input was processed but resulted in None image data.")
        
        # --- 2. CORE PROCESSING DELEGATION ---
        
        # The Orchestrator now receives the standardized NumPy array
        processed_output = self.preprocessor.run(image_data)
        
        # --- 3. FRAMEWORK ADAPTER (FINAL STEP) ---
        
        # Final step to convert NumPy array to batched Tensor for PyTorch
        if isinstance(processed_output, np.ndarray):
             # Add batch dimension and convert to PyTorch Tensor
             # HWC -> CHW (permute), add Batch dimension (unsqueeze)
             tensor_input = torch.from_numpy(processed_output).permute(2, 0, 1).unsqueeze(0).float()
             return tensor_input.to(self.device)
             
        return processed_output 

    def predict(self, model_input: ModelInput) -> Any:
        """
        Performs the core model inference step. (Logic remains the same)
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        logger.debug(f"[{self.predictor_id}] Running core model inference.")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad(): 
                if isinstance(self.model, torch.nn.Module):
                    raw_output = self.model(model_input)
                else:
                    raw_output = self.model.predict(model_input.cpu().numpy())
            
        return raw_output

    def postprocess(self, raw_output: Any) -> Any:
        """
        Converts the raw model output (Tensors/Logits) into a framework-agnostic 
        intermediate format (NumPy/Dictionary). (Logic remains the same)
        """
        logger.debug(f"[{self.predictor_id}] Running Predictor-level postprocessing (Tensor to NumPy/Standardization).")
        
        if isinstance(raw_output, torch.Tensor):
             return raw_output.cpu().numpy()
        
        if isinstance(raw_output, dict):
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in raw_output.items()}
            
        return raw_output