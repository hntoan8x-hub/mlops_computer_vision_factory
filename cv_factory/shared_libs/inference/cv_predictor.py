# cv_factory/shared_libs/inference/cv_predictor.py (HARDENED - SRP Compliant)

import logging
from typing import Any, Dict, List, Union
import numpy as np
import warnings
import torch 


# --- Import Abstractions and Core Services ---
from .base_cv_predictor import BaseCVPredictor, RawInput, ModelInput, PredictionOutput 
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator
# THÊM: Import utility đã tách biệt
from shared_libs.core_utils.image_decoding_utils import decode_base64_to_numpy 

logger = logging.getLogger(__name__)

class CVPredictor(BaseCVPredictor):
    """
    A concrete implementation of BaseCVPredictor.
    
    HARDENED: Only handles the prediction pipeline (preprocess, predict, postprocess).
    Model loading is delegated (via injected_model), and image decoding is delegated (via core_utils).
    """

    def __init__(self, predictor_id: str, config: Dict[str, Any], postprocessor: Any, loaded_model: Any): 
        # 1. Base Init (Tiêm postprocessor và model đã tải)
        super().__init__(predictor_id, config, postprocessor, loaded_model) 

        self.model_config = self.config.get('model', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        # 2. Initialize Preprocessing Orchestrator
        self.preprocessor = CVPreprocessingOrchestrator(
            config=self.preprocessing_config,
            context='inference'
        )
        
        # Thiết bị được xác định trong quá trình tải model, nhưng cần để xử lý Tensor
        self.device = torch.device(self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
    # --- BaseCVPredictor Contract Implementation ---

    def preprocess(self, raw_input: RawInput) -> ModelInput:
        """
        Applies necessary transformations, acting as the Adapter/Router 
        to standardize input formats (Base64/JSON) into a NumPy array.
        """
        logger.debug(f"[{self.predictor_id}] Running preprocessing pipeline.")
        
        # --- 1. ADAPTER/ROUTER LOGIC ---
        image_data: Union[np.ndarray, List[np.ndarray], None] = None
        
        if isinstance(raw_input, (np.ndarray, list)):
            image_data = raw_input
            logger.debug("Input type is native NumPy/List. Skipping Base64 decode.")
            
        elif isinstance(raw_input, dict):
            base64_string = raw_input.get('image_base64') or raw_input.get('data')
            if base64_string and isinstance(base64_string, str):
                try:
                    # SỬ DỤNG UTILITY ĐÃ TÁCH BIỆT
                    image_data = decode_base64_to_numpy(base64_string)
                    logger.info("Decoded Base64 input successfully.")
                except Exception as e:
                    # Reroute exception từ utility
                    raise ValueError(f"Input dictionary contains invalid Base64 image data: {e}") from e
            else:
                raise ValueError("Input dictionary from API/Kafka does not contain valid Base64 image data.")
        
        else:
            raise TypeError(f"Unsupported RawInput type: {type(raw_input)}. Must be NumPy array, List[NumPy array], or Dict[Base64 string].")

        if image_data is None:
             raise ValueError("Raw input was processed but resulted in None image data.")
        
        # --- 2. CORE PROCESSING DELEGATION ---
        processed_output = self.preprocessor.run(image_data)
        
        # --- 3. FRAMEWORK ADAPTER (FINAL STEP) ---
        if isinstance(processed_output, np.ndarray):
             tensor_input = torch.from_numpy(processed_output).permute(2, 0, 1).unsqueeze(0).float()
             return tensor_input.to(self.device)
             
        return processed_output 

    def predict(self, model_input: ModelInput) -> Any:
        """
        Performs the core model inference step.
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
                    raw_output = self.model.predict(model_input.cpu().numpy() if isinstance(model_input, torch.Tensor) else model_input)
            
        return raw_output

    def postprocess(self, raw_output: Any) -> Any:
        """
        Converts the raw model output (Tensors/Logits) into a framework-agnostic 
        intermediate format (NumPy/Dictionary).
        """
        logger.debug(f"[{self.predictor_id}] Running Predictor-level postprocessing (Tensor to NumPy/Standardization).")
        
        if isinstance(raw_output, torch.Tensor):
             return raw_output.cpu().numpy()
        
        if isinstance(raw_output, dict):
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in raw_output.items()}
            
        return raw_output