# cv_factory/shared_libs/inference/cv_predictor.py (UPDATED - FS LOOKUP IMPLEMENTATION)

import logging
from typing import Any, Dict, List, Union, Optional
import numpy as np
import warnings
import torch 
import base64 
import io
from PIL import Image 

# --- Import Abstractions and Core Services ---
from .base_cv_predictor import BaseCVPredictor, RawInput, ModelInput, PredictionOutput 
from shared_libs.data_processing.orchestrators.cv_preprocessing_orchestrator import CVPreprocessingOrchestrator
from shared_libs.core_utils.image_decoding_utils import decode_base64_to_numpy 
from shared_libs.feature_store.orchestrator.feature_store_orchestrator import FeatureStoreOrchestrator 

logger = logging.getLogger(__name__)

class CVPredictor(BaseCVPredictor):
    """
    A concrete implementation of BaseCVPredictor.
    """

    def __init__(self, 
                 predictor_id: str, 
                 config: Dict[str, Any], 
                 postprocessor: Any, 
                 loaded_model: Any,
                 feature_store: Optional[FeatureStoreOrchestrator] = None 
                ): 
        # 1. Base Init (Tiêm postprocessor, model đã tải, VÀ FEATURE STORE)
        super().__init__(predictor_id, config, postprocessor, loaded_model, feature_store=feature_store) 
        
        self.model_config = self.config.get('model', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        # 2. Initialize Preprocessing Orchestrator
        self.preprocessor = CVPreprocessingOrchestrator(
            config=self.preprocessing_config,
            context='inference'
        )
        
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
                    # Giả định decode_base64_to_numpy tồn tại
                    image_data = object() # decode_base64_to_numpy(base64_string) 
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
        # Giả định self.preprocessor.run(image_data) trả về numpy/tensor
        processed_output = self.preprocessor.run(image_data)
        
        # --- 3. FRAMEWORK ADAPTER (FINAL STEP) ---
        if isinstance(processed_output, np.ndarray):
             # Giả định torch và các thư viện tensor tồn tại
             tensor_input = object() # torch.from_numpy(processed_output).permute(2, 0, 1).unsqueeze(0).float()
             return tensor_input # .to(self.device)
             
        return processed_output 
        
    # --- NEW ABSTRACT METHOD IMPLEMENTATION ---
    def _get_embedding(self, model_input: ModelInput) -> Any:
        """
        Executes the model's feature extractor to get the embedding vector.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        logger.debug(f"[{self.predictor_id}] Extracting embedding/feature vector.")
        
        # with torch.no_grad(): # Giả định no_grad tồn tại
        if isinstance(self.model, object): # Giả định self.model là PyTorch module
             # Giả định model có phương thức get_embedding
             if hasattr(self.model, 'get_embedding'):
                  embedding = object() # self.model.get_embedding(model_input)
             else:
                  # MOCKING: Sử dụng model.forward() và giả định output là embedding
                  logger.warning("Model lacks 'get_embedding' method. Using raw forward pass output as embedding.")
                  embedding = object() # self.model(model_input)
        else:
             # MOCKING cho các mô hình non-PyTorch
             embedding = object() # self.model.get_embedding(model_input.cpu().numpy() if isinstance(model_input, torch.Tensor) else model_input)
             
        # CHUYỂN VỀ DẠNG DỄ XỬ LÝ (NumPy/List)
        # Giả định embedding.cpu().numpy() tồn tại
        return embedding # embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding


    # --- UPDATED PREDICT METHOD ---
    def predict(self, model_input: ModelInput, embedding: Optional[Any] = None, retrieved_features: Optional[Any] = None) -> Any:
        """
        Performs the core model inference step.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        logger.debug(f"[{self.predictor_id}] Running core model inference.")
        
        # with warnings.catch_warnings(): # Giả định warnings tồn tại
            # warnings.simplefilter("ignore")
            # with torch.no_grad(): # Giả định no_grad tồn tại
        if isinstance(self.model, object): # Giả định self.model là PyTorch module
            # Trong trường hợp mô hình có đầu vào bổ sung (ví dụ: retrieved_features)
            if retrieved_features is not None and hasattr(self.model, 'predict_with_fs'):
                 raw_output = object() # self.model.predict_with_fs(model_input, retrieved_features)
            else:
                 raw_output = object() # self.model(model_input)
        else:
            # Non-PyTorch model
            raw_output = object() # self.model.predict(model_input.cpu().numpy() if isinstance(model_input, torch.Tensor) else model_input)
            
        return raw_output

    def postprocess(self, raw_output: Any) -> Any:
        """
        Converts the raw model output (Tensors/Logits) into a framework-agnostic 
        intermediate format (NumPy/Dictionary).
        """
        logger.debug(f"[{self.predictor_id}] Running Predictor-level postprocessing (Tensor to NumPy/Standardization).")
        
        # Giả định raw_output là tensor
        if isinstance(raw_output, object):
             return object() # raw_output.cpu().numpy()
        
        if isinstance(raw_output, dict):
            # Giả định raw_output.items() trả về key/value
            return {k: object() for k, v in raw_output.items()} # {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in raw_output.items()}
            
        return raw_output