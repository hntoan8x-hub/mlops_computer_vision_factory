# cv_factory/shared_libs/inference/base_cv_predictor.py (UPDATED - ADDING FS LOOKUP LOGIC)

import abc
from typing import Any, Dict, List, Union, Optional
import logging
from shared_libs.feature_store.orchestrator.feature_store_orchestrator import FeatureStoreOrchestrator 

# Define standardized input/output data types
RawInput = Any
ModelInput = Any
PredictionOutput = Union[List[Dict[str, Any]], Dict[str, Any]]

logger = logging.getLogger(__name__)

class BaseCVPredictor(abc.ABC):
    """
    Abstract Base Class for all Computer Vision model prediction handlers.
    
    HARDENED: Accepts the fully loaded Model object and FeatureStoreOrchestrator via DI.
    """

    def __init__(self, 
                 predictor_id: str, 
                 config: Dict[str, Any], 
                 postprocessor: Any, 
                 loaded_model: Any,
                 feature_store: Optional[FeatureStoreOrchestrator] = None): 
        
        self.predictor_id = predictor_id
        self.config = config
        self.postprocessor = postprocessor  
        self.model: Any = loaded_model      
        self.is_loaded = (loaded_model is not None)
        self.feature_store = feature_store # LƯU TRỮ ORCHESTRATOR

    # NEW ABSTRACT METHOD: Buộc lớp con phải biết cách trích xuất Embedding
    @abc.abstractmethod
    def _get_embedding(self, model_input: ModelInput) -> Any:
        """
        [MANDATORY] Executes the model's feature extractor to get the embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, raw_input: RawInput) -> ModelInput:
        """
        Applies necessary transformations to the raw input data to match the model's 
        required input format.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, model_input: ModelInput, embedding: Optional[Any] = None, retrieved_features: Optional[Any] = None) -> Any:
        """
        Performs the core model inference step. Updated to accept FS features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """
        Converts the raw model output (logits, tensors) into a framework-agnostic 
        intermediate format (e.g., NumPy array/dictionary).
        """
        raise NotImplementedError
        
    def predict_pipeline(self, raw_input: RawInput) -> PredictionOutput:
        """
        [UPDATED] Executes the entire inference pipeline, including Feature Store Lookup.
        """
        if not self.is_loaded:
            raise RuntimeError(f"[{self.predictor_id}] Model is not loaded. Initialization failed.")
            
        # 1. Preprocess
        model_input = self.preprocess(raw_input)
        
        # 2. Extract Embedding
        embedding = self._get_embedding(model_input)
        
        # 3. GLUE: Feature Store Lookup (Nếu Feature Store được tiêm)
        retrieved_features = None
        if self.feature_store:
            logger.info(f"[{self.predictor_id}] Querying Feature Store...")
            # Sử dụng Context Manager để đảm bảo phiên làm việc được đóng
            with self.feature_store as fs:
                 retrieved_features = fs.retrieve(embedding)
            
        # 4. Predict (Core Model Inference)
        # Truyền cả embedding và retrieved_features vào hàm predict
        raw_output = self.predict(model_input, embedding=embedding, retrieved_features=retrieved_features)
        
        # 5. Predictor Postprocess (Framework-specific cleanup)
        intermediate_output = self.postprocess(raw_output)
        
        # 6. Domain Postprocess (Business/Domain specific logic)
        logger.debug(f"[{self.predictor_id}] Applying INJECTED domain postprocessor.")
        final_output = self.postprocessor.run(
            intermediate_output, 
            config=self.config.get('postprocess_config', {}),
            retrieved_features=retrieved_features # Pass FS features to domain postprocessor
        )
        
        return final_output

    def unload_model(self) -> None:
        """
        Releases the model from memory.
        """
        if self.is_loaded:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info(f"[{self.predictor_id}] Model successfully unloaded.")