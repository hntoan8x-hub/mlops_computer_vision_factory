# cv_factory/shared_libs/inference/base_cv_predictor.py


import abc
from typing import Any, Dict, List, Union, Optional
import logging

# Define standardized input/output data types
RawInput = Any
ModelInput = Any
PredictionOutput = Union[List[Dict[str, Any]], Dict[str, Any]]

logger = logging.getLogger(__name__)

class BaseCVPredictor(abc.ABC):
    """
    Abstract Base Class for all Computer Vision model prediction handlers.

    Enforces a standardized inference lifecycle and utilizes Dependency Injection 
    to handle domain-specific Postprocessing logic.
    """

    def __init__(self, predictor_id: str, config: Dict[str, Any], postprocessor: Any):
        """
        Initializes the base predictor and injects the domain-specific postprocessor.
        
        Args:
            predictor_id (str): A unique identifier for the predictor instance.
            config (Dict[str, Any]): Configuration specific to model loading and inference.
            postprocessor (Any): The concrete, domain-specific post-processing object 
                                 (INJECTED DEPENDENCY). This object handles business rules and final output formatting.
        """
        self.predictor_id = predictor_id
        self.config = config
        self.postprocessor = postprocessor  # <<< DEPENDENCY INJECTION >>>
        self.model: Optional[Any] = None
        self.is_loaded = False

    @abc.abstractmethod
    def load_model(self, model_uri: str) -> None:
        """
        Loads the trained model artifact from a specified location.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, raw_input: RawInput) -> ModelInput:
        """
        Applies necessary transformations to the raw input data to match the model's 
        required input format (leveraging CVPreprocessingOrchestrator).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, model_input: ModelInput) -> Any:
        """
        Performs the core model inference step.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """
        Converts the raw model output (logits, tensors) into a framework-agnostic 
        intermediate format (e.g., NumPy array/dictionary) before domain rules are applied.
        
        Returns:
            Any: Intermediate output format (ready for the injected postprocessor).
        """
        raise NotImplementedError
        
    def predict_pipeline(self, raw_input: RawInput) -> PredictionOutput:
        """
        Executes the entire inference pipeline in sequence, applying the INJECTED 
        Postprocessor at the final stage.
        """
        if not self.is_loaded:
            raise RuntimeError(f"[{self.predictor_id}] Model is not loaded. Call load_model() first.")
            
        # 1. Preprocess
        model_input = self.preprocess(raw_input)
        
        # 2. Predict
        raw_output = self.predict(model_input)
        
        # 3. Predictor Postprocess (Framework-specific cleanup, e.g., Tensor -> NumPy)
        intermediate_output = self.postprocess(raw_output)
        
        # 4. Domain Postprocess (Business/Domain specific logic, e.g., NMS, medical rules)
        logger.debug(f"[{self.predictor_id}] Applying INJECTED domain postprocessor.")
        final_output = self.postprocessor.run(
            intermediate_output, 
            config=self.config.get('postprocess_config', {})
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