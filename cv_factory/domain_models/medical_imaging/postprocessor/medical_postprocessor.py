# cv_factory/domain_models/medical_imaging/medical_postprocessor.py 

import logging
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

class MedicalPostprocessor:
    """
    Domain-specific post-processing logic for medical imaging tasks.

    This class enforces business rules and formats the final output 
    based on the intermediate results (e.g., probability tensors or bounding boxes).
    """

    def __init__(self, domain_id: str = "medical"):
        self.domain_id = domain_id
        logger.info(f"Initialized Medical Postprocessor.")

    def run(self, intermediate_output: Union[Dict[str, Any], List[Dict[str, Any]]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies domain rules (e.g., thresholding, applying clinical filters) 
        and formats the final JSON output.
        
        Args:
            intermediate_output: The framework-agnostic output from CVPredictor (e.g., raw NumPy logits).
            config: Post-processing configuration (e.g., confidence thresholds, NMS settings).
            
        Returns:
            Dict[str, Any]: The final, human-readable, domain-specific result.
        """
        threshold = config.get('confidence_threshold', 0.5)
        
        final_results = []
        
        # NOTE: Giả định intermediate_output là một NumPy array chứa xác suất (probability)
        if isinstance(intermediate_output, list):
            # Handle batch output
            intermediate_output = intermediate_output[0] # Lấy kết quả đầu tiên cho ví dụ đơn giản
            
        if intermediate_output.ndim == 1:
            # Giả sử đây là một vector xác suất duy nhất
            prediction_score = intermediate_output[0] 
        else:
            prediction_score = intermediate_output.flatten()[0]


        diagnosis = "Positive Finding (High Confidence)" if prediction_score >= threshold else "Negative Finding (Normal)"
        
        result = {
            "domain": self.domain_id,
            "status": "success",
            "prediction_score": float(prediction_score),
            "diagnosis": diagnosis,
            "threshold_applied": threshold
        }
        
        logger.info(f"Diagnosis complete: {diagnosis} (Score: {prediction_score:.4f})")
        return result