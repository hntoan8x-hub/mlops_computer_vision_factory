# cv_factory/shared_libs/ml_core/configs/evaluator_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, conint, conlist, constr
from typing import List, Dict, Any, Union, Literal, Optional

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Metric Configuration Schema ---

class MetricConfig(BaseModel):
    """Schema for configuring a single metric."""
    name: constr(to_lower=True) = Field(..., description="The name of the metric (e.g., 'accuracy', 'map', 'miou').")
    
    # Custom parameters for the metric instance (e.g., threshold, averaging method)
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the Metric Class constructor.")

    @validator('name')
    def validate_metric_name(cls, v):
        """Rule: Check if the metric name is supported by MetricFactory."""
        supported_metrics = [
            'accuracy', 'f1_score', 'map', 'miou', 'dice'
        ]
        if v not in supported_metrics:
            raise ValueError(f"Unsupported metric name: '{v}'. Supported are: {supported_metrics}")
        return v


# --- 2. Explainability Configuration Schema ---

class ExplainerConfig(BaseConfig):
    """Schema for configuring model explainability methods (e.g., LIME, SHAP)."""
    
    method: Literal["gradcam", "lime", "shap", "ig"] = Field(..., description="The explanation method to use.")
    sample_size: NonNegativeInt = Field(100, description="Number of samples to run the explanation method on (for speed/cost control).")
    
    @validator('method')
    def validate_explainability_method(cls, v, values):
        """Rule: Ensure specific parameters exist for resource-intensive methods."""
        if v == "shap" and values.get('params', {}).get('background_data_samples', 0) < 10:
            logger.warning("SHAP/LIME methods require sufficient background or sample data for meaningful explanation.")
        return v
    
    
# --- 3. Master Evaluator Configuration ---

class EvaluatorConfig(BaseConfig):
    """Master Schema for the Evaluation Orchestrator."""
    
    # Metrics
    primary_metric: str = Field(..., description="The metric used for model selection/comparison (e.g., 'f1_score' or 'map@0.5').")
    metrics: conlist(MetricConfig, min_length=1) = Field(..., description="List of all metrics to compute.")
    
    # Explainability and Reporting
    explainability: ExplainerConfig = Field(..., description="Model interpretability configuration.")
    reporting_level: Literal["basic", "detailed", "full"] = Field("detailed", description="Verbosity of the evaluation report.")

    @validator('primary_metric')
    def validate_primary_metric_exists(cls, v, values):
        """Rule: The primary metric must be present in the list of metrics to compute."""
        if not values.get('metrics'):
            raise ValueError("Cannot validate primary_metric; 'metrics' list is empty.")
            
        metric_names = [m.name for m in values.get('metrics', [])]
        
        # Check if the primary_metric base name exists in the metrics list
        primary_base_name = v.split('@')[0] # Handles 'map@0.5' -> 'map'
        if primary_base_name not in metric_names and v not in metric_names:
            raise ValueError(
                f"Primary metric '{v}' must be listed in the 'metrics' section to be computed."
            )
        return v