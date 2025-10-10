# cv_factory/shared_libs/ml_core/configs/selector_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Rule Definitions ---

class MetricRule(BaseModel):
    """Defines a rule based on a single metric threshold."""
    metric_name: str = Field(..., description="The metric to check (e.g., 'mAP@0.5', 'latency_ms').")
    operator: Literal["gt", "lt", "ge", "le"] = Field(..., description="Comparison operator (>, <, >=, <=).")
    threshold_value: float = Field(..., description="The value to compare the metric against.")

class CustomRule(BaseModel):
    """Defines a non-metric based business rule."""
    name: constr(min_length=1) = Field(..., description="Unique name for the custom rule (e.g., 'max_model_size_mb').")
    value: Any = Field(..., description="Target value for the custom rule check.")

# --- 2. Strategy Schemas ---

class MetricBasedSelectorConfig(BaseConfig):
    """Configuration for selecting the best model purely on numerical metrics."""
    strategy: Literal["metric_based"] = "metric_based"
    primary_metric: str = Field(..., description="The metric used for final ranking (e.g., 'mAP@0.5').")
    maximize: bool = Field(True, description="If True, maximize the primary metric; otherwise, minimize.")
    minimum_acceptance_rules: Optional[List[MetricRule]] = Field(None, description="Metrics that must be satisfied for a model to be considered valid.")

class RuleBasedSelectorConfig(BaseConfig):
    """Configuration for applying predefined business/compliance rules."""
    strategy: Literal["rule_based"] = "rule_based"
    acceptance_rules: List[Union[MetricRule, CustomRule]] = Field(..., description="A list of rules that all candidate models must satisfy.")
    
class EnsembleSelectorConfig(BaseConfig):
    """Configuration for choosing models based on a weighted ensemble approach."""
    strategy: Literal["ensemble_selector"] = "ensemble_selector"
    weighting_schema: Dict[str, float] = Field(..., description="Weights assigned to different evaluation metrics or model properties.")
    
    @validator('weighting_schema')
    def check_weights_sum(cls, v):
        """Rule: Ensure all weights sum up to 1.0 (for normalization)."""
        if abs(sum(v.values()) - 1.0) > 1e-4:
            raise ValueError("Weighting schema values must sum up to 1.0.")
        return v

# --- 3. Master Selector Configuration ---

# The Master Selector can choose one of the defined strategies
SelectorStrategy = Union[MetricBasedSelectorConfig, RuleBasedSelectorConfig, EnsembleSelectorConfig]

class SelectorConfig(BaseConfig):
    """
    Master Schema for the Model Selector Orchestrator.
    This schema dictates the logic used to promote a model to Production/Staging.
    """
    
    active_strategy: SelectorStrategy = Field(..., description="The active strategy used to select the best model.")
    
    # Global Rules (Applies regardless of strategy)
    fallback_model_uri: Optional[str] = Field(None, description="URI of a safe, known model to use if no new model meets the rules.")

    @validator('active_strategy', pre=True)
    def validate_strategy_type(cls, v):
        """Rule: Enforce that the strategy dictionary contains a valid 'strategy' key."""
        if 'strategy' not in v:
            raise ValueError("Strategy configuration must contain a 'strategy' field to identify its type.")
        return v