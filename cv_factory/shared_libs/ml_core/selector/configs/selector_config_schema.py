from typing import Dict, Any, List, Literal, Union, Optional
from pydantic import BaseModel, Field, validator

class MetricRuleConfig(BaseModel):
    metric: str
    op: Literal[">=", "<=", "=="]
    value: Union[float, int]

class RuleBasedConfig(BaseModel):
    rules: List[MetricRuleConfig]

class SelectorConfig(BaseModel):
    type: str = Field(..., description="The type of the selector.")
    params: Dict[str, Any] = Field({}, description="Parameters for the selector.")

    @validator('type')
    def validate_selector_type(cls, v):
        allowed_types = {"metric_based", "rule_based", "ensemble"}
        if v.lower() not in allowed_types:
            raise ValueError(f"Invalid selector type: '{v}'. Supported types are: {allowed_types}")
        return v