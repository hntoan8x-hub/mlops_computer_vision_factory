from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class EvaluationReport(BaseModel):
    """
    Schema for a standardized evaluation report.

    This ensures that all evaluation results are logged consistently,
    facilitating easy comparison and analysis.
    """
    report_id: str = Field(..., description="A unique identifier for the evaluation report.")
    model_name: str = Field(..., description="The name of the model being evaluated.")
    model_version: str = Field(..., description="The version of the model.")
    task_type: str = Field(..., description="The type of the task (e.g., 'classification').")
    dataset_info: Dict[str, Any] = Field({}, description="Information about the dataset used for evaluation.")
    metrics: Dict[str, float] = Field({}, description="A dictionary of computed performance metrics.")
    explainability_reports: Optional[Dict[str, Any]] = Field({}, description="Reports from explainability methods.")
    timestamp: str = Field(..., description="The timestamp of the evaluation run.")
    additional_info: Optional[Dict[str, Any]] = Field({}, description="Any extra information.")