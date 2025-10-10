from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, validator

class TriggerConfig(BaseModel):
    type: Literal["drift", "time", "performance"]
    params: Dict[str, Any] = {}

class NotificationConfig(BaseModel):
    slack_webhook_url: str = Field(..., description="Slack webhook URL for notifications.")

class JobConfig(BaseModel):
    training_config_path: str = Field(..., description="Path to the training pipeline configuration.")
    params: Dict[str, Any] = Field({}, description="Parameters for the job submission.")

class RetrainConfig(BaseModel):
    triggers: List[TriggerConfig]
    notification: NotificationConfig
    job_config: JobConfig