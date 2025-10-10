# cv_factory/shared_libs/ml_core/configs/retrain_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Retraining Trigger Rules ---

class DriftTriggerConfig(BaseModel):
    """Rules for triggering retraining based on Data or Model Drift."""
    data_drift_metric: Literal["ks_test", "chi_squared"] = Field("ks_test", description="Statistical test for data drift.")
    data_drift_threshold: Field(default=0.1, ge=0.0, le=1.0) = Field(description="Max allowable drift magnitude before triggering.")
    model_performance_drop: Field(default=0.05, ge=0.0, le=1.0) = Field(description="Max allowable drop in primary metric (e.g., AUC, mAP).")
    
class TimeTriggerConfig(BaseModel):
    """Rules for triggering retraining based on a fixed schedule (time)."""
    schedule_interval: str = Field(..., description="Cron-style schedule string (e.g., '0 0 * * 0' for weekly).")
    max_model_age_days: PositiveInt = Field(90, description="Max age of the current production model.")
    
# Union of all possible triggers
TriggerConfig = Union[DriftTriggerConfig, TimeTriggerConfig]


# --- 2. Master Retraining Configuration ---

class RetrainingConfig(BaseConfig):
    """
    Master Schema for the Retraining Orchestrator.
    This schema dictates the logic for deciding WHEN and HOW to initiate a new training run.
    """
    
    # Logic Control
    retrain_if_model_fails_test: bool = Field(True, description="If True, always retrain if model quality test fails during inference monitoring.")
    
    # Triggers List
    triggers: conlist(TriggerConfig, min_length=1) = Field(..., description="A list of active triggers. Any trigger violation initiates a retrain.")
    
    # Data Selection (What data to use for the new run)
    data_selection_method: Literal["latest_n_days", "drifted_samples_only", "full_dataset"] = Field(
        "latest_n_days", 
        description="Method for selecting data for the new training run."
    )
    
    # Notification Settings
    notification_email: Optional[str] = Field(None, description="Email address to notify upon successful or failed retraining.")
    
    # Custom Validation Rules (Rules of the Game)
    
    @validator('triggers')
    def validate_trigger_uniqueness(cls, v):
        """Rule: Ensure only one time-based trigger is specified."""
        time_triggers = [t for t in v if isinstance(t, TimeTriggerConfig)]
        if len(time_triggers) > 1:
            raise ValueError("Only one time-based retraining schedule is allowed to prevent job overlap.")
        return v