from typing import Dict, Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator

class MonitorConfig(BaseModel):
    """Configuration for a single monitor instance."""
    type: Literal["feature_drift", "prediction_drift", "metric", "latency", "fairness", "shap"]
    params: Dict[str, Any] = Field({}, description="Parameters specific to the monitor type.")

class ReporterConfig(BaseModel):
    """Configuration for a single reporter instance."""
    type: Literal["prometheus", "grafana", "log"]
    params: Dict[str, Any] = Field({}, description="Parameters specific to the reporter type.")

class AlertConfig(BaseModel):
    """Configuration for alerts."""
    slack_webhook_url: Optional[str] = None
    email_recipients: Optional[List[str]] = None

class MonitoringConfig(BaseModel):
    """The main configuration schema for the monitoring module."""
    monitors: List[MonitorConfig]
    reporters: List[ReporterConfig]
    alerts: AlertConfig
    scheduling: Dict[str, Any] = Field({}, description="Scheduling information (e.g., cron expression).")