import logging
from typing import Dict, Any, Union
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from shared_libs.ml_core.monitoring.base.base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class PrometheusReporter(BaseReporter):
    """
    A concrete reporter that pushes metrics to a Prometheus Pushgateway.
    """
    def __init__(self, config: Dict[str, Any]):
        self.gateway_url = config.get("gateway_url")
        self.job_name = config.get("job_name", "model_monitor")
        if not self.gateway_url:
            raise ValueError("Prometheus gateway URL must be provided.")
        self.registry = CollectorRegistry()
        self.metrics = {}
        logger.info(f"PrometheusReporter initialized for gateway: {self.gateway_url}")

    def report(self, report_name: str, report_data: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Reports monitoring data by pushing metrics to the Prometheus gateway.
        """
        for key, value in report_data.items():
            metric_name = f"{report_name}_{key}".replace(" ", "_").lower()
            if metric_name not in self.metrics:
                self.metrics[metric_name] = Gauge(metric_name, f"Metric for {key}", registry=self.registry)
            
            # Assuming value is a single float.
            if isinstance(value, (int, float)):
                self.metrics[metric_name].set(value)
        
        try:
            push_to_gateway(self.gateway_url, job=self.job_name, registry=self.registry)
            logger.info(f"Metrics for report '{report_name}' successfully pushed to Prometheus.")
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus gateway: {e}")
            raise