import logging
import time
from typing import Any, Dict
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Define Prometheus metrics. These should be defined once.
PIPELINE_LATENCY = Histogram('pipeline_step_latency_seconds', 'Latency of pipeline steps', ['component_type', 'step_name'])
PIPELINE_ERRORS = Counter('pipeline_step_errors_total', 'Total errors in pipeline steps', ['component_type', 'step_name'])

def measure_latency(component_type: str, step_name: str, func: callable) -> callable:
    """
    A decorator to measure the execution time of a pipeline component's method.

    Args:
        component_type (str): The type of the component (e.g., "resizer").
        step_name (str): The name of the method being timed (e.g., "transform").
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            PIPELINE_LATENCY.labels(component_type=component_type, step_name=step_name).observe(elapsed_time)
            logger.debug(f"Component '{component_type}.{step_name}' took {elapsed_time:.4f}s.")
            return result
        except Exception as e:
            PIPELINE_ERRORS.labels(component_type=component_type, step_name=step_name).inc()
            logger.error(f"Error in component '{component_type}.{step_name}': {e}")
            raise
    return wrapper

def emit_custom_metric(metric_name: str, value: float, labels: Dict[str, str] = {}) -> None:
    """
    Emits a custom metric to Prometheus.

    Args:
        metric_name (str): The name of the metric.
        value (float): The value of the metric.
        labels (Dict[str, str]): A dictionary of labels.
    """
    try:
        # Prometheus Gauge is a good choice for general-purpose metrics
        metric = Gauge(metric_name, 'A custom metric from the pipeline', list(labels.keys()))
        metric.labels(**labels).set(value)
        logger.info(f"Emitted custom metric '{metric_name}' with value {value}.")
    except Exception as e:
        logger.error(f"Failed to emit custom metric '{metric_name}': {e}")