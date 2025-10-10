import logging
import time
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics for orchestration
ORCHESTRATOR_RUNS = Counter('orchestrator_runs_total', 'Total number of orchestrator runs', ['orchestrator_type'])
ORCHESTRATOR_LATENCY = Histogram('orchestrator_latency_seconds', 'Latency of orchestrator runs', ['orchestrator_type'])
ORCHESTRATOR_ERRORS = Counter('orchestrator_errors_total', 'Total errors in orchestrators', ['orchestrator_type', 'error_type'])

def measure_orchestrator_latency(orchestrator_type: str) -> Callable:
    """A decorator to measure the execution time of an orchestrator's run method."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                ORCHESTRATOR_LATENCY.labels(orchestrator_type).observe(elapsed_time)
                ORCHESTRATOR_RUNS.labels(orchestrator_type).inc()
                logger.info(f"Orchestrator '{orchestrator_type}' run completed in {elapsed_time:.4f}s.")
                return result
            except Exception as e:
                ORCHESTRATOR_ERRORS.labels(orchestrator_type, type(e).__name__).inc()
                raise
        return wrapper
    return decorator