import logging
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceException

logger = logging.getLogger(__name__)

class TrackerFactory:
    """
    A factory class for creating different types of ML experiment trackers.

    This class centralizes the creation logic, allowing for a config-driven
    approach to experiment tracking.
    """
    _TRACKER_MAP: Dict[str, Type[BaseTracker]] = {
        "mlflow": MLflowLogger,
        # Add other trackers here (e.g., "wandb": WAndBLogger)
    }

    @classmethod
    def create(cls, tracker_type: str, config: Optional[Dict[str, Any]] = None) -> BaseTracker:
        """
        Creates and returns a tracker instance based on its type.

        Args:
            tracker_type (str): The type of tracker to create (e.g., "mlflow").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the tracker's constructor.

        Returns:
            BaseTracker: An instance of the requested tracker.

        Raises:
            MLflowServiceException: If the specified tracker_type is not supported.
        """
        config = config or {}
        tracker_cls = cls._TRACKER_MAP.get(tracker_type.lower())
        
        if not tracker_cls:
            supported_trackers = ", ".join(cls._TRACKER_MAP.keys())
            logger.error(f"Unsupported tracker type: '{tracker_type}'. Supported types are: {supported_trackers}")
            raise MLflowServiceException(f"Unsupported tracker type: '{tracker_type}'. Supported types are: {supported_trackers}")

        logger.info(f"Creating instance of {tracker_cls.__name__}...")
        try:
            return tracker_cls(**config)
        except Exception as e:
            logger.error(f"Failed to instantiate tracker '{tracker_type}' with config {config}: {e}")
            raise