# shared_libs/ml_core/mlflow_service/factories/tracker_factory.py (HARDENED - DYNAMIC IMPORT)

import logging
import importlib 
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
# LOẠI BỎ: from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger 
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceException

logger = logging.getLogger(__name__)

class TrackerFactory:
    """
    A factory class for creating different types of ML experiment trackers.

    HARDENED: Now uses dynamic import to load implementation classes, eliminating 
    the need to modify the factory code when adding new tracker types (e.g., W&B).
    """
    # LOẠI BỎ: _TRACKER_MAP tĩnh

    @staticmethod
    def _get_class_from_config(tracker_type: str, config: Dict[str, Any]) -> Type[BaseTracker]:
        """
        Dynamically imports the concrete Tracker class based on configuration.
        """
        # Giả định config chứa cấu trúc cho Dynamic Import, ví dụ:
        # config['mlflow'] = {'module_path': 'shared_libs.ml_core.mlflow_service.implementations.mlflow_logger', 'class_name': 'MLflowLogger'}
        
        tracker_config = config.get(tracker_type.lower(), {})
        module_path = tracker_config.get('module_path')
        class_name = tracker_config.get('class_name')

        if not module_path or not class_name:
            raise MLflowServiceException(
                f"Dynamic configuration for tracker '{tracker_type}' is missing 'module_path' or 'class_name'."
            )

        try:
            module = importlib.import_module(module_path)
            tracker_cls = getattr(module, class_name)
            if not issubclass(tracker_cls, BaseTracker):
                 raise TypeError(f"Class '{class_name}' must inherit from BaseTracker.")
            return tracker_cls
            
        except ImportError as e:
            logger.error(f"Module import failed for tracker '{tracker_type}': {e}")
            raise MLflowServiceException(f"Failed to import tracker class: {e}")
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_path}'.")
            raise MLflowServiceException(f"Tracker class not found: {e}")


    @classmethod
    def create(cls, tracker_type: str, config: Optional[Dict[str, Any]] = None) -> BaseTracker:
        """
        Creates and returns a tracker instance based on its type.

        Args:
            tracker_type (str): The type of tracker to create (e.g., "mlflow").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                which MUST contain dynamic path info under
                                                the tracker_type key (e.g., config['mlflow']).

        Returns:
            BaseTracker: An instance of the requested tracker.

        Raises:
            MLflowServiceException: If the specified tracker_type is not supported or instantiation fails.
        """
        config = config or {}
        
        # 1. Dynamically find the class
        tracker_cls = cls._get_class_from_config(tracker_type, config)
        
        # 2. Extract constructor parameters (Assuming they are flat in the root of config or a designated key)
        # Lấy tham số cấu hình cần thiết để khởi tạo Tracker (Ví dụ: tracking_uri)
        # Ta giả định config['params'] chứa các tham số khởi tạo.
        constructor_params = config.get('params', {}) 

        logger.info(f"Creating instance of {tracker_cls.__name__} using dynamic configuration...")
        try:
            return tracker_cls(**constructor_params)
        except Exception as e:
            logger.error(f"Failed to instantiate tracker '{tracker_type}' with params {constructor_params}: {e}")
            raise MLflowServiceException(f"Tracker creation failed for type '{tracker_type}': {e}")