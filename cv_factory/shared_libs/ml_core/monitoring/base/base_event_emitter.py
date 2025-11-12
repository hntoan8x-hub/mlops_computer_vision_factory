# shared_libs/infra/monitoring/base_event_emitter.py

import abc
from typing import Dict, Any, Optional

class BaseEventEmitter(abc.ABC):
    """
    Abstract Base Class cho các dịch vụ phát sự kiện (e.g., Kafka, Pub/Sub, Prometheus metrics).
    """

    @abc.abstractmethod
    def emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Phát một sự kiện có cấu trúc.
        
        Args:
            event_name (str): Tên sự kiện (ví dụ: 'model_registered').
            payload (Dict[str, Any]): Dữ liệu liên quan đến sự kiện.
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Phương thức tiện ích để log metrics cho hệ thống monitoring (nếu khác MLflow).
        """
        raise NotImplementedError