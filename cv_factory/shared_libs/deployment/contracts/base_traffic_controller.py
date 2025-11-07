# shared_libs/deployment/contracts/base_traffic_controller.py

import abc
from typing import Dict, Any, Optional

class BaseTrafficController(abc.ABC):
    """
    Contract cho mọi bộ điều khiển lưu lượng (Service Mesh/Load Balancer API).
    """

    @abc.abstractmethod
    def __init__(self, endpoint_name: str, config: Dict[str, Any]):
        self.endpoint_name = endpoint_name
        self.config = config
        pass

    @abc.abstractmethod
    async def async_set_traffic(self, new_version: str, new_traffic_percentage: int, stable_version: str) -> bool:
        """
        Thiết lập tỷ lệ phần trăm lưu lượng truy cập cho phiên bản mới.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_confirm_traffic_split(self, new_traffic_percentage: int, timeout: int = 30) -> bool:
        """
        Polling API để xác nhận rằng tỷ lệ lưu lượng đã thực sự được áp dụng. (CRITICAL HARDENING)
        """
        raise NotImplementedError