# shared_libs/deployment/traffic_controller.py

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger("TRAFFIC_CONTROLLER")

class TrafficController:
    """
    Mô phỏng một bộ điều khiển lưu lượng bất đồng bộ (Async Traffic Controller).
    Nó quản lý việc phân phối lưu lượng giữa các phiên bản dịch vụ (Stable vs. New).
    """

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.current_traffic: Dict[str, float] = {}

    async def async_set_traffic(self, new_version: str, new_traffic_percentage: int, stable_version: str) -> bool:
        """
        Thiết lập tỷ lệ phần trăm lưu lượng truy cập cho phiên bản mới.
        
        Args:
            new_version (str): Tag của phiên bản Canary.
            new_traffic_percentage (int): Tỷ lệ lưu lượng (0-100) chuyển sang phiên bản mới.
            stable_version (str): Tag của phiên bản ổn định.

        Returns:
            bool: True nếu cấu hình thành công.
        """
        if not 0 <= new_traffic_percentage <= 100:
            logger.error("Traffic percentage must be between 0 and 100.")
            return False

        stable_traffic = 100 - new_traffic_percentage
        
        logger.info(
            f"Applying traffic split for {self.endpoint_name}: "
            f"NEW ({new_version}): {new_traffic_percentage}% | "
            f"STABLE ({stable_version}): {stable_traffic}%"
        )

        # Mô phỏng thời gian chờ của API Service Mesh/Load Balancer
        await asyncio.sleep(2) 
        
        # Cập nhật trạng thái hiện tại (MOCK)
        self.current_traffic = {
            new_version: new_traffic_percentage,
            stable_version: stable_traffic
        }

        # Trong production: Gọi API của Istio, Linkerd, hoặc Cloud Load Balancer
        # Ví dụ: await requests.post(self.config.traffic_api, json=split_config)
        
        return True

    async def async_get_current_traffic(self) -> Dict[str, float]:
        """Trả về tỷ lệ lưu lượng hiện tại."""
        return self.current_traffic