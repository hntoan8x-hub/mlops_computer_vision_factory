# shared_libs/deployment/factory/traffic_controller_factory.py

import logging
from typing import Dict, Any, Type
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController
from shared_libs.deployment.implementations.istio_traffic_controller import IstioTrafficController
from shared_libs.exceptions import UnsupportedProviderError # Giả định exception

logger = logging.getLogger(__name__)

# Registry của các Traffic Controller được hỗ trợ
CONTROLLER_REGISTRY: Dict[str, Type[BaseTrafficController]] = {
    "istio": IstioTrafficController,
    # "nginx": NginxTrafficController,  # Có thể mở rộng
    # "aws_alb": AwsAlbController,
}

class TrafficControllerFactory:
    """
    Factory để tạo ra các Traffic Controller Adapter (ví dụ: Istio, Nginx)
    dựa trên cấu hình.
    """

    @staticmethod
    def create_controller(controller_type: str, endpoint_name: str, config: Dict[str, Any]) -> BaseTrafficController:
        """
        Tạo và trả về một instance Traffic Controller đã được khởi tạo.
        """
        provider_key = controller_type.lower()
        
        ControllerClass = CONTROLLER_REGISTRY.get(provider_key)
        
        if ControllerClass is None:
            available_keys = list(CONTROLLER_REGISTRY.keys())
            raise UnsupportedProviderError(
                f"Traffic Controller provider '{controller_type}' không được hỗ trợ. Các loại có sẵn: {available_keys}"
            )
        
        logger.info(f"🏭 Creating Traffic Controller instance: {ControllerClass.__name__}")
        
        try:
            # Truyền endpoint_name (service ID) và config vào __init__
            return ControllerClass(endpoint_name=endpoint_name, config=config)
        except Exception as e:
            logger.critical(f"Failed to instantiate controller '{controller_type}': {e}")
            raise RuntimeError(f"Traffic Controller Factory failed: {e}")