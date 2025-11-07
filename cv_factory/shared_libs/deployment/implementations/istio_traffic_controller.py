# shared_libs/deployment/implementations/istio_traffic_controller.py (FINAL FULL PRODUCTION CODE)

import logging
import asyncio
import httpx 
import time
from typing import Dict, Any
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController
from shared_libs.exceptions import DeploymentError # Giả định GenAI Factory Exception

logger = logging.getLogger("ISTIO_CONTROLLER")

class IstioTrafficController(BaseTrafficController):
    """
    Adapter cho Istio Service Mesh, cấu hình VirtualService để chia traffic.
    """

    def __init__(self, endpoint_name: str, config: Dict[str, Any]):
        super().__init__(endpoint_name, config)
        self.istio_api_url = config['api_endpoint'] 
        self.namespace = config['namespace']
        self.virtual_service_name = config.get('virtual_service_name', f"{endpoint_name}-vs")
        # Hardening: Lấy token từ config (đã được tiêm từ Secret Manager)
        self.api_token = config.get('api_token') 
        self.headers = {'Authorization': f"Bearer {self.api_token}"}
        
        # Hardening: Khởi tạo Client
        self.client = httpx.AsyncClient(
            headers=self.headers, 
            base_url=self.istio_api_url,
            timeout=10.0, # Thêm timeout cho request
            verify=config.get('ssl_verify', True) # Tùy chọn: SSL verification
        )

    async def async_set_traffic(self, new_version: str, new_traffic_percentage: int, stable_version: str) -> bool:
        stable_traffic = 100 - new_traffic_percentage
        
        # Hardening 1: Kiểm tra đầu vào
        if not (0 <= new_traffic_percentage <= 100):
            logger.error("Traffic percentage must be between 0 and 100.")
            return False

        # --- Logic tạo Istio VirtualService Patch ---
        patch_data = {
            "spec": {
                "http": [{
                    "route": [
                        {"destination": {"host": self.endpoint_name, "subset": new_version}, "weight": new_traffic_percentage},
                        {"destination": {"host": self.endpoint_name, "subset": stable_version}, "weight": stable_traffic},
                    ]
                }]
            }
        }
        
        # Hardening 2: Sử dụng đường dẫn API đúng (Istio Custom Resource API)
        api_path = f"/apis/networking.istio.io/v1alpha3/namespaces/{self.namespace}/virtualservices/{self.virtual_service_name}"
        
        try:
            response = await self.client.patch(
                api_path,
                json=patch_data,
                headers={"Content-Type": "application/json-patch+json"} 
            )
            response.raise_for_status() 

            logger.info(f"Istio VS patch requested. NEW: {new_traffic_percentage}%. Waiting for confirmation...")
            
            # Chuyển sang bước xác nhận trạng thái (CRITICAL)
            return await self.async_confirm_traffic_split(new_traffic_percentage, new_version)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to patch Istio VirtualService (Status {e.response.status_code}): {e.response.text}")
            raise DeploymentError(f"Istio Traffic Patch Failed: {e.response.text}")
        except Exception as e:
            logger.error(f"Critical error during Istio API call: {e}")
            raise DeploymentError(f"Istio API communication error: {e}")

    async def async_confirm_traffic_split(self, expected_percentage: int, version_tag: str, timeout: int = 60) -> bool:
        """
        Hardening 3: Polling API để xác nhận tỷ lệ traffic đã được áp dụng.
        """
        start_time = time.time()
        poll_interval = 5 
        
        api_path = f"/apis/networking.istio.io/v1alpha3/namespaces/{self.namespace}/virtualservices/{self.virtual_service_name}"

        while time.time() - start_time < timeout:
            await asyncio.sleep(poll_interval)
            
            try:
                # GET VirtualService hiện tại
                response = await self.client.get(api_path)
                response.raise_for_status()
                vs_data = response.json()
                
                # Hardening 4: Logic kiểm tra trọng số (weight) thực tế
                current_routes = vs_data['spec']['http'][0]['route']
                
                new_variant_weight = 0
                for route in current_routes:
                    if route['destination']['subset'] == version_tag:
                        new_variant_weight = route.get('weight', 0)
                        break
                
                if new_variant_weight == expected_percentage:
                    logger.info(f"SUCCESS: Istio confirmed traffic split. Version {version_tag} is at {new_variant_weight}%.")
                    return True
                
                logger.info(f"Traffic not fully applied yet. Version {version_tag} is at {new_variant_weight}% (Expected: {expected_percentage}%). Retrying...")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.critical("VirtualService not found during polling.")
                    break
                logger.warning(f"Error during polling (GET VS): {e}")
            except Exception as e:
                logger.error(f"Unexpected error during polling: {e}")

        logger.critical(f"Traffic split confirmation FAILED. Weight for {version_tag} did not reach {expected_percentage}% within {timeout} seconds.")
        return False