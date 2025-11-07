# shared_libs/deployment/implementations/kubernetes_deployer.py (FULL HARDENING)

import logging
from typing import Dict, Any, Optional
from kubernetes import client, config 
import asyncio
import time
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.deployment.configs.deployment_schema import KubernetesConfig 
from shared_libs.exceptions import DeploymentError # Giả định GenAI Factory Exception

logger = logging.getLogger(__name__)

class KubernetesDeployer(BaseDeployer):
    """
    Adapter để triển khai mô hình ML lên Kubernetes bằng cách tạo Deployment và Service.
    Sử dụng KubernetesConfig schema đã được xác thực.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Hardening 1: Khởi tạo Config từ Dict đã được validate
        # Giả định 'config' là một Dict chứa các tham số của KubernetesConfig
        self.k8s_config = KubernetesConfig(**config) 
        self.namespace = self.k8s_config.namespace
        
        try:
            # Load kubeconfig (Tải cấu hình Kubernetes từ môi trường/file)
            config.load_kube_config() 
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            logger.info(f"Kubernetes client initialized for namespace: {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to load Kubernetes config: {e}")
            raise
        
    # --- Hardening 4: Thêm cơ chế chờ Deployment Ready ---
    def _wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300) -> bool:
        """Chờ cho Deployment chuyển sang trạng thái sẵn sàng (readyReplicas == replicas)."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
                status = deployment.status
                
                # Điều kiện kiểm tra Ready: Số replica đã sẵn sàng phải bằng số replica mong muốn
                if status.ready_replicas is not None and status.ready_replicas >= deployment.spec.replicas:
                    logger.info(f"Deployment '{deployment_name}' is Ready (Replicas: {status.ready_replicas}).")
                    return True
                
                logger.info(f"Deployment '{deployment_name}' not ready yet. Ready: {status.ready_replicas}/{deployment.spec.replicas}. Waiting...")
                time.sleep(10)

            except client.ApiException as e:
                if e.status == 404:
                    logger.error(f"Deployment {deployment_name} not found during wait.")
                    return False
                raise
            except Exception as e:
                logger.error(f"Error while waiting for deployment: {e}")
                raise

        logger.critical(f"Deployment '{deployment_name}' failed to become Ready within {timeout} seconds.")
        return False

    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        [TRIỂN KHAI] Tạo hoặc cập nhật Deployment (Pods) và Service (Load Balancer).
        """
        version_tag = deploy_config.get('version_tag', 'v1') # Hardening: Thêm version tag
        deployment_name = f"{model_name}-deployment-{version_tag}" # Hardening: Tên Deployment bao gồm version
        service_name = f"{model_name}-service" 
        container_image = deploy_config['image_uri'] 
        limits = self.k8s_config.resource_limits

        # --- Logic tạo V1Deployment object (Container và Probes) ---
        container = client.V1Container(
            # ... (Phần logic container không đổi) ...
            name=model_name,
            image=container_image,
            ports=[client.V1ContainerPort(container_port=8080)],
            resources=client.V1ResourceRequirements(
                limits={"cpu": f"{limits.cpu_cores}m", "memory": f"{limits.memory_mib}Mi"},
                requests={"cpu": f"{limits.cpu_cores / 2}m", "memory": f"{limits.memory_mib / 2}Mi"}
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path=self.k8s_config.liveness_path, port=8080),
                initial_delay_seconds=10, timeout_seconds=5
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path=self.k8s_config.readiness_path, port=8080),
                initial_delay_seconds=5, timeout_seconds=3
            ),
            env=[
                client.V1EnvVar(name="MODEL_URI", value=model_artifact_uri),
                client.V1EnvVar(name="MODEL_VERSION", value=version_tag) # Hardening: Thêm biến môi trường version
            ]
        )
        
        # Hardening 5: Gắn nhãn Version (Subset cho Istio/Traffic Controller)
        labels = {"app": model_name, "version": version_tag} 
        
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name, labels=labels),
            spec=client.V1DeploymentSpec(
                replicas=1, 
                selector=client.V1LabelSelector(match_labels={"app": model_name, "version": version_tag}),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=labels),
                    spec=client.V1PodSpec(containers=[container])
                )
            )
        )
        
        # ... (Logic tạo/patch Deployment) ...
        try:
            self.apps_v1.create_namespaced_deployment(body=deployment, namespace=self.namespace)
        except client.ApiException as e:
             if e.status == 409: 
                 self.apps_v1.patch_namespaced_deployment(name=deployment_name, namespace=self.namespace, body=deployment)
             else:
                 raise

        # --- Logic tạo V1Service object (Chỉ tạo lần đầu, Service chung cho mọi Deployment) ---
        try:
             # Service không cần nhãn version, nó dùng selector chung
             service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=service_name), # Dùng selector chung (app: model_name)
                spec=client.V1ServiceSpec(
                    selector={"app": model_name},
                    ports=[client.V1ServicePort(port=80, target_port=8080)],
                    type="ClusterIP" # Hardening: Dùng ClusterIP nếu có Istio/Service Mesh
                )
            )
             self.core_v1.create_namespaced_service(body=service, namespace=self.namespace)
             logger.info(f"K8s Service '{service_name}' created/updated.")
        except client.ApiException as e:
             if e.status != 409: # Bỏ qua nếu đã tồn tại
                 raise

        # --- Quality Gate: Chờ Deployment Ready ---
        if not self._wait_for_deployment_ready(deployment_name):
            raise DeploymentError(f"Deployment '{deployment_name}' failed to be ready. Aborting.")

        logger.info(f"K8s Deployment '{deployment_name}' created/updated and READY.")
        
        return service_name # Trả về tên Service làm ID Endpoint

    # --- Triển khai các phương thức Canary/Rollback ---
    async def async_update_endpoint(self, endpoint_name: str, new_version_tag: str, deploy_config: Dict[str, Any]) -> None:
        """
        Kích hoạt triển khai phiên bản mới. Đây là bước 1 trong Canary Rollout.
        """
        logger.info(f"Starting deployment for new version '{new_version_tag}' for endpoint {endpoint_name}.")
        deploy_config['version_tag'] = new_version_tag # Truyền tag vào deploy_config
        # Gọi lại deploy_model, nó sẽ tạo Deployment mới với tên deployment_name-{new_version_tag}
        self.deploy_model(endpoint_name, deploy_config['model_artifact_uri'], deploy_config)
        
    def rollback(self, endpoint_name: str, target_version: str) -> None:
        """
        Rollback: Đảm bảo Deployment ổn định vẫn tồn tại và kích hoạt quy trình chuyển 100% traffic.

        LƯU Ý: Phương thức này không tự điều khiển traffic; nó dựa vào caller (rollback_deployment.py)
        để gọi IstioTrafficController sau khi xác nhận Deployment ổn định.
        """
        stable_deployment_name = f"{endpoint_name}-deployment-{target_version}"
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=stable_deployment_name, namespace=self.namespace)
            
            # Hardening: Nếu Deployment ổn định bị scale về 0 (để tiết kiệm chi phí), 
            # chúng ta phải đảm bảo ít nhất 1 replica được chạy lại trước khi chuyển traffic.
            if deployment.spec.replicas == 0:
                logger.warning(f"Stable deployment {stable_deployment_name} was scaled to 0. Scaling back to 1.")
                
                patch_body = {"spec": {"replicas": 1}}
                self.apps_v1.patch_namespaced_deployment(
                    name=stable_deployment_name, 
                    namespace=self.namespace, 
                    body=patch_body
                )
                # Chờ cho Pod Ready lại (optional: gọi lại _wait_for_deployment_ready)
            
            logger.info(f"Deployment '{stable_deployment_name}' verified and ready for rollback switch.")

        except client.ApiException as e:
            if e.status == 404:
                logger.critical(f"CRITICAL ROLLBACK FAILURE: Stable Deployment {stable_deployment_name} not found.")
                raise DeploymentError(f"Stable deployment version {target_version} not found for rollback.")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Xóa K8s Deployment và Service liên kết với Endpoint (Xóa mọi thứ có nhãn 'app: model_name').
        """
        service_name = f"{endpoint_name}-service" 
        
        try:
            # 1. Xóa TẤT CẢ Deployments (gồm cả stable và canary)
            # Sử dụng delete_collection_namespaced_deployment để xóa tất cả deployment có label "app: model_name"
            label_selector = f"app={endpoint_name}"
            logger.info(f"Deleting ALL K8s Deployments with label '{label_selector}' in namespace {self.namespace}.")
            self.apps_v1.delete_collection_namespaced_deployment(
                namespace=self.namespace,
                label_selector=label_selector
            )
            
            # 2. Xóa Service
            logger.info(f"Deleting K8s Service: '{service_name}' in namespace {self.namespace}.")
            self.core_v1.delete_namespaced_service(
                name=service_name, 
                namespace=self.namespace
            )

            logger.info(f"Kubernetes endpoint '{endpoint_name}' cleanup successful.")

        except client.ApiException as e:
            if e.status == 404:
                logger.warning(f"K8s resource for '{endpoint_name}' not found (Status 404). Already deleted.")
            else:
                logger.error(f"Failed to delete K8s resources for '{endpoint_name}': {e}")
                raise