# shared_libs/deployment/configs/deployment_schema.py (FINAL FULL SCHEMA)

# Cần cài đặt: pip install pydantic
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, conint, confloat

# -----------------------------------------------------
# 1. Base Components (Không đổi)
# -----------------------------------------------------

class ResourceLimits(BaseModel):
    """Định nghĩa tài nguyên CPU và Memory cho container."""
    cpu_cores: confloat(gt=0) = Field(..., description="CPU limits in cores (e.g., 1.0)")
    memory_mib: conint(gt=0) = Field(..., description="Memory limits in MiB (e.g., 2048)")


# -----------------------------------------------------
# 2. Hardening: Config cho các Nền tảng Đám mây
# -----------------------------------------------------

class KubernetesConfig(BaseModel):
    """Cấu hình cho K8s Deployer."""
    namespace: str = "mlops-production"
    readiness_path: str = "/health/ready"
    liveness_path: str = "/health/live"
    resource_limits: ResourceLimits
    secret_name: Optional[str] = Field(None, description="Key/Name of K8s Secret containing credentials or config.")

class SageMakerConfig(BaseModel):
    """Cấu hình cho AWS SageMaker Deployer."""
    region_name: str
    instance_type: str = "ml.m5.large"
    instance_count: conint(ge=1) = 1

# --- Hardening 5: Định nghĩa Config cho AzureML ---
class AzureConfig(BaseModel):
    """Cấu hình cho AzureML Deployer (Managed Endpoint)."""
    workspace_name: str = Field(..., description="Tên Azure ML Workspace.")
    resource_group: str
    compute_target: str = Field(..., description="Tên AzureML Compute Cluster/Instance.")
    sku_name: str = Field("Standard_DS3_v2", description="VM SKU cho Managed Endpoint.")
    instance_count: conint(ge=1) = 1

# --- Hardening 6: Định nghĩa Config cho GCP Vertex AI ---
class GcpConfig(BaseModel):
    """Cấu hình cho GCP Vertex AI Deployer (Managed Endpoint)."""
    project_id: str
    region: str
    machine_type: str = Field("n1-standard-4", description="GCP Machine Type cho Endpoint.")
    min_replicas: conint(ge=1) = 1
    max_replicas: conint(ge=1) = 3


# -----------------------------------------------------
# 3. Hardening: Config cho On-Premise
# -----------------------------------------------------

class OnPremiseConfig(BaseModel):
    """Cấu hình cho On-Premise Deployer."""
    
    # Hardening 7.1: Sử dụng Literal để giới hạn phương thức
    method: Literal["script", "api"] = Field(..., description="Phương thức triển khai: 'script' (shell) hoặc 'api' (gọi Internal API).")
    
    # Hardening 7.2: Tham số cấu hình tùy thuộc vào phương thức
    deployment_script_path: Optional[str] = Field(None, description="Đường dẫn tuyệt đối đến script triển khai (nếu method='script').")
    internal_api_endpoint: Optional[str] = Field(None, description="URL của Internal Deployment API (nếu method='api').")
    
    # Thêm cấu hình Secret Manager Key cho token API nội bộ
    api_secret_key: Optional[str] = Field(None, description="Tên Secret Key cho Internal API Token.")


# -----------------------------------------------------
# 4. Schema Chính (FINAL)
# -----------------------------------------------------

class DeploymentTargetSchema(BaseModel):
    """Schema cấu hình chung cho mọi nền tảng triển khai (Quality Gate)."""
    
    platform_type: Literal["kubernetes", "aws", "gcp", "azure", "on_premise"]
    endpoint_name: str = Field(..., description="Tên duy nhất của dịch vụ/endpoint.")
    
    # Cấu hình cụ thể cho từng nền tảng (Optional)
    kubernetes: Optional[KubernetesConfig] = None
    aws: Optional[SageMakerConfig] = None
    azure: Optional[AzureConfig] = None
    gcp: Optional[GcpConfig] = None
    on_premise: Optional[OnPremiseConfig] = None
    
    # Hardening 8: Thêm Custom Validation (Ví dụ cho Pydantic v2)
    # @model_validator(mode='after')
    # def check_platform_config(self) -> 'DeploymentTargetSchema':
    #     if self.platform_type == "kubernetes" and not self.kubernetes:
    #         raise ValueError("Must provide 'kubernetes' config when platform_type is 'kubernetes'")
    #     # ... tương tự cho các platform khác
    #     return self