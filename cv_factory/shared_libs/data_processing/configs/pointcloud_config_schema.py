# shared_libs/data_processing/configs/pointcloud_config_schema.py
import logging
from pydantic import Field, validator, NonNegativeInt, BaseModel, constr, conint
from typing import List, Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)

# --- Base Schema (Giả định được kế thừa từ một BaseConfig chung) ---
class BaseConfig(BaseModel):
    class Config:
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag để bật/tắt component này.")
    params: Optional[Dict[str, Any]] = Field(None, description="Các tham số cụ thể của component.")

# --- 1. Atomic Component Step Schema (Định nghĩa riêng cho Point Cloud) ---

class PointcloudComponentStepConfig(BaseConfig):
    """Schema cho một bước xử lý Point Cloud (Loader, Normalizer, Augmenter, Voxelizer)."""
    type: constr(to_lower=True) = Field(..., description="Tên/loại component PC (ví dụ: 'pc_loader', 'pc_voxelizer').")
    
    @validator('type')
    def validate_known_pc_component(cls, v):
        """Xác thực rằng loại component Point Cloud được hỗ trợ."""
        supported_pc_types = [
            'pc_loader', 'pc_normalizer', 'pc_augmenter', 'pc_voxelizer'
        ]
        if v not in supported_pc_types:
            raise ValueError(f"Unknown Point Cloud component type: {v}. Must be one of: {', '.join(supported_pc_types)}")
        return v

    @validator('params')
    def validate_voxelizer_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Thi hành các quy tắc consistency cho PC Voxelizer."""
        if values.get('type') == 'pc_voxelizer' and v and v.get('enabled', False):
            if 'voxel_size' not in v or v.get('voxel_size', 0.0) <= 0:
                raise ValueError("PC Voxelizer: 'voxel_size' phải là giá trị dương.")
            
            grid_shape = v.get('grid_shape')
            if not isinstance(grid_shape, list) or len(grid_shape) != 3:
                raise ValueError("PC Voxelizer: 'grid_shape' phải là list có 3 phần tử (L, W, H).")
            
        return v
        
# --- 2. Point Cloud Processing Configuration (Schema Chính) ---

class PointcloudProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Point Cloud Processing pipeline. 
    Quản lý luồng xử lý dữ liệu 3D rời rạc (LiDAR, RGB-D).
    """
    
    # 1. Policy Mode: 
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy mode: 'default' sequential execution, hoặc 'conditional_metadata' (adaptive)."
    )
    
    # 2. Các bước xử lý cụ thể (Sử dụng tên chuẩn hóa 'pc_')
    
    loader: PointcloudComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho PointcloudLoader (tải, downsampling/padding). Loại: 'pc_loader'."
    )
    
    normalizer: PointcloudComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho PointcloudNormalizer (Centering, Scaling, Intensity). Loại: 'pc_normalizer'."
    )
    
    augmenter: Optional[PointcloudComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho PointcloudAugmenter (Rotation 3D, Jittering). Loại: 'pc_augmenter'."
    )
    
    # Voxelizer thường là bước cuối cùng và tùy chọn, chuyển đổi PC sang Grid
    voxelizer: Optional[PointcloudComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho PointcloudVoxelizer (chuyển đổi sang 3D Grid). Loại: 'pc_voxelizer'."
    )

    # 3. Quy tắc Governance
    
    @validator('loader', 'normalizer', 'augmenter', 'voxelizer', always=True)
    def validate_pc_component_types(cls, v: Optional[PointcloudComponentStepConfig], values: Dict[str, Any], field: Any) -> Optional[PointcloudComponentStepConfig]:
        """Rule: Đảm bảo các component Point Cloud có type đúng."""
        if v is None:
            return v
        
        expected_type = f"pc_{field.name}"
        if v.type != expected_type:
            raise ValueError(
                f"PointcloudProcessingConfig field '{field.name}' requires type '{expected_type}', but got '{v.type}'."
            )
        return v

    @validator('loader')
    def validate_loader_params(cls, v: PointcloudComponentStepConfig) -> PointcloudComponentStepConfig:
        """Rule: Point Cloud Loader phải có max_points được định nghĩa."""
        if not v.params or 'max_points' not in v.params:
            raise ValueError("PointcloudLoader requires 'max_points' parameter for fixed-size output.")
        return v